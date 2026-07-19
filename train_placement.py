"""Train the graph DQN on generated placement problems."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from rl_qtranspiler.curriculum import (
    Curriculum,
    CurriculumStage,
    expert_action_accuracy,
    supervised_warm_start,
)
from rl_qtranspiler.generators import SUPPORTED_FAMILIES, generate_circuit
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.model import GraphDQN
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import PlacementProblem, build_placement_problem
from rl_qtranspiler.trainer import DoubleDQNTrainer


def connected_nodes(
    graph: nx.Graph, count: int, generator: random.Random
) -> tuple[int, ...]:
    start = generator.randrange(graph.number_of_nodes())
    selected = [start]
    seen = {start}
    queue = [start]
    while queue and len(selected) < count:
        node = queue.pop(0)
        neighbors = list(graph.neighbors(node))
        generator.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.append(neighbor)
            queue.append(neighbor)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError("Could not sample a sufficiently large subgraph.")
    return tuple(sorted(selected))


def make_problem(
    num_qubits: int,
    index: int,
    *,
    hardware,
    family: str,
    seed: int,
    exact_subgraph: bool = False,
) -> PlacementProblem:
    circuit = generate_circuit(
        family, num_qubits, layers=4, seed=seed + index
    )
    preprocessed = preprocess_for_swap_routing(circuit)
    allowed = None
    if exact_subgraph:
        allowed = connected_nodes(
            hardware.as_networkx(),
            num_qubits,
            random.Random(seed + index),
        )
    return build_placement_problem(
        preprocessed,
        hardware,
        problem_id=f"{family}-{num_qubits}-{index}",
        allowed_physical_qubits=allowed,
    )


def validation_qubit_counts(
    stage: CurriculumStage,
    count: int = len(SUPPORTED_FAMILIES),
) -> tuple[int, ...]:
    """Return deterministic sizes spanning the complete curriculum stage."""
    if count <= 0:
        raise ValueError("Validation problem count must be positive.")
    if count == 1:
        return ((stage.minimum_qubits + stage.maximum_qubits) // 2,)
    span = stage.maximum_qubits - stage.minimum_qubits
    return tuple(
        round(stage.minimum_qubits + span * index / (count - 1))
        for index in range(count)
    )


def validation_set_matches_stage(
    problems: list[PlacementProblem],
    stage: CurriculumStage,
) -> bool:
    return tuple(
        problem.num_logical_qubits for problem in problems
    ) == validation_qubit_counts(stage)


def make_validation_problems(
    stage: CurriculumStage,
    *,
    hardware,
    seed: int,
) -> list[PlacementProblem]:
    return [
        make_problem(
            num_qubits,
            1_000_000 + index,
            hardware=hardware,
            family=family,
            seed=seed,
        )
        for index, (num_qubits, family) in enumerate(
            zip(
                validation_qubit_counts(stage),
                SUPPORTED_FAMILIES,
                strict=True,
            )
        )
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (default: CUDA, then MPS, then CPU).",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("placement_dqn.pt"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, replay, RNG, curriculum, and validation state.",
    )
    parser.add_argument("--warm-start-problems", type=int, default=3)
    parser.add_argument("--warm-start-epochs", type=int, default=5)
    parser.add_argument(
        "--warm-start-batch-size",
        type=int,
        default=128,
        help=(
            "Warm-start examples per optimizer update (default: 128)."
        ),
    )
    parser.add_argument(
        "--warm-start-updates",
        type=int,
        default=None,
        help=(
            "Total warm-start optimizer updates; by default preserve the "
            "legacy epochs-times-examples budget."
        ),
    )
    parser.add_argument("--evaluation-interval", type=int, default=1000)
    parser.add_argument("--curriculum-patience", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    hardware = load_ibm_boston()
    trainer = DoubleDQNTrainer(GraphDQN(), device=arguments.device)
    curriculum = Curriculum(patience=arguments.curriculum_patience)
    validation_problems: list[PlacementProblem] = []
    episode = 0
    next_evaluation = arguments.evaluation_interval

    if arguments.resume:
        if not arguments.checkpoint.exists():
            raise FileNotFoundError(arguments.checkpoint)
        extra = trainer.load(arguments.checkpoint)
        if "curriculum" in extra:
            curriculum.load_state_dict(extra["curriculum"])
            curriculum.patience = arguments.curriculum_patience
        validation_problems = list(extra.get("validation_problems", []))
        if not validation_set_matches_stage(
            validation_problems, curriculum.current
        ):
            validation_problems.clear()
            curriculum.reset_validation_tracking()
            sizes = validation_qubit_counts(curriculum.current)
            print(
                "Reset incompatible validation set; "
                f"stage={curriculum.current.name} sizes={sizes}"
            )
        episode = int(extra.get("episode", 0))
        next_evaluation = int(
            extra.get(
                "next_evaluation",
                trainer.environment_steps + arguments.evaluation_interval,
            )
        )
        print(
            f"Resumed step={trainer.environment_steps} episode={episode} "
            f"stage={curriculum.current.name} replay={len(trainer.replay)}"
        )
    elif arguments.warm_start_problems:
        exact_problems = [
            make_problem(
                4 + index % 5,
                index,
                hardware=hardware,
                family=SUPPORTED_FAMILIES[index % len(SUPPORTED_FAMILIES)],
                seed=arguments.seed,
                exact_subgraph=True,
            )
            for index in range(arguments.warm_start_problems)
        ]
        losses = supervised_warm_start(
            trainer,
            exact_problems,
            epochs=arguments.warm_start_epochs,
            batch_size=arguments.warm_start_batch_size,
            updates=arguments.warm_start_updates,
        )
        teacher_accuracy = expert_action_accuracy(
            trainer,
            exact_problems,
            batch_size=arguments.warm_start_batch_size,
        )
        if losses:
            print(
                f"Warm-start loss: {losses[-1]:.6f} "
                f"updates={len(losses)} "
                f"teacher_accuracy={teacher_accuracy:.3f}"
            )
        else:
            print(
                "Warm start completed without optimizer updates; "
                f"teacher_accuracy={teacher_accuracy:.3f}"
            )

    while trainer.environment_steps < arguments.environment_steps:
        stage = curriculum.current
        num_qubits = random.randint(stage.minimum_qubits, stage.maximum_qubits)
        family = SUPPORTED_FAMILIES[episode % len(SUPPORTED_FAMILIES)]
        problem = make_problem(
            num_qubits,
            episode,
            hardware=hardware,
            family=family,
            seed=arguments.seed,
        )
        reward = trainer.collect_episode(problem)

        if trainer.environment_steps >= next_evaluation:
            if not validation_problems:
                validation_problems = make_validation_problems(
                    stage,
                    hardware=hardware,
                    seed=arguments.seed,
                )
                print(
                    f"Validation stage={stage.name} sizes="
                    f"{validation_qubit_counts(stage)}"
                )
            validation = -float(
                np.mean(
                    [
                        trainer.collect_episode(
                            item, explore=False, optimize=False
                        )
                        for item in validation_problems
                    ]
                )
            )
            advanced = (
                curriculum.observe(validation)
                if trainer.has_started_learning
                else False
            )
            if advanced:
                validation_problems.clear()
            print(
                f"step={trainer.environment_steps} episode={episode} "
                f"stage={curriculum.current.name} "
                f"reward={reward:.5f} validation={validation:.5f} "
                f"epsilon={trainer.epsilon:.3f} replay={len(trainer.replay)}"
            )
            trainer.save(
                arguments.checkpoint,
                extra_state={
                    "curriculum": curriculum.state_dict(),
                    "validation_problems": validation_problems,
                    "episode": episode + 1,
                    "next_evaluation": (
                        next_evaluation + arguments.evaluation_interval
                    ),
                },
            )
            next_evaluation += arguments.evaluation_interval
        episode += 1
    trainer.save(
        arguments.checkpoint,
        extra_state={
            "curriculum": curriculum.state_dict(),
            "validation_problems": validation_problems,
            "episode": episode,
            "next_evaluation": next_evaluation,
        },
    )


if __name__ == "__main__":
    main()
