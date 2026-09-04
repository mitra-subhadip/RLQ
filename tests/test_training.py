from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from rl_qtranspiler.curriculum import (
    Curriculum,
    expert_action_accuracy,
    expert_trajectory,
    solve_exact,
    supervised_warm_start,
)
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.evaluation import RoutedMetrics
from rl_qtranspiler.model import GraphDQN
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem
from rl_qtranspiler.trainer import DoubleDQNTrainer, TrainerConfig
from train_placement import (
    validation_qubit_counts,
    validation_set_matches_stage,
)


def test_default_curriculum_advances_after_five_stale_evaluations():
    curriculum = Curriculum()

    assert not curriculum.observe(0.1)
    for _ in range(4):
        assert not curriculum.observe(0.1)
    assert curriculum.observe(0.1)
    assert curriculum.current.name == "medium"


def test_validation_sizes_span_each_curriculum_stage():
    curriculum = Curriculum()
    expected = {
        "small": (4, 6, 8),
        "medium": (9, 12, 16),
        "large": (17, 86, 156),
        "mixed": (4, 80, 156),
    }

    for stage in curriculum.stages:
        sizes = validation_qubit_counts(stage)
        assert sizes == expected[stage.name]
        assert sizes[0] == stage.minimum_qubits
        assert sizes[-1] == stage.maximum_qubits
        problems = [
            SimpleNamespace(num_logical_qubits=size)
            for _family in range(6)
            for size in sizes
        ]
        assert validation_set_matches_stage(problems, stage)


def test_reset_validation_tracking_for_new_distribution():
    curriculum = Curriculum()
    curriculum.observe(0.1)
    curriculum.observe(0.2)

    curriculum.reset_validation_tracking()

    assert curriculum.best_validation_score == float("inf")
    assert curriculum.stale_evaluations == 0


def test_double_dqn_update_and_checkpoint(tmp_path: Path):
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="training-smoke",
    )
    config = TrainerConfig(
        replay_capacity=32,
        warmup_transitions=2,
        batch_size=2,
        target_update_steps=1,
        epsilon_decay_steps=10,
        training_frequency=1,
        routed_reward_enabled=False,
    )
    trainer = DoubleDQNTrainer(GraphDQN(hidden_dim=16), config=config)
    reward = trainer.collect_episode(problem)

    assert reward <= 0
    assert len(trainer.replay) == 2
    assert trainer.optimization_steps == 1

    checkpoint = tmp_path / "smoke.pt"
    trainer.save(checkpoint, extra_state={"episode": 7})
    restored = DoubleDQNTrainer(GraphDQN(hidden_dim=16), config=config)
    extra = restored.load(checkpoint)
    assert restored.environment_steps == trainer.environment_steps
    assert restored.optimization_steps == trainer.optimization_steps
    assert len(restored.replay) == len(trainer.replay)
    assert restored.config == config
    assert set(restored.problems) == set(trainer.problems)
    assert extra == {"episode": 7}


def test_routed_terminal_correction_replaces_proxy_return(monkeypatch):
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="routed-reward",
    )
    routed = RoutedMetrics(
        swap_count=2,
        two_qubit_gate_count=7,
        depth=9,
        estimated_log_infidelity=0.123,
        normalized_log_infidelity=0.123,
        estimated_success_probability=float(np.exp(-0.123)),
        runtime_seconds=0.01,
    )
    monkeypatch.setattr(
        "rl_qtranspiler.trainer.evaluate_problem_mapping",
        lambda *args, **kwargs: routed,
    )
    trainer = DoubleDQNTrainer(
        GraphDQN(hidden_dim=8),
        config=TrainerConfig(
            replay_capacity=8,
            warmup_transitions=100,
            batch_size=2,
        ),
    )

    outcome = trainer.run_episode(problem, optimize=False)
    stored_rewards = [
        transition.reward
        for transition in trainer.replay._data
        if transition is not None
    ]

    assert outcome.routed_metrics is routed
    assert np.isclose(outcome.total_reward, -0.123)
    assert np.isclose(sum(stored_rewards), -0.123)
    assert trainer.training_episodes == 1


def test_large_routed_reward_frequency_is_respected():
    circuit = QuantumCircuit(17)
    circuit.cz(0, 1)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="routing-frequency",
    )
    trainer = DoubleDQNTrainer(
        GraphDQN(hidden_dim=8),
        config=TrainerConfig(large_routed_reward_frequency=5),
    )

    assert trainer._should_route_training_episode(problem)
    trainer.training_episodes = 1
    assert not trainer._should_route_training_episode(problem)
    trainer.training_episodes = 5
    assert trainer._should_route_training_episode(problem)


def test_old_reward_checkpoint_is_rejected(tmp_path: Path):
    config = TrainerConfig(replay_capacity=8, batch_size=2)
    trainer = DoubleDQNTrainer(GraphDQN(hidden_dim=8), config=config)
    checkpoint = tmp_path / "new.pt"
    old_checkpoint = tmp_path / "old.pt"
    trainer.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop("reward_version")
    torch.save(payload, old_checkpoint)

    restored = DoubleDQNTrainer(GraphDQN(hidden_dim=8), config=config)
    with pytest.raises(ValueError, match="old placement-proxy reward"):
        restored.load(old_checkpoint)


def test_exact_teacher_and_supervised_warm_start():
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="exact-smoke",
        allowed_physical_qubits=(0, 1),
    )
    solution = solve_exact(problem)
    trajectory = expert_trajectory(problem, solution)
    trainer = DoubleDQNTrainer(
        GraphDQN(hidden_dim=16),
        config=TrainerConfig(replay_capacity=8, batch_size=2),
    )
    losses = supervised_warm_start(trainer, [problem], epochs=1)

    assert len(set(solution.logical_to_physical)) == 2
    assert len(trajectory) == 2
    assert len(losses) == len(trajectory)
    assert losses[0] >= 0
    optimizer_steps = {
        int(state["step"].item())
        for state in trainer.optimizer.state.values()
    }
    assert optimizer_steps == {len(trajectory)}
    accuracy = expert_action_accuracy(trainer, [problem])
    assert 0.0 <= accuracy <= 1.0

    batched_trainer = DoubleDQNTrainer(
        GraphDQN(hidden_dim=16),
        config=TrainerConfig(replay_capacity=8, batch_size=2),
    )
    supervised_warm_start(
        batched_trainer,
        [problem],
        epochs=1,
        batch_size=2,
        updates=1,
    )
    batched_steps = {
        int(state["step"].item())
        for state in batched_trainer.optimizer.state.values()
    }
    assert batched_steps == {1}


def test_problem_registry_tracks_only_replay_references():
    hardware = load_ibm_boston()
    config = TrainerConfig(
        replay_capacity=4,
        warmup_transitions=100,
        batch_size=2,
    )
    trainer = DoubleDQNTrainer(GraphDQN(hidden_dim=8), config=config)
    for index in range(6):
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        problem = build_placement_problem(
            preprocess_for_swap_routing(circuit),
            hardware,
            problem_id=f"bounded-{index}",
        )
        trainer.collect_episode(problem, optimize=False)

    referenced = {
        transition.problem_id
        for transition in trainer.replay._data
        if transition is not None
    }
    assert set(trainer.problems) == referenced
    assert len(trainer.problems) <= 2
