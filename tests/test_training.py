from pathlib import Path

from qiskit import QuantumCircuit

from rl_qtranspiler.curriculum import (
    Curriculum,
    expert_trajectory,
    solve_exact,
    supervised_warm_start,
)
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.model import GraphDQN
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem
from rl_qtranspiler.trainer import DoubleDQNTrainer, TrainerConfig


def test_default_curriculum_advances_after_two_stale_evaluations():
    curriculum = Curriculum()

    assert not curriculum.observe(0.1)
    assert not curriculum.observe(0.1)
    assert curriculum.observe(0.1)
    assert curriculum.current.name == "medium"


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
    assert len(losses) == 1
    assert losses[0] >= 0


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
