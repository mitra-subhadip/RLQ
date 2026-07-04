import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch not installed")


def test_graph_dqn_forward_and_mask():
    import torch
    from qiskit import QuantumCircuit

    from rl_qtranspiler.environment import PlacementEnvironment
    from rl_qtranspiler.hardware import load_ibm_boston
    from rl_qtranspiler.model import GraphDQN
    from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
    from rl_qtranspiler.problem import build_placement_problem

    circuit = QuantumCircuit(3)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit), load_ibm_boston()
    )
    state = PlacementEnvironment(problem).reset()
    model = GraphDQN(hidden_dim=32)
    q_values = model(problem, state)
    assert q_values.shape == (156,)
    assert torch.isfinite(q_values).all()

    environment = PlacementEnvironment(problem)
    environment.step(0)
    q_values = model(problem, environment.state)
    assert q_values[0] < -1e20


def test_batched_forward_matches_individual_and_encodes_allowed_nodes():
    import numpy as np
    import torch
    from qiskit import QuantumCircuit

    from rl_qtranspiler.environment import (
        PlacementEnvironment,
        build_state_features,
    )
    from rl_qtranspiler.hardware import load_ibm_boston
    from rl_qtranspiler.model import GraphDQN
    from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
    from rl_qtranspiler.problem import build_placement_problem

    circuit = QuantumCircuit(3)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        allowed_physical_qubits=(0, 1, 2),
    )
    environment = PlacementEnvironment(problem)
    state_a = environment.reset()
    environment.step(0)
    state_b = environment.state
    physical_features, _ = build_state_features(problem, state_a)

    assert np.array_equal(
        physical_features[:, 7], problem.allowed_physical_mask
    )
    model = GraphDQN(hidden_dim=16).eval()
    with torch.no_grad():
        batched = model.forward_batch(
            [problem, problem], [state_a, state_b]
        )
        individual_a = model(problem, state_a)
        individual_b = model(problem, state_b)
    assert torch.allclose(batched[0], individual_a, atol=1e-6)
    assert torch.allclose(batched[1], individual_b, atol=1e-6)
    assert torch.all(batched[:, 3:] < -1e20)


def test_swapping_assigned_logical_identities_changes_observation():
    import torch
    from qiskit import QuantumCircuit

    from rl_qtranspiler.environment import PlacementState
    from rl_qtranspiler.hardware import load_ibm_boston
    from rl_qtranspiler.model import GraphDQN
    from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
    from rl_qtranspiler.problem import build_placement_problem

    circuit = QuantumCircuit(5)
    circuit.cz(0, 2)
    circuit.cz(1, 2)
    circuit.cz(0, 3)
    circuit.cz(1, 4)
    circuit.cz(3, 4)
    circuit.cz(2, 3)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit), load_ibm_boston()
    )
    first, second = 0, 1
    logical_a = [-1] * problem.num_logical_qubits
    logical_b = [-1] * problem.num_logical_qubits
    physical_a = [-1] * problem.hardware.num_qubits
    physical_b = [-1] * problem.hardware.num_qubits
    logical_a[first], logical_a[second] = 0, 10
    logical_b[first], logical_b[second] = 10, 0
    physical_a[0], physical_a[10] = first, second
    physical_b[0], physical_b[10] = second, first
    state_a = PlacementState(tuple(logical_a), tuple(physical_a), 2)
    state_b = PlacementState(tuple(logical_b), tuple(physical_b), 2)

    torch.manual_seed(3)
    model = GraphDQN(hidden_dim=16).eval()
    with torch.no_grad():
        values = model.forward_batch(
            [problem, problem], [state_a, state_b]
        )
    assert not torch.allclose(values[0], values[1])
