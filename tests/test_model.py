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
