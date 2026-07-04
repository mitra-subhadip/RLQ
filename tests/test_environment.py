import numpy as np
from qiskit import QuantumCircuit

from rl_qtranspiler.baselines import greedy_mapping, sabre_mapping
from rl_qtranspiler.environment import PlacementEnvironment
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem


def make_problem():
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(0, 1)
    circuit.h(3)
    circuit.cz(1, 2)
    return build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="environment-test",
    )


def test_problem_weights_order_and_isolated_qubit():
    problem = make_problem()
    assert problem.interaction_counts[0, 1] == 2
    assert np.isclose(problem.interaction_weights[0, 1], 2.0)
    assert problem.placement_order[-1] == 3
    assert set(problem.placement_order) == {0, 1, 2, 3}


def test_rewards_sum_to_negative_terminal_score():
    problem = make_problem()
    mapping = greedy_mapping(problem)
    environment = PlacementEnvironment(problem)
    state = environment.reset()
    rewards = []
    for logical in problem.placement_order:
        state, reward, _, _ = environment.step(mapping[logical])
        rewards.append(reward)

    independent = environment.score_mapping(mapping)
    assert state.done
    assert len(set(mapping)) == problem.num_logical_qubits
    assert np.isclose(sum(rewards), -independent.combined)
    assert np.isclose(environment.score.combined, independent.combined)


def test_invalid_occupied_action_is_rejected():
    problem = make_problem()
    environment = PlacementEnvironment(problem)
    environment.step(0)
    try:
        environment.step(0)
    except ValueError:
        pass
    else:
        raise AssertionError("An occupied action should be rejected.")


def test_sabre_baseline_produces_injective_mapping():
    problem = make_problem()
    mapping = sabre_mapping(
        problem_id_to_circuit(), problem, heuristic="lookahead"
    )
    assert len(mapping) == problem.num_logical_qubits
    assert len(set(mapping)) == problem.num_logical_qubits


def problem_id_to_circuit():
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(0, 1)
    circuit.h(3)
    circuit.cz(1, 2)
    return circuit
