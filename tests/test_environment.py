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
    assert np.isclose(problem.interaction_weights[0, 1], 1.0 + 0.99)
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


def test_greedy_mapping_starts_from_a_top_five_degree_anchor():
    problem = make_problem()
    mapping = greedy_mapping(problem)
    degree = problem.hardware.static_node_features[:, 0]
    allowed = np.flatnonzero(problem.allowed_physical_mask).tolist()
    top_five = sorted(
        allowed,
        key=lambda physical: (-float(degree[physical]), physical),
    )[:5]

    first_logical = problem.placement_order[0]
    assert mapping[first_logical] in top_five


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


def test_barriers_are_not_interactions_and_fused_gates_advance_time():
    hardware = load_ibm_boston()
    barrier_circuit = QuantumCircuit(2)
    barrier_circuit.barrier(0, 1)
    barrier_problem = build_placement_problem(
        preprocess_for_swap_routing(barrier_circuit), hardware
    )
    assert barrier_problem.interaction_weights[0, 1] == 0

    fused = QuantumCircuit(2)
    fused.cz(0, 1)
    fused.cz(0, 1)
    fused_problem = build_placement_problem(
        preprocess_for_swap_routing(fused),
        hardware,
        temporal_discount=0.5,
    )
    assert np.isclose(fused_problem.interaction_weights[0, 1], 1.5)


def test_disjoint_gate_order_does_not_change_temporal_weights():
    hardware = load_ibm_boston()

    first = QuantumCircuit(4)
    first.cz(0, 1)
    first.cz(2, 3)
    first.cz(1, 2)

    reordered = QuantumCircuit(4)
    reordered.cz(2, 3)
    reordered.cz(0, 1)
    reordered.cz(1, 2)

    first_problem = build_placement_problem(
        preprocess_for_swap_routing(first),
        hardware,
        temporal_discount=0.5,
    )
    reordered_problem = build_placement_problem(
        preprocess_for_swap_routing(reordered),
        hardware,
        temporal_discount=0.5,
    )

    assert np.array_equal(
        first_problem.interaction_weights,
        reordered_problem.interaction_weights,
    )
    assert first_problem.interaction_weights[0, 1] == 1.0
    assert first_problem.interaction_weights[2, 3] == 1.0
    assert first_problem.interaction_weights[1, 2] == 0.5


def problem_id_to_circuit():
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(0, 1)
    circuit.h(3)
    circuit.cz(1, 2)
    return circuit
