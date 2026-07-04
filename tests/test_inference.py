import numpy as np
from qiskit import QuantumCircuit

from rl_qtranspiler.environment import PlacementEnvironment
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.inference import place
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem


class CentralityModel:
    def action_values(self, problem, state):
        values = problem.hardware.static_node_features[:, 0].copy()
        values[~problem.valid_action_mask(state.physical_to_logical)] = -np.inf
        return values


def test_beam_result_is_valid_and_not_worse_than_greedy():
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="inference-test",
    )
    model = CentralityModel()
    greedy = place(model, problem, beam_width=1, expansions_per_state=1)
    beam = place(model, problem, beam_width=4, expansions_per_state=3)

    assert len(set(beam.logical_to_physical.values())) == 4
    assert beam.combined_score <= greedy.combined_score + 1e-12
    score = PlacementEnvironment(problem).score_mapping(
        tuple(beam.logical_to_physical[index] for index in range(4))
    )
    assert np.isclose(score.combined, beam.combined_score)
