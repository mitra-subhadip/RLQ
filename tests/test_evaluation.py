import numpy as np
from qiskit import qasm2

from rl_qtranspiler.evaluation import evaluate_routed_circuit
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem


def test_legacy_cu1_routes_on_undirected_boston_graph():
    circuit = qasm2.loads(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[0];
        cu1(pi/2) q[1],q[0];
        cu1(pi/4) q[2],q[0];
        """
    )
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        load_ibm_boston(),
        problem_id="cu1-direction-regression",
    )
    metrics = evaluate_routed_circuit(circuit, problem, (0, 1, 2))
    assert metrics.two_qubit_gate_count >= 2
    assert problem.original_two_qubit_gate_count == 2
    assert np.isclose(
        metrics.normalized_log_infidelity,
        metrics.estimated_log_infidelity / 2,
    )
    assert np.isclose(
        metrics.estimated_success_probability,
        np.exp(-metrics.estimated_log_infidelity),
    )
