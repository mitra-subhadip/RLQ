from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from rl_qtranspiler.preprocessing import (
    preprocess_for_swap_routing,
    restore_without_routing,
)


def test_extract_fuse_and_restore_equivalence():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cz(1, 0)
    circuit.rz(0.4, 2)
    circuit.cx(1, 2)
    circuit.x(1)
    circuit.cx(1, 2)

    result = preprocess_for_swap_routing(circuit)
    restored = restore_without_routing(result)

    assert len(result.one_qubit_gates) == 3
    assert len(result.instructions) == 3
    assert result.instructions[0].source_indices == (1, 2)
    assert result.instructions[1].source_indices == (4,)
    assert result.instructions[2].source_indices == (6,)
    assert Operator(circuit).equiv(Operator(restored))


def test_removed_gate_breaks_fusion_adjacency():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.cz(0, 1)
    result = preprocess_for_swap_routing(circuit)
    assert len(result.instructions) == 2
