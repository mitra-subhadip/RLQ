"""Circuit preprocessing used by the placement and routing stages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator


@dataclass(frozen=True)
class StoredOneQGate:
    operation: Gate
    logical_qubit: int
    slot: int
    source_index: int


@dataclass(frozen=True)
class BackboneInstruction:
    operation: Any
    qargs: tuple[int, ...]
    cargs: tuple[int, ...]
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessingResult:
    original: QuantumCircuit
    backbone: QuantumCircuit
    one_qubit_gates: tuple[StoredOneQGate, ...]
    instructions: tuple[BackboneInstruction, ...]


def _is_gate_of_width(operation: Any, width: int) -> bool:
    return isinstance(operation, Gate) and operation.num_qubits == width


def _fuse_two_qubit_gates(
    left: BackboneInstruction,
    right_operation: Gate,
    right_qargs: tuple[int, int],
) -> UnitaryGate:
    local = QuantumCircuit(2)
    local.append(left.operation, [0, 1])
    right_local_qargs = [left.qargs.index(qubit) for qubit in right_qargs]
    local.append(right_operation, right_local_qargs)
    label = (
        f"fused({getattr(left.operation, 'name', 'gate')},"
        f"{right_operation.name})"
    )
    return UnitaryGate(Operator(local).data, label=label)


def preprocess_for_swap_routing(circuit: QuantumCircuit) -> PreprocessingResult:
    """Extract one-qubit gates and fuse adjacent two-qubit gates safely."""
    entries: list[BackboneInstruction] = []
    stored: list[StoredOneQGate] = []
    previous_input_was_two_qubit_gate = False

    for source_index, instruction in enumerate(circuit.data):
        operation = instruction.operation
        qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        cargs = tuple(circuit.find_bit(c).index for c in instruction.clbits)

        if _is_gate_of_width(operation, 1) and not cargs:
            stored.append(
                StoredOneQGate(
                    operation=operation,
                    logical_qubit=qargs[0],
                    slot=len(entries),
                    source_index=source_index,
                )
            )
            previous_input_was_two_qubit_gate = False
            continue

        is_two_qubit_gate = _is_gate_of_width(operation, 2) and not cargs
        same_pair_as_previous = (
            previous_input_was_two_qubit_gate
            and bool(entries)
            and len(entries[-1].qargs) == 2
            and set(entries[-1].qargs) == set(qargs)
        )

        if is_two_qubit_gate and same_pair_as_previous:
            try:
                fused = _fuse_two_qubit_gates(entries[-1], operation, qargs)
            except (QiskitError, TypeError, ValueError):
                entries.append(
                    BackboneInstruction(operation, qargs, cargs, (source_index,))
                )
            else:
                entries[-1] = replace(
                    entries[-1],
                    operation=fused,
                    source_indices=entries[-1].source_indices + (source_index,),
                )
        else:
            entries.append(
                BackboneInstruction(operation, qargs, cargs, (source_index,))
            )
        previous_input_was_two_qubit_gate = is_two_qubit_gate

    backbone = circuit.copy_empty_like(name=f"{circuit.name}_2q_backbone")
    for entry in entries:
        backbone.append(entry.operation, list(entry.qargs), list(entry.cargs))

    return PreprocessingResult(
        original=circuit.copy(),
        backbone=backbone,
        one_qubit_gates=tuple(stored),
        instructions=tuple(entries),
    )


def restore_without_routing(result: PreprocessingResult) -> QuantumCircuit:
    """Restore extracted gates when no physical remapping has occurred."""
    restored = result.backbone.copy_empty_like(name="restored_preprocessed")
    gates_by_slot: dict[int, list[StoredOneQGate]] = {}
    for gate in result.one_qubit_gates:
        gates_by_slot.setdefault(gate.slot, []).append(gate)

    for slot in range(len(result.instructions) + 1):
        for gate in gates_by_slot.get(slot, []):
            restored.append(gate.operation, [gate.logical_qubit])
        if slot < len(result.instructions):
            entry = result.instructions[slot]
            restored.append(entry.operation, list(entry.qargs), list(entry.cargs))
    return restored


def mapped_one_qubit_gates_for_slot(
    result: PreprocessingResult,
    slot: int,
    logical_to_physical: Mapping[int, int] | Sequence[int],
) -> list[tuple[Gate, int]]:
    return [
        (gate.operation, logical_to_physical[gate.logical_qubit])
        for gate in result.one_qubit_gates
        if gate.slot == slot
    ]
