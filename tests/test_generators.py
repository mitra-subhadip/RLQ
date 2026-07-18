import pytest

from rl_qtranspiler.generators import generate_layered_random_circuit


def test_layered_random_circuit_has_requested_size_depth_and_gate_width():
    circuit = generate_layered_random_circuit(
        50,
        40,
        two_qubit_density=0.6,
        seed=7,
    )

    assert circuit.num_qubits == 50
    assert circuit.depth() == 40
    assert max(instruction.operation.num_qubits for instruction in circuit.data) == 2
    assert circuit.count_ops()["cz"] == 15 * 40


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_qubits": 1, "depth": 40}, "at least two"),
        ({"num_qubits": 50, "depth": 0}, "positive"),
        (
            {"num_qubits": 50, "depth": 40, "two_qubit_density": 1.1},
            r"\[0, 1\]",
        ),
    ],
)
def test_layered_random_circuit_validates_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_layered_random_circuit(**kwargs)
