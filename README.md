# RL-QTranspiler

Calibration-aware initial qubit placement for the 156-qubit IBM Boston Heron r3
topology. The agent uses an edge-aware graph Double DQN to assign logical
qubits sequentially. SWAP insertion is intentionally left to a downstream
router.

## Setup

```bash
conda activate qiskit
python -m pip install -e ".[dev]"
```

## Core API

```python
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.inference import place
from rl_qtranspiler.model import GraphDQN
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem

preprocessed = preprocess_for_swap_routing(circuit)
problem = build_placement_problem(preprocessed, load_ibm_boston())
result = place(trained_model, problem, beam_width=8)
print(result.logical_to_physical)
```

Train with `train_placement.py`; place an OpenQASM circuit with
`place_circuit.py`; compare identity, random, degree, greedy, SABRE
basic/lookahead/decay, and Graph-DQN mappings with `benchmark_placement.py`.
Installed wheels also provide `rlq-train`, `rlq-place`, and `rlq-benchmark`.
The trainer uses prioritized replay, batched Double DQN updates, dueling action
values, temporal interaction weights, bounded problem retention, and the
staged curriculum described in the implementation plan.

Resume a complete checkpoint—including replay, RNG, curriculum, and validation
state—with:

```bash
rlq-train --checkpoint placement_dqn.pt --resume
```

The calibration snapshot is static. Refresh
`ibm_boston_connectivity_snapshot.py` before experiments that require current
hardware errors.
