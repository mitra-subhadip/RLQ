# RL-QTranspiler

Calibration-aware initial qubit placement for the 156-qubit IBM Boston Heron r3
topology. The agent uses an edge-aware graph Double DQN to assign logical
qubits sequentially. SWAP insertion is intentionally left to a downstream
router.

## Setup

Ensure you're using the pip>=26.1

```bash
conda activate <environment-name>
python -m pip install -r pylock.toml
python -m pip install -e
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
Training automatically selects CUDA, then Apple MPS, then CPU; pass
`--device cpu` (or another explicit PyTorch device) to override it.
Warm-start training batches up to 128 expert examples while preserving the
legacy optimizer-update budget by default (75 updates with the standard three
problems and five epochs). Use `--warm-start-updates` to set that budget
directly and `--warm-start-batch-size` to tune throughput.
The trainer uses prioritized replay, batched Double DQN updates, dueling action
values, temporal interaction weights, bounded problem retention, and the
staged curriculum described in the implementation plan.

Training is fidelity-first. A calibration-heavy dense proxy supplies stepwise
feedback, and routed episodes receive a terminal correction so their complete
return equals the negative SABRE-routed log-infidelity per original two-qubit
gate. Circuits of at most 16 qubits are routed every episode; larger circuits
are routed every fifth episode by default. Set
`--large-routed-reward-frequency 1` to route every episode or use
`--disable-routed-reward` for proxy-only ablations. Validation always routes a
fixed 18-circuit, size-by-family set and runs every 5,000 steps by default.
Placement inference also selects its final beam candidate by routed fidelity;
pass `--selection-metric proxy` for faster proxy-only selection.

Generate ten random 50–60-qubit benchmark circuits with depths between 40 and
100 using only RZ, SX, X, and CZ gates:

```bash
python generate_random_circuits.py --count 10 --output-dir random_circuits
```

Use `--seed` for reproducibility, `--two-qubit-density` to control the fraction
of qubits participating in CZ gates per layer (default: `0.25`), and
`--format qasm2` when OpenQASM 2 output is preferred. The command prints a table
containing each file's seed, size, exact depth, and gate counts.

Resume a complete fidelity-reward checkpoint—including replay, RNG, and
curriculum state—with:

```bash
rlq-train --checkpoint placement_dqn.pt --resume
```

Checkpoints created before the fidelity reward are intentionally rejected
because their replay entries contain incompatible rewards. Start without
`--resume` to replace an old checkpoint.

The calibration snapshot is static. Refresh
`ibm_boston_connectivity_snapshot.py` before experiments that require current
hardware errors.

Benchmark results are printed as aligned placement/proxy and routed-circuit
tables. Pass `--format json` when machine-readable output is needed.
