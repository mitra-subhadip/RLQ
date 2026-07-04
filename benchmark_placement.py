#!/usr/bin/env python3
"""Compare graph-DQN placement with deterministic, random, and SABRE layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from qiskit import qasm2, qasm3

from rl_qtranspiler.baselines import (
    degree_centrality_mapping,
    greedy_mapping,
    identity_mapping,
    random_mapping,
    sabre_mapping,
)
from rl_qtranspiler.evaluation import evaluate_mapping, evaluate_routed_circuit
from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.inference import place
from rl_qtranspiler.model import GraphDQN
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-routing", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    text = arguments.circuit.read_text(encoding="utf-8")
    circuit = qasm3.loads(text) if "OPENQASM 3" in text else qasm2.loads(text)
    hardware = load_ibm_boston()
    problem = build_placement_problem(
        preprocess_for_swap_routing(circuit),
        hardware,
        problem_id=arguments.circuit.stem,
    )

    mappings: dict[str, tuple[int, ...]] = {
        "identity": identity_mapping(problem),
        "random": random_mapping(problem, seed=arguments.seed),
        "degree": degree_centrality_mapping(problem),
        "greedy": greedy_mapping(problem),
    }
    for heuristic in ("basic", "lookahead", "decay"):
        mappings[f"sabre_{heuristic}"] = sabre_mapping(
            circuit,
            problem,
            heuristic=heuristic,
            seed=arguments.seed,
        )

    dqn_metadata = None
    if arguments.checkpoint:
        model = GraphDQN().to(arguments.device)
        checkpoint = torch.load(
            arguments.checkpoint,
            map_location=arguments.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        result = place(model, problem)
        mappings["graph_dqn"] = tuple(
            result.logical_to_physical[index]
            for index in range(problem.num_logical_qubits)
        )
        dqn_metadata = {
            "beam_width": result.beam_width,
            "runtime_seconds": result.runtime_seconds,
        }

    report: dict[str, object] = {
        "backend": hardware.name,
        "circuit": arguments.circuit.name,
        "placements": {},
    }
    for name, mapping in mappings.items():
        started = perf_counter()
        proxy = evaluate_mapping(problem, mapping, started_at=started)
        metrics: dict[str, object] = {
            "mapping": mapping,
            "proxy": proxy.__dict__,
        }
        if not arguments.skip_routing:
            metrics["routed"] = evaluate_routed_circuit(
                circuit, problem, mapping, seed=arguments.seed
            ).__dict__
        report["placements"][name] = metrics
    if dqn_metadata:
        report["graph_dqn_inference"] = dqn_metadata
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
