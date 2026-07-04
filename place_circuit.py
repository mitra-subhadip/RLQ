#!/usr/bin/env python3
"""Load a trained model and place a QASM circuit on IBM Boston."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qiskit import qasm2, qasm3

from rl_qtranspiler.hardware import load_ibm_boston
from rl_qtranspiler.inference import place
from rl_qtranspiler.model import GraphDQN
from rl_qtranspiler.preprocessing import preprocess_for_swap_routing
from rl_qtranspiler.problem import build_placement_problem


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--expansions", type=int, default=4)
    parser.add_argument("--device", default="cpu")
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

    model = GraphDQN().to(arguments.device)
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=arguments.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    result = place(
        model,
        problem,
        beam_width=arguments.beam_width,
        expansions_per_state=arguments.expansions,
    )
    print(
        json.dumps(
            {
                "logical_to_physical": result.logical_to_physical,
                "physical_to_logical": result.physical_to_logical,
                "placement_order": result.placement_order,
                "distance_score": result.distance_score,
                "calibration_score": result.calibration_score,
                "combined_score": result.combined_score,
                "beam_width": result.beam_width,
                "runtime_seconds": result.runtime_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
