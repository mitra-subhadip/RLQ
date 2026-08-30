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
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format (default: table)",
    )
    return parser.parse_args()


def _format_number(value: object) -> str:
    """Format a benchmark value compactly without hiding small measurements."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_table(
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, object]],
) -> None:
    formatted_rows = [
        {key: _format_number(row[key]) for _, key in columns} for row in rows
    ]
    widths = {
        key: max(
            len(heading),
            *(len(row[key]) for row in formatted_rows),
        )
        for heading, key in columns
    }
    numeric_columns = {
        key
        for _, key in columns
        if all(isinstance(row[key], (int, float)) for row in rows)
    }

    def render(values: dict[str, str]) -> str:
        return "  ".join(
            (
                values[key].rjust(widths[key])
                if key in numeric_columns
                else values[key].ljust(widths[key])
            )
            for _, key in columns
        )

    print(render({key: heading for heading, key in columns}))
    print("  ".join("-" * widths[key] for _, key in columns))
    for row in formatted_rows:
        print(render(row))


def print_benchmark_report(report: dict[str, object]) -> None:
    """Print the benchmark report as readable terminal tables."""
    print(f"Backend: {report['backend']}")
    print(f"Circuit: {report['circuit']}")

    placements = report["placements"]
    assert isinstance(placements, dict)
    placement_rows: list[dict[str, object]] = []
    routed_rows: list[dict[str, object]] = []
    for method, metrics in placements.items():
        assert isinstance(metrics, dict)
        proxy = metrics["proxy"]
        assert isinstance(proxy, dict)
        placement_rows.append(
            {
                "method": method,
                "placement": metrics["placement_runtime_seconds"],
                "distance": proxy["distance_score"],
                "calibration": proxy["calibration_score"],
                "combined": proxy["combined_score"],
                "scoring": proxy["scoring_runtime_seconds"],
            }
        )
        routed = metrics.get("routed")
        if isinstance(routed, dict):
            routed_rows.append(
                {
                    "method": method,
                    "swaps": routed["swap_count"],
                    "two_qubit": routed["two_qubit_gate_count"],
                    "depth": routed["depth"],
                    "log_infidelity": routed["estimated_log_infidelity"],
                    "normalized": routed["normalized_log_infidelity"],
                    "success": routed["estimated_success_probability"],
                    "routing": routed["runtime_seconds"],
                }
            )

    print("\nPlacement and proxy metrics")
    _print_table(
        (
            ("Method", "method"),
            ("Placement (s)", "placement"),
            ("Distance", "distance"),
            ("Calibration", "calibration"),
            ("Combined", "combined"),
            ("Scoring (s)", "scoring"),
        ),
        placement_rows,
    )
    if routed_rows:
        print("\nRouted-circuit metrics")
        _print_table(
            (
                ("Method", "method"),
                ("Swaps", "swaps"),
                ("2Q gates", "two_qubit"),
                ("Depth", "depth"),
                ("Log infidelity", "log_infidelity"),
                ("Normalized", "normalized"),
                ("Success prob.", "success"),
                ("Routing (s)", "routing"),
            ),
            routed_rows,
        )


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

    placement_functions = {
        "identity": lambda: identity_mapping(problem),
        "random": lambda: random_mapping(problem, seed=arguments.seed),
        "degree": lambda: degree_centrality_mapping(problem),
        "greedy": lambda: greedy_mapping(problem),
    }
    for heuristic in ("basic", "lookahead", "decay"):
        placement_functions[f"sabre_{heuristic}"] = (
            lambda selected=heuristic: sabre_mapping(
                circuit,
                problem,
                heuristic=selected,
                seed=arguments.seed,
            )
        )

    if arguments.checkpoint:
        model = GraphDQN().to(arguments.device)
        checkpoint = torch.load(
            arguments.checkpoint,
            map_location=arguments.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])

        def graph_dqn_mapping() -> tuple[int, ...]:
            result = place(model, problem, routing_seed=arguments.seed)
            return tuple(
                result.logical_to_physical[index]
                for index in range(problem.num_logical_qubits)
            )

        placement_functions["graph_dqn"] = graph_dqn_mapping

    report: dict[str, object] = {
        "backend": hardware.name,
        "circuit": arguments.circuit.name,
        "placements": {},
    }
    for name, placement_function in placement_functions.items():
        started = perf_counter()
        mapping = placement_function()
        placement_seconds = perf_counter() - started
        proxy = evaluate_mapping(problem, mapping)
        metrics: dict[str, object] = {
            "mapping": mapping,
            "placement_runtime_seconds": placement_seconds,
            "proxy": proxy.__dict__,
        }
        if not arguments.skip_routing:
            metrics["routed"] = evaluate_routed_circuit(
                circuit, problem, mapping, seed=arguments.seed
            ).__dict__
        report["placements"][name] = metrics
    if arguments.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print_benchmark_report(report)


if __name__ == "__main__":
    main()

