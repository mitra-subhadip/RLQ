#!/usr/bin/env python3
"""Build IBM Boston's connectivity graph with calibrated two-qubit errors.

The script queries the current IBM Quantum backend calibration, creates an
undirected NetworkX graph, writes it as JSON, and optionally renders a PNG.

Requirements:
    python -m pip install qiskit-ibm-runtime networkx matplotlib

Authentication:
    Use a saved Qiskit Runtime account, or set QISKIT_IBM_TOKEN.  Optionally
    set QISKIT_IBM_INSTANCE to an IBM Cloud instance CRN.

Examples:
    conda run -n qiskit python ibm_heron3_connectivity.py
    conda run -n qiskit python ibm_heron3_connectivity.py --edge-labels
    conda run -n qiskit python ibm_heron3_connectivity.py --gate cz --no-plot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


DEFAULT_BACKEND = "ibm_boston"
GATE_PREFERENCE = {"cz": 3, "ecr": 2, "cx": 1}


def connect_to_ibm(account: str | None, instance: str | None) -> Any:
    """Create a Qiskit Runtime service without putting secrets in source code."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise SystemExit(
            "qiskit-ibm-runtime is not installed. In the qiskit environment run:\n"
            "  python -m pip install qiskit-ibm-runtime"
        ) from exc

    kwargs: dict[str, str] = {}
    if account:
        kwargs["name"] = account
    else:
        token = os.environ.get("QISKIT_IBM_TOKEN")
        if token:
            kwargs["token"] = token

    selected_instance = instance or os.environ.get("QISKIT_IBM_INSTANCE")
    if selected_instance:
        kwargs["instance"] = selected_instance

    # With no kwargs, Qiskit Runtime loads the default saved account.
    return QiskitRuntimeService(**kwargs)


def calibrated_two_qubit_gates(target: Any) -> dict[str, dict[tuple[int, int], Any]]:
    """Return two-qubit target-property mappings keyed by operation name."""
    candidates: dict[str, dict[tuple[int, int], Any]] = {}
    for name in target.operation_names:
        operation = target.operation_from_name(name)
        if getattr(operation, "num_qubits", None) != 2:
            continue

        properties = {
            tuple(qargs): instruction_properties
            for qargs, instruction_properties in target[name].items()
            if qargs is not None and len(qargs) == 2
        }
        if properties:
            candidates[name] = properties
    return candidates


def choose_entangling_gate(
    candidates: dict[str, dict[tuple[int, int], Any]],
    requested_gate: str | None,
) -> str:
    """Choose the requested gate or the best-calibrated two-qubit operation."""
    if requested_gate:
        if requested_gate not in candidates:
            available = ", ".join(sorted(candidates)) or "(none)"
            raise ValueError(
                f"Two-qubit gate {requested_gate!r} is unavailable. "
                f"Available gates: {available}"
            )
        return requested_gate

    if not candidates:
        raise ValueError("The backend target contains no two-qubit operations.")

    def score(item: tuple[str, dict[tuple[int, int], Any]]) -> tuple[int, int, int]:
        name, properties = item
        calibrated = sum(
            getattr(value, "error", None) is not None
            for value in properties.values()
        )
        return calibrated, len(properties), GATE_PREFERENCE.get(name, 0)

    return max(candidates.items(), key=score)[0]


def build_connectivity_graph(
    backend: Any,
    gate_name: str | None = None,
) -> tuple[nx.Graph, str]:
    """Build an undirected coupling graph with error data on every edge.

    Edge attribute ``error`` is the mean error over calibrated orientations of
    the selected gate. ``directional_errors`` retains each orientation, so no
    information is lost if a backend reports asymmetric calibration data.
    """
    candidates = calibrated_two_qubit_gates(backend.target)
    selected_gate = choose_entangling_gate(candidates, gate_name)
    gate_properties = candidates[selected_gate]

    coupling_map = backend.coupling_map
    if coupling_map is None:
        raise ValueError(f"{backend.name} does not expose a coupling map.")

    graph = nx.Graph(
        backend=backend.name,
        num_qubits=backend.num_qubits,
        two_qubit_gate=selected_gate,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    graph.add_nodes_from(range(backend.num_qubits))

    directed_errors: dict[
        tuple[int, int], list[tuple[tuple[int, int], float]]
    ] = defaultdict(list)
    for qargs, properties in gate_properties.items():
        error = getattr(properties, "error", None)
        if error is not None:
            pair = tuple(sorted(qargs))
            directed_errors[pair].append((qargs, float(error)))

    # IBM coupling maps may list both orientations. NetworkX Graph collapses
    # them into one physical connection while directional_errors preserves both.
    physical_edges = {tuple(sorted(edge)) for edge in coupling_map.get_edges()}
    for q0, q1 in sorted(physical_edges):
        measurements = directed_errors.get((q0, q1), [])
        errors = [error for _, error in measurements]
        by_direction = {
            f"{source}->{target}": error
            for (source, target), error in measurements
        }
        graph.add_edge(
            q0,
            q1,
            gate=selected_gate,
            error=(sum(errors) / len(errors)) if errors else None,
            error_min=min(errors) if errors else None,
            error_max=max(errors) if errors else None,
            directional_errors=by_direction,
        )

    return graph, selected_gate


def graph_as_json_data(graph: nx.Graph) -> dict[str, Any]:
    """Return a stable, human-readable JSON representation."""
    return {
        "metadata": dict(graph.graph),
        "nodes": [{"qubit": int(node)} for node in sorted(graph.nodes)],
        "edges": [
            {
                "qubits": [int(q0), int(q1)],
                **attributes,
            }
            for q0, q1, attributes in sorted(graph.edges(data=True))
        ],
    }


def save_json(graph: nx.Graph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph_as_json_data(graph), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def plot_graph(
    graph: nx.Graph,
    output_path: Path,
    show_edge_labels: bool,
) -> None:
    """Render connectivity; edge color represents the calibrated error."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(22, 16), constrained_layout=True)
    positions = nx.spring_layout(
        graph,
        seed=7,
        k=2.0 / math.sqrt(max(graph.number_of_nodes(), 1)),
        iterations=350,
    )

    calibrated_errors = [
        attributes["error"]
        for _, _, attributes in graph.edges(data=True)
        if attributes["error"] is not None
    ]
    if calibrated_errors:
        error_min = min(calibrated_errors)
        error_max = max(calibrated_errors)
        # Normalize requires a nonzero range.
        if math.isclose(error_min, error_max):
            error_max = error_min + sys.float_info.epsilon
        normalization = Normalize(vmin=error_min, vmax=error_max)
        color_map = colormaps["viridis"]
        edge_colors = [
            color_map(normalization(attributes["error"]))
            if attributes["error"] is not None
            else (0.65, 0.65, 0.65, 1.0)
            for _, _, attributes in graph.edges(data=True)
        ]
    else:
        normalization = Normalize(vmin=0.0, vmax=1.0)
        color_map = colormaps["viridis"]
        edge_colors = ["0.65"] * graph.number_of_edges()

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=310,
        node_color="#d8ecff",
        edgecolors="#174a70",
        linewidths=0.8,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=7, ax=axis)
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=edge_colors,
        width=2.2,
        ax=axis,
    )

    if show_edge_labels:
        labels = {
            (q0, q1): (
                f"{attributes['error']:.2e}"
                if attributes["error"] is not None
                else "n/a"
            )
            for q0, q1, attributes in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=labels,
            font_size=4.5,
            rotate=False,
            ax=axis,
        )

    if calibrated_errors:
        color_bar = figure.colorbar(
            ScalarMappable(norm=normalization, cmap=color_map),
            ax=axis,
            shrink=0.72,
        )
        color_bar.set_label(f"{graph.graph['two_qubit_gate']} gate error")

    axis.set_title(
        f"{graph.graph['backend']} connectivity "
        f"({graph.graph['num_qubits']} qubits) — "
        f"{graph.graph['two_qubit_gate']} calibration error"
    )
    axis.set_axis_off()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch an IBM backend's connectivity and calibrated two-qubit "
            "gate errors."
        )
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument(
        "--gate",
        help="Two-qubit gate to use (default: choose the best-covered gate).",
    )
    parser.add_argument(
        "--account",
        help="Name of a saved Qiskit Runtime account (default: saved default).",
    )
    parser.add_argument(
        "--instance",
        help="IBM Cloud instance CRN; may also use QISKIT_IBM_INSTANCE.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("ibm_boston_connectivity.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("ibm_boston_connectivity.png"),
        help="PNG output path.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not render the PNG.",
    )
    parser.add_argument(
        "--edge-labels",
        action="store_true",
        help="Print numeric errors on plotted edges (crowded for 156 qubits).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        service = connect_to_ibm(arguments.account, arguments.instance)
        backend = service.backend(arguments.backend)
        graph, selected_gate = build_connectivity_graph(backend, arguments.gate)
        save_json(graph, arguments.json)
        if not arguments.no_plot:
            plot_graph(graph, arguments.plot, arguments.edge_labels)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    calibrated = sum(
        attributes["error"] is not None
        for _, _, attributes in graph.edges(data=True)
    )
    print(f"Backend: {backend.name}")
    print(f"Qubits: {graph.number_of_nodes()}")
    print(f"Couplings: {graph.number_of_edges()}")
    print(f"Error source: {selected_gate}")
    print(f"Edges with calibration error: {calibrated}")
    print(f"JSON: {arguments.json.resolve()}")
    if not arguments.no_plot:
        print(f"Plot: {arguments.plot.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
