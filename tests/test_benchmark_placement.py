from benchmark_placement import print_benchmark_report


def test_benchmark_report_prints_metric_tables(capsys):
    report = {
        "backend": "test_backend",
        "circuit": "example.qasm",
        "placements": {
            "identity": {
                "mapping": (0, 1),
                "placement_runtime_seconds": 0.00123456,
                "proxy": {
                    "distance_score": 2.5,
                    "calibration_score": 0.125,
                    "combined_score": 2.625,
                    "scoring_runtime_seconds": 0.002,
                },
                "routed": {
                    "swap_count": 3,
                    "two_qubit_gate_count": 8,
                    "depth": 12,
                    "estimated_log_infidelity": 0.05,
                    "normalized_log_infidelity": 0.025,
                    "estimated_success_probability": 0.951229,
                    "runtime_seconds": 0.4,
                },
            }
        },
    }

    print_benchmark_report(report)

    output = capsys.readouterr().out
    assert "Backend: test_backend" in output
    assert "Circuit: example.qasm" in output
    assert "Placement and proxy metrics" in output
    assert "Routed-circuit metrics" in output
    assert "Method" in output
    assert "identity" in output
    assert "Success prob." in output
    assert '"placements"' not in output


def test_benchmark_report_omits_routed_table_when_routing_is_skipped(capsys):
    report = {
        "backend": "test_backend",
        "circuit": "example.qasm",
        "placements": {
            "greedy": {
                "mapping": (0, 1),
                "placement_runtime_seconds": 0.01,
                "proxy": {
                    "distance_score": 1.0,
                    "calibration_score": 0.2,
                    "combined_score": 1.2,
                    "scoring_runtime_seconds": 0.003,
                },
            }
        },
    }

    print_benchmark_report(report)

    output = capsys.readouterr().out
    assert "greedy" in output
    assert "Placement and proxy metrics" in output
    assert "Routed-circuit metrics" not in output

