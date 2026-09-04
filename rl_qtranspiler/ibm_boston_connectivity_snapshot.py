"""IBM Boston (Heron r3) connectivity and calibration snapshot.

Source: https://quantum.cloud.ibm.com/computers?system=ibm_boston
Retrieved: 2026-07-04
Calibration age shown by IBM at retrieval: "Last updated: 1 hour ago"

Each tuple in COUPLING_DATA is:
    (qubit_1, qubit_2, CZ gate error)

The connections are undirected.  CZ error is the error displayed in IBM's
calibration table and is the appropriate native two-qubit edge cost for the
processor at the time of this snapshot.
"""

from __future__ import annotations

from typing import Final


BACKEND_NAME: Final = "ibm_boston"
PROCESSOR_TYPE: Final = "Heron r3"
NUM_QUBITS: Final = 156
NUM_COUPLERS: Final = 176
CZ_ERROR_MEDIAN_REPORTED: Final = 1.188e-3
SOURCE_URL: Final = (
    "https://quantum.cloud.ibm.com/computers?system=ibm_boston"
)

COUPLING_DATA: Final[list[tuple[int, int, float]]] = [
    (0, 1, 0.0007435),
    (1, 2, 0.01839),
    (2, 3, 0.007568),
    (3, 4, 0.0007511),
    (3, 16, 0.0007535),
    (4, 5, 0.0007529),
    (5, 6, 0.001461),
    (6, 7, 0.002142),
    (7, 8, 0.005055),
    (7, 17, 0.001583),
    (8, 9, 0.009062),
    (9, 10, 0.01983),
    (10, 11, 0.006824),
    (11, 12, 0.007111),
    (11, 18, 0.001324),
    (12, 13, 0.001049),
    (13, 14, 0.000989),
    (14, 15, 0.002345),
    (15, 19, 0.002394),
    (16, 23, 0.0009951),
    (17, 27, 0.001759),
    (18, 31, 0.0009674),
    (19, 35, 0.001195),
    (20, 21, 0.0009091),
    (21, 22, 0.0009316),
    (21, 36, 0.0008855),
    (22, 23, 0.0006518),
    (23, 24, 0.0008668),
    (24, 25, 0.0008333),
    (25, 26, 0.0009068),
    (25, 37, 0.004278),
    (26, 27, 0.001615),
    (27, 28, 0.00481),
    (28, 29, 0.001092),
    (29, 30, 0.001712),
    (29, 38, 0.0008866),
    (30, 31, 0.001276),
    (31, 32, 0.0006529),
    (32, 33, 0.001251),
    (33, 34, 0.002395),
    (33, 39, 0.000976),
    (34, 35, 0.001503),
    (36, 41, 0.001299),
    (37, 45, 0.001378),
    (38, 49, 0.001065),
    (39, 53, 0.0007434),
    (40, 41, 0.001227),
    (41, 42, 0.0014),
    (42, 43, 0.0009806),
    (43, 44, 0.001015),
    (43, 56, 0.00174),
    (44, 45, 0.0009697),
    (45, 46, 0.01953),
    (46, 47, 0.008349),
    (47, 48, 0.001311),
    (47, 57, 0.001348),
    (48, 49, 0.001465),
    (49, 50, 0.00133),
    (50, 51, 0.001398),
    (51, 52, 0.0009494),
    (51, 58, 0.000997),
    (52, 53, 0.000817),
    (53, 54, 0.000908),
    (54, 55, 0.0006124),
    (55, 59, 0.001541),
    (56, 63, 0.0009875),
    (57, 67, 0.001263),
    (58, 71, 0.001109),
    (59, 75, 0.001607),
    (60, 61, 0.0006916),
    (61, 62, 0.0007161),
    (61, 76, 0.001066),
    (62, 63, 0.0006731),
    (63, 64, 0.006154),
    (64, 65, 0.008597),
    (65, 66, 0.001172),
    (65, 77, 0.0009631),
    (66, 67, 0.001936),
    (67, 68, 0.0008067),
    (68, 69, 0.0009011),
    (69, 70, 0.001204),
    (69, 78, 0.0007839),
    (70, 71, 0.001013),
    (71, 72, 0.0006685),
    (72, 73, 0.00118),
    (73, 74, 0.0008643),
    (73, 79, 0.00146),
    (74, 75, 0.0007294),
    (76, 81, 0.001218),
    (77, 85, 0.08971),
    (78, 89, 0.001247),
    (79, 93, 0.0009318),
    (80, 81, 0.0009361),
    (81, 82, 0.0009731),
    (82, 83, 0.002159),
    (83, 84, 0.002491),
    (83, 96, 0.0009788),
    (84, 85, 0.05562),
    (85, 86, 0.121),
    (86, 87, 0.001028),
    (87, 88, 0.0007168),
    (87, 97, 0.001525),
    (88, 89, 0.002331),
    (89, 90, 0.003476),
    (90, 91, 0.001545),
    (91, 92, 0.000751),
    (91, 98, 0.0008687),
    (92, 93, 0.0006974),
    (93, 94, 0.001053),
    (94, 95, 0.001527),
    (95, 99, 0.001492),
    (96, 103, 0.002296),
    (97, 107, 0.00198),
    (98, 111, 0.0007595),
    (99, 115, 0.001196),
    (100, 101, 0.001345),
    (101, 102, 0.006026),
    (101, 116, 0.001374),
    (102, 103, 0.00111),
    (103, 104, 0.002091),
    (104, 105, 0.003096),
    (105, 106, 0.0009215),
    (105, 117, 0.001459),
    (106, 107, 0.001076),
    (107, 108, 0.001024),
    (108, 109, 0.0008871),
    (109, 110, 0.001053),
    (109, 118, 0.002232),
    (110, 111, 0.00188),
    (111, 112, 0.002076),
    (112, 113, 0.002087),
    (113, 114, 0.001331),
    (113, 119, 0.001301),
    (114, 115, 0.001232),
    (116, 121, 0.003432),
    (117, 125, 0.0009397),
    (118, 129, 0.0007444),
    (119, 133, 0.0009232),
    (120, 121, 0.001823),
    (121, 122, 0.0009299),
    (122, 123, 0.0009655),
    (123, 124, 0.0007019),
    (123, 136, 0.002153),
    (124, 125, 0.0007537),
    (125, 126, 0.0009237),
    (126, 127, 0.001034),
    (127, 128, 0.001092),
    (127, 137, 0.001735),
    (128, 129, 0.001042),
    (129, 130, 0.001049),
    (130, 131, 0.0007953),
    (131, 132, 0.0007372),
    (131, 138, 0.0008748),
    (132, 133, 0.0009827),
    (133, 134, 0.001181),
    (134, 135, 0.001329),
    (135, 139, 0.001145),
    (136, 143, 0.001553),
    (137, 147, 0.001392),
    (138, 151, 0.001111),
    (139, 155, 0.0009739),
    (140, 141, 0.001335),
    (141, 142, 0.001369),
    (142, 143, 0.001066),
    (143, 144, 0.0009366),
    (144, 145, 0.0007463),
    (145, 146, 0.053),
    (146, 147, 0.08767),
    (147, 148, 0.0007878),
    (148, 149, 0.00153),
    (149, 150, 0.001529),
    (150, 151, 0.005382),
    (151, 152, 0.001239),
    (152, 153, 0.001506),
    (153, 154, 0.0009196),
    (154, 155, 0.000869),
]

# Convenient representations for routing and lookup.
CONNECTIVITY_EDGES: Final[list[tuple[int, int]]] = [
    (q1, q2) for q1, q2, _ in COUPLING_DATA
]
CZ_ERROR_BY_EDGE: Final[dict[tuple[int, int], float]] = {
    (q1, q2): error for q1, q2, error in COUPLING_DATA
}


def floyd_warshall_distances() -> list[list[int]]:
    """Compute all-pairs shortest hop distances with Floyd-Warshall.

    Returns:
        A NUM_QUBITS x NUM_QUBITS matrix. ``distances[i][j]`` is the
        minimum number of coupling edges between physical qubits i and j.

    Complexity:
        O(NUM_QUBITS**3) time and O(NUM_QUBITS**2) memory.
    """
    unreachable = NUM_QUBITS + 1
    distances = [
        [0 if source == target else unreachable for target in range(NUM_QUBITS)]
        for source in range(NUM_QUBITS)
    ]

    for qubit_1, qubit_2 in CONNECTIVITY_EDGES:
        distances[qubit_1][qubit_2] = 1
        distances[qubit_2][qubit_1] = 1

    for intermediate in range(NUM_QUBITS):
        from_intermediate = distances[intermediate]
        for source in range(NUM_QUBITS):
            source_distances = distances[source]
            distance_to_intermediate = source_distances[intermediate]
            if distance_to_intermediate == unreachable:
                continue

            for target in range(NUM_QUBITS):
                candidate = distance_to_intermediate + from_intermediate[target]
                if candidate < source_distances[target]:
                    source_distances[target] = candidate

    if any(unreachable in row for row in distances):
        raise ValueError("The coupling graph is disconnected.")
    return distances


# Complete 156 x 156 shortest-hop matrix, computed once when this module loads.
SHORTEST_HOP_DISTANCES: Final[list[list[int]]] = floyd_warshall_distances()


def shortest_distance(qubit_1: int, qubit_2: int) -> int:
    """Return the minimum number of couplers between two physical qubits."""
    if not 0 <= qubit_1 < NUM_QUBITS or not 0 <= qubit_2 < NUM_QUBITS:
        raise ValueError(f"Qubit indices must be between 0 and {NUM_QUBITS - 1}")
    return SHORTEST_HOP_DISTANCES[qubit_1][qubit_2]


def get_cz_error(qubit_1: int, qubit_2: int) -> float:
    """Return the CZ error for an edge, accepting either qubit order."""
    edge = tuple(sorted((qubit_1, qubit_2)))
    try:
        return CZ_ERROR_BY_EDGE[edge]
    except KeyError as exc:
        raise ValueError(f"Qubits {qubit_1} and {qubit_2} are not connected") from exc


def as_networkx_graph():
    """Return a NetworkX graph whose edge weight is the calibrated CZ error."""
    import networkx as nx

    graph = nx.Graph(
        backend=BACKEND_NAME,
        processor=PROCESSOR_TYPE,
        calibration_snapshot="2026-07-04",
    )
    graph.add_nodes_from(range(NUM_QUBITS))
    graph.add_weighted_edges_from(COUPLING_DATA, weight="cz_error")
    return graph


assert len(COUPLING_DATA) == NUM_COUPLERS
assert len(CZ_ERROR_BY_EDGE) == NUM_COUPLERS
assert len(SHORTEST_HOP_DISTANCES) == NUM_QUBITS
assert all(len(row) == NUM_QUBITS for row in SHORTEST_HOP_DISTANCES)
