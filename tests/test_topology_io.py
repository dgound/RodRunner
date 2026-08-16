import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.topology_io import (
    COORDINATE_FRAME,
    LINK_POINT_FIELDS,
    LINK_SUMMARY_FIELDS,
    OUTPUT_NAMES,
    SCHEMA_VERSION,
    THETA_POINT_FIELDS,
    THETA_SUMMARY_FIELDS,
    TopologyIOError,
    load_link_loops,
    load_theta_yamada,
    write_topology_outputs,
)


def _theta_summary():
    return {
        "Dataset": "synthetic", "Location": "Location-01", "WalkID": 7,
        "GraphID": "theta-7", "ThetaStatus": "nominal_theta_graph", "FailureReason": "",
        "NumDoubledLoci": 2, "AssignmentCost": 12.5,
        "AssignmentMargin": 4.25, "YamadaPolynomial": "-A^2-A^-2",
        "YamadaNormalized": "A^4+1", "ProjectionStable": 1,
        "PerturbationStable": 1, "ForkModelStable": 1,
    }


def _theta_row(strand, order, xyz, *, tp="", tp_index=""):
    endpoint = order in (0, 3)
    return {
        "Dataset": "synthetic", "Location": "Location-01", "WalkID": 7,
        "GraphID": "theta-7", "StrandID": strand, "PointOrder": order,
        "PointRole": "fork_1" if order == 0 else "fork_2" if order == 3 else "locus",
        "TP": tp, "TPIndex": tp_index, "GenomicBP": "" if endpoint else order * 1000,
        "CoordinateSource": "inferred_fork" if endpoint else "v3_fit",
        "ComponentID": "" if endpoint else ("single" if strand == "unreplicated" else strand[-1]),
        "XNm": xyz[0], "YNm": xyz[1], "ZNm": xyz[2],
        "XSEnm": 20.0, "YSEnm": 20.0, "ZSEnm": 40.0,
        "Observed": 0 if endpoint else 1, "InferredFork": 1 if endpoint else 0,
    }


def _theta_points():
    start, end = (0.0, 0.0, 0.0), (30.0, 0.0, 0.0)
    interiors = {
        "unreplicated": [(10.0, -5.0, 0.0), (20.0, -5.0, 0.0)],
        "sister_A": [(10.0, 5.0, 1.0), (20.0, 5.0, 1.0)],
        "sister_B": [(10.0, 0.0, 6.0), (20.0, 0.0, 6.0)],
    }
    rows = []
    for strand in reversed(("unreplicated", "sister_A", "sister_B")):
        # Deliberately reverse input order: the writer must be deterministic.
        part = [
            _theta_row(strand, 0, start),
            _theta_row(strand, 1, interiors[strand][0], tp="TP017", tp_index=17),
            _theta_row(strand, 2, interiors[strand][1], tp="TP018", tp_index=18),
            _theta_row(strand, 3, end),
        ]
        rows.extend(reversed(part))
    return rows


def _link_summary():
    return {
        "Dataset": "synthetic", "Location": "Location-02", "WalkID": 9,
        "LinkID": "link-9", "LinkStatus": "nominal_linked_polygon", "FailureReason": "",
        "NumLoci": 3, "AssignmentCost": 20.0, "AssignmentMargin": 7.0,
        "LinkingNumber": 1, "MinInterLoopDistanceNm": 75.0,
        "AssignmentStable": 1, "PerturbationStable": 1, "DisjointStable": 1,
    }


def _link_row(loop, order, xyz):
    return {
        "Dataset": "synthetic", "Location": "Location-02", "WalkID": 9,
        "LinkID": "link-9", "LoopID": loop, "PointOrder": order,
        "PointRole": "locus", "TP": f"TP{order + 2:03d}",
        "TPIndex": order + 2, "GenomicBP": order * 1000,
        "CoordinateSource": "v3_two_component", "ComponentID": loop[-1],
        "XNm": xyz[0], "YNm": xyz[1], "ZNm": xyz[2],
        "XSEnm": 20.0, "YSEnm": 20.0, "ZSEnm": 40.0, "Closed": 1,
    }


def _link_points():
    coordinates = {
        "sister_A": [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 8.0, 0.0)],
        "sister_B": [(0.0, 0.0, 5.0), (10.0, 0.0, 5.0), (5.0, 8.0, 5.0)],
    }
    return [
        _link_row(loop, order, coordinates[loop][order])
        for loop in reversed(("sister_A", "sister_B"))
        for order in reversed(range(3))
    ]


class TopologyIOTest(unittest.TestCase):
    def _write(self, root, *, theta_points=None, link_points=None):
        return write_topology_outputs(
            root,
            theta_summaries=[_theta_summary()],
            theta_points=_theta_points() if theta_points is None else theta_points,
            link_summaries=[_link_summary()],
            link_points=_link_points() if link_points is None else link_points,
            provenance={"input_sha256": "abc123", "software": {"version": "test"}},
        )

    def test_round_trip_is_yamada_compatible_and_xyz_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write(directory)
            strands = load_theta_yamada(paths["theta_points"], "theta-7")
            self.assertEqual(len(strands), 3)
            self.assertTrue(all(value.shape == (4, 3) for value in strands))
            np.testing.assert_array_equal(strands[0][0], [0.0, 0.0, 0.0])
            np.testing.assert_array_equal(strands[1][1], [10.0, 5.0, 1.0])
            for strand in strands[1:]:
                np.testing.assert_array_equal(strand[0], strands[0][0])
                np.testing.assert_array_equal(strand[-1], strands[0][-1])

    def test_link_loader_appends_exactly_one_closing_vertex(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write(directory)
            loop_a, loop_b = load_link_loops(paths["link_points"], "link-9")
            self.assertEqual(loop_a.shape, (4, 3))
            self.assertEqual(loop_b.shape, (4, 3))
            np.testing.assert_array_equal(loop_a[0], loop_a[-1])
            np.testing.assert_array_equal(loop_b[0], loop_b[-1])
            self.assertFalse(np.array_equal(loop_a[-2], loop_a[0]))

    def test_writer_emits_only_five_named_outputs_and_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write(directory)
            self.assertEqual(set(path.name for path in paths.values()), set(OUTPUT_NAMES.values()))
            self.assertEqual(set(path.name for path in Path(directory).iterdir()), set(OUTPUT_NAMES.values()))
            manifest = json.loads(paths["manifest"].read_text())
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(manifest["coordinate_frame"], COORDINATE_FRAME)
            self.assertEqual(manifest["files"]["theta_points"]["rows"], 12)
            self.assertEqual(len(manifest["files"]["theta_points"]["sha256"]), 64)

    def test_headers_and_file_order_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write(directory)
            with paths["theta_points"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(tuple(rows[0]), THETA_POINT_FIELDS)
            self.assertEqual(
                [(row["StrandID"], int(row["PointOrder"])) for row in rows[:5]],
                [("unreplicated", 0), ("unreplicated", 1), ("unreplicated", 2),
                 ("unreplicated", 3), ("sister_A", 0)],
            )

    def test_wrong_row_schema_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = _theta_points()
            malformed[0] = dict(malformed[0])
            malformed[0]["Surprise"] = 1
            with self.assertRaisesRegex(TopologyIOError, "wrong schema"):
                self._write(directory, theta_points=malformed)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_nonfinite_coordinate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = _link_points()
            malformed[0] = dict(malformed[0], XNm=float("nan"))
            with self.assertRaisesRegex(TopologyIOError, "finite"):
                self._write(directory, link_points=malformed)

    def test_theta_forks_must_be_shared_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = _theta_points()
            target = next(
                row for row in malformed
                if row["StrandID"] == "sister_B" and row["PointOrder"] == 0)
            target["XNm"] = 0.000001
            with self.assertRaisesRegex(TopologyIOError, "exactly shared first fork"):
                self._write(directory, theta_points=malformed)

    def test_duplicate_and_missing_point_orders_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            malformed = _link_points()
            target = next(
                row for row in malformed
                if row["LoopID"] == "sister_A" and row["PointOrder"] == 2)
            target["PointOrder"] = 1
            with self.assertRaisesRegex(TopologyIOError, "duplicate link point key"):
                self._write(directory, link_points=malformed)

    def test_loader_rejects_wrong_column_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            fields = list(THETA_POINT_FIELDS)
            fields[0], fields[1] = fields[1], fields[0]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(_theta_points())
            with self.assertRaisesRegex(TopologyIOError, "header must be exactly"):
                load_theta_yamada(path, "theta-7")

    def test_absent_identifiers_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write(directory)
            with self.assertRaisesRegex(TopologyIOError, "is absent"):
                load_theta_yamada(paths["theta_points"], "missing")
            with self.assertRaisesRegex(TopologyIOError, "is absent"):
                load_link_loops(paths["link_points"], "missing")

    def test_public_schema_constants_are_fixed_tuples(self):
        self.assertIsInstance(THETA_SUMMARY_FIELDS, tuple)
        self.assertIsInstance(THETA_POINT_FIELDS, tuple)
        self.assertIsInstance(LINK_SUMMARY_FIELDS, tuple)
        self.assertIsInstance(LINK_POINT_FIELDS, tuple)


if __name__ == "__main__":
    unittest.main()
