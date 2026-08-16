"""Reducing the default output set must change WHICH files appear -- nothing else.

Trace salvage used to write six dataset-level files plus one per Location, all
unconditionally. Four are the deliverables; the per-locus audit CSV, the reduced
walk summary and the per-Location comprehensive files carried facts that are
already on every row of the comprehensive CSV, so they were a second
representation that could drift from the first.

This file proves the reduction is presentational: coordinates, the global/
OriginalWalkID mapping, walk membership and every salvage decision are identical
with and without the diagnostics flag.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_REPO / "tests"))

import trace_salvage as ts  # noqa: E402
from test_trace_salvage import TraceSalvageTest  # noqa: E402

#: Comprehensive-CSV columns whose values must be bit-identical either way.
_MEANING_FIELDS = ("TraceZNm", "TraceYNm", "TraceXNm",
                   "OriginalZNm", "OriginalYNm", "OriginalXNm",
                   "z_nm", "y_nm", "x_nm",
                   "WalkID", "OriginalWalkID", "Dataset", "Location", "tp",
                   "TraceSalvaged", "TraceSalvageReason", "TraceSalvageStatus",
                   "TraceSalvagedSingleSpotComplete", "TraceSalvagedMultiSpot")

DEFAULT_OUTPUTS = ("full_schema", "full_schema_complete", "policy", "manifest")
DIAGNOSTIC_OUTPUTS = ("audit", "walk_summary")


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class OutputReductionEquivalenceTest(unittest.TestCase):
    """Uses the existing fixture builder so the data is the real thing."""

    def setUp(self):
        self._fixture = TraceSalvageTest("test_salvages_isolated_call_using_xy_and_emits_full_xyz")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        config = self._fixture._config(self.root, [
            self._fixture._location(self.root, "A", candidate=True),
            self._fixture._location(self.root, "B"),
        ])
        self.lean = ts.run_dataset(config, self.root / "lean", "p")
        self.full = ts.run_dataset(config, self.root / "full", "p",
                                   diagnostics=True)

    def _csv(self, which, name):
        return self.root / which / f"p_trace_salvage_{name}.csv"

    def test_default_writes_exactly_the_four_deliverables(self):
        present = sorted(p.name for p in (self.root / "lean").iterdir()
                         if p.is_file())
        self.assertEqual(present, sorted([
            "p_trace_salvage_full_schema.csv",
            "p_trace_salvage_full_schema_complete.csv",
            "p_trace_salvage_manifest.json",
            "p_trace_salvage_policy.json"]))

    def test_no_audit_or_walk_summary_or_location_dir_by_default(self):
        for name in ("p_trace_salvage_audit.csv",
                     "p_trace_salvage_walk_summary.csv"):
            self.assertFalse((self.root / "lean" / name).exists(), name)
        self.assertFalse((self.root / "lean" / "locations").exists())

    def test_diagnostics_adds_them_back(self):
        for name in ("p_trace_salvage_audit.csv",
                     "p_trace_salvage_walk_summary.csv"):
            self.assertTrue((self.root / "full" / name).exists(), name)
        self.assertTrue((self.root / "full" / "locations").is_dir())

    # ---- the equivalence proof -------------------------------------------
    def test_the_comprehensive_csv_is_byte_identical(self):
        for name in ("full_schema", "full_schema_complete"):
            with self.subTest(name):
                self.assertEqual(
                    self._csv("lean", name).read_bytes(),
                    self._csv("full", name).read_bytes(),
                    "reducing the output set must not touch the deliverable")

    def test_every_coordinate_and_decision_field_matches(self):
        lean, full = (_rows(self._csv(w, "full_schema"))
                      for w in ("lean", "full"))
        self.assertEqual(len(lean), len(full))
        self.assertGreater(len(lean), 0)
        for a, b in zip(lean, full):
            for field in _MEANING_FIELDS:
                if field in a:
                    with self.subTest(field=field):
                        self.assertEqual(a[field], b[field])

    def test_the_global_walkid_mapping_is_identical(self):
        def mapping(which):
            return {(r["Dataset"], r["Location"], r["OriginalWalkID"]):
                    r["WalkID"] for r in _rows(self._csv(which, "full_schema"))}
        self.assertEqual(mapping("lean"), mapping("full"))

    def test_walk_membership_is_identical(self):
        def walks(which, name):
            return {(r["Dataset"], r["Location"], r["WalkID"])
                    for r in _rows(self._csv(which, name))}
        for name in ("full_schema", "full_schema_complete"):
            with self.subTest(name):
                self.assertEqual(walks("lean", name), walks("full", name))

    def test_the_manifest_counts_agree(self):
        self.assertEqual(self.lean["counts"], self.full["counts"])
        self.assertEqual(self.lean["decision_counts"],
                         self.full["decision_counts"])

    def test_the_manifest_records_provenance_of_dropped_intermediates(self):
        """Dropping the per-Location files must not lose their provenance."""
        for location, entry in self.lean["location_outputs"].items():
            with self.subTest(location):
                self.assertIn("path", entry)
                self.assertFalse(entry["retained"])
                self.assertEqual(
                    entry["rows"],
                    self.full["location_outputs"][location]["rows"])
        for entry in self.full["location_outputs"].values():
            self.assertTrue(entry["retained"])

    def test_the_manifest_lists_only_the_files_it_wrote(self):
        self.assertEqual(sorted(self.lean["outputs"]), sorted(DEFAULT_OUTPUTS))
        self.assertEqual(sorted(self.full["outputs"]),
                         sorted([*DEFAULT_OUTPUTS, *DIAGNOSTIC_OUTPUTS]))
        for name in DIAGNOSTIC_OUTPUTS:
            self.assertNotIn(name, self.lean["outputs"],
                             "a file that was not written must not be listed")


class SharedWalkLevelReaderTest(unittest.TestCase):
    """Walk-level info is DERIVED from the comprehensive CSV, by one reader."""

    def setUp(self):
        self._fixture = TraceSalvageTest("test_salvages_isolated_call_using_xy_and_emits_full_xyz")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        config = self._fixture._config(self.root, [
            self._fixture._location(self.root, "A", candidate=True),
            self._fixture._location(self.root, "B"),
        ])
        ts.run_dataset(config, self.root / "out", "p", diagnostics=True)
        self.comprehensive = (
            self.root / "out" / "p_trace_salvage_full_schema.csv")
        self.walk_summary = (
            self.root / "out" / "p_trace_salvage_walk_summary.csv")

    def test_derived_rows_match_the_diagnostic_walk_summary(self):
        derived = {(r["Dataset"], r["Location"], r["WalkID"]): r
                   for r in ts.walk_level_rows(self.comprehensive)}
        written = {(r["Dataset"], r["Location"], r["WalkID"]): r
                   for r in _rows(self.walk_summary)}
        self.assertEqual(set(derived), set(written))
        for key, row in written.items():
            for field in ts.DERIVED_WALK_FIELDS:
                if field in row and field in derived[key]:
                    with self.subTest(f"{key} {field}"):
                        self.assertEqual(derived[key][field], row[field])

    def test_one_row_per_walk(self):
        rows = ts.walk_level_rows(self.comprehensive)
        keys = [(r["Dataset"], r["Location"], r["WalkID"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))

    def test_a_walk_invariant_field_that_varies_is_an_error(self):
        """Silently taking the first value would hide real corruption."""
        rows = _rows(self.comprehensive)
        field = next(f for f in ts.DERIVED_WALK_FIELDS if f in rows[0])
        same_walk = [i for i, r in enumerate(rows)
                     if (r["Location"], r["WalkID"])
                     == (rows[0]["Location"], rows[0]["WalkID"])]
        self.assertGreater(len(same_walk), 1, "need a multi-row walk")
        rows[same_walk[1]][field] = "CORRUPTED"
        corrupted = self.root / "corrupted.csv"
        with open(corrupted, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ts.TraceSalvageError):
            ts.walk_level_rows(corrupted)

    def test_a_file_without_walk_identity_is_refused(self):
        bad = self.root / "no_identity.csv"
        bad.write_text("tp,z_nm\nTP002,1.0\n")
        with self.assertRaises(ts.TraceSalvageError):
            ts.walk_level_rows(bad)



class AnnotationFieldsSurviveTest(unittest.TestCase):
    """The regression this reader nearly caused: annotate needs MORE than
    DERIVED_WALK_FIELDS.

    The campaign's annotate stage requires `CalibratedReplicatingComplexID` and
    copies every remaining walk-summary column onto each localization row. A
    projection limited to this module's own derived fields dropped the upstream
    summary fields, so annotate would have failed closed the moment trace salvage
    was enabled.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "comprehensive.csv"

    def _write(self, rows):
        with open(self.path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _row(self, tp, **over):
        row = {"Dataset": "rep3", "Location": "Location-01", "WalkID": "1",
               "OriginalWalkID": "0", "tp": tp,
               # an upstream walk-level field annotate requires
               "CalibratedReplicatingComplexID": "7",
               "TraceSalvageStatus": "resolved",
               # a genuinely per-locus column
               "TraceZNm": {"TP002": "10.0", "TP003": "20.0"}[tp]}
        row.update(over)
        return row

    def test_an_upstream_walk_level_field_is_carried(self):
        self._write([self._row("TP002"), self._row("TP003")])
        rows = ts.walk_level_rows(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["CalibratedReplicatingComplexID"], "7")

    def test_a_per_locus_column_is_dropped(self):
        self._write([self._row("TP002"), self._row("TP003")])
        rows = ts.walk_level_rows(self.path)
        self.assertNotIn("TraceZNm", rows[0],
                         "a column that varies within the walk has no "
                         "walk-level value")
        self.assertNotIn("tp", rows[0])

    def test_a_varying_declared_walk_field_is_still_an_error(self):
        rows = [self._row("TP002"),
                self._row("TP003", TraceSalvageStatus="unresolved")]
        self._write(rows)
        with self.assertRaises(ts.TraceSalvageError):
            ts.walk_level_rows(self.path)

    def test_two_walks_are_projected_independently(self):
        rows = [self._row("TP002"), self._row("TP003"),
                {**self._row("TP002"), "WalkID": "2", "OriginalWalkID": "1",
                 "CalibratedReplicatingComplexID": "9"},
                {**self._row("TP003"), "WalkID": "2", "OriginalWalkID": "1",
                 "CalibratedReplicatingComplexID": "9"}]
        self._write(rows)
        derived = {r["WalkID"]: r for r in ts.walk_level_rows(self.path)}
        self.assertEqual(set(derived), {"1", "2"})
        self.assertEqual(derived["1"]["CalibratedReplicatingComplexID"], "7")
        self.assertEqual(derived["2"]["CalibratedReplicatingComplexID"], "9")

if __name__ == "__main__":
    unittest.main()
