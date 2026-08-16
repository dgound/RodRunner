import csv
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.trace_salvage import (
    TraceSalvageError,
    _canonical_tp,
    apply_policy,
    fit_policy,
    load_config,
    run_dataset,
)


ORDER = list(range(5))


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TraceSalvageTest(unittest.TestCase):
    def test_rep1_cycle_suffix_maps_to_its_genomic_timepoint(self):
        self.assertEqual(_canonical_tp("TP000_CYC000"), "TP000")
        self.assertEqual(_canonical_tp("TP017_CYC123"), "TP017")

    def _location(self, root, location, *, candidate=False, endpoint=False,
                  broken_neighbor=False, wide_clean=False, schema="v2",
                  omit_drift=False):
        evidence_path = root / f"{location}_evidence.csv"
        coordinates_path = root / f"{location}_coordinates.csv"
        summary_path = root / f"{location}_summary.csv"
        evidence, coordinates = [], []
        for walk_id in (1, 2):
            for tp in ORDER:
                row = {"WalkID": walk_id, "tp": f"TP{tp:03d}"}
                if schema == "v2":
                    row.update({"replication_confirmed": 0,
                                "status": "tested_not_confirmed"})
                    prefix = "m2"
                else:
                    row.update({"ResolvedLateralTwoComponentPrimary": 0,
                                "v3_locus_evaluable": 1,
                                "v3_aggregation_status": "evaluable"})
                    prefix = "v3_m2"
                is_candidate = candidate and walk_id == 1 and tp == 2
                is_endpoint = endpoint and walk_id == 1 and tp == 0
                is_broken_neighbor = broken_neighbor and walk_id == 1 and tp == 1
                if is_candidate or is_endpoint or is_broken_neighbor:
                    if schema == "v2":
                        row.update({"replication_confirmed": 1,
                                    "status": "confirmed"})
                    else:
                        row["ResolvedLateralTwoComponentPrimary"] = 1
                if is_candidate:
                    # c1 follows x=10 -> 20 -> 30; c2 is deliberately far.
                    row.update({
                        f"{prefix}_c1_z": 1, f"{prefix}_c1_y": 0,
                        f"{prefix}_c1_x": 20,
                        f"{prefix}_c2_z": 4, f"{prefix}_c2_y": 0,
                        f"{prefix}_c2_x": 80,
                    })
                evidence.append(row)
                x = tp * 10.0
                if wide_clean and walk_id == 2 and tp >= 3:
                    x += 1000.0
                coord = {"WalkID": walk_id, "tp": f"TP{tp:03d}",
                         "z_nm": float(tp), "y_nm": 0.0, "x_nm": x}
                if not omit_drift:
                    coord.update({"drift_z": 0.0, "drift_y": 0.0,
                                  "drift_x": 0.0})
                coordinates.append(coord)
        evidence_fields = sorted({field for row in evidence for field in row})
        coord_fields = sorted({field for row in coordinates for field in row})
        _write_csv(evidence_path, evidence_fields, evidence)
        _write_csv(coordinates_path, coord_fields, coordinates)
        _write_csv(summary_path, ["WalkID", "legacy_field"], [
            {"WalkID": 1, "legacy_field": f"{location}-one"},
            {"WalkID": 2, "legacy_field": f"{location}-two"},
        ])
        return {"location": location, "evidence_csv": str(evidence_path),
                "coordinates_csv": str(coordinates_path),
                "walk_summary_csv": str(summary_path)}

    def _config(self, root, locations, *, schema="v2", min_links=2):
        path = root / "trace_config.json"
        path.write_text(json.dumps({
            "dataset": "unit", "schema": schema,
            "chromosome_order": ORDER, "ori_tp": 3, "ter_tp": 0,
            "xy_pixel_nm": 1.0, "z_step_nm": 1.0,
            "min_calibration_links": min_links,
            "clean_link_quantile": 0.95, "winner_ratio": 0.8,
            "locations": locations,
        }))
        return load_config(path)

    def test_salvages_isolated_call_using_xy_and_emits_full_xyz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, [
                self._location(root, "A", candidate=True),
                self._location(root, "B"),
            ])
            manifest = run_dataset(config, root / "out", "unit", diagnostics=True)
            with (root / "out" /
                  "unit_trace_salvage_full_schema.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            target = next(row for row in rows if row["Location"] == "A"
                          and row["WalkID"] == "1" and row["tp"] == "TP002")
            self.assertEqual(target["RawTwoComponentCall"], "1")
            self.assertEqual(target["TraceSalvaged"], "1")
            self.assertEqual(target["TraceSelectedComponent"], "1")
            self.assertEqual(target["TraceXNm"], "20")
            self.assertEqual(target["TraceZNm"], "1")
            self.assertEqual(target["TraceSalvageReason"], "salvaged")
            self.assertEqual(target["m2_c1_x"], "20")
            self.assertEqual(target["legacy_field"], "A-one")
            self.assertEqual(target["WalkID"], "1")
            self.assertEqual(target["OriginalWalkID"], "1")
            complete_path = (root / "out" /
                             "unit_trace_salvage_full_schema_complete.csv")
            with complete_path.open() as handle:
                complete_rows = list(csv.DictReader(handle))
            self.assertEqual(len(complete_rows), 20)
            with (root / "out" / "unit_trace_salvage_walk_summary.csv").open() as handle:
                walks = list(csv.DictReader(handle))
            summary = next(row for row in walks if row["Location"] == "A"
                           and row["WalkID"] == "1")
            self.assertEqual(summary["legacy_field"], "A-one")
            self.assertEqual(summary["TraceSalvagedMultiSpot"], "0")
            self.assertEqual(summary["TraceSalvagedSingleSpotComplete"], "1")
            self.assertEqual(manifest["counts"]["salvaged_loci"], 1)

    def test_full_schema_is_built_from_coordinate_rows_and_canonicalizes_tp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locations = [
                self._location(root, "A", candidate=True),
                self._location(root, "B"),
            ]

            # Model the rep1 spelling and include acquired timepoints outside
            # the five-locus salvage interval.  An evidence-only TP024 must not
            # create a synthetic output row.
            coordinate_path = Path(locations[0]["coordinates_csv"])
            with coordinate_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                coordinate_rows = list(reader)
                coordinate_fields = list(reader.fieldnames or [])
            coordinate_fields = [
                "TP" if field == "tp" else field for field in coordinate_fields
            ] + ["z_raw", "y_raw", "x_raw"]
            for row in coordinate_rows:
                tp = row.pop("tp")
                row["TP"] = f"{tp}_CYC{int(tp[2:]):03d}"
                row.update({"z_raw": "1", "y_raw": "2", "x_raw": "3"})
            for tp, x_nm in ((22, 2200.0), (23, 2300.0)):
                coordinate_rows.append({
                    "TP": f"TP{tp:03d}_CYC{tp:03d}", "WalkID": 1,
                    "z_nm": 10.0, "y_nm": 20.0, "x_nm": x_nm,
                    "drift_z": 0.0, "drift_y": 0.0, "drift_x": 0.0,
                    "z_raw": 10.0, "y_raw": 20.0, "x_raw": x_nm / 10,
                })
            _write_csv(coordinate_path, coordinate_fields, coordinate_rows)

            evidence_path = Path(locations[0]["evidence_csv"])
            with evidence_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                evidence_rows = list(reader)
                evidence_fields = list(reader.fieldnames or [])
            evidence_rows.extend([
                {"tp": "TP022_CYC022", "WalkID": 1,
                 "replication_confirmed": 0,
                 "status": "tested_not_confirmed"},
                {"tp": "TP023_CYC023", "WalkID": 1,
                 "replication_confirmed": 0,
                 "status": "tested_not_confirmed"},
                {"tp": "TP024_CYC024", "WalkID": 1,
                 "replication_confirmed": 0,
                 "status": "tested_not_confirmed"},
                {"tp": "TP002_CYC002", "WalkID": 99,
                 "replication_confirmed": 0,
                 "status": "tested_not_confirmed"},
            ])
            _write_csv(evidence_path, evidence_fields, evidence_rows)

            summary_path = Path(locations[0]["walk_summary_csv"])
            with summary_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                summary_rows = list(reader)
                summary_fields = list(reader.fieldnames or [])
            summary_rows.append({"WalkID": 99, "legacy_field": "no-coordinates"})
            _write_csv(summary_path, summary_fields, summary_rows)

            config = self._config(root, locations)
            manifest = run_dataset(config, root / "out", "unit", diagnostics=True)
            full_path = root / "out" / "unit_trace_salvage_full_schema.csv"
            with full_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)

            self.assertEqual(len(rows), 22)
            self.assertEqual(manifest["counts"]["full_schema_rows"], 22)
            self.assertEqual(manifest["counts"]["walks"], 4)
            self.assertEqual(manifest["counts"]["decision_walks"], 5)
            self.assertEqual(
                manifest["counts"]["walks_without_coordinate_rows"], 1)
            self.assertEqual(manifest["counts"]["complete_clean"], 3)
            self.assertEqual(manifest["decision_counts"]["complete_clean"], 3)
            self.assertIn("tp", fields)
            self.assertNotIn("TP", fields)
            self.assertNotIn("OriginalTP", fields)
            self.assertFalse(any(not row["tp"] for row in rows))
            self.assertIn("TP022", {row["tp"] for row in rows})
            self.assertIn("TP023", {row["tp"] for row in rows})
            self.assertNotIn("TP024", {row["tp"] for row in rows})
            self.assertEqual(sorted({int(row["WalkID"]) for row in rows}),
                             [1, 2, 3, 4])

            outside = next(
                row for row in rows
                if row["Location"] == "A" and row["OriginalWalkID"] == "1"
                and row["tp"] == "TP023")
            self.assertEqual(outside["OriginalXNm"], "2300")
            self.assertEqual(outside["TraceXNm"], "2300")
            self.assertEqual(outside["x_nm"], "2300")
            self.assertEqual(float(outside["x_raw"]), 230.0)
            self.assertEqual(outside["TraceSalvageReason"],
                             "outside_salvage_range")

            location_path = (
                root / "out" / "locations" /
                "A_trace_salvage_full_schema.csv")
            with location_path.open(newline="") as handle:
                location_reader = csv.DictReader(handle)
                location_fields = list(location_reader.fieldnames or [])
                location_rows = list(location_reader)
            self.assertEqual(len(location_rows), 12)
            self.assertNotIn("OriginalWalkID", location_fields)
            self.assertEqual({row["WalkID"] for row in location_rows},
                             {"1", "2"})

            complete_path = (
                root / "out" /
                "unit_trace_salvage_full_schema_complete.csv")
            with complete_path.open(newline="") as handle:
                complete_rows = list(csv.DictReader(handle))
            self.assertEqual(len(complete_rows), 22)
            self.assertEqual(
                {(row["Location"], row["OriginalWalkID"], row["tp"])
                 for row in complete_rows},
                {(row["Location"], row["OriginalWalkID"], row["tp"])
                 for row in rows})
            with (root / "out" /
                  "unit_trace_salvage_walk_summary.csv").open(
                      newline="") as handle:
                walk_rows = list(csv.DictReader(handle))
            self.assertNotIn("99", {row["OriginalWalkID"] for row in walk_rows})

    def test_endpoints_and_nonisolated_calls_are_never_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, [
                self._location(root, "A", candidate=True, endpoint=True,
                               broken_neighbor=True),
                self._location(root, "B"),
            ])
            run_dataset(config, root / "out", "unit", diagnostics=True)
            with (root / "out" / "unit_trace_salvage_audit.csv").open() as handle:
                rows = {(row["Location"], row["WalkID"], row["TP"]): row
                        for row in csv.DictReader(handle)}
            self.assertEqual(rows[("A", "1", "TP000")]["TraceSalvageReason"],
                             "endpoint_never_salvaged")
            self.assertEqual(rows[("A", "1", "TP002")]["TraceSalvageReason"],
                             "not_isolated_between_confident_singles")

    def test_loo_calibration_excludes_target_location_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, [
                self._location(root, "A", candidate=True, wide_clean=True),
                self._location(root, "B"),
            ])
            policy = fit_policy(config)
            self.assertEqual(policy["calibration"]["locations"]["A"]
                             ["source_locations"], ["B"])
            self.assertEqual(policy["calibration"]["locations"]["A"]
                             ["max_link_xy_nm"], 10.0)
            run_dataset(config, root / "out", "unit", diagnostics=True)
            with (root / "out" / "unit_trace_salvage_audit.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            target = next(row for row in rows if row["Location"] == "A"
                          and row["WalkID"] == "1" and row["TP"] == "TP002")
            self.assertEqual(target["TraceSalvageReason"], "salvaged")

            insufficient = self._config(root, [
                self._location(root, "C", candidate=True),
                self._location(root, "D"),
            ], min_links=10)
            run_dataset(insufficient, root / "out2", "unit", diagnostics=True)
            with (root / "out2" / "unit_trace_salvage_audit.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            target = next(row for row in rows if row["Location"] == "C"
                          and row["WalkID"] == "1" and row["TP"] == "TP002")
            self.assertEqual(target["TraceSalvageReason"], "calibration_unavailable")

    def test_v3_adapter_and_frozen_policy_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locations = [self._location(root, "A", candidate=True, schema="v3"),
                         self._location(root, "B", schema="v3")]
            config = self._config(root, locations, schema="v3")
            policy = fit_policy(config)
            subset = self._config(root, [locations[0]], schema="v3")
            manifest = apply_policy(subset, policy, root / "out", "unit",
                                    diagnostics=True)
            self.assertEqual(manifest["counts"]["salvaged_loci"], 1)

    def test_missing_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, [
                self._location(root, "A", candidate=True, omit_drift=True),
                self._location(root, "B", omit_drift=True),
            ])
            with self.assertRaisesRegex(TraceSalvageError, "missing drift_z"):
                fit_policy(config)


if __name__ == "__main__":
    unittest.main()
