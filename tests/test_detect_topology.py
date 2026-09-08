import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import sys
import unittest

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
from detect_topology import ENGINE_SO, TopologyDetectionError, detect  # noqa: E402


ORDER = tuple(f"TP{i:03d}" for i in range(6))


class DetectTopologyTest(unittest.TestCase):
    def setUp(self):
        if not ENGINE_SO.is_file():
            self.skipTest(f"{ENGINE_SO} not built")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "chromosome_order": list(ORDER),
            "ori_tp": "TP000", "ter_tp": "TP003",
            "geometry": {"xy_pixel_nm": 1.0, "z_step_nm": 1.0},
            "power_lcb_min": 0.8, "exploratory_bh_q": 0.05,
            "family_id": "synthetic-v3",
        }))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write(path, fields, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _inputs(self, *, duplicate=False, v2=False):
        locus_path = self.root / "loci.csv"
        coordinate_path = self.root / "coordinates.csv"
        locus_fields = [
            "Dataset", "Location", "WalkID", "TP", "v3_schema_version",
            "v3_locus_evaluable", "ResolvedLateralTwoComponentPrimary",
            "v3_reduction_phase", "v3_profile_hash", "v3_profile_id",
            "v3_geometry_id", "v3_power_lcb_effective", "v3_locus_holm_p",
            "z", "y", "x",
            "v3_m2_c1_z", "v3_m2_c1_y", "v3_m2_c1_x",
            "v3_m2_c2_z", "v3_m2_c2_y", "v3_m2_c2_x",
        ]
        coordinate_fields = [
            "Dataset", "Location", "WalkID", "OriginalWalkID", "TP",
            "OriginalZNm", "OriginalYNm", "OriginalXNm",
            "z_nm", "y_nm", "x_nm", "drift_z", "drift_y", "drift_x",
            "z_raw", "y_raw", "x_raw",
            "CenterOwnershipValid", "x_err", "y_err", "z_err",
        ]
        locus_rows = []
        coordinate_rows = []

        theta_a = {
            "TP000": (50.0, 20.0, 0.0),
            "TP001": (0.0, 10.0, 0.0),
            "TP005": (100.0, 10.0, 0.0),
        }
        theta_b = {
            "TP000": (50.0, -20.0, 0.0),
            "TP001": (0.0, -10.0, 0.0),
            "TP005": (100.0, -10.0, 0.0),
        }
        theta_single = {
            "TP000": (50.0, 0.0, 0.0),
            "TP001": (0.0, 0.0, 0.0),
            "TP002": (15.0, 0.0, 8.0),
            "TP003": (50.0, 0.0, 15.0),
            "TP004": (85.0, 0.0, 8.0),
            "TP005": (100.0, 0.0, 0.0),
        }
        angle = [2.0 * math.pi * index / len(ORDER) for index in range(len(ORDER))]
        link_a = [(100.0 * math.cos(a), 100.0 * math.sin(a), 0.0) for a in angle]
        link_b = [(100.0 + 100.0 * math.cos(a), 0.0,
                   100.0 * math.sin(a)) for a in angle]

        for walk in (7, 9):
            for index, tp in enumerate(ORDER):
                if walk == 7:
                    doubled = tp in theta_a
                    one = theta_single[tp]
                    first = theta_a.get(tp, one)
                    second = theta_b.get(tp, one)
                else:
                    doubled = True
                    first, second = link_a[index], link_b[index]
                    one = tuple((a + b) / 2.0 for a, b in zip(first, second))
                def zyx(point):
                    return point[2], point[1], point[0]
                first_zyx, second_zyx = zyx(first), zyx(second)
                locus_rows.append({
                    "Dataset": "new-dataset", "Location": "FOV-A",
                    "WalkID": walk, "TP": tp,
                    "v3_schema_version": "replication_qc_v2" if v2 else "replication_qc_v3",
                    "v3_locus_evaluable": 1,
                    "ResolvedLateralTwoComponentPrimary": int(doubled),
                    "v3_reduction_phase": "final",
                    "v3_profile_hash": "synthetic-profile-hash",
                    "v3_profile_id": "synthetic-profile",
                    "v3_geometry_id": hashlib.sha256(
                        b'{"xy_pixel_nm":1.0,"z_step_nm":1.0}').hexdigest()[:16],
                    "v3_power_lcb_effective": 0.95,
                    "v3_locus_holm_p": 0.001 if doubled else 1.0,
                    "z": one[2], "y": one[1], "x": one[0],
                    "v3_m2_c1_z": first_zyx[0] if doubled else "",
                    "v3_m2_c1_y": first_zyx[1] if doubled else "",
                    "v3_m2_c1_x": first_zyx[2] if doubled else "",
                    "v3_m2_c2_z": second_zyx[0] if doubled else "",
                    "v3_m2_c2_y": second_zyx[1] if doubled else "",
                    "v3_m2_c2_x": second_zyx[2] if doubled else "",
                })
                coordinate_rows.append({
                    "Dataset": "new-dataset", "Location": "FOV-A",
                    # Prove the merged/global ID is not used for the join.
                    "WalkID": 1000 + walk, "OriginalWalkID": walk, "TP": tp,
                    "OriginalZNm": one[2], "OriginalYNm": one[1],
                    "OriginalXNm": one[0],
                    "z_nm": one[2], "y_nm": one[1], "x_nm": one[0],
                    "drift_z": 0, "drift_y": 0, "drift_x": 0,
                    "z_raw": one[2], "y_raw": one[1], "x_raw": one[0],
                    "CenterOwnershipValid": 1,
                    "x_err": 0.2, "y_err": 0.2, "z_err": 0.4,
                })
        if duplicate:
            locus_rows.append(dict(locus_rows[0]))
        self._write(locus_path, locus_fields, locus_rows)
        self._write(coordinate_path, coordinate_fields, coordinate_rows)
        self.reduction_manifest = self.root / "publication_v1_manifest.json"
        self.reduction_manifest.write_text(json.dumps({
            "schema_version": "replication_publication_v1_results_v1",
            "family_id": "synthetic-v3", "reduction_phase": "final",
            "dataset": "new-dataset", "chromosome_order": list(ORDER),
            "ori_tp": "TP000", "ter_tp": "TP003",
            "thresholds": {
                "exploratory": {"method": "BH", "q": 0.05},
                "power_lcb_min": 0.8,
            },
            "inputs": {"config_sha256": self._sha256(self.config)},
            "outputs": {"locus_results_sha256": self._sha256(locus_path)},
        }))
        return locus_path, coordinate_path

    @staticmethod
    def _read(path):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or ()), [dict(row) for row in reader]

    def test_end_to_end_outputs_reconstruct_xyz_graph_and_loops(self):
        loci, coordinates = self._inputs()
        paths = detect(
            locus_results=loci, coordinates_csv=coordinates,
            config_path=self.config, reduction_manifest=self.reduction_manifest,
            output_dir=self.root / "out",
            yamada_seeds=(0, 1), yamada_tries=20,
            link_projections=32)
        _, theta_points = self._read(paths["theta_points"])
        strands = {}
        for row in theta_points:
            self.assertEqual(row["GraphID"], "theta:new-dataset:FOV-A:7")
            strands.setdefault(row["StrandID"], []).append(
                (row["XNm"], row["YNm"], row["ZNm"]))
        self.assertEqual(sorted(strands), ["sister_A", "sister_B", "unreplicated"])
        for points in strands.values():
            self.assertEqual(points[0], strands["unreplicated"][0])
            self.assertEqual(points[-1], strands["unreplicated"][-1])
        _, link_points = self._read(paths["link_points"])
        loops = {}
        for row in link_points:
            self.assertEqual(row["LinkID"], "link:new-dataset:FOV-A:9")
            self.assertEqual(row["Closed"], "1")
            loops.setdefault(row["LoopID"], []).append(row["TP"])
        self.assertEqual(loops["sister_A"], list(ORDER))
        self.assertEqual(loops["sister_B"], list(ORDER))
        manifest = json.loads(paths["manifest"].read_text())
        self.assertEqual(manifest["provenance"]["walks_total"], 2)
        self.assertEqual(manifest["provenance"]["theta_candidates"], 1)
        self.assertEqual(manifest["provenance"]["link_candidates"], 1)
        with paths["theta_summary"].open(newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["ThetaStatus"],
                             "nominal_theta_graph")
        with paths["link_summary"].open(newline="") as handle:
            link = next(csv.DictReader(handle))
            self.assertEqual(link["LinkStatus"], "nominal_linked_polygon")
            self.assertEqual(abs(int(link["LinkingNumber"])), 1)

    def test_legacy_full_schema_mode_is_explicit_and_reconstructs_geometry(self):
        loci, coordinates = self._inputs()
        _, locus_rows = self._read(loci)
        _, coordinate_rows = self._read(coordinates)
        coordinate_by_key = {
            (row["OriginalWalkID"], row["TP"]): row
            for row in coordinate_rows
        }
        fields = [
            "Dataset", "Location", "WalkID", "OriginalWalkID", "tp",
            "RawTwoComponentCall", "RawEvaluable", "RawConfidentSingle",
            "x_raw", "y_raw", "z_raw", "drift_x", "drift_y", "drift_z",
            "OriginalXNm", "OriginalYNm", "OriginalZNm",
            "x_err", "y_err", "z_err",
            "m2_c1_z", "m2_c1_y", "m2_c1_x",
            "m2_c2_z", "m2_c2_y", "m2_c2_x",
        ]
        rows = []
        for locus in locus_rows:
            coordinate = coordinate_by_key[(locus["WalkID"], locus["TP"])]
            doubled = locus["ResolvedLateralTwoComponentPrimary"]
            rows.append({
                "Dataset": locus["Dataset"], "Location": locus["Location"],
                "WalkID": coordinate["WalkID"],
                "OriginalWalkID": coordinate["OriginalWalkID"],
                "tp": locus["TP"], "RawTwoComponentCall": doubled,
                "RawEvaluable": 1,
                "RawConfidentSingle": int(not int(doubled)),
                "x_raw": coordinate["x_raw"],
                "y_raw": coordinate["y_raw"],
                "z_raw": coordinate["z_raw"],
                "drift_x": coordinate["drift_x"],
                "drift_y": coordinate["drift_y"],
                "drift_z": coordinate["drift_z"],
                "OriginalXNm": coordinate["OriginalXNm"],
                "OriginalYNm": coordinate["OriginalYNm"],
                "OriginalZNm": coordinate["OriginalZNm"],
                "x_err": coordinate["x_err"],
                "y_err": coordinate["y_err"],
                "z_err": coordinate["z_err"],
                **{
                    f"m2_c{component}_{axis}":
                    locus[f"v3_m2_c{component}_{axis}"]
                    for component in (1, 2) for axis in ("z", "y", "x")
                },
            })
        legacy = self.root / "legacy_full_schema.csv"
        self._write(legacy, fields, rows)
        paths = detect(
            locus_results=legacy, coordinates_csv=legacy,
            config_path=self.config, reduction_manifest=None,
            output_dir=self.root / "legacy-out",
            evidence_mode="legacy-full-schema",
            evaluate_yamada_polynomial=False, link_projections=16)
        manifest = json.loads(paths["manifest"].read_text())["provenance"]
        self.assertFalse(manifest["publication_calibrated"])
        self.assertFalse(manifest["ownership_required"])
        self.assertFalse(manifest["yamada_evaluated"])
        _, theta = self._read(paths["theta_summary"])
        _, links = self._read(paths["link_summary"])
        self.assertEqual(theta[0]["ThetaStatus"], "nominal_theta_geometry")
        self.assertEqual(links[0]["LinkStatus"], "nominal_linked_polygon")

    def test_v2_and_duplicate_locus_inputs_fail_closed(self):
        loci, coordinates = self._inputs(v2=True)
        with self.assertRaisesRegex(TopologyDetectionError, "unexpected v3_schema"):
            detect(locus_results=loci, coordinates_csv=coordinates,
                   config_path=self.config, reduction_manifest=self.reduction_manifest,
                   output_dir=self.root / "bad-v2")
        loci, coordinates = self._inputs(duplicate=True)
        with self.assertRaisesRegex(TopologyDetectionError, "duplicate locus key"):
            detect(locus_results=loci, coordinates_csv=coordinates,
                   config_path=self.config, reduction_manifest=self.reduction_manifest,
                   output_dir=self.root / "duplicate")

    def test_underpowered_non_discovery_does_not_complete_theta(self):
        loci, coordinates = self._inputs()
        fields, rows = self._read(loci)
        target = next(row for row in rows
                      if row["WalkID"] == "7" and row["TP"] == "TP002")
        target["v3_power_lcb_effective"] = "0.2"
        self._write(loci, fields, rows)
        manifest = json.loads(self.reduction_manifest.read_text())
        manifest["outputs"]["locus_results_sha256"] = self._sha256(loci)
        self.reduction_manifest.write_text(json.dumps(manifest))
        paths = detect(
            locus_results=loci, coordinates_csv=coordinates,
            config_path=self.config, reduction_manifest=self.reduction_manifest,
            output_dir=self.root / "underpowered",
            yamada_seeds=(0,), yamada_tries=10,
            link_projections=16)
        _, summaries = self._read(paths["theta_summary"])
        self.assertEqual(summaries[0]["ThetaStatus"], "pattern_not_traceable")
        _, points = self._read(paths["theta_points"])
        self.assertEqual(points, [])

    def test_provisional_geometry_and_partial_coordinates_fail_closed(self):
        loci, coordinates = self._inputs()
        fields, rows = self._read(loci)
        rows[0]["v3_reduction_phase"] = "provisional"
        self._write(loci, fields, rows)
        with self.assertRaisesRegex(TopologyDetectionError, "must be 'final'"):
            detect(locus_results=loci, coordinates_csv=coordinates,
                   config_path=self.config, reduction_manifest=self.reduction_manifest,
                   output_dir=self.root / "provisional")

        loci, coordinates = self._inputs()
        fields, rows = self._read(coordinates)
        rows[0]["OriginalXNm"] = ""
        self._write(coordinates, fields, rows)
        with self.assertRaisesRegex(TopologyDetectionError, "partially populated"):
            detect(locus_results=loci, coordinates_csv=coordinates,
                   config_path=self.config, reduction_manifest=self.reduction_manifest,
                   output_dir=self.root / "partial")

    def test_manifest_and_coordinate_source_are_bound(self):
        loci, coordinates = self._inputs()
        manifest = json.loads(self.reduction_manifest.read_text())
        manifest["thresholds"]["power_lcb_min"] = 0.1
        self.reduction_manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(TopologyDetectionError, "thresholds differ"):
            detect(locus_results=loci, coordinates_csv=coordinates,
                   config_path=self.config, reduction_manifest=self.reduction_manifest,
                   output_dir=self.root / "wrong-policy")

        loci, coordinates = self._inputs()
        fields, rows = self._read(coordinates)
        target = next(row for row in rows
                      if row["WalkID"] == "1007" and row["TP"] == "TP000")
        target["x_raw"] = str(float(target["x_raw"]) + 0.01)
        self._write(coordinates, fields, rows)
        paths = detect(locus_results=loci, coordinates_csv=coordinates,
                       config_path=self.config,
                       reduction_manifest=self.reduction_manifest,
                       output_dir=self.root / "wrong-coordinates")
        _, summaries = self._read(paths["theta_summary"])
        theta = next(row for row in summaries if row["WalkID"] == "7")
        self.assertEqual(theta["ThetaStatus"], "geometry_incomplete")
        self.assertIn("raw centre differs", theta["FailureReason"])

    def test_manifest_only_unevaluable_locus_may_have_blank_profile(self):
        loci, coordinates = self._inputs()
        fields, rows = self._read(loci)
        target = next(row for row in rows
                      if row["WalkID"] == "7" and row["TP"] == "TP004")
        target["v3_locus_evaluable"] = "0"
        target["v3_schema_version"] = ""
        target["v3_profile_hash"] = ""
        target["v3_profile_id"] = ""
        target["v3_geometry_id"] = ""
        target["v3_power_lcb_effective"] = ""
        self._write(loci, fields, rows)
        manifest = json.loads(self.reduction_manifest.read_text())
        manifest["outputs"]["locus_results_sha256"] = self._sha256(loci)
        self.reduction_manifest.write_text(json.dumps(manifest))
        paths = detect(
            locus_results=loci, coordinates_csv=coordinates,
            config_path=self.config, reduction_manifest=self.reduction_manifest,
            output_dir=self.root / "manifest-only",
            yamada_seeds=(0,), yamada_tries=10, link_projections=16)
        _, summaries = self._read(paths["theta_summary"])
        self.assertEqual(summaries[0]["ThetaStatus"], "pattern_not_traceable")


if __name__ == "__main__":
    unittest.main()
