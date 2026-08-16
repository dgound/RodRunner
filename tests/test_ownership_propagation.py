"""Test G: centre ownership must survive every production output stage.

This exercises the REAL modules rather than re-implementing their logic:
``pipeline/drift_correct.py`` as a subprocess, ``classify_replication.main``,
``drivers/merge_all_walks.main`` and ``pipeline.trace_salvage`` in process. The
point is not that a helper computes the right number, it is that the columns and
the decision actually arrive in the files an analyst opens.

Stages covered, in pipeline order:

  1. localizations_640.csv                       (localizer schema)
  2. localizations_640_drift_corrected*.csv      (real drift_correct)
  3. replication classification + QC side-car    (real classify_replication)
  4. per-Location <Loc>_trace_salvage_full_schema.csv
  5. all_locations_640_valid_walks.csv           (real merge_all_walks)
  6. *_trace_salvage_full_schema.csv             (unfiltered comprehensive)
  7. *_trace_salvage_full_schema_complete.csv    (filtered comprehensive)
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline import center_ownership as _own
from pipeline import trace_salvage as _ts

PY = sys.executable
TPS = [f"TP{index:03d}" for index in range(24)]
#: Configured genomic loci, matching the widefield presets.
LOCI = list(range(2, 22))
ORI_TP, TER_TP = 17, 7

LEGACY_LOC_FIELDS = [
    "tp", "WalkID", "z", "y", "x", "sigma_z", "sigma_y", "sigma_x",
    "amplitude", "background", "z_err", "y_err", "x_err",
]


def _read(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ownership_row(valid=True, category=None):
    """Per-locus ownership fields exactly as the localizer writes them."""
    category = category or (_own.OWN_MASK if valid else _own.BACKGROUND)
    return _own.row_fields(
        _own.POLICY_STRICT, category, final=category,
        registration_applied=True, reference_nm="405", target_nm="640")


class GStageOneAndTwoTest(unittest.TestCase):
    """Stages 1-2: the localizer schema, and the real drift-correction run."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name) / "run"
        cls.root = root
        drift = {tp: (0.1 * i, 0.2 * i, -0.15 * i)
                 for i, tp in enumerate(TPS)}

        fiducials = []
        for roi in range(60):
            z0, y0, x0 = 5.0, 300 + roi * 20.0, 300 + roi * 19.0
            for tp in TPS:
                dz, dy, dx = drift[tp]
                fiducials.append({
                    "tp": tp, "WalkID": roi, "z": z0 + dz, "y": y0 + dy,
                    "x": x0 + dx, "sigma_z": 1.5, "sigma_y": 1.2,
                    "sigma_x": 1.2, "amplitude": 100, "background": 10,
                    "z_err": 0.01, "y_err": 0.01, "x_err": 0.01,
                })
        _write(root / "probe_1/localizations/localizations_488.csv",
               LEGACY_LOC_FIELDS, fiducials)

        # Walk 0 fully owned; walk 1 has one locus recorded as NOT owned, which
        # is what `allow` produces (strict would have dropped the row).
        cls.expected = {}
        rows = []
        for walk in range(3):
            for tp in TPS:
                dz, dy, dx = drift[tp]
                valid = not (walk == 1 and tp == "TP005")
                own = _ownership_row(valid=valid)
                cls.expected[(walk, tp)] = own
                rows.append({
                    "tp": tp, "WalkID": walk, "z": 5.0 + dz,
                    "y": 500 + walk * 30 + dy, "x": 500 + walk * 31 + dx,
                    "sigma_z": 1.5, "sigma_y": 1.2, "sigma_x": 1.2,
                    "amplitude": 200, "background": 10,
                    "z_err": 0.02, "y_err": 0.02, "x_err": 0.02, **own,
                })
        cls.loc_csv = root / "probe_0/localizations/localizations_640.csv"
        _write(cls.loc_csv,
               LEGACY_LOC_FIELDS + list(_own.OWNERSHIP_ROW_FIELDS), rows)

        env = dict(os.environ, PIPELINE_OUT=str(root))
        cls.proc = subprocess.run(
            [PY, str(_REPO / "pipeline/drift_correct.py"), "--fov_size", "1866"],
            env=env, capture_output=True, text=True, timeout=900)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_real_drift_correction_run_succeeded(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])

    def test_stage_1_localizations_keep_the_legacy_prefix_then_ownership(self):
        fields, _ = _read(self.loc_csv)
        self.assertEqual(fields[:len(LEGACY_LOC_FIELDS)], LEGACY_LOC_FIELDS,
                         "legacy column prefix must be byte-position stable")
        self.assertEqual(fields[len(LEGACY_LOC_FIELDS):],
                         list(_own.OWNERSHIP_ROW_FIELDS))

    def test_stage_2_all_three_drift_products_carry_all_eleven_fields(self):
        base = self.root / "probe_0/localizations"
        for name in ("localizations_640_drift_corrected.csv",
                     "localizations_640_drift_corrected_full_coverage.csv",
                     "localizations_640_drift_corrected_filtered.csv"):
            with self.subTest(name):
                fields, rows = _read(base / name)
                for column in _own.OWNERSHIP_ROW_FIELDS:
                    self.assertIn(column, fields)
                self.assertTrue(rows, f"{name} is empty")

    def test_stage_2_values_are_carried_not_regenerated(self):
        _, rows = _read(self.root / "probe_0/localizations"
                        / "localizations_640_drift_corrected.csv")
        checked = 0
        for row in rows:
            key = (int(row["WalkID"]), row["tp"])
            if key not in self.expected:
                continue
            for column, value in self.expected[key].items():
                self.assertEqual(row[column], str(value),
                                 f"{column} changed for {key}")
            checked += 1
        self.assertGreater(checked, 0)

    def test_stage_2_preserved_the_recorded_rejection(self):
        _, rows = _read(self.root / "probe_0/localizations"
                        / "localizations_640_drift_corrected.csv")
        rejected = [row for row in rows
                    if row["CenterOwnershipValid"] in ("0", 0)]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(int(rejected[0]["WalkID"]), 1)
        self.assertEqual(rejected[0]["tp"], "TP005")
        self.assertEqual(rejected[0]["CenterOwnershipFailureReason"],
                         "center_background")
        self.assertNotEqual(rejected[0]["z"], "",
                            "the unconstrained coordinate is retained for audit")


def _qc_row(walk, tp_index, *, confirmed=0, status="tested_not_confirmed",
            ownership_unevaluable=0, reason=""):
    return {
        "tp": f"TP{tp_index:03d}", "WalkID": walk,
        "status": status, "replication_confirmed": confirmed,
        "pinned_double_fit_suspect": 0, "merged_mask_suspect": 0,
        "spot_sep_xy_lcb_nm": 400.0, "delta_BIC": 3000.0, "amp_ratio": 0.8,
        "ownership_unevaluable": ownership_unevaluable,
        "ownership_failure_reason": reason,
    }


QC_FIELDS = list(_qc_row(0, 2).keys())


class GStageThreeReplicationTest(unittest.TestCase):
    """Stage 3: ownership must reshape the walk-level replication call."""

    def _classify(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        qc = Path(tmp.name) / "localizations_640_replication_qc.csv"
        _write(qc, QC_FIELDS, rows)
        argv = [
            "classify_replication", "--replication_qc_csv", str(qc),
            "--outdir", tmp.name, "--no_annotate",
            "--ori_tp", str(ORI_TP), "--ter_tp", str(TER_TP),
            "--replication_tp_start", str(LOCI[0]),
            "--replication_tp_end", str(LOCI[-1]),
        ]
        from pipeline import classify_replication
        old = sys.argv
        sys.argv = argv
        try:
            classify_replication.main()
        finally:
            sys.argv = old
        _, out = _read(
            Path(tmp.name) / "localizations_640_replication_classification.csv")
        return {int(row["WalkID"]): row for row in out}

    def test_an_ownership_invalid_ori_is_not_reported_non_replicating(self):
        rows = [_qc_row(0, tp) for tp in LOCI if tp != ORI_TP]
        rows.append(_qc_row(0, ORI_TP, status="ownership_unevaluable",
                            ownership_unevaluable=1,
                            reason="m2_c1_center_background"))
        state = self._classify(rows)[0]["replication_state"]
        self.assertNotEqual(state, "non_replicating")
        self.assertEqual(state, "indeterminate_ori")

    def test_an_ownership_invalid_locus_cannot_set_multispot(self):
        # A confirmed two-spot that ownership rejected. Belt-and-braces: the
        # column alone must veto it even if `replication_confirmed` still says 1.
        rows = [_qc_row(0, tp) for tp in LOCI if tp != 11]
        rows.append(_qc_row(0, 11, confirmed=1, status="confirmed",
                            ownership_unevaluable=1,
                            reason="m2_c2_center_other_active_cell"))
        walk = self._classify(rows)[0]
        self.assertEqual(int(walk["MultiSpot"]), 0)
        self.assertEqual(int(walk["Replicating"]), 0)

    def test_an_ownership_invalid_locus_is_not_evidence_of_a_single_spot(self):
        rows = [_qc_row(0, tp) for tp in LOCI if tp != 11]
        rows.append(_qc_row(0, 11, status="ownership_unevaluable",
                            ownership_unevaluable=1, reason="m2_c1_center_background"))
        walk = self._classify(rows)[0]
        self.assertEqual(int(walk["replication_pattern_complete"]), 0,
                         "an unevaluable locus must not complete the pattern")

    def test_a_fully_owned_walk_keeps_its_expected_classification(self):
        rows = [_qc_row(0, tp) for tp in LOCI]
        walk = self._classify(rows)[0]
        self.assertEqual(walk["replication_state"], "non_replicating")
        self.assertEqual(int(walk["replication_pattern_complete"]), 1)
        self.assertEqual(int(walk["MultiSpot"]), 0)

    def test_a_genuine_two_spot_still_sets_multispot(self):
        rows = [_qc_row(0, tp) for tp in LOCI if tp != 11]
        rows.append(_qc_row(0, 11, confirmed=1, status="confirmed"))
        self.assertEqual(int(self._classify(rows)[0]["MultiSpot"]), 1)


class GStageFourToSevenTest(unittest.TestCase):
    """Stages 4-7: per-Location, merged, and both trace-salvage products.

    The comprehensive (unfiltered) trace-salvage CSV must be self-contained and
    lossless: one row per localization ATTEMPT, whatever became of it. The
    fixture therefore contains every way an attempt can fail, with location-local
    WalkIDs deliberately reused across Locations.
    """

    LOCATIONS = ("Location-01", "Location-02", "Location-03")
    WALKS = tuple(range(1, 11))

    #: Every special case lives in Location-02, so 01 and 03 are clean controls.
    REJECTED = ("Location-02", 3, "TP009")      # 19 accepted + 1 rejected
    MISSING = ("Location-02", 5, "TP012")       # never submitted at all
    REJECTED_ORI = ("Location-02", 6, f"TP{ORI_TP:03d}")
    M2_INVALID = ("Location-02", 7, "TP011")    # centre owned, M2 component not
    FIT_FAILED = ("Location-02", 8, "TP004")    # no centre, so no verdict
    #: Every configured locus accepted and evaluable, but the walk is absent from
    #: the full-coverage decision input -- what happens in production when a walk
    #: misses a CONTROL acquisition outside the configured genomic interval.
    COVERAGE_DROPPED = 9
    #: Every locus ownership-rejected: no coordinates, no replication side-car
    #: rows, and therefore no replication measurement of any kind. It must still
    #: be classified, and its attempts must still be auditable.
    ALL_REJECTED = 10

    #: Walks that survive every filter and must still reach the filtered output.
    CLEAN_WALKS = frozenset(
        [(location, walk)
         for location in ("Location-01", "Location-03")
         for walk in range(1, 9)]
        + [("Location-02", 1), ("Location-02", 2), ("Location-02", 4)])

    COORD_FIELDS = (
        ["tp", "WalkID", "z_nm", "y_nm", "x_nm", "z_raw", "y_raw", "x_raw",
         "drift_z", "drift_y", "drift_x",
         "PinnedDoubleFitSuspect", "MergedMaskSuspect",
         # classify_replication annotates these onto the drift-corrected CSVs;
         # the merge requires them.
         "Replicating", "MultiSpot", "replication_state",
         "replication_pattern_complete", "ReplicationClassificationReason"]
        + list(_own.OWNERSHIP_ROW_FIELDS))

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name) / "outputs"
        self.outdir = Path(self._tmp.name) / "merged"
        self._build()

    # -- fixture -----------------------------------------------------------
    def _role(self, location, walk, tp):
        for name, (loc, wid, label) in (
                ("rejected", self.REJECTED),
                ("missing", self.MISSING),
                ("rejected_ori", self.REJECTED_ORI),
                ("m2_invalid", self.M2_INVALID),
                ("fit_failed", self.FIT_FAILED)):
            if (location, walk, tp) == (loc, wid, label):
                return name
        return "accepted"

    def _build(self, *, reject=True):
        """Write one synthetic finalized tree.

        ``reject=False`` reproduces legacy ownership-``allow`` behaviour: every
        attempt keeps its coordinate, so it is the backward-compatibility arm.
        """
        for location in self.LOCATIONS:
            d = self.base / location / "probe_0/localizations"
            unfiltered, qc, cls_rows, audit = [], [], [], []
            coverage_walks = []
            for walk in self.WALKS:
                walk_rows, walk_complete = [], True
                for tp_i in LOCI:
                    tp = f"TP{tp_i:03d}"
                    role = self._role(location, walk, tp)
                    if role == "missing":
                        # Never submitted: absent from the ledger too. Nothing may
                        # invent a row for it anywhere downstream.
                        walk_complete = False
                        continue
                    rejected = reject and (
                        role in ("rejected", "rejected_ori")
                        or walk == self.ALL_REJECTED)
                    failed = reject and role == "fit_failed"
                    z_raw, y_raw, x_raw = (
                        5.0 + tp_i * 0.01, 4.0 + walk * 0.1, 10.0 + tp_i * 2.5)
                    entry = {
                        "tp": tp, "WalkID": walk,
                        "fit_ret_code": -1 if failed else 0,
                        "localization_written": int(not (rejected or failed)),
                        "center_ownership_initial": "",
                        "center_ownership_final": "",
                        "center_ownership_pass": 0 if (rejected or failed) else 1,
                        "center_refit_attempted": 0,
                        "center_refit_success": 0,
                        "unevaluable_reason": (
                            "center_background" if rejected
                            else "fit_failed:-1" if failed else ""),
                    }
                    if not failed:
                        entry.update({
                            "center_fit_z_px": f"{z_raw:.4f}",
                            "center_fit_y_px": f"{y_raw:.4f}",
                            "center_fit_x_px": f"{x_raw:.4f}",
                        })
                        entry.update(_ownership_row(valid=not rejected))
                        entry["center_ownership_initial"] = (
                            _own.BACKGROUND if rejected else _own.OWN_MASK)
                        entry["center_ownership_final"] = (
                            entry["center_ownership_initial"])
                    audit.append(entry)
                    if rejected or failed:
                        # STRICT drops the coordinate row entirely; a failed fit
                        # never produced one. Both must still be represented.
                        walk_complete = False
                        continue
                    walk_rows.append({
                        "tp": tp, "WalkID": walk,
                        "z_nm": 500.0, "y_nm": 400.0 + walk * 10.0,
                        "x_nm": 1000.0 + tp_i * 250.0,
                        "z_raw": f"{z_raw:.4f}", "y_raw": f"{y_raw:.4f}",
                        "x_raw": f"{x_raw:.4f}",
                        "drift_z": 0.0, "drift_y": 0.0, "drift_x": 0.0,
                        "PinnedDoubleFitSuspect": 0, "MergedMaskSuspect": 0,
                        "Replicating": 0, "MultiSpot": 0,
                        "replication_state": "non_replicating",
                        "replication_pattern_complete": 1,
                        "ReplicationClassificationReason": "",
                        **_ownership_row(valid=True),
                    })
                    if reject and role == "m2_invalid":
                        # The centre is owned; the second fitted component is not.
                        # replication_qc records that as unevaluable, never as a
                        # negative single-spot result.
                        qc.append(_qc_row(
                            walk, tp_i, status="ownership_unevaluable",
                            ownership_unevaluable=1,
                            reason="m2_c2_center_other_active_cell"))
                    else:
                        qc.append(_qc_row(walk, tp_i))
                unfiltered.extend(walk_rows)
                if walk_complete and walk != self.COVERAGE_DROPPED:
                    coverage_walks.append(walk)
                wholly_unevaluable = reject and walk == self.ALL_REJECTED
                cls_rows.append({
                    "WalkID": walk, "Replicating": 0, "MultiSpot": 0,
                    "replication_state": ("indeterminate_ori"
                                          if wholly_unevaluable
                                          else "non_replicating"),
                    "replication_pattern_complete": (
                        0 if wholly_unevaluable else int(walk_complete)),
                    "ReplicationClassificationReason": (
                        "all_localizations_ownership_unevaluable"
                        if wholly_unevaluable else ""),
                    "PinnedDoubleFitSuspect": 0, "MergedMaskSuspect": 0,
                })
            covered = [row for row in unfiltered
                       if int(row["WalkID"]) in set(coverage_walks)]
            _write(d / "localizations_640_drift_corrected.csv",
                   self.COORD_FIELDS, unfiltered)
            _write(d / "localizations_640_drift_corrected_full_coverage.csv",
                   self.COORD_FIELDS, covered)
            _write(d / "localizations_640_drift_corrected_filtered.csv",
                   self.COORD_FIELDS, covered)
            _write(d / "localizations_640_replication_qc.csv", QC_FIELDS, qc)
            _write(d / "localizations_640_replication_classification.csv",
                   list(cls_rows[0].keys()), cls_rows)
            _write(d / "localizations_640_z_gate_status.csv",
                   sorted({key for row in audit for key in row}), audit)
            with open(d / "localizations_640_replication_policy.json", "w") as h:
                json.dump({"hierarchy": {
                    "ori_tp": ORI_TP, "ter_tp": TER_TP,
                    "replication_tp_start": LOCI[0],
                    "replication_tp_end": LOCI[-1]}}, h)

    def _merge(self, *extra):
        from drivers import merge_all_walks
        argv = [
            "merge_all_walks", "--base", str(self.base),
            "--outdir", str(self.outdir), "--location_prefix", "Location-",
            "--trace_salvage_diagnostics",
            *extra,
        ]
        old = sys.argv
        sys.argv = argv
        try:
            merge_all_walks.main()
        finally:
            sys.argv = old

    def _merge_with_salvage(self):
        universe = Path(self._tmp.name) / "expected_locations.json"
        universe.write_text(json.dumps(list(self.LOCATIONS)))
        self._merge(
            "--trace_salvage", "--trace_salvage_dataset", "custom",
            "--trace_salvage_chromosome_order",
            ",".join(str(tp) for tp in LOCI),
            "--trace_salvage_ori_tp", str(ORI_TP),
            "--trace_salvage_ter_tp", str(TER_TP),
            "--trace_salvage_xy_pixel_nm", "103.28",
            "--trace_salvage_z_step_nm", "100.0",
            "--trace_salvage_expected_locations", str(universe),
            "--trace_salvage_output_prefix", "all_locations_640",
        )

    def _comprehensive(self):
        return _read(
            self.outdir / "all_locations_640_trace_salvage_full_schema.csv")

    def _filtered(self):
        return _read(
            self.outdir
            / "all_locations_640_trace_salvage_full_schema_complete.csv")

    @staticmethod
    def _pick(rows, location, walk, tp=None):
        return [row for row in rows
                if row["Location"] == location
                and int(row["OriginalWalkID"]) == walk
                and (tp is None or row["tp"] == tp)]

    #: Ownership-rejected attempts the fixture creates: the single rejected
    #: locus and the rejected Ori in Location-02, plus every locus of the
    #: all-rejected walk in all three Locations.
    EXPECTED_REJECTED = 2 + 3 * len(LOCI)

    #: Canonical, downstream-consumable coordinate columns. A rejected attempt
    #: must carry NONE of them.
    CANONICAL_COORDINATES = (
        "z_nm", "y_nm", "x_nm", "z_raw", "y_raw", "x_raw",
        "OriginalZNm", "OriginalYNm", "OriginalXNm",
        "TraceZNm", "TraceYNm", "TraceXNm",
    )

    # -- stage 5: the legacy merged full-coverage product ------------------
    def test_stage_5_merged_output_carries_locus_and_walk_ownership(self):
        self._merge()
        fields, rows = _read(self.outdir / "all_locations_640_valid_walks.csv")
        for column in _own.OWNERSHIP_ROW_FIELDS:
            self.assertIn(column, fields)
        for column in _own.OWNERSHIP_WALK_FIELDS:
            self.assertIn(column, fields)
        self.assertTrue(rows)

    def test_stage_5_expected_count_is_the_configured_universe(self):
        self._merge()
        _, rows = _read(self.outdir / "all_locations_640_valid_walks.csv")
        counts = {row["CenterOwnershipExpectedCount"] for row in rows}
        self.assertEqual(counts, {str(len(LOCI))})

    def test_stage_5_an_unaffected_walk_is_ownership_complete(self):
        self._merge()
        _, rows = _read(self.outdir / "all_locations_640_valid_walks.csv")
        clean = [row for row in rows if row["Location"] == "Location-01"]
        self.assertTrue(clean)
        self.assertEqual({row["CenterOwnershipComplete"] for row in clean},
                         {"1"})

    def test_stage_5_the_ownership_join_does_not_change_row_multiplicity(self):
        _, before = _read(
            self.base / "Location-01/probe_0/localizations"
            / "localizations_640_drift_corrected_full_coverage.csv")
        self._merge()
        _, rows = _read(self.outdir / "all_locations_640_valid_walks.csv")
        first = [row for row in rows if row["Location"] == "Location-01"]
        self.assertEqual(len(first), len(before))

    def test_stage_5_global_walkid_is_unique_and_continuous(self):
        self._merge()
        _, rows = _read(self.outdir / "all_locations_640_valid_walks.csv")
        ids = sorted({int(row["WalkID"]) for row in rows})
        self.assertEqual(ids, list(range(1, len(ids) + 1)))
        by_id = {}
        for row in rows:
            by_id.setdefault(int(row["WalkID"]), set()).add(row["Location"])
        self.assertTrue(all(len(v) == 1 for v in by_id.values()))

    # -- 1. attempt preservation ------------------------------------------
    def test_a_rejected_attempt_is_present_exactly_once_with_its_reason(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        location, walk, tp = self.REJECTED
        walk_rows = self._pick(full, location, walk)
        self.assertEqual(len(walk_rows), len(LOCI),
                         "19 accepted + 1 rejected must be 20 rows")
        hit = self._pick(full, location, walk, tp)
        self.assertEqual(len(hit), 1, "exactly one row for the rejected locus")
        row = hit[0]
        self.assertEqual(row["tp"], tp)
        self.assertTrue(row["tp"].strip())
        self.assertEqual(row["CenterOwnershipValid"], "0")
        self.assertEqual(row["CenterOwnershipFailureReason"], "center_background")
        self.assertEqual(row["LocalizationAttemptStatus"], "ownership_rejected")
        self.assertEqual(row["LocalizationAttemptReason"], "center_background")

    def test_a_rejected_attempt_keeps_only_audit_coordinates(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        row = self._pick(full, *self.REJECTED)[0]
        for field in _ts.AUDIT_COORDINATE_FIELDS:
            self.assertNotEqual(row[field].strip(), "",
                                f"{field} must retain the unconstrained fit")
        for field in self.CANONICAL_COORDINATES:
            self.assertEqual(row[field].strip(), "",
                             f"{field} must be unusable for a rejected attempt")

    def test_the_audit_coordinate_is_the_fitted_centre_in_nanometres(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        row = self._pick(full, *self.REJECTED)[0]
        _, ledger = _read(
            self.base / self.REJECTED[0] / "probe_0/localizations"
            / "localizations_640_z_gate_status.csv")
        entry = [item for item in ledger
                 if int(item["WalkID"]) == self.REJECTED[1]
                 and item["tp"] == self.REJECTED[2]][0]
        self.assertAlmostEqual(float(row["UnconstrainedFitZNm"]),
                               float(entry["center_fit_z_px"]) * 100.0, places=6)
        self.assertAlmostEqual(float(row["UnconstrainedFitXNm"]),
                               float(entry["center_fit_x_px"]) * 103.28,
                               places=6)

    def test_an_accepted_attempt_carries_the_same_audit_schema(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        row = self._pick(full, "Location-01", 1, "TP002")[0]
        self.assertEqual(row["LocalizationAttemptStatus"], "accepted")
        self.assertAlmostEqual(float(row["UnconstrainedFitXNm"]),
                               float(row["x_raw"]) * 103.28, places=6)
        self.assertNotEqual(row["x_nm"].strip(), "")

    def test_a_failed_fit_is_represented_without_a_centre(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        location, walk, tp = self.FIT_FAILED
        self.assertEqual(len(self._pick(full, location, walk)), len(LOCI))
        row = self._pick(full, location, walk, tp)[0]
        self.assertEqual(row["LocalizationAttemptStatus"], "fit_failed")
        self.assertEqual(row["LocalizationAttemptReason"], "fit_failed:-1")
        for field in (*_ts.AUDIT_COORDINATE_FIELDS,
                      *self.CANONICAL_COORDINATES):
            self.assertEqual(row[field].strip(), "")

    def test_a_locus_that_was_never_submitted_is_never_invented(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        location, walk, tp = self.MISSING
        self.assertEqual(len(self._pick(full, location, walk)), len(LOCI) - 1)
        self.assertEqual(self._pick(full, location, walk, tp), [])

    # -- 2. scientific exclusion ------------------------------------------
    def test_a_rejected_attempt_is_not_evidence_and_is_never_salvaged(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        row = self._pick(full, *self.REJECTED)[0]
        self.assertEqual(row["RawTwoComponentCall"], "0")
        self.assertEqual(row["RawEvaluable"], "0")
        self.assertEqual(row["RawConfidentSingle"], "0")
        self.assertEqual(row["TraceSalvaged"], "0")
        self.assertEqual(row["CenterOwnershipComplete"], "0")

    def test_the_comprehensive_row_agrees_with_the_salvage_audit(self):
        """One (walk, locus) must not get two different TraceSalvageReasons."""
        self._merge_with_salvage()
        _, full = self._comprehensive()
        _, audit = _read(
            self.outdir / "all_locations_640_trace_salvage_audit.csv")
        reasons = {(row["Location"], row["WalkID"], row["TP"]):
                   row["TraceSalvageReason"] for row in audit}
        checked = 0
        for row in full:
            key = (row["Location"], row["OriginalWalkID"], row["tp"])
            if key not in reasons:
                continue
            self.assertEqual(row["TraceSalvageReason"], reasons[key], key)
            checked += 1
        self.assertGreater(checked, 0)
        row = self._pick(full, *self.REJECTED)[0]
        self.assertEqual(row["TraceSalvageReason"], "missing_or_unevaluable")

    def test_an_m2_ownership_rejection_stays_unevaluable(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        location, walk, tp = self.M2_INVALID
        row = self._pick(full, location, walk, tp)[0]
        self.assertEqual(row["RawEvidenceStatus"], "ownership_unevaluable")
        self.assertEqual(row["RawTwoComponentCall"], "0")
        self.assertEqual(row["RawConfidentSingle"], "0")
        # The CENTRE was owned, so walk-level ownership is complete: the two
        # decisions are distinct and must not be conflated.
        self.assertEqual(row["CenterOwnershipComplete"], "1")
        self.assertEqual(row["TraceSalvagedSingleSpotComplete"], "0")
        self.assertEqual(self._pick(self._filtered()[1], location, walk), [])

    def test_an_ownership_invalid_ori_blocks_the_walk(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        location, walk, tp = self.REJECTED_ORI
        rows = self._pick(full, location, walk)
        self.assertEqual(len(rows), len(LOCI))
        self.assertEqual({row["TraceSalvageStatus"] for row in rows},
                         {"ori_not_confident_single"})
        self.assertEqual({row["TraceSalvagedSingleSpotComplete"]
                          for row in rows}, {"0"})
        self.assertEqual(self._pick(self._filtered()[1], location, walk), [])

    def test_every_affected_walk_contributes_zero_filtered_rows(self):
        self._merge_with_salvage()
        _, complete = self._filtered()
        for location, walk, _tp in (self.REJECTED, self.MISSING,
                                    self.REJECTED_ORI, self.M2_INVALID,
                                    self.FIT_FAILED):
            with self.subTest(f"{location}/{walk}"):
                self.assertEqual(self._pick(complete, location, walk), [])

    def test_a_coverage_dropped_walk_gains_rows_but_not_completeness(self):
        """Reintroducing rows must not widen the filtered scientific set."""
        self._merge_with_salvage()
        _, full = self._comprehensive()
        rows = self._pick(full, "Location-01", self.COVERAGE_DROPPED)
        self.assertEqual(len(rows), len(LOCI))
        # Its accepted coordinates ARE available and carried.
        self.assertEqual({row["LocalizationAttemptStatus"] for row in rows},
                         {"accepted"})
        self.assertTrue(all(row["z_nm"].strip() for row in rows))
        # Ownership is complete, and yet no derived trace was constructed.
        self.assertEqual({row["CenterOwnershipComplete"] for row in rows}, {"1"})
        self.assertEqual({row["TraceZNm"].strip() for row in rows}, {""})
        self.assertEqual({row["TraceSalvagedSingleSpotComplete"]
                          for row in rows}, {"0"})
        self.assertEqual(
            self._pick(self._filtered()[1], "Location-01",
                       self.COVERAGE_DROPPED), [])

    def test_a_wholly_unevaluable_walk_stays_in_the_comprehensive_output(self):
        """The denominator bug: it used to have no rows anywhere at all."""
        self._merge_with_salvage()
        _, full = self._comprehensive()
        for location in self.LOCATIONS:
            with self.subTest(location):
                rows = self._pick(full, location, self.ALL_REJECTED)
                self.assertEqual(len(rows), len(LOCI),
                                 "every attempt must be represented")
                self.assertEqual(
                    {row["LocalizationAttemptStatus"] for row in rows},
                    {"ownership_rejected"})
                # Every attempt row carries the SAME walk-level indeterminate
                # classification, not a blank one.
                self.assertEqual({row["replication_state"] for row in rows},
                                 {"indeterminate_ori"})
                self.assertEqual(
                    {row["ReplicationClassificationReason"] for row in rows},
                    {"all_localizations_ownership_unevaluable"})
                for field in ("Replicating", "MultiSpot",
                              "replication_pattern_complete",
                              "CenterOwnershipComplete",
                              "TraceSalvagedSingleSpotComplete"):
                    self.assertEqual({row[field] for row in rows}, {"0"}, field)
                self.assertEqual(
                    {row["CenterOwnershipValidCount"] for row in rows}, {"0"})
                self.assertEqual(
                    {row["CenterOwnershipRejectedCount"] for row in rows},
                    {str(len(LOCI))})
                for row in rows:
                    for field in self.CANONICAL_COORDINATES:
                        self.assertEqual(row[field].strip(), "", field)
                    self.assertTrue(all(row[f].strip()
                                        for f in _ts.AUDIT_COORDINATE_FIELDS))
                self.assertEqual(
                    self._pick(self._filtered()[1], location,
                               self.ALL_REJECTED), [])

    def test_no_comprehensive_row_lacks_its_walk_classification(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        blank = [row for row in full if not row["replication_state"].strip()]
        self.assertEqual(blank, [],
                         "a walk without accepted localizations must still be "
                         "classified, not left blank")
        states = {(row["Location"], row["OriginalWalkID"]): row
                  for row in full}
        self.assertEqual(len(states),
                         len(self.LOCATIONS) * len(self.WALKS))

    def test_a_fully_accepted_walk_still_reaches_the_filtered_output(self):
        self._merge_with_salvage()
        _, complete = self._filtered()
        pairs = {(row["Location"], int(row["OriginalWalkID"]))
                 for row in complete}
        self.assertEqual(pairs, self.CLEAN_WALKS)
        self.assertEqual(len(complete), len(self.CLEAN_WALKS) * len(LOCI))

    # -- 3. correct aggregation -------------------------------------------
    def test_walk_ownership_counts_separate_rejected_from_missing(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        expectations = {
            self.REJECTED[:2]: (len(LOCI), len(LOCI) - 1, 1, "0"),
            self.MISSING[:2]: (len(LOCI), len(LOCI) - 1, 0, "0"),
            self.FIT_FAILED[:2]: (len(LOCI), len(LOCI) - 1, 0, "0"),
            ("Location-01", 1): (len(LOCI), len(LOCI), 0, "1"),
            self.M2_INVALID[:2]: (len(LOCI), len(LOCI), 0, "1"),
        }
        for (location, walk), values in expectations.items():
            expected, valid, rejected, complete = values
            with self.subTest(f"{location}/{walk}"):
                row = self._pick(full, location, walk)[0]
                self.assertEqual(row["CenterOwnershipExpectedCount"],
                                 str(expected))
                self.assertEqual(row["CenterOwnershipValidCount"], str(valid))
                self.assertEqual(row["CenterOwnershipRejectedCount"],
                                 str(rejected))
                self.assertEqual(row["CenterOwnershipComplete"], complete)

    # -- 4. schema and identifiers ----------------------------------------
    def test_stages_4_6_7_all_salvage_products_carry_ownership(self):
        self._merge_with_salvage()
        paths = [
            self.outdir / "locations/Location-01_trace_salvage_full_schema.csv",
            self.outdir / "all_locations_640_trace_salvage_full_schema.csv",
            self.outdir
            / "all_locations_640_trace_salvage_full_schema_complete.csv",
        ]
        for path in paths:
            with self.subTest(path.name):
                fields, rows = _read(path)
                for column in (*_own.OWNERSHIP_ROW_FIELDS,
                               *_own.OWNERSHIP_WALK_FIELDS,
                               *_ts.ATTEMPT_FIELDS,
                               *_ts.AUDIT_COORDINATE_FIELDS):
                    self.assertIn(column, fields)
                self.assertTrue(rows, f"{path.name} is empty")

    def test_filtered_and_unfiltered_share_one_full_schema(self):
        self._merge_with_salvage()
        full, _ = self._comprehensive()
        complete, _ = self._filtered()
        self.assertEqual(full, complete)

    def test_filtered_output_is_exactly_the_complete_walks(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        _, complete = self._filtered()
        expected = {row["WalkID"] for row in full
                    if row["TraceSalvagedSingleSpotComplete"] == "1"}
        self.assertEqual({row["WalkID"] for row in complete}, expected)
        self.assertEqual({row["TraceSalvagedSingleSpotComplete"]
                          for row in complete}, {"1"})

    def test_ownership_complete_is_required_for_salvaged_complete(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        for row in full:
            if row["TraceSalvagedSingleSpotComplete"] == "1":
                self.assertEqual(row["CenterOwnershipComplete"], "1")

    def test_identifiers_do_not_collide_across_locations(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        for location in self.LOCATIONS:
            local = {int(row["OriginalWalkID"]) for row in full
                     if row["Location"] == location}
            self.assertEqual(local, set(self.WALKS), location)
        pairs = {}
        for row in full:
            pairs.setdefault(
                (row["Location"], row["OriginalWalkID"]), set()).add(
                    row["WalkID"])
        self.assertTrue(all(len(v) == 1 for v in pairs.values()))
        self.assertEqual(len(set().union(*pairs.values())), len(pairs))
        ids = sorted({int(row["WalkID"]) for row in full})
        self.assertEqual(ids, list(range(1, len(ids) + 1)))

    def test_every_attempt_appears_once_and_nothing_is_fabricated(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        keys = [(row["Location"], row["OriginalWalkID"], row["tp"])
                for row in full]
        self.assertEqual(len(keys), len(set(keys)), "no Cartesian expansion")
        self.assertEqual([row for row in full if not row["tp"].strip()], [])
        expected = 0
        for location in self.LOCATIONS:
            _, ledger = _read(
                self.base / location / "probe_0/localizations"
                / "localizations_640_z_gate_status.csv")
            expected += len(ledger)
        self.assertEqual(len(full), expected,
                         "one row per ledger attempt, no more and no fewer")

    def test_no_accepted_coordinate_row_disappears(self):
        self._merge_with_salvage()
        _, full = self._comprehensive()
        accepted = {(row["Location"], row["OriginalWalkID"], row["tp"])
                    for row in full
                    if row["LocalizationAttemptStatus"] == "accepted"}
        source = set()
        for location in self.LOCATIONS:
            _, rows = _read(
                self.base / location / "probe_0/localizations"
                / "localizations_640_drift_corrected.csv")
            source.update((location, row["WalkID"], row["tp"]) for row in rows)
        self.assertEqual(accepted, source)

    def test_the_manifest_records_the_attempt_universe_and_the_rule(self):
        self._merge_with_salvage()
        with open(self.outdir
                  / "all_locations_640_trace_salvage_manifest.json") as handle:
            manifest = json.load(handle)
        block = manifest["center_ownership"]
        self.assertTrue(block["enforced"])
        self.assertEqual(set(block["sources"].values()),
                         {"ownership_audit_csv"})
        self.assertTrue(
            "cannot be salvaged" in block["rule"]
            or "censored" in block["rule"],
            f"unexpected rule text: {block['rule']!r}")
        attempts = manifest["attempt_universe"]
        self.assertEqual(attempts["source"], ["attempt_ledger"])
        self.assertEqual(attempts["counts"]["ownership_rejected"],
                         self.EXPECTED_REJECTED)
        self.assertEqual(attempts["counts"]["fit_failed"], 1)
        self.assertEqual(
            attempts["counts"]["accepted"]
            + attempts["counts"]["ownership_rejected"]
            + attempts["counts"]["fit_failed"],
            sum(int(item["rows"])
                for item in manifest["location_outputs"].values()))

    # -- 5. the comprehensive file alone is sufficient ---------------------
    def test_the_comprehensive_csv_needs_no_sidecar(self):
        self._merge_with_salvage()
        path = (self.outdir
                / "all_locations_640_trace_salvage_full_schema.csv")
        _, rows = _read(path)
        # Deliberately reads ONLY the comprehensive file.
        rejected = [row for row in rows
                    if row["LocalizationAttemptStatus"] == "ownership_rejected"]
        self.assertEqual(len(rejected), self.EXPECTED_REJECTED)
        self.assertEqual({row["CenterOwnershipFailureReason"]
                          for row in rejected}, {"center_background"})
        self.assertTrue(all(row["UnconstrainedFitZNm"].strip()
                            for row in rejected))
        per_walk = {}
        for row in rows:
            key = (row["Location"], int(row["OriginalWalkID"]))
            per_walk.setdefault(key, {"valid": 0, "rejected": 0})
            if row["CenterOwnershipValid"] == "1":
                per_walk[key]["valid"] += 1
            elif row["LocalizationAttemptStatus"] == "ownership_rejected":
                per_walk[key]["rejected"] += 1
        complete = {key for key, counts in per_walk.items()
                    if counts["valid"] == len(LOCI)}
        declared = {(row["Location"], int(row["OriginalWalkID"]))
                    for row in rows
                    if row["CenterOwnershipComplete"] == "1"}
        self.assertEqual(complete, declared,
                         "completeness recomputed from the file alone must "
                         "match the column")
        selected = {(row["Location"], int(row["OriginalWalkID"]))
                    for row in rows
                    if row["TraceSalvagedSingleSpotComplete"] == "1"}
        self.assertEqual(selected, self.CLEAN_WALKS)

    # -- 6. backward compatibility ----------------------------------------
    def test_legacy_allow_values_and_row_order_are_unchanged(self):
        """Ownership-allow: the audit columns are additive, nothing moves."""
        import shutil
        shutil.rmtree(self.base)
        self._build(reject=False)
        self._merge_with_salvage()
        _, first = self._comprehensive()
        source = []
        for location in self.LOCATIONS:
            _, rows = _read(
                self.base / location / "probe_0/localizations"
                / "localizations_640_drift_corrected.csv")
            source.extend((location, row) for row in rows)
        self.assertEqual(len(first), len(source))
        for (location, want), got in zip(source, first):
            self.assertEqual(got["Location"], location)
            self.assertEqual(got["tp"], want["tp"])
            self.assertEqual(got["OriginalWalkID"], want["WalkID"])
            for field in ("z_nm", "y_nm", "x_nm", "z_raw", "y_raw", "x_raw"):
                self.assertEqual(float(got[field]), float(want[field]))
            self.assertEqual(got["LocalizationAttemptStatus"], "accepted")
        shutil.rmtree(self.outdir)
        self._merge_with_salvage()
        _, again = self._comprehensive()
        self.assertEqual(first, again, "output ordering must be deterministic")

    def test_a_config_without_a_ledger_keeps_the_historical_universe(self):
        """No ledger declared: the coordinate rows remain the row universe."""
        self._merge_with_salvage()
        _, ledger_driven = self._comprehensive()
        config_path = (self.outdir
                       / "all_locations_640_trace_salvage_config.json")
        with open(config_path) as handle:
            payload = json.load(handle)
        for entry in payload["locations"]:
            entry.pop("ownership_audit_csv", None)
            entry.pop("all_coordinates_csv", None)
        legacy_dir = Path(self._tmp.name) / "legacy"
        legacy_dir.mkdir()
        legacy_config = legacy_dir / "config.json"
        with open(legacy_config, "w") as handle:
            json.dump(payload, handle)
        config = _ts.load_config(legacy_config)
        _ts.run_dataset(config, outdir=legacy_dir, output_prefix="legacy",
                        diagnostics=True)
        _, legacy = _read(legacy_dir / "legacy_trace_salvage_full_schema.csv")
        covered = 0
        for location in self.LOCATIONS:
            _, rows = _read(
                self.base / location / "probe_0/localizations"
                / "localizations_640_drift_corrected_full_coverage.csv")
            covered += len(rows)
        self.assertEqual(len(legacy), covered)
        self.assertLess(len(legacy), len(ledger_driven))
        keyed = {(row["Location"], row["OriginalWalkID"], row["tp"]): row
                 for row in ledger_driven}
        for row in legacy:
            key = (row["Location"], row["OriginalWalkID"], row["tp"])
            self.assertIn(key, keyed)
            for field in ("z_nm", "y_nm", "x_nm"):
                self.assertEqual(float(row[field]), float(keyed[key][field]))


class GNoOwnershipSchemaTest(unittest.TestCase):
    """A dataset predating the schema must not be reported as a clean pass."""

    def test_absent_ownership_is_blank_rather_than_valid(self):
        rows = [{"WalkID": 0, "tp": f"TP{tp:03d}", "z": 1.0} for tp in LOCI]
        with self.assertRaises(_own.OwnershipSchemaMissing):
            _own.walk_ownership_summary(rows, LOCI)


if __name__ == "__main__":
    unittest.main()
