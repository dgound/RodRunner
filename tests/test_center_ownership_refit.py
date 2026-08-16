#!/usr/bin/env python3
"""Tests for centre-ownership policy semantics.

Corrected `strict` is a FAIL-CLOSED FILTER: the ordinary unconstrained centre is
classified against the registered effective mask and accepted only if it already
lies in the cell's own label. It never moves, projects, clamps, snaps or re-fits
a coordinate.

These tests replace an earlier suite that asserted the OPPOSITE -- that a
constrained fit landing at x ~= 8.50, exactly on the owned mask boundary, was a
success, when the true source sat at x ~= 14 well outside the mask. That was
boundary clamping through optimization, and on real data it "recovered" 100% of
out-of-support centres while destroying 53-86% of complete traces downstream.

Fits run against the real compiled ``gauss3d_fit.so``: the behaviour under test
lives in the optimizer, so a mocked fitter would prove nothing.
"""

import csv
import ctypes
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import center_ownership as co  # noqa: E402

LOCALIZER = ROOT / "pipeline" / "localize_probe0.py"
C_SRC = ROOT / "pipeline" / "gauss3d_fit.c"
C_LIB = ROOT / "pipeline" / "gauss3d_fit.so"


def _load_lib():
    """Load (building if stale) the fitter, with both entry points declared."""
    if not C_LIB.exists() or C_SRC.stat().st_mtime > C_LIB.stat().st_mtime:
        if os.system(f"gcc -O3 -fopenmp -shared -fPIC -o {C_LIB} {C_SRC} -lm") != 0:
            raise unittest.SkipTest("cannot build gauss3d_fit.so")
    lib = ctypes.CDLL(str(C_LIB))
    lib.batch_fit.restype = None
    lib.batch_fit.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
    ]
    co.declare_batch_fit_ex(lib)
    return lib


def _unconstrained(lib, image, labels, bboxes, n):
    results = np.zeros((n, 11), dtype=np.float64)
    success = np.zeros(n, dtype=np.int32)
    image = np.ascontiguousarray(image, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    nz, ny, nx = labels.shape
    lib.batch_fit(
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nz, ny, nx,
        np.ascontiguousarray(bboxes, dtype=np.int32)
        .ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n,
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        success.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return results, success


def _off_centre_scene():
    """One cell whose brightest structure sits OUTSIDE its own mask.

    The mask is a slab at x in [4, 8]; the Gaussian source is centred at x = 14.
    Inside the owned support the intensity rises monotonically toward +x, so the
    best unconstrained single Gaussian puts its centre well outside the mask.
    """
    nz, ny, nx = 7, 21, 21
    labels = np.zeros((nz, ny, nx), dtype=np.int32)
    labels[:, 6:15, 4:9] = 1
    zz, yy, xx = np.indices((nz, ny, nx), dtype=float)
    image = 100.0 + 6000.0 * np.exp(
        -0.5 * (((zz - 3) / 1.2) ** 2
                + ((yy - 10) / 2.0) ** 2
                + ((xx - 14) / 2.2) ** 2))
    bboxes = np.array([[0, nz - 1, 3, 17, 1, 19]], dtype=np.int32)
    return image, labels, bboxes


class TestStrictRejectsOutsideCentres(unittest.TestCase):
    """Requirement A: the adversarial case that the old tests got WRONG.

    The previous suite asserted that a constrained fit landing at x ~= 8.50 --
    the mask boundary -- was a SUCCESS. That is boundary clamping through
    optimization: the true source is at x ~= 14, far outside the owned mask, and
    no honest procedure turns that into a valid localization for this cell.
    Corrected `strict` classifies the ORIGINAL unconstrained centre and rejects.
    """

    def test_unconstrained_centre_is_outside_and_strict_rejects_it(self):
        lib = _load_lib()
        image, labels, bboxes = _off_centre_scene()
        results, success = _unconstrained(lib, image, labels, bboxes, 1)
        self.assertEqual(int(success[0]), 0, "the unconstrained fit should succeed")

        z, y, x = results[0][1], results[0][2], results[0][3]
        self.assertGreater(x, 12.0, "the true source sits at x ~= 14")
        category = co.classify_center(labels, 0, z, y, x)
        self.assertEqual(category, co.BACKGROUND)

        fields = co.row_fields(co.POLICY_STRICT, category)
        self.assertEqual(fields["CenterOwnershipValid"], 0)
        self.assertEqual(fields["CenterOwnershipFailureReason"], "center_background")
        # The defining invariant: strict never refits and never moves anything.
        self.assertEqual(fields["CenterRefitAttempted"], 0)
        self.assertEqual(fields["CenterRefitAccepted"], 0)
        self.assertEqual(fields["CenterOwnershipFinal"], category)

    def test_the_original_coordinate_is_not_replaced_by_a_boundary_coordinate(self):
        """The rejected coordinate must stay where the honest fit put it."""
        lib = _load_lib()
        image, labels, bboxes = _off_centre_scene()
        results, _ = _unconstrained(lib, image, labels, bboxes, 1)
        x_original = results[0][3]

        # What the discredited constrained refit would have produced.
        refit, codes = co.constrained_refit(lib, image, labels, bboxes, [0])
        self.assertEqual(int(codes[0]), 0)
        x_boundary = refit[0][3]
        self.assertLess(x_boundary, 8.5, "the old path pinned it to the mask edge")
        self.assertGreater(x_boundary, 8.0)

        # strict must not adopt that value. The unconstrained result is unchanged.
        again, _ = _unconstrained(lib, image, labels, bboxes, 1)
        self.assertEqual(again[0][3], x_original)
        self.assertNotAlmostEqual(x_original, x_boundary, places=1)

    def test_a_rejected_locus_cannot_make_a_walk_ownership_complete(self):
        rows = [{"tp": f"TP{i:03d}", "WalkID": 0, "center_ownership_pass": "1"}
                for i in range(2, 21)]
        rows.append({"tp": "TP021", "WalkID": 0, "center_ownership_pass": "0"})
        summary = co.walk_completeness(rows, 2, 21)
        self.assertEqual(summary[0]["CenterOwnershipComplete"], 0)
        self.assertEqual(summary[0]["n_loci_own_mask"], 19)


class TestStrictAcceptsGenuinelyOwnedCentres(unittest.TestCase):
    """Requirement B: a source clearly inside its own mask is accepted intact."""

    def test_owned_source_is_accepted_with_the_unconstrained_coordinate(self):
        lib = _load_lib()
        nz, ny, nx = 7, 21, 21
        labels = np.zeros((nz, ny, nx), dtype=np.int32)
        labels[:, 6:15, 6:15] = 1
        zz, yy, xx = np.indices((nz, ny, nx), dtype=float)
        image = 100.0 + 6000.0 * np.exp(
            -0.5 * (((zz - 3) / 1.2) ** 2 + ((yy - 10) / 1.5) ** 2
                    + ((xx - 10) / 1.5) ** 2))
        bboxes = np.array([[0, nz - 1, 3, 17, 3, 17]], dtype=np.int32)

        results, success = _unconstrained(lib, image, labels, bboxes, 1)
        self.assertEqual(int(success[0]), 0)
        z, y, x = results[0][1], results[0][2], results[0][3]
        category = co.classify_center(labels, 0, z, y, x)
        self.assertEqual(category, co.OWN_MASK)

        fields = co.row_fields(co.POLICY_STRICT, category)
        self.assertEqual(fields["CenterOwnershipValid"], 1)
        self.assertEqual(fields["CenterOwnershipFailureReason"], "")
        self.assertEqual(fields["CenterRefitAttempted"], 0)
        # Accepted coordinate == the ordinary unconstrained result, exactly.
        self.assertAlmostEqual(x, 10.0, delta=0.5)
        self.assertAlmostEqual(y, 10.0, delta=0.5)


class TestStrictRejectsAnotherCell(unittest.TestCase):
    """Requirement C: a centre inside a neighbouring active label fails."""

    def test_center_in_neighbouring_label_is_rejected_with_reason(self):
        nz, ny, nx = 5, 21, 21
        labels = np.zeros((nz, ny, nx), dtype=np.int32)
        labels[:, 4:9, 4:9] = 1        # own cell   (walk 0)
        labels[:, 4:9, 12:17] = 2      # neighbour  (walk 1)
        # A centre sitting in the neighbour's label.
        category = co.classify_center(labels, 0, 2, 6, 14)
        self.assertEqual(category, co.OTHER_ACTIVE_CELL)

        fields = co.row_fields(co.POLICY_STRICT, category)
        self.assertEqual(fields["CenterOwnershipValid"], 0)
        self.assertEqual(fields["CenterOwnershipFailureReason"],
                         "center_other_active_cell")
        self.assertEqual(fields["CenterRefitAttempted"], 0)
        # Background is likewise not legal spot-centre territory.
        self.assertEqual(
            co.row_fields(co.POLICY_STRICT,
                          co.classify_center(labels, 0, 2, 2, 2))["CenterOwnershipValid"],
            0)


class TestPolicySeparation(unittest.TestCase):
    """strict must never select the experimental constrained path."""

    def test_policies_are_distinct_and_strict_is_not_the_experimental_one(self):
        self.assertEqual(co.POLICY_STRICT, "strict")
        self.assertEqual(co.POLICY_ALLOW, "allow")
        self.assertEqual(co.POLICY_CONSTRAINED_EXPERIMENTAL,
                         "constrained_experimental")
        self.assertNotEqual(co.POLICY_STRICT, co.POLICY_CONSTRAINED_EXPERIMENTAL)
        self.assertIn(co.POLICY_STRICT, co.POLICIES)

    def test_row_fields_rejects_an_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "unknown ownership policy"):
            co.row_fields("lenient", co.OWN_MASK)

    def test_allow_retains_an_out_of_support_fit(self):
        """Backward compatibility: allow records, it does not filter."""
        fields = co.row_fields(co.POLICY_ALLOW, co.BACKGROUND)
        self.assertEqual(fields["CenterOwnershipValid"], 0)
        self.assertEqual(fields["CenterRefitAttempted"], 0)
        # The caller keeps the row; validity is a recorded diagnostic only.
        self.assertEqual(fields["CenterOwnershipPolicy"], "allow")

    def test_production_drivers_never_select_the_experimental_policy(self):
        import subprocess
        out = subprocess.run(
            ["grep", "-rn", "constrained_experimental",
             str(ROOT / "drivers")], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "",
                         "no production driver may select the experimental policy")


class TestWalkCompleteness(unittest.TestCase):
    """Requirement 4: walk-level CenterOwnershipComplete over the locus range."""

    def _rows(self, spec):
        return [{"tp": tp, "WalkID": walk, "center_ownership_pass": flag}
                for walk, tp, flag in spec]

    def test_complete_only_when_every_configured_locus_passes(self):
        rows = self._rows(
            [(0, f"TP{i:03d}", "1") for i in range(2, 22)]
            + [(1, f"TP{i:03d}", "1") for i in range(2, 21)]
            + [(1, "TP021", "0")])
        summary = {r["WalkID"]: r for r in co.walk_completeness(rows, 2, 21)}
        self.assertEqual(summary[0]["CenterOwnershipComplete"], 1)
        self.assertEqual(summary[0]["n_loci_own_mask"], 20)
        self.assertEqual(summary[1]["CenterOwnershipComplete"], 0)
        self.assertEqual(summary[1]["n_loci_own_mask"], 19)

    def test_a_missing_locus_row_also_breaks_completeness(self):
        rows = self._rows([(0, f"TP{i:03d}", "1") for i in range(2, 21)])
        summary = co.walk_completeness(rows, 2, 21)
        self.assertEqual(summary[0]["n_loci_attempted"], 19)
        self.assertEqual(summary[0]["n_loci_expected"], 20)
        self.assertEqual(summary[0]["CenterOwnershipComplete"], 0)

    def test_loci_outside_the_configured_interval_are_ignored(self):
        """TP names are never hardcoded: only the configured range counts.

        TP000/TP001 lead-in and the TP022/TP023 re-image controls must not be
        able to make a walk complete or incomplete.
        """
        rows = self._rows(
            [(0, f"TP{i:03d}", "1") for i in range(2, 22)]
            + [(0, "TP000", "0"), (0, "TP001", "0"),
               (0, "TP022", "0"), (0, "TP023", "0")])
        summary = co.walk_completeness(rows, 2, 21)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["CenterOwnershipComplete"], 1)
        self.assertEqual(summary[0]["n_loci_attempted"], 20)

    def test_a_different_configured_interval_shifts_the_verdict(self):
        rows = self._rows([(0, f"TP{i:03d}", "1") for i in range(5, 10)])
        self.assertEqual(
            co.walk_completeness(rows, 5, 9)[0]["CenterOwnershipComplete"], 1)
        self.assertEqual(
            co.walk_completeness(rows, 5, 10)[0]["CenterOwnershipComplete"], 0)

    def test_tp_index_parses_the_suffix(self):
        self.assertEqual(co.tp_index("TP017"), 17)
        self.assertEqual(co.tp_index("tp007"), 7)
        with self.assertRaises(ValueError):
            co.tp_index("")

    def test_tp_index_accepts_the_rep1_cycle_suffix(self):
        """rep1's TIFF-era tree labels every acquisition TP###_CYC###.

        A parser that accepts only bare TP### drops every rep1 row, which is how
        this was originally broken.
        """
        self.assertEqual(co.tp_index("TP000_CYC000"), 0)
        self.assertEqual(co.tp_index("TP017_CYC000"), 17)
        self.assertEqual(co.tp_index("TP021_CYC003"), 21)
        with self.assertRaises(ValueError):
            co.tp_index("TP017_FOO000")

    def test_rep1_labels_produce_a_real_result(self):
        rows = self._rows([(0, f"TP{i:03d}_CYC000", "1") for i in range(2, 22)])
        summary = co.walk_completeness(rows, 2, 21)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["CenterOwnershipComplete"], 1)
        self.assertEqual(summary[0]["n_loci_own_mask"], 20)

    def test_wholly_unparseable_labels_raise_instead_of_reporting_zero(self):
        """The silent-empty failure mode must be impossible.

        Reporting "0 walks" and exiting successfully is worse than crashing: it
        looks like a real answer.
        """
        rows = self._rows([(0, f"CYCLE{i}", "1") for i in range(2, 22)])
        with self.assertRaisesRegex(ValueError, "parseable TP label"):
            co.walk_completeness(rows, 2, 21)

    def test_a_few_odd_labels_do_not_raise(self):
        """Only a WHOLLY unparseable set is a contract break."""
        rows = self._rows(
            [(0, f"TP{i:03d}", "1") for i in range(2, 22)] + [(0, "junk", "1")])
        summary = co.walk_completeness(rows, 2, 21)
        self.assertEqual(summary[0]["CenterOwnershipComplete"], 1)

    def test_missing_tp_column_raises(self):
        with self.assertRaisesRegex(ValueError, "no 'tp' column"):
            co.walk_completeness([{"WalkID": 0, "center_ownership_pass": "1"}],
                                 2, 21)


class TestRoundingMatchesTheCConstraint(unittest.TestCase):
    """A half-integer centre must be resolved the same way on both sides.

    Otherwise the constrained refit can accept a centre that the classifier then
    reports as out-of-support, and the "unreachable" refit-outside branch fires.
    """

    def test_half_rounds_away_from_zero_like_lround(self):
        # The rules differ exactly when the integer below the half is EVEN:
        # np.rint takes the even neighbour, lround always rounds up.
        for value, expected in ((980.5, 981), (0.5, 1), (2.5, 3), (4.5, 5)):
            self.assertEqual(int(co.nearest_voxel(value)), expected)
            self.assertNotEqual(int(np.rint(value)), expected)
        # Where the lower integer is odd the two agree, and must keep agreeing.
        for value in (1.5, 3.5, 981.5):
            self.assertEqual(int(co.nearest_voxel(value)), int(np.rint(value)))

    def test_classifier_uses_the_c_rule_on_a_label_boundary(self):
        """The rep3 Location-01 TP010 case, reduced to its essentials.

        A centre sitting exactly on the boundary between a neighbour and its own
        cell must be resolved the way the optimizer's constraint resolved it.
        """
        labels = np.zeros((3, 5, 6), dtype=np.int32)
        labels[:, :, :3] = 7      # neighbour: x 0..2
        labels[:, :, 3:] = 8      # own cell for walk 7: x 3..5
        # x = 2.5: lround -> 3 -> label 8 -> own. np.rint -> 2 -> label 7.
        self.assertEqual(co.classify_center(labels, 7, 1, 2, 2.5), co.OWN_MASK)
        self.assertEqual(
            int(labels[1, 2, int(np.rint(2.5))]), 7,
            "np.rint would have called this centre the neighbour's")

    def test_ownership_survives_a_csv_round_trip_exactly(self):
        """The invariant the smoke test asserts: no accepted centre drifts out.

        Ownership is decided on the value as SERIALIZED, so a consumer that
        recomputes the verdict from the written coordinate gets the same answer.
        This replaces an earlier test that pinned the opposite as a "documented
        limitation" -- it was a defect, and it is fixed rather than tolerated.
        """
        labels = np.zeros((3, 9, 9), dtype=np.int32)
        labels[:, 2:7, 2:7] = 1                       # own cell, walk 0
        for x in (4.0, 6.4, 6.49999, 6.5, 6.50001, 1.5, 1.49999):
            full = co.classify_center(labels, 0, 1, 4, x)
            written = float(f"{co.quantize_for_output(x):.{co.COORD_DECIMALS}f}")
            reread = co.classify_center(labels, 0, 1, 4, written)
            self.assertEqual(full, reread, f"verdict changed for x={x}")

    def test_quantize_matches_the_written_precision(self):
        self.assertEqual(co.COORD_DECIMALS, 4)
        self.assertAlmostEqual(float(co.quantize_for_output(8.49999816)), 8.5)
        self.assertAlmostEqual(float(co.quantize_for_output(1.234549)), 1.2345)
        # Non-finite input is passed through untouched for the caller to reject.
        self.assertFalse(np.isfinite(co.quantize_for_output(float("nan"))))

    def test_the_experimental_refit_output_does_not_survive_serialization(self):
        """Why constrained_experimental is unusable, stated as a test.

        Its centre converges against the owned boundary at x ~= 8.49999816. Once
        written at 4 decimals that becomes 8.5000, which rounds to voxel 9 --
        outside the mask. So the "recovered" coordinate is not genuinely owned;
        the review measured ~47% of such centres rounding outside.
        """
        lib = _load_lib()
        image, labels, bboxes = _off_centre_scene()
        refit, codes = co.constrained_refit(lib, image, labels, bboxes, [0])
        self.assertEqual(int(codes[0]), 0, "the C refit reports success")
        x = refit[0][3]
        self.assertGreater(x, 8.0)
        # The optimizer says owned; the serialized coordinate is not.
        self.assertEqual(
            co.classify_center(labels, 0, refit[0][1], refit[0][2], x),
            co.BACKGROUND,
            "a boundary-pinned centre is not owned once serialized")


if __name__ == "__main__":
    unittest.main(verbosity=2)
