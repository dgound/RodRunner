#!/usr/bin/env python3
"""Tests for chromatic channel registration and centre-ownership classification."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

import center_ownership as co  # noqa: E402
import chromatic_registration as cr  # noqa: E402


def _plane(m00=1.0, m01=0.0, m02=0.0, m10=0.0, m11=1.0, m12=0.0,
           nonlinear=None):
    value = {
        "Matrix00": m00, "Matrix01": m01, "Matrix02": m02,
        "Matrix10": m10, "Matrix11": m11, "Matrix12": m12,
    }
    value["NonlinearTransform"] = {
        "type": "NonlinearTransform",
        "value": nonlinear or {"DimX": 0, "DimY": 0, "Coefficients": []},
    }
    return {"type": "PlaneTransformation", "value": value}


def write_metadata(path, transforms, lasers=("640", "488", "405"),
                   px=(100.0, 100.0), z_offsets=None, rms=None):
    """Build a minimal widefield.json with the real nesting shape."""
    doc = {"value": {
        "Configuration": {
            "PixelSizeX": px[0],
            "PixelSizeY": px[1],
            "Probes": {"type": "vector_Probe", "value": [
                {"type": "Probe", "value": {"WidefieldLaser": laser}}
                for laser in lasers]},
        },
        "ChromaticOffset": {
            "Transforms": {"type": "vector_PlaneTransformation",
                           "value": list(transforms)},
            "ZOffsets": {"type": "vector_double",
                         "value": list(z_offsets or [0.0] * len(transforms))},
            "RMSErrors": {"type": "vector_double",
                          "value": list(rms or [0.0] * len(transforms))},
        },
    }}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
    return path


class TestIdentity(unittest.TestCase):
    """(1) An identity affine must leave masks bit-for-bit unchanged."""

    def test_identity_leaves_labels_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(os.path.join(tmp, "widefield.json"),
                               [_plane(), _plane(), _plane()])
            calib = cr.load_calibration(p)
            self.assertTrue(cr.is_identity(calib, "640"))
            self.assertTrue(cr.is_identity(calib, "488"))

            rng = np.random.default_rng(0)
            labels = rng.integers(0, 6, size=(4, 32, 32)).astype(np.int32)
            for wavelength in ("640", "488", "405"):
                out = cr.resample_labels_to_channel(labels, calib, wavelength)
                np.testing.assert_array_equal(out, labels)


class TestKnownTransform(unittest.TestCase):
    """(2) A known translation/scale must land labels on the expected pixels."""

    def test_pure_translation_shifts_by_expected_pixels(self):
        # T_405 translates by +5 px in x, -3 px in y at 100 nm pixels.
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(), _plane(m02=500.0, m12=-300.0)],
                px=(100.0, 100.0))
            calib = cr.load_calibration(p)

            labels = np.zeros((1, 20, 20), dtype=np.int32)
            labels[0, 10, 10] = 7          # DAPI-frame position

            out = cr.resample_labels_to_channel(labels, calib, "640")
            # p_405 = T_405 @ p_640 -> the 640 pixel whose DAPI image is (10,10)
            # satisfies x_405 = x_640 + 5, y_405 = y_640 - 3, so x_640 = 5, y_640 = 13.
            hits = np.argwhere(out == 7)
            self.assertEqual(len(hits), 1, f"expected one labelled pixel, got {hits}")
            self.assertEqual(tuple(hits[0]), (0, 13, 5))

    def test_translation_is_physical_not_pixels(self):
        """The same nm translation must move twice as far at half the pixel size."""
        with tempfile.TemporaryDirectory() as tmp:
            coarse = cr.load_calibration(write_metadata(
                os.path.join(tmp, "a.json"),
                [_plane(), _plane(), _plane(m02=400.0)], px=(100.0, 100.0)))
            fine = cr.load_calibration(write_metadata(
                os.path.join(tmp, "b.json"),
                [_plane(), _plane(), _plane(m02=400.0)], px=(50.0, 50.0)))
            m_coarse = cr.dapi_lookup_matrix(coarse, "640")
            m_fine = cr.dapi_lookup_matrix(fine, "640")
            x_coarse, _ = cr.apply_matrix_px(m_coarse, [0.0], [0.0],
                                             coarse.pixel_size_nm)
            x_fine, _ = cr.apply_matrix_px(m_fine, [0.0], [0.0], fine.pixel_size_nm)
            self.assertAlmostEqual(float(x_coarse[0]), 4.0)
            self.assertAlmostEqual(float(x_fine[0]), 8.0)

    def test_scale_maps_expected_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(), _plane(m00=2.0, m11=2.0)],
                px=(100.0, 100.0)))
            m = cr.dapi_lookup_matrix(calib, "640")
            x, y = cr.apply_matrix_px(m, [10.0], [4.0], calib.pixel_size_nm)
            self.assertAlmostEqual(float(x[0]), 20.0)
            self.assertAlmostEqual(float(y[0]), 8.0)


class TestCategorical(unittest.TestCase):
    """(3) Label IDs and background must stay categorical."""

    def test_no_interpolated_labels_appear(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                # deliberately awkward: sub-pixel shift + slight scale
                [_plane(), _plane(), _plane(m00=1.0031, m11=1.0027,
                                            m02=-51.3, m12=137.9)],
                px=(103.0, 103.5)))
            labels = np.zeros((2, 40, 40), dtype=np.int32)
            labels[:, 5:15, 5:15] = 3
            labels[:, 20:30, 20:30] = 9
            out = cr.resample_labels_to_channel(labels, calib, "640")

            self.assertEqual(out.dtype, labels.dtype)
            self.assertTrue(set(np.unique(out)).issubset({0, 3, 9}),
                            f"interpolated labels appeared: {np.unique(out)}")

    def test_background_stays_zero_outside_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            # huge translation pushes the whole DAPI image off the target grid
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(), _plane(m02=1e6)], px=(100.0, 100.0)))
            labels = np.full((1, 16, 16), 4, dtype=np.int32)
            out = cr.resample_labels_to_channel(labels, calib, "640")
            self.assertTrue(np.all(out == 0))


class TestComposition(unittest.TestCase):
    """(4) 405->640 and 405->488 compositions must be correct and distinct."""

    def test_composition_matches_manual_product(self):
        t488 = _plane(m00=1.0033, m01=-0.0007, m02=-78.4,
                      m10=-1.9e-05, m11=1.0025, m12=-287.4)
        t405 = _plane(m00=1.0027, m01=-0.0008, m02=21.3,
                      m10=5.6e-04, m11=1.0035, m12=-460.3)
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), t488, t405], px=(102.96875, 103.59375)))

            m640 = cr.dapi_lookup_matrix(calib, "640")
            m488 = cr.dapi_lookup_matrix(calib, "488")

            T405 = calib.matrix("405")
            T488 = calib.matrix("488")
            T640 = calib.matrix("640")
            np.testing.assert_allclose(m640, T405 @ np.linalg.inv(T640))
            np.testing.assert_allclose(m488, T405 @ np.linalg.inv(T488))
            # T_640 is identity here, so the 640 lookup is T_405 itself
            np.testing.assert_allclose(m640, T405)
            self.assertFalse(np.allclose(m640, m488))

    def test_channel_map_comes_from_probe_laser_not_index(self):
        """Reordering the probes must reorder the wavelength binding."""
        a = _plane(m02=111.0)
        b = _plane(m02=222.0)
        with tempfile.TemporaryDirectory() as tmp:
            normal = cr.load_calibration(write_metadata(
                os.path.join(tmp, "n.json"), [_plane(), a, b],
                lasers=("640", "488", "405")))
            swapped = cr.load_calibration(write_metadata(
                os.path.join(tmp, "s.json"), [_plane(), a, b],
                lasers=("640", "405", "488")))
            self.assertAlmostEqual(normal.matrix("405")[0, 2], 222.0)
            self.assertAlmostEqual(swapped.matrix("405")[0, 2], 111.0)
            self.assertEqual(normal.transform_index["405"], 2)
            self.assertEqual(swapped.transform_index["405"], 1)


class TestDriftUnaffected(unittest.TestCase):
    """(5) A constant channel translation cancels in 488(t) - 488(reference)."""

    def test_constant_translation_cancels_in_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(m02=-78.4, m12=-287.4), _plane(m02=21.3)],
                px=(103.0, 103.0)))
            # a pure-translation channel transform
            trans = np.eye(3)
            trans[0, 2], trans[1, 2] = -78.4, -287.4

            rng = np.random.default_rng(3)
            ref_x, ref_y = rng.uniform(0, 1800, 50), rng.uniform(0, 1800, 50)
            drift_x, drift_y = 2.5, -1.25
            tp_x, tp_y = ref_x + drift_x, ref_y + drift_y

            raw = (tp_x - ref_x, tp_y - ref_y)
            ax, ay = cr.apply_matrix_px(trans, tp_x, tp_y, calib.pixel_size_nm)
            bx, by = cr.apply_matrix_px(trans, ref_x, ref_y, calib.pixel_size_nm)
            np.testing.assert_allclose(ax - bx, raw[0], atol=1e-9)
            np.testing.assert_allclose(ay - by, raw[1], atol=1e-9)
            np.testing.assert_allclose(ax - bx, np.full(50, drift_x), atol=1e-9)


class TestMetadataValidation(unittest.TestCase):
    """Requested registration must fail loudly, never fall back to identity."""

    def test_missing_file_raises(self):
        with self.assertRaises(cr.ChromaticRegistrationError):
            cr.load_calibration("/nonexistent/widefield.json")

    def test_nonzero_z_offset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(os.path.join(tmp, "widefield.json"),
                               [_plane(), _plane(), _plane()],
                               z_offsets=[0.0, 0.0, 120.0])
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(p)
            self.assertIn("ZOffset", str(ctx.exception))

    def test_nonlinear_term_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(),
                 _plane(nonlinear={"DimX": 3, "DimY": 3,
                                   "Coefficients": [1.0] * 9})])
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(p)
            self.assertIn("nonlinear", str(ctx.exception).lower())

    def test_missing_dapi_transform_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(os.path.join(tmp, "widefield.json"),
                               [_plane(), _plane()], lasers=("640", "488"))
            with self.assertRaises(cr.ChromaticRegistrationError):
                cr.load_calibration(p)

    def test_probe_transform_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(os.path.join(tmp, "widefield.json"),
                               [_plane(), _plane()],
                               lasers=("640", "488", "405"))
            with self.assertRaises(cr.ChromaticRegistrationError):
                cr.load_calibration(p)

    def test_singular_matrix_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_metadata(os.path.join(tmp, "widefield.json"),
                               [_plane(), _plane(),
                                _plane(m00=0.0, m01=0.0, m10=0.0, m11=0.0)])
            with self.assertRaises(cr.ChromaticRegistrationError):
                cr.load_calibration(p)

    def test_provenance_records_the_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(m02=-78.4), _plane(m02=21.3)],
                px=(102.96875, 103.59375), rms=[0.0, 8.17, 1.76]))
            rec = cr.provenance(calib, {"probe_0": "640", "probe_1": "488"})
            self.assertEqual(rec["schema_version"], cr.PROVENANCE_SCHEMA)
            self.assertEqual(rec["transform_direction"],
                             "reference_plane_to_channel_plane")
            self.assertEqual(rec["interpolation"], cr.INTERPOLATION)
            self.assertEqual(rec["translation_units"], "nanometre_physical")
            self.assertAlmostEqual(rec["pixel_size_nm"]["x"], 102.96875)
            self.assertIn("405", rec["channel_map"])
            self.assertEqual(rec["channel_map"]["488"]["rms_error_nm"], 8.17)
            self.assertIn("probe_0", rec["applied"])
            self.assertEqual(rec["applied"]["probe_0"]["target_wavelength"], "640")
            # must be JSON-serialisable
            json.dumps(rec)


class TestCenterOwnership(unittest.TestCase):
    """Ownership classification, including the 2.5D z-gated case (9)."""

    def setUp(self):
        self.labels = np.zeros((5, 20, 20), dtype=np.int32)
        self.labels[:, 4:8, 4:8] = 1     # WalkID 0
        self.labels[:, 12:16, 12:16] = 2  # WalkID 1

    def test_label_convention(self):
        self.assertEqual(co.label_for_walk(0), 1)
        self.assertEqual(co.label_for_walk(41), 42)

    def test_own_background_and_other(self):
        self.assertEqual(co.classify_center(self.labels, 0, 2, 5, 5), co.OWN_MASK)
        self.assertEqual(co.classify_center(self.labels, 0, 2, 10, 10),
                         co.BACKGROUND)
        self.assertEqual(co.classify_center(self.labels, 0, 2, 13, 13),
                         co.OTHER_ACTIVE_CELL)

    def test_out_of_bounds_and_nonfinite_are_distinguished(self):
        """An off-grid centre is a real position; a non-finite one is not.

        Both are unevaluable under strict, but the recorded reason differs so a
        numerically failed fit is not confused with a spot outside the field.
        """
        self.assertEqual(co.classify_center(self.labels, 0, 2, 5, 999),
                         co.OUT_OF_BOUNDS)
        self.assertEqual(co.classify_center(self.labels, 0, 99, 5, 5),
                         co.OUT_OF_BOUNDS)
        self.assertEqual(co.classify_center(self.labels, 0, np.nan, 5, 5),
                         co.UNCLASSIFIABLE)
        self.assertEqual(co.classify_center(self.labels, 0, np.inf, 5, 5),
                         co.UNCLASSIFIABLE)
        # Neither passes, and each carries a distinct failure reason.
        for cat in (co.OUT_OF_BOUNDS, co.UNCLASSIFIABLE):
            self.assertNotIn(cat, co.PASSING)
            self.assertEqual(co.failure_reason(cat), f"center_{cat}")

    def test_nearest_voxel_not_interpolated(self):
        # 3.6 rounds to 4, which is inside the cell; 3.4 rounds to 3, outside.
        self.assertEqual(co.classify_center(self.labels, 0, 2, 3.6, 5),
                         co.OWN_MASK)
        self.assertEqual(co.classify_center(self.labels, 0, 2, 3.4, 5),
                         co.BACKGROUND)

    def test_zgated_axial_restriction_changes_ownership(self):
        """(9) The same XY centre is owned inside the gate, background outside."""
        gated = self.labels.copy()
        gated[3:, :, :] = 0          # this timepoint's gate keeps planes 0-2
        self.assertEqual(co.classify_center(gated, 0, 1, 5, 5), co.OWN_MASK)
        self.assertEqual(co.classify_center(gated, 0, 4, 5, 5), co.BACKGROUND)
        # and the ungated volume still owns it, proving the gate did the work
        self.assertEqual(co.classify_center(self.labels, 0, 4, 5, 5), co.OWN_MASK)

    def test_registration_then_gating_composes(self):
        """Registering the 2D map then gating must equal gating the registered map."""
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                [_plane(), _plane(), _plane(m02=200.0, m12=-100.0)],
                px=(100.0, 100.0)))
            labels_2d = self.labels[0]
            reg_2d = cr.resample_labels_to_channel(labels_2d, calib, "640")
            stack_then_gate = np.repeat(reg_2d[None], 5, axis=0)
            stack_then_gate[3:] = 0

            reg_3d = cr.resample_labels_to_channel(self.labels, calib, "640")
            gate_then = reg_3d.copy()
            gate_then[3:] = 0
            np.testing.assert_array_equal(stack_then_gate, gate_then)

    def test_owned_voxel_seed_picks_brightest_owned(self):
        image = np.zeros_like(self.labels, dtype=float)
        image[:] = 1.0
        image[2, 13, 13] = 1000.0   # brightest overall, but the OTHER cell
        image[3, 6, 6] = 50.0       # brightest owned by WalkID 0
        seed = co.owned_voxel_seed(image, self.labels, 0)
        self.assertEqual(seed, (3, 6, 6))

    def test_owned_voxel_seed_none_when_gate_excludes_cell(self):
        gated = self.labels.copy()
        gated[gated == 1] = 0
        self.assertIsNone(co.owned_voxel_seed(
            np.ones_like(gated, dtype=float), gated, 0))

    def test_summarize_reports_all_categories(self):
        counts = co.summarize([co.OWN_MASK, co.OWN_MASK, co.BACKGROUND])
        self.assertEqual(counts[co.OWN_MASK], 2)
        self.assertEqual(counts[co.BACKGROUND], 1)
        self.assertEqual(counts[co.OTHER_ACTIVE_CELL], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
