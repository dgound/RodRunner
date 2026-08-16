#!/usr/bin/env python3
"""Registration direction, wavelength selection, and Z-metadata validation.

Covers required tests D (direction/composition), E (wavelength selection, no
probe0->640 hardcoding) and F (Z-offset metadata validation).
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import chromatic_registration as cr  # noqa: E402


def _plane(m00=1.0, m01=0.0, m02=0.0, m10=0.0, m11=1.0, m12=0.0):
    return {"type": "PlaneTransformation", "value": {
        "Matrix00": m00, "Matrix01": m01, "Matrix02": m02,
        "Matrix10": m10, "Matrix11": m11, "Matrix12": m12,
        "NonlinearTransform": {"type": "NonlinearTransform",
                               "value": {"DimX": 0, "DimY": 0,
                                         "Coefficients": []}}}}


def write_metadata(path, lasers, transforms, *, px=100.0, py=100.0,
                   zoffsets="auto", rms=None):
    """Vutara-shaped widefield.json. ``zoffsets='auto'`` -> complete zeros."""
    if zoffsets == "auto":
        zoffsets = [0.0] * len(transforms)
    chrom = {"Transforms": {"type": "list", "value": transforms},
             "RMSErrors": {"type": "list",
                           "value": rms if rms is not None else [0.0] * len(transforms)}}
    if zoffsets is not None:
        chrom["ZOffsets"] = {"type": "list", "value": zoffsets}
    doc = {"Configuration": {
               "PixelSizeX": px, "PixelSizeY": py,
               "Probes": {"type": "list", "value": [
                   {"type": "Probe", "value": {"WidefieldLaser": laser}}
                   for laser in lasers]}},
           "ChromaticOffset": chrom}
    with open(path, "w") as handle:
        json.dump(doc, handle)
    return path


class TestRegistrationDirection(unittest.TestCase):
    """Requirement D: verified transform direction and composition."""

    def _calib(self, tmp, **kw):
        # Reference 405 carries a known PHYSICAL translation; target 640 identity.
        return cr.load_calibration(write_metadata(
            os.path.join(tmp, "widefield.json"),
            ["640", "488", "405"],
            [_plane(), _plane(m02=500.0), _plane(m02=300.0, m12=-200.0)],
            **kw))

    def test_composition_is_reference_times_inverse_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calib(tmp)
            for target in ("640", "488"):
                expected = (calib.matrix("405")
                            @ np.linalg.inv(calib.matrix(target)))
                np.testing.assert_allclose(
                    cr.lookup_matrix(calib, target, "405"), expected, atol=1e-12)

    def test_known_translation_lands_on_the_expected_pixel(self):
        """300 nm x / -200 nm y at 100 nm pixels = +3 / -2 pixels."""
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calib(tmp)
            m = cr.lookup_matrix(calib, "640", "405")
            x, y = cr.apply_matrix_px(m, 10.0, 10.0, calib.pixel_size_nm)
            self.assertAlmostEqual(float(x), 13.0, places=9)
            self.assertAlmostEqual(float(y), 8.0, places=9)

            labels = np.zeros((1, 20, 20), dtype=np.int32)
            labels[0, 8, 13] = 7
            out = cr.resample_labels_to_channel(labels, calib, "640", "405")
            # Target pixel (10,10) pulls from reference (13, 8) -> label 7 there.
            self.assertEqual(int(out[0, 10, 10]), 7)

    def test_direction_is_not_symmetric(self):
        """Swapping reference and target must NOT give the same matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calib(tmp)
            forward = cr.lookup_matrix(calib, "640", "405")
            reverse = cr.lookup_matrix(calib, "405", "640")
            self.assertFalse(np.allclose(forward, reverse))
            np.testing.assert_allclose(forward @ reverse, np.eye(3), atol=1e-9)

    def test_labels_are_never_interpolated(self):
        """A half-pixel shift must still yield only original label values."""
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                ["640", "405"],
                [_plane(), _plane(m02=50.0)]))     # 0.5 px at 100 nm
            labels = np.zeros((1, 12, 12), dtype=np.int32)
            labels[0, 4:8, 4:8] = 3
            labels[0, 4:8, 8:11] = 9
            out = cr.resample_labels_to_channel(labels, calib, "640", "405")
            self.assertEqual(set(np.unique(out).tolist()) - {0}, {3, 9})
            self.assertEqual(out.dtype, labels.dtype)

    def test_rotation_composes_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            theta = np.deg2rad(10.0)
            c, s = np.cos(theta), np.sin(theta)
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                ["640", "405"],
                [_plane(), _plane(m00=c, m01=-s, m10=s, m11=c)]))
            m = cr.lookup_matrix(calib, "640", "405")
            np.testing.assert_allclose(m[:2, :2], np.array([[c, -s], [s, c]]),
                                       atol=1e-12)


class TestWavelengthSelection(unittest.TestCase):
    """Requirement E: explicit wavelengths, no probe0->640 hardcoding."""

    def test_widefield_and_confocal_defaults_differ(self):
        self.assertEqual(cr.modality_default("widefield", "probe"), "640")
        self.assertEqual(cr.modality_default("confocal", "probe"), "561")
        self.assertEqual(cr.modality_default("widefield", "fiducial"), "488")
        self.assertEqual(cr.modality_default("widefield", "reference"), "405")
        self.assertEqual(cr.modality_default("confocal", "reference"), "405")

    def test_unknown_modality_or_role_fails_clearly(self):
        with self.assertRaises(cr.ChromaticRegistrationError):
            cr.modality_default("lightsheet", "probe")
        with self.assertRaises(cr.ChromaticRegistrationError):
            cr.modality_default("widefield", "dapi")

    def test_widefield_405_to_640_and_405_to_488(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                ["640", "488", "405"],
                [_plane(), _plane(m02=100.0), _plane(m02=200.0)]))
            for target in ("640", "488"):
                m = cr.lookup_matrix(calib, target, "405")
                self.assertEqual(m.shape, (3, 3))
            # The two targets give DIFFERENT corrections.
            self.assertFalse(np.allclose(cr.lookup_matrix(calib, "640", "405"),
                                         cr.lookup_matrix(calib, "488", "405")))

    def test_confocal_405_to_561(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                ["561", "488", "405"],
                [_plane(), _plane(m02=100.0), _plane(m02=250.0)],
                px=81.25, py=81.25))
            m = cr.lookup_matrix(calib, "561", "405")
            x, _ = cr.apply_matrix_px(m, 0.0, 0.0, calib.pixel_size_nm)
            self.assertAlmostEqual(float(x), 250.0 / 81.25, places=9)

    def test_requesting_640_on_confocal_metadata_fails(self):
        """A confocal calibration has no 640 entry: silently using one is the bug."""
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                ["561", "488", "405"],
                [_plane(), _plane(m02=100.0), _plane(m02=250.0)]))
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.lookup_matrix(calib, "640", "405")
            self.assertIn("640", str(ctx.exception))

    def test_missing_reference_wavelength_fails_at_load(self):
        """Absent segmentation channel is caught when parsing, not at use."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(write_metadata(
                    os.path.join(tmp, "widefield.json"),
                    ["640", "488"], [_plane(), _plane(m02=100.0)]))
            self.assertIn("405", str(ctx.exception))

    def test_missing_target_wavelength_fails_at_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(write_metadata(
                os.path.join(tmp, "widefield.json"),
                ["488", "405"], [_plane(), _plane(m02=100.0)]))
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.lookup_matrix(calib, "561", "405")
            self.assertIn("561", str(ctx.exception))

    def test_duplicate_wavelength_is_ambiguous_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(write_metadata(
                    os.path.join(tmp, "widefield.json"),
                    ["640", "640", "405"],
                    [_plane(), _plane(m02=1.0), _plane(m02=2.0)]))
            self.assertIn("more than once", str(ctx.exception))

    def test_no_probe_index_is_consulted(self):
        """Reordering the probes must not change the 405->640 correction."""
        with tempfile.TemporaryDirectory() as tmp:
            a = cr.load_calibration(write_metadata(
                os.path.join(tmp, "a.json"), ["640", "488", "405"],
                [_plane(), _plane(m02=100.0), _plane(m02=200.0)]))
            b = cr.load_calibration(write_metadata(
                os.path.join(tmp, "b.json"), ["405", "488", "640"],
                [_plane(m02=200.0), _plane(m02=100.0), _plane()]))
            np.testing.assert_allclose(cr.lookup_matrix(a, "640", "405"),
                                       cr.lookup_matrix(b, "640", "405"),
                                       atol=1e-12)
            self.assertNotEqual(a.transform_index["640"], b.transform_index["640"])


class TestZOffsetValidation(unittest.TestCase):
    """Requirement F: Z metadata must be complete or absent, never guessed."""

    def _write(self, tmp, zoffsets):
        return write_metadata(os.path.join(tmp, "widefield.json"),
                              ["640", "488", "405"],
                              [_plane(), _plane(m02=1.0), _plane(m02=2.0)],
                              zoffsets=zoffsets)

    def test_complete_zero_offsets_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(self._write(tmp, [0.0, 0.0, 0.0]))
            self.assertEqual(set(calib.z_offsets_nm.values()), {0.0})

    def test_wholly_absent_field_is_accepted_as_no_axial_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = cr.load_calibration(self._write(tmp, None))
            self.assertEqual(set(calib.z_offsets_nm.values()), {0.0})

    def test_incomplete_offsets_fail_rather_than_default_to_zero(self):
        """The original defect: a short array silently became zeros."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(self._write(tmp, [0.0, 0.0]))
            self.assertIn("ZOffsets", str(ctx.exception))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError):
                cr.load_calibration(self._write(tmp, []))

    def test_malformed_offsets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError):
                cr.load_calibration(self._write(tmp, [0.0, "x", 0.0]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError):
                cr.load_calibration(self._write(tmp, [0.0, float("nan"), 0.0]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(self._write(tmp, {"a": 1}))
            self.assertIn("expected a list", str(ctx.exception))

    def test_nonzero_offsets_are_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.load_calibration(self._write(tmp, [0.0, 120.0, 0.0]))
            self.assertIn("non-zero axial", str(ctx.exception))


class TestProvenanceIsolation(unittest.TestCase):
    """Requirement 7: one stage must never overwrite another's provenance."""

    def _calib(self, tmp):
        return cr.load_calibration(write_metadata(
            os.path.join(tmp, "widefield.json"), ["640", "488", "405"],
            [_plane(), _plane(m02=1.0), _plane(m02=2.0)]))

    def test_record_carries_stage_hash_and_wavelengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calib(tmp)
            rec = cr.provenance(calib, {"probe_0": "640"}, stage="localize_probe0",
                                reference_wavelength="405",
                                ownership_policy="strict",
                                cli_args={"modality": "widefield"})
            self.assertEqual(rec["stage"], "localize_probe0")
            self.assertEqual(rec["reference_wavelength_nm"], "405")
            self.assertEqual(rec["ownership_policy"], "strict")
            self.assertEqual(len(rec["source_calibration_sha256"]), 64)
            self.assertEqual(rec["applied"]["probe_0"]["target_wavelength"], "640")
            self.assertEqual(rec["cli_args"]["modality"], "widefield")

    def test_writing_a_different_stage_to_the_same_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calib(tmp)
            path = os.path.join(tmp, "shared.json")
            cr.write_provenance(path, cr.provenance(
                calib, {"probe_0": "640"}, stage="localize_probe0"))
            with self.assertRaises(cr.ChromaticRegistrationError) as ctx:
                cr.write_provenance(path, cr.provenance(
                    calib, {"probe_0": "640"}, stage="replication_qc"))
            self.assertIn("refusing to overwrite", str(ctx.exception))

    def test_rewriting_the_same_stage_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._calib(tmp)
            path = os.path.join(tmp, "same.json")
            for _ in range(2):
                cr.write_provenance(path, cr.provenance(
                    calib, {"probe_0": "640"}, stage="localize_probe0"))
            with open(path) as handle:
                self.assertEqual(json.load(handle)["stage"], "localize_probe0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
