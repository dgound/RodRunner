import argparse
import contextlib
import ctypes
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
PIPELINE = Path(__file__).resolve().parents[1] / 'pipeline'
sys.path.insert(0, str(PIPELINE))
import replication_qc as rq  # noqa: E402
import z_support_2p5d as zs  # noqa: E402


def _gaussian_grid(shape, amp, center, sigma):
    z, y, x = np.indices(shape, dtype=np.float64)
    cz, cy, cx = center
    sz, sy, sx = sigma
    return amp * np.exp(-0.5 * (
        ((z - cz) / sz) ** 2
        + ((y - cy) / sy) ** 2
        + ((x - cx) / sx) ** 2
    ))


def _run_batch(*, second_center=None, sigma_z=5.0, sigma_z_max=0.0,
               first_amp=1000.0, second_amp=800.0, noise_sd=0.5,
               external=False):
    shape = (21, 17, 23 if external else 17)
    image = 100.0 + _gaussian_grid(shape, first_amp, (10, 8, 5),
                                   (sigma_z, 1.1, 1.1))
    if second_center is not None:
        image += _gaussian_grid(shape, second_amp, second_center,
                                (sigma_z, 1.1, 1.1))
    if noise_sd:
        image += np.random.default_rng(7).normal(0.0, noise_sd, shape)
    image = np.ascontiguousarray(image, dtype=np.float64)

    labels = np.zeros(shape, dtype=np.int32)
    labels[:, 2:15, 2:(14 if external else 15)] = 1
    bbox = np.array([0, shape[0] - 1, 0, shape[1] - 1,
                     0, shape[2] - 3 if external else shape[2] - 1],
                    dtype=np.int32)
    has_fit1 = np.array([1], dtype=np.int32)
    fit1 = np.array([1200.0, 10.0, 8.0, 8.0 if second_center else 5.0,
                     sigma_z, 1.1, 3.2 if second_center else 1.1, 100.0],
                    dtype=np.float64)
    out_floats = np.zeros(41, dtype=np.float64)
    out_ints = np.zeros(24, dtype=np.int32)

    rq._BACKEND.batch_replication_qc(
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        *shape,
        bbox.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        has_fit1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        fit1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        1,
        2.5, 1.5, 0.05, 2.0, 0.25,
        20.0, 500.0, 0.3, 0.5,
        0.0, 0.0, 0.0,
        sigma_z_max, 110.0, 100.0, 103.28,
        20, 200,
        out_floats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_ints.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return out_floats, out_ints


class ReplicationQCTest(unittest.TestCase):
    def test_zgated_qc_blocks_inactive_owned_xy_from_background_shell(self):
        labels_2d = np.array([[1, 2], [0, 0]], dtype=np.int32)
        effective = np.zeros((3, 2, 2), dtype=np.int32)
        effective[1, 0, 0] = 1
        effective[2, 0, 1] = 2
        protected = rq._protect_inactive_xy_ownership(
            effective, labels_2d, n_cells=2)
        self.assertTrue(np.array_equal(protected[:, 0, 0], [3, 1, 3]))
        self.assertTrue(np.array_equal(protected[:, 0, 1], [3, 3, 2]))
        self.assertTrue(np.all(protected[:, 1, :] == 0))

    def test_zgated_localization_provenance_mismatch_is_rejected(self):
        gate = {
            'status': np.array(['ok']),
            'fallback': np.array(['none']),
            'z_min': np.array([2]),
            'z_max': np.array([5]),
            'quarantined': np.array([False]),
            'evaluable': np.array([True]),
            'fit_allowed': np.array([True]),
        }
        fit = {
            'z_gate_status': 'ok', 'z_gate_fallback': 'none',
            'z_gate_min': 2, 'z_gate_max': 6,
            'z_gate_quarantined': False, 'z_gate_evaluable': True,
            'z_gate_fit_allowed': True,
        }
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            rq._validate_fit_gate_provenance(fit, gate, 0, 'TP000', 0)

    def test_zgated_localization_csv_requires_gate_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'localizations.csv'
            with open(path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'tp', 'WalkID', 'z', 'y', 'x', 'sigma_z', 'sigma_y',
                    'sigma_x', 'amplitude', 'background',
                ])
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, 'missing columns'):
                rq._load_1spot_csv(path, require_z_gate=True)

    def test_pooled_null_qvalues_use_family_null_tail(self):
        observed = [20.0, 30.0, 10.0]
        null = np.full((3, 10), -np.inf)
        q_values, raw_fdr, exceedances = rq._pooled_null_qvalues(
            observed, null)
        self.assertTrue(np.allclose(raw_fdr, [0.05, 0.1, 1.0 / 30.0]))
        self.assertTrue(np.allclose(q_values, np.full(3, 1.0 / 30.0)))
        self.assertTrue(np.array_equal(exceedances, np.zeros(3, dtype=int)))

    def test_bootstrap_support_uses_physical_shell_and_excludes_other_cells(self):
        labels = np.zeros((3, 3, 3), dtype=np.int32)
        labels[1, 1, 1] = 1
        labels[1, 1, 2] = 2
        support = rq._owned_support(labels, 1, 110.0, 100.0, 103.28)
        self.assertTrue(support[1, 1, 1])
        self.assertTrue(support[0, 1, 1])
        self.assertTrue(support[1, 0, 1])
        self.assertFalse(support[0, 0, 1])
        self.assertFalse(support[1, 1, 2])

    def test_uncapped_rep3_recovers_broad_double(self):
        floats, ints = _run_batch(second_center=(10, 8, 11), sigma_z=5.0)
        self.assertEqual(ints[0], 4)
        self.assertEqual(ints[3], 1)
        self.assertEqual(ints[13], 0)
        self.assertEqual(ints[15], 4)
        self.assertEqual(ints[18], 0)
        self.assertEqual(ints[19], 1)
        self.assertAlmostEqual(floats[8], 5.0, delta=0.1)
        self.assertGreaterEqual(floats[32], 500.0)

    def test_replicate1_cap_is_censoring_not_a_veto(self):
        floats, ints = _run_batch(second_center=(10, 8, 11), sigma_z=5.0,
                                  sigma_z_max=3.5)
        self.assertAlmostEqual(floats[8], 3.5, delta=0.05)
        self.assertEqual(ints[18], 1)
        self.assertEqual(ints[19], 0)
        self.assertGreaterEqual(floats[34], 500.0)
        self.assertEqual(ints[3], 1)

    def test_broad_single_spot_is_not_confirmed(self):
        _, ints = _run_batch(second_center=None, sigma_z=5.0, noise_sd=1.0)
        self.assertEqual(ints[3], 0)

    def test_external_focus_beyond_owned_shell_is_not_confirmed(self):
        _, ints = _run_batch(second_center=(10, 8, 17), sigma_z=5.0,
                             second_amp=1000.0, external=True)
        self.assertEqual(ints[3], 0)

    def test_two_spot_result_is_invariant_to_amplitude_label_order(self):
        f1, i1 = _run_batch(second_center=(10, 8, 11), first_amp=1000,
                            second_amp=800, noise_sd=0.0)
        f2, i2 = _run_batch(second_center=(10, 8, 11), first_amp=800,
                            second_amp=1000, noise_sd=0.0)
        self.assertEqual(i1[3], 1)
        self.assertEqual(i2[3], 1)
        self.assertAlmostEqual(f1[3], f2[3], delta=0.1)
        self.assertTrue(np.allclose(np.sort(f1[4:6]), np.sort(f2[4:6]),
                                    rtol=1e-4, atol=1e-3))

    def test_calibration_config_must_meet_false_positive_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'calibration.json'
            path.write_text(json.dumps({
                'schema_version': 1,
                'validated': True,
                'calibration_version': 'rep3-test-v1',
                'model_policy': {
                    'sigma_z_max': 0.0, 'center_shell_nm': 110.0,
                    'z_step_nm': 100.0, 'xy_pixel_nm': 103.28,
                    'require_two_lobe_for_confirmation': True,
                    'min_spot_sep_xy_nm': 0.0,
                },
                'validation': {'max_tp_false_positive_rate': 0.008},
                'thresholds': {
                    'delta_bic_thresh': 25, 'min_spot_sep_nm': 500,
                    'min_amp_ratio': 0.4, 'min_amp_recovery': 0.5,
                    'min_peak_support_c2': 0.7, 'max_valley_frac': 0,
                    'min_resid_core_frac': 0.05,
                },
            }))
            args = SimpleNamespace(calibration_config=str(path),
                                   delta_bic_thresh=15.0, min_amp_ratio=0.3,
                                   sigma_z_max=0.0, center_shell_nm=110.0,
                                   z_step_nm=100.0, xy_pixel_nm=103.28,
                                   require_two_lobe_for_confirmation=True,
                                   min_spot_sep_xy_nm=0.0)
            result = rq._apply_calibration_config(args, argparse.ArgumentParser())
            self.assertEqual(result.calibration_version, 'rep3-test-v1')
            self.assertEqual(result.delta_bic_thresh, 25.0)
            self.assertEqual(result.min_amp_ratio, 0.4)

            payload = json.loads(path.read_text())
            payload['validation']['max_tp_false_positive_rate'] = 0.02
            path.write_text(json.dumps(payload))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rq._apply_calibration_config(args, argparse.ArgumentParser())

            payload['validation']['max_tp_false_positive_rate'] = 0.008
            payload['model_policy']['require_two_lobe_for_confirmation'] = 'true'
            path.write_text(json.dumps(payload))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rq._apply_calibration_config(args, argparse.ArgumentParser())

            payload['model_policy']['require_two_lobe_for_confirmation'] = True
            payload['thresholds']['delta_bic_thresh'] = float('nan')
            path.write_text(json.dumps(payload))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rq._apply_calibration_config(args, argparse.ArgumentParser())

    def test_end_to_end_writes_v2_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / 'probe_0'
            outdir = root / 'out'
            probe.mkdir()
            shape = (21, 17, 17)
            image = 100.0 + _gaussian_grid(shape, 1000.0, (10, 8, 5),
                                           (5.0, 1.1, 1.1))
            image += _gaussian_grid(shape, 800.0, (10, 8, 11),
                                    (5.0, 1.1, 1.1))
            mask = np.zeros(shape, dtype=np.int32)
            mask[:, 2:15, 2:15] = 1
            tifffile.imwrite(probe / 'TP000.tif', image.astype(np.uint16))
            tifffile.imwrite(root / 'mask.tif', mask)
            with open(root / 'localizations.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'tp', 'WalkID', 'z', 'y', 'x', 'sigma_z', 'sigma_y',
                    'sigma_x', 'amplitude', 'background',
                ])
                writer.writeheader()
                writer.writerow({
                    'tp': 'TP000', 'WalkID': 0, 'z': 10, 'y': 8, 'x': 8,
                    'sigma_z': 5, 'sigma_y': 1.1, 'sigma_x': 3.2,
                    'amplitude': 1200, 'background': 100,
                })
            cmd = [
                sys.executable, str(PIPELINE / 'replication_qc.py'),
                '--input_csv', str(root / 'localizations.csv'),
                '--probe0_dir', str(probe), '--mask_path', str(root / 'mask.tif'),
                '--outdir', str(outdir), '--output_prefix', 'test',
                '--segmentation-mode', '3d',
                '--sigma_z_max', '0', '--limit_tps', '1',
                '--replication_tp_start', '0', '--replication_tp_end', '19',
                '--delta_bic_thresh', '20', '--min_spot_sep_nm', '500',
                '--min_amp_recovery', '0.5',
                '--auto_null_bootstrap_reps', '16',
                '--auto_null_max_reps', '128', '--auto_fdr_q', '0.01',
            ]
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            with open(outdir / 'test_replication_qc.csv') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['schema_version'], 'replication_qc_v2')
            self.assertEqual(rows[0]['fit2_n_starts'], '4')
            self.assertGreater(float(rows[0]['sigma_z']), 4.5)
            self.assertEqual(rows[0]['auto_null_tested'], '1')
            # One candidate needs 100 empty-tail draws to resolve q=0.01.
            self.assertEqual(rows[0]['auto_null_reps'], '100')
            self.assertTrue(0.0 < float(rows[0]['auto_null_p']) <= 1.0)
            self.assertEqual(rows[0]['auto_calibration_pass'], '1')
            self.assertEqual(rows[0]['replication_confirmed'], '1')
            self.assertEqual(float(rows[0]['auto_pooled_fdr']), 0.01)
            with open(outdir / 'test_auto_calibration.json') as f:
                auto_calibration = json.load(f)
            self.assertEqual(
                auto_calibration['schema'],
                'replication_qc_auto_calibration_v2')
            self.assertEqual(
                auto_calibration['summary']['provisional_candidates'], 1)
            self.assertEqual(
                auto_calibration['model_policy']['segmentation_mode'], '3d')
            self.assertEqual(
                auto_calibration['timepoints'][0]['effective_replicates'], 100)

            tifffile.imwrite(probe / 'TP000.tif', np.zeros((20, 17, 17), np.uint16))
            mismatch = subprocess.run(cmd, text=True, capture_output=True, check=False)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn('Shape mismatch', mismatch.stderr + mismatch.stdout)

    def test_zgated_quarantine_is_explicit_and_never_multispot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / 'probe_0'
            mask_dir = root / 'cellpose_masks'
            outdir = root / 'out'
            probe.mkdir()
            mask_dir.mkdir()
            shape = (5, 9, 9)
            labels_2d = np.zeros(shape[1:], dtype=np.int32)
            labels_2d[2:7, 2:7] = 1
            masks = np.repeat(labels_2d[None, :, :], shape[0], axis=0)
            image = np.full(shape, 100, dtype=np.uint16)
            tifffile.imwrite(probe / 'TP000.tif', image)
            tifffile.imwrite(mask_dir / 'masks_final.tif', masks)
            tifffile.imwrite(mask_dir / 'masks_2p5d_2d.tif', labels_2d)
            support = zs.compute_z_support(
                labels_2d, [('TP000', image)], failure_policy='quarantine')
            self.assertTrue(bool(support.quarantined[0, 0]))
            zs.save_z_support(
                support, mask_dir / 'z_support_2p5d.npz',
                mask_dir / 'z_support_2p5d_summary.json')

            localizations = root / 'localizations.csv'
            fields = [
                'tp', 'WalkID', 'z', 'y', 'x', 'sigma_z', 'sigma_y',
                'sigma_x', 'amplitude', 'background',
                'z_gate_status', 'z_gate_min', 'z_gate_max',
                'z_gate_fallback',
                'z_gate_quarantined', 'z_gate_evaluable',
                'z_gate_fit_allowed',
            ]
            with open(localizations, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    'tp': 'TP000', 'WalkID': 0, 'z': 2, 'y': 4, 'x': 4,
                    'sigma_z': 1, 'sigma_y': 1, 'sigma_x': 1,
                    'amplitude': 10, 'background': 100,
                    'z_gate_status': str(support.status[0, 0]),
                    'z_gate_fallback': str(support.fallback[0, 0]),
                    'z_gate_min': int(support.z_min[0, 0]),
                    'z_gate_max': int(support.z_max[0, 0]),
                    'z_gate_quarantined': 1,
                    'z_gate_evaluable': 0,
                    'z_gate_fit_allowed': 1,
                })
            cmd = [
                sys.executable, str(PIPELINE / 'replication_qc.py'),
                '--input_csv', str(localizations),
                '--probe0_dir', str(probe),
                '--mask_path', str(mask_dir / 'masks_final.tif'),
                '--outdir', str(outdir), '--output_prefix', 'zgated',
                '--segmentation-mode', '2p5d-zgated',
                '--z-gate-failure-policy', 'quarantine',
                '--limit_tps', '1', '--replication_tp_start', '0',
                '--replication_tp_end', '19',
            ]
            result = subprocess.run(
                cmd, text=True, capture_output=True, check=False)
            self.assertEqual(
                result.returncode, 0, msg=result.stderr + result.stdout)
            with open(outdir / 'zgated_replication_qc.csv') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['z_gate_quarantined'], '1')
            self.assertEqual(rows[0]['z_gate_fallback'], 'full_z')
            self.assertEqual(rows[0]['z_gate_evaluable'], '0')
            self.assertTrue(rows[0]['status'].startswith(
                'z_gate_quarantined:'))
            self.assertEqual(rows[0]['replication_confirmed'], '0')
            self.assertEqual(rows[0]['n_spots_best'], '')
            with open(outdir / 'zgated_walk_replication_summary.csv') as handle:
                summary = list(csv.DictReader(handle))[0]
            self.assertEqual(summary['MultiSpot'], '0')
            self.assertEqual(summary['Replicating'], '0')

            # The fail policy writes an explicit unevaluable QC row even
            # though the localizer correctly emitted no numeric fit row.
            fail_support = zs.compute_z_support(
                labels_2d, [('TP000', image)], failure_policy='fail')
            zs.save_z_support(
                fail_support, mask_dir / 'z_support_2p5d.npz',
                mask_dir / 'z_support_2p5d_summary.json')
            with open(localizations, 'w', newline='') as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
            fail_outdir = root / 'fail_out'
            fail_cmd = list(cmd)
            fail_cmd[fail_cmd.index(str(outdir))] = str(fail_outdir)
            fail_cmd[fail_cmd.index('quarantine')] = 'fail'
            result = subprocess.run(
                fail_cmd, text=True, capture_output=True, check=False)
            self.assertEqual(
                result.returncode, 0, msg=result.stderr + result.stdout)
            with open(fail_outdir / 'zgated_replication_qc.csv') as handle:
                fail_row = list(csv.DictReader(handle))[0]
            self.assertEqual(fail_row['z_gate_fallback'], 'disallowed')
            self.assertEqual(fail_row['z_gate_quarantined'], '0')
            self.assertEqual(fail_row['z_gate_evaluable'], '0')
            self.assertTrue(fail_row['status'].startswith('z_gate_failed:'))
            self.assertEqual(fail_row['replication_confirmed'], '0')


if __name__ == '__main__':
    unittest.main()
