import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from pipeline.z_support_2p5d import compute_z_support, save_z_support
from pipeline.center_ownership import OWNERSHIP_ROW_FIELDS


ROOT = Path(__file__).resolve().parents[1]
LOCALIZER = ROOT / "pipeline" / "localize_probe0.py"
LOCALIZER_488 = ROOT / "pipeline" / "localize_488.py"


class ZGatedLocalizationIntegrationTests(unittest.TestCase):
    def _fixture(self, root):
        probe0 = root / "probe_0"
        probe1 = root / "probe_1"
        mask_dir = root / "probe_2" / "cellpose_masks"
        probe0.mkdir(parents=True)
        probe1.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        shape = (7, 15, 15)
        labels_2d = np.zeros(shape[1:], dtype=np.int32)
        labels_2d[3:12, 3:12] = 1
        masks_3d = np.repeat(labels_2d[None, :, :], shape[0], axis=0)
        tifffile.imwrite(mask_dir / "masks_2p5d_2d.tif", labels_2d)
        tifffile.imwrite(mask_dir / "masks_final.tif", masks_3d)

        zz, yy, xx = np.indices(shape, dtype=np.float64)
        image = 100.0 + 5000.0 * np.exp(
            -0.5
            * (
                ((zz - 3.0) / 1.0) ** 2
                + ((yy - 7.0) / 1.5) ** 2
                + ((xx - 7.0) / 1.5) ** 2
            )
        )
        tifffile.imwrite(probe0 / "TP000.tif", image.astype(np.uint16))
        tifffile.imwrite(probe1 / "TP000.tif", image.astype(np.uint16))
        return probe0, mask_dir, labels_2d, image.astype(np.uint16)

    def _run(
        self, root, policy="quarantine", localizer=LOCALIZER,
        mode="2p5d-zgated"
    ):
        env = os.environ.copy()
        env["PIPELINE_OUT"] = str(root)
        return subprocess.run(
            [
                sys.executable,
                str(localizer),
                "--segmentation-mode",
                mode,
                "--z-gate-failure-policy",
                policy,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_legacy_modes_keep_the_original_localization_schema(self):
        """Legacy columns stay an unchanged PREFIX; ownership columns append.

        Ownership must be identifiable from the comprehensive CSV itself, not only
        from a sidecar, so the audit columns now travel on every localization row.
        They are appended after the historical block, so any reader that indexes
        the legacy prefix by position is unaffected.
        """
        legacy_fields = [
            "tp", "WalkID", "z", "y", "x", "sigma_z", "sigma_y",
            "sigma_x", "amplitude", "background", "z_err", "y_err", "x_err",
        ]
        for mode in ("stitched", "2p5d", "3d"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._fixture(root)
                result = self._run(root, mode=mode)
                self.assertEqual(
                    result.returncode, 0, result.stderr + result.stdout)
                loc_dir = root / "probe_0" / "localizations"
                with open(loc_dir / "localizations_640.csv") as handle:
                    reader = csv.DictReader(handle)
                    names = reader.fieldnames
                    rows = list(reader)
                self.assertEqual(names[:len(legacy_fields)], legacy_fields)
                self.assertEqual(names[len(legacy_fields):],
                                 list(OWNERSHIP_ROW_FIELDS))
                self.assertEqual(len(rows), 1)
                # Default policy is 'allow' with registration off: the legacy
                # behaviour, recorded rather than acted on.
                self.assertEqual(rows[0]["CenterOwnershipPolicy"], "allow")
                self.assertEqual(rows[0]["CenterOwnershipValid"], "1")
                self.assertEqual(rows[0]["CenterRefitAttempted"], "0")
                self.assertEqual(rows[0]["CenterRefitAccepted"], "0")
                self.assertEqual(rows[0]["ChromaticRegistrationApplied"], "0")
                self.assertEqual(
                    rows[0]["OwnershipCoordinateFrame"],
                    "raw_target_channel_pixels")
                # The per-attempt status table is now written in EVERY mode: it
                # is the single auditable record of centre ownership, which is
                # classified regardless of segmentation mode. Its z_gate_*
                # columns are blank when the mode has no dynamic gate.
                status = loc_dir / "localizations_640_z_gate_status.csv"
                self.assertTrue(status.exists())
                with open(status) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["z_gate_status"], "")
                self.assertEqual(rows[0]["center_ownership_initial"], "own_mask")
                self.assertEqual(rows[0]["center_ownership_pass"], "1")
                self.assertEqual(rows[0]["center_refit_attempted"], "0")

    def test_good_gate_is_used_and_written_to_localization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mask_dir, labels_2d, _ = self._fixture(root)
            dapi = np.full((7, 15, 15), 100, dtype=np.uint16)
            dapi[3, labels_2d == 1] = 2000
            support = compute_z_support(
                labels_2d, [("TP000", dapi)], failure_policy="quarantine"
            )
            self.assertTrue(bool(support.evaluable[0, 0]))
            save_z_support(
                support,
                mask_dir / "z_support_2p5d.npz",
                mask_dir / "z_support_2p5d_summary.json",
            )

            result = self._run(root, "quarantine")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with open(
                root / "probe_0" / "localizations" / "localizations_640.csv"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["z_gate_status"], str(support.status[0, 0]))
            self.assertEqual(rows[0]["z_gate_fallback"], "none")
            self.assertEqual(rows[0]["z_gate_evaluable"], "1")
            z = float(rows[0]["z"])
            self.assertGreaterEqual(z, int(support.z_min[0, 0]))
            self.assertLessEqual(z, int(support.z_max[0, 0]))

            result = self._run(root, "quarantine", LOCALIZER_488)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with open(
                root / "probe_1" / "localizations" / "localizations_488.csv"
            ) as handle:
                rows_488 = list(csv.DictReader(handle))
            self.assertEqual(len(rows_488), 1)
            self.assertEqual(rows_488[0]["z_gate_fallback"], "none")
            self.assertEqual(rows_488[0]["z_gate_evaluable"], "1")

    def test_fail_policy_records_disallowed_gate_without_numeric_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mask_dir, labels_2d, _ = self._fixture(root)
            flat_dapi = np.full((7, 15, 15), 100, dtype=np.uint16)
            support = compute_z_support(
                labels_2d, [("TP000", flat_dapi)], failure_policy="fail"
            )
            self.assertFalse(bool(support.fit_allowed[0, 0]))
            save_z_support(
                support,
                mask_dir / "z_support_2p5d.npz",
                mask_dir / "z_support_2p5d_summary.json",
            )

            result = self._run(root, "fail")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            loc_dir = root / "probe_0" / "localizations"
            with open(loc_dir / "localizations_640.csv") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with open(loc_dir / "localizations_640_z_gate_status.csv") as handle:
                audit = list(csv.DictReader(handle))
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["z_gate_fallback"], "disallowed")
            self.assertEqual(audit[0]["z_gate_fit_allowed"], "0")
            self.assertEqual(audit[0]["localization_written"], "0")


if __name__ == "__main__":
    unittest.main()
