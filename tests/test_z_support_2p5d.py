import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.z_support_2p5d import (
    FALLBACK_CONSENSUS,
    FALLBACK_DISALLOWED,
    FALLBACK_FULL_Z,
    STATUS_AMBIGUOUS,
    STATUS_EDGE_CLIPPED,
    STATUS_GOOD,
    STATUS_NONFINITE,
    STATUS_WEAK,
    _fill_single_plane_gaps,
    compute_z_support,
    load_z_support,
    materialize_labels,
    save_z_support,
)


def single_label_image(label_id=1):
    labels = np.zeros((8, 9), dtype=np.uint16)
    labels[2:6, 3:7] = label_id
    return labels


def stack_from_profiles(labels, profiles, background=7.0):
    """Build a stack where ``profiles`` maps label ID to signal above background."""
    nz = len(next(iter(profiles.values())))
    stack = np.full((nz,) + labels.shape, background, dtype=np.float64)
    for label_id, profile in profiles.items():
        profile = np.asarray(profile, dtype=np.float64)
        stack[:, labels == label_id] += profile[:, None]
    return stack


class ZSupportGateTests(unittest.TestCase):
    def test_centered_peak_produces_contiguous_narrow_gate(self):
        labels = single_label_image()
        profile = np.zeros(15)
        profile[6:9] = [5, 12, 5]
        support = compute_z_support(
            labels, [("DAPI_001", stack_from_profiles(labels, {1: profile}))]
        )

        tp = support.select_tp("DAPI_001")
        self.assertEqual(tp.status[0], STATUS_GOOD)
        self.assertLessEqual(tp.z_min[0], 7)
        self.assertGreaterEqual(tp.z_max[0], 7)
        self.assertGreaterEqual(tp.z_max[0] - tp.z_min[0] + 1, 5)
        self.assertLess(tp.z_max[0] - tp.z_min[0] + 1, 15)
        self.assertTrue(tp.evaluable[0])
        self.assertTrue(tp.fit_allowed[0])
        self.assertFalse(tp.quarantined[0])

    def test_boundary_peak_is_evaluable_and_records_edge_clipping(self):
        labels = single_label_image()
        lower = np.zeros(12)
        lower[0] = 10
        upper = np.zeros(12)
        upper[-1] = 10
        support = compute_z_support(
            labels,
            [
                ("lower", stack_from_profiles(labels, {1: lower})),
                ("upper", stack_from_profiles(labels, {1: upper})),
            ],
        )
        lower_tp = support.select_tp("lower")
        upper_tp = support.select_tp("upper")
        self.assertEqual(lower_tp.status[0], STATUS_EDGE_CLIPPED)
        self.assertEqual(lower_tp.z_min[0], 0)
        self.assertEqual(upper_tp.status[0], STATUS_EDGE_CLIPPED)
        self.assertEqual(upper_tp.z_max[0], 11)
        self.assertTrue(lower_tp.edge_clipped[0])
        self.assertTrue(upper_tp.edge_clipped[0])
        self.assertTrue(lower_tp.evaluable[0])
        self.assertTrue(upper_tp.evaluable[0])

    def test_one_plane_internal_gap_is_filled(self):
        np.testing.assert_array_equal(
            _fill_single_plane_gaps([False, True, False, True, False]),
            [False, True, True, True, False],
        )

    def test_secondary_component_below_half_is_not_ambiguous(self):
        labels = single_label_image()
        profile = np.zeros(24)
        profile[4:7] = 10
        profile[16:19] = 3
        support = compute_z_support(
            labels, [("below", stack_from_profiles(labels, {1: profile}))]
        )
        tp = support.select_tp("below")
        self.assertIn(tp.status[0], (STATUS_GOOD, STATUS_EDGE_CLIPPED))
        self.assertLess(tp.competing_signal_ratio[0], 0.5)
        self.assertFalse(tp.quarantined[0])

    def test_secondary_component_at_half_is_ambiguous(self):
        labels = single_label_image()
        profile = np.zeros(24)
        profile[4:7] = 10
        profile[16:19] = 6
        support = compute_z_support(
            labels, [("above", stack_from_profiles(labels, {1: profile}))],
            failure_policy="fail",
        )
        tp = support.select_tp("above")
        self.assertEqual(tp.status[0], STATUS_AMBIGUOUS)
        self.assertGreaterEqual(tp.competing_signal_ratio[0], 0.5)
        self.assertFalse(tp.evaluable[0])
        self.assertFalse(tp.fit_allowed[0])
        self.assertEqual(tp.fallback[0], FALLBACK_DISALLOWED)

    def test_flat_and_nonfinite_profiles_have_explicit_statuses(self):
        labels = single_label_image()
        flat = np.full(10, 4.0)
        bad = np.zeros(10)
        bad[4] = np.nan
        support = compute_z_support(
            labels,
            [
                ("flat", stack_from_profiles(labels, {1: flat})),
                ("bad", stack_from_profiles(labels, {1: bad})),
            ],
            failure_policy="fail",
        )
        self.assertEqual(support.select_tp("flat").status[0], STATUS_WEAK)
        self.assertEqual(support.select_tp("bad").status[0], STATUS_NONFINITE)
        self.assertFalse(np.any(support.fit_allowed))

    def test_quarantine_uses_consensus_then_full_z_if_none_valid(self):
        labels = np.zeros((8, 12), dtype=np.uint16)
        labels[1:4, 1:4] = 1
        labels[1:4, 7:10] = 9
        p1 = np.zeros(16)
        p1[4] = 10
        p2 = np.zeros(16)
        p2[8] = 10
        weak = np.ones(16)
        stacks = [
            ("good_a", stack_from_profiles(labels, {1: p1, 9: weak})),
            ("good_b", stack_from_profiles(labels, {1: p2, 9: weak})),
            ("weak", stack_from_profiles(labels, {1: weak, 9: weak})),
        ]
        support = compute_z_support(labels, stacks, failure_policy="quarantine")
        weak_tp = support.select_tp("weak")

        expected_min = int(np.rint(np.median(support.z_min[:2, 0])))
        expected_max = int(np.rint(np.median(support.z_max[:2, 0])))
        self.assertEqual((weak_tp.z_min[0], weak_tp.z_max[0]), (expected_min, expected_max))
        self.assertEqual(weak_tp.fallback[0], FALLBACK_CONSENSUS)
        self.assertTrue(weak_tp.quarantined[0])
        self.assertFalse(weak_tp.evaluable[0])
        self.assertTrue(weak_tp.fit_allowed[0])

        for tp_name in support.tp_names:
            tp = support.select_tp(str(tp_name))
            self.assertEqual((tp.z_min[1], tp.z_max[1]), (0, 15))
            self.assertEqual(tp.fallback[1], FALLBACK_FULL_Z)
            self.assertTrue(tp.quarantined[1])

    def test_duplicate_or_missing_exact_time_point_name_is_rejected(self):
        labels = single_label_image()
        stack = stack_from_profiles(labels, {1: np.arange(8)})
        with self.assertRaisesRegex(ValueError, "duplicate time point"):
            compute_z_support(labels, [("same", stack), ("same", stack)])
        support = compute_z_support(labels, [("DAPI_01", stack)])
        with self.assertRaisesRegex(KeyError, "exact time point"):
            support.select_tp("dapi_01")


class ZSupportMaterializationTests(unittest.TestCase):
    def test_materialization_preserves_noncontiguous_ids_and_bounds(self):
        labels = np.array(
            [[0, 2, 2, 0], [7, 7, 0, 41], [7, 0, 41, 41]], dtype=np.uint16
        )
        volume = materialize_labels(
            labels,
            nz=6,
            z_min=np.array([1, 0, 3]),
            z_max=np.array([2, 5, 4]),
            fit_allowed=np.array([True, False, True]),
        )
        self.assertEqual(volume.dtype, labels.dtype)
        self.assertTrue(np.all(volume[1:3, labels == 2] == 2))
        self.assertTrue(np.all(volume[[0, 3, 4, 5]][:, labels == 2] == 0))
        self.assertFalse(np.any(volume[:, labels == 7]))
        self.assertTrue(np.all(volume[3:5, labels == 41] == 41))
        self.assertTrue(np.all(volume[:3, labels == 41] == 0))
        self.assertTrue(np.all(volume[5:, labels == 41] == 0))
        self.assertEqual(set(np.unique(volume).tolist()), {0, 2, 41})

    def test_materialization_rejects_bad_bounds(self):
        labels = single_label_image()
        with self.assertRaisesRegex(ValueError, "outside"):
            materialize_labels(labels, 5, np.array([2]), np.array([5]))


class ZSupportSerializationTests(unittest.TestCase):
    def setUp(self):
        self.labels = single_label_image(13)
        peak = np.zeros(11)
        peak[5] = 10
        self.support = compute_z_support(
            self.labels,
            [
                ("DAPI exact 01", stack_from_profiles(self.labels, {13: peak})),
                ("DAPI exact 02", stack_from_profiles(self.labels, {13: np.ones(11)})),
            ],
        )

    def test_pickle_free_round_trip_and_json_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "z_support_2p5d.npz"
            json_path = Path(tmpdir) / "z_support_2p5d_summary.json"
            save_z_support(
                self.support,
                npz_path,
                json_path,
                provenance={"cellpose_model": "cpsam_v2", "source": "tiff"},
            )
            with np.load(npz_path, allow_pickle=False) as raw:
                for name in raw.files:
                    raw[name]
                self.assertEqual(raw["tp_names"].dtype.kind, "U")
                self.assertEqual(raw["status"].dtype.kind, "U")
            loaded = load_z_support(
                npz_path,
                self.labels,
                expected_tp_names=["DAPI exact 01", "DAPI exact 02"],
                expected_nz=11,
                expected_failure_policy="quarantine",
            )
            np.testing.assert_array_equal(loaded.status, self.support.status)
            np.testing.assert_array_equal(loaded.z_min, self.support.z_min)
            self.assertEqual(loaded.select_tp("DAPI exact 02").status[0], STATUS_WEAK)
            summary = json.loads(json_path.read_text())
            self.assertEqual(summary["schema_version"], "1.0")
            self.assertEqual(summary["provenance"]["cellpose_model"], "cpsam_v2")
            self.assertEqual(sum(summary["status_counts"].values()), 2)

    def test_load_rejects_time_depth_mask_and_policy_mismatches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "support.npz"
            save_z_support(self.support, npz_path, Path(tmpdir) / "support.json")
            with self.assertRaisesRegex(ValueError, "time points mismatch"):
                load_z_support(npz_path, self.labels, expected_tp_names=["different"])
            with self.assertRaisesRegex(ValueError, "z depth mismatch"):
                load_z_support(npz_path, self.labels, expected_nz=99)
            with self.assertRaisesRegex(ValueError, "failure policy mismatch"):
                load_z_support(npz_path, self.labels, expected_failure_policy="fail")
            changed_mask = self.labels.copy()
            changed_mask[2, 3] = 0
            with self.assertRaisesRegex(ValueError, "ownership map hash|label identity"):
                load_z_support(npz_path, changed_mask)

    def test_load_rejects_missing_fields_and_unsupported_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            malformed = Path(tmpdir) / "malformed.npz"
            with malformed.open("wb") as handle:
                np.savez(handle, schema_version=np.asarray("1.0"))
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_z_support(malformed, self.labels)

            valid = Path(tmpdir) / "valid.npz"
            save_z_support(self.support, valid, Path(tmpdir) / "valid.json")
            with np.load(valid, allow_pickle=False) as raw:
                arrays = {name: raw[name].copy() for name in raw.files}
            arrays["schema_version"] = np.asarray("999")
            unsupported = Path(tmpdir) / "unsupported.npz"
            with unsupported.open("wb") as handle:
                np.savez(handle, **arrays)
            with self.assertRaisesRegex(ValueError, "unsupported z-support schema"):
                load_z_support(unsupported, self.labels)


if __name__ == "__main__":
    unittest.main()
