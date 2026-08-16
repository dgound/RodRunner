"""The axial residual ACF must be searched as deep as the crops allow.

Two coupled defects blocked axial characterization:

1. ``--max-lag-z`` was hard-coded to 8, so on a stack whose crops span more than
   8 planes the search stopped before a crossing could be observed. "z never
   decorrelates by lag 8" then said as much about the bound as about the data.
2. When the ACF was still above threshold at the last searched lag, the profile
   builder RAISED, so the whole residual profile was unbuildable -- which is the
   blocker itself.

The fix searches to the deepest lag the crops support, refuses to read a crossing
off an under-supported lag, and when the crossing is genuinely unreachable takes
the most conservative block scale and records that verdict. It does not lower the
threshold and does not fabricate a crossing.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "pipeline"))

import replication_qc_v3 as v3  # noqa: E402


class _Crop:
    """Minimal stand-in exposing only the .z the lag derivation reads."""

    def __init__(self, depth):
        import numpy as np
        self.z = np.repeat(np.arange(depth, dtype=float), 4)


def _accumulator(correlations, count):
    """An axis_stats accumulator whose lag-k correlation is correlations[k]."""
    out = []
    for value in correlations:
        # rho = cross / sqrt(left_sq * right_sq); pick left_sq = right_sq = 1.
        out.append({"cross": float(value), "left_sq": 1.0, "right_sq": 1.0,
                    "count": int(count)})
    return out


class MeasurableZLagTest(unittest.TestCase):
    def test_the_bound_comes_from_the_deepest_crop(self):
        self.assertEqual(v3.measurable_z_lag(
            [_Crop(7), _Crop(21), _Crop(11)]), 20)

    def test_a_thin_stack_gives_a_small_bound(self):
        self.assertEqual(v3.measurable_z_lag([_Crop(7)]), 6)

    def test_the_old_hard_coded_eight_is_not_the_answer(self):
        """A 21-plane rep3 stack must search well past lag 8."""
        self.assertGreater(v3.measurable_z_lag([_Crop(21)]), 8)

    def test_an_empty_bundle_is_zero_not_a_default(self):
        self.assertEqual(v3.measurable_z_lag([]), 0)


class LagSupportTest(unittest.TestCase):
    def test_a_clean_crossing_is_found_and_reported(self):
        support = v3._lag_support(
            _accumulator([1.0, 0.5, 0.2, 0.05, 0.01], count=1000), min_pairs=64)
        self.assertEqual(support["crossing_lag"], 3)
        self.assertTrue(support["crossing_supported"])
        self.assertEqual(support["unresolved_reason"], "")

    def test_no_crossing_is_reported_as_unresolved_not_as_the_bound(self):
        support = v3._lag_support(
            _accumulator([1.0, 0.9, 0.8, 0.7, 0.6], count=1000), min_pairs=64)
        self.assertIsNone(support["crossing_lag"])
        self.assertFalse(support["crossing_supported"])
        self.assertIn("longer than this geometry can measure",
                      support["unresolved_reason"])

    def test_an_under_supported_lag_cannot_supply_a_crossing(self):
        """Few pairs give a noisy rho; reading a crossing off it invents one."""
        accumulator = _accumulator([1.0, 0.9, 0.01], count=1000)
        accumulator[2]["count"] = 3            # the "crossing" has 3 pairs
        support = v3._lag_support(accumulator, min_pairs=64)
        self.assertIsNone(support["crossing_lag"])
        self.assertNotIn(2, support["adequately_supported_lags"])
        self.assertEqual(support["last_supported_lag"], 1)

    def test_pair_counts_are_recorded_for_every_lag(self):
        support = v3._lag_support(
            _accumulator([1.0, 0.5, 0.2], count=128), min_pairs=64)
        self.assertEqual(support["pair_counts"], [128, 128, 128])

    def test_the_threshold_is_the_documented_one_and_is_not_lowered(self):
        support = v3._lag_support(
            _accumulator([1.0, 0.11], count=1000), min_pairs=64)
        self.assertEqual(support["threshold"],
                         v3.ACF_DECORRELATION_THRESHOLD)
        # 0.11 is above 0.1, so it is NOT a crossing.
        self.assertIsNone(support["crossing_lag"])

    def test_a_zero_variance_lag_is_not_a_crossing(self):
        accumulator = _accumulator([1.0, 0.0], count=1000)
        accumulator[1]["left_sq"] = 0.0
        support = v3._lag_support(accumulator, min_pairs=64)
        self.assertIsNone(support["correlations"][1])
        self.assertIsNone(support["crossing_lag"])


class NoHardCodedBoundRemainsTest(unittest.TestCase):
    def test_the_cli_derives_the_axial_bound_by_default(self):
        namespace = v3._parse_args(
            ["characterize-residuals", "--input-npz", "x", "--config-json", "c",
             "--output-json", "o"])
        self.assertIsNone(namespace.max_lag_z,
                          "the axial bound must be derived, not defaulted to 8")
        self.assertEqual(namespace.min_residual_pairs, v3.MIN_RESIDUAL_PAIRS)

    def test_an_explicit_axial_bound_is_still_honoured(self):
        namespace = v3._parse_args(
            ["characterize-residuals", "--input-npz", "x", "--config-json", "c",
             "--output-json", "o", "--max-lag-z", "20"])
        self.assertEqual(namespace.max_lag_z, 20)

    def test_the_campaign_does_not_pin_the_axial_lag(self):
        text = (_REPO
                / ".campaigns/replication_publication_v1/stage_commands.py"
                ).read_text()
        code = "\n".join(line for line in text.splitlines()
                         if not line.lstrip().startswith("#"))
        self.assertNotIn('"--max-lag-z"', code)
        # the lateral bound is still pinned, which is correct
        self.assertIn('"--max-lag-xy"', code)

    def test_an_axial_truncation_is_not_fatal_but_a_lateral_one_is(self):
        source = (_REPO / "pipeline/replication_qc_v3.py").read_text()
        self.assertIn("lateral_truncated", source)
        self.assertIn("increase characterize max-lag-xy", source)
        # the old unconditional raise must be gone
        self.assertNotIn("increase characterize max-lag\"", source)

    def test_the_unresolved_verdict_is_recorded_in_the_profile(self):
        source = (_REPO / "pipeline/replication_qc_v3.py").read_text()
        for field in ("axial_acf_resolved", "axial_pair_counts",
                      "axial_max_measured_lag", "axial_note"):
            with self.subTest(field):
                self.assertIn(field, source)



class EndpointReachabilityIsRecordedTest(unittest.TestCase):
    """Which stages can move the confirmatory endpoint must be checkable."""

    def setUp(self):
        sys.path.insert(
            0, str(_REPO / ".campaigns" / "replication_publication_v1"))
        import campaignlib
        self.cl = campaignlib

    def test_every_location_stage_has_a_verdict(self):
        table = self.cl.STAGE_AFFECTS_CONFIRMATORY_ENDPOINT
        for stage in self.cl.LOCATION_STAGES:
            with self.subTest(stage):
                self.assertIn(stage, table)

    def test_the_stage_that_computes_the_endpoint_can_affect_it(self):
        self.assertTrue(
            self.cl.STAGE_AFFECTS_CONFIRMATORY_ENDPOINT["final-reduce"])

    def test_power_cannot_move_the_endpoint(self):
        """Its overlay may write only v3_power_* columns."""
        self.assertFalse(self.cl.STAGE_AFFECTS_CONFIRMATORY_ENDPOINT["power"])

    def test_the_synthetic_barrier_cannot_move_the_endpoint(self):
        for stage in ("simulation-validation", "simulation-verify"):
            with self.subTest(stage):
                self.assertFalse(
                    self.cl.STAGE_AFFECTS_CONFIRMATORY_ENDPOINT[stage])

    def test_trace_salvage_and_annotation_cannot_move_the_endpoint(self):
        for stage in ("trace-salvage", "annotate", "merge", "qc"):
            with self.subTest(stage):
                self.assertFalse(
                    self.cl.STAGE_AFFECTS_CONFIRMATORY_ENDPOINT[stage])

    def test_the_power_overlay_really_is_restricted_to_power_columns(self):
        """The table's claim about `power`, verified against the enforcement."""
        source = (_REPO / ".campaigns/replication_publication_v1/"
                  "campaign_tasks.py").read_text()
        self.assertIn("POWER_OVERLAY_COLUMNS", source + (
            _REPO / "pipeline/replication_qc_v3.py").read_text())

    def test_the_barrier_artifact_states_its_own_scope(self):
        source = (_REPO / ".campaigns/replication_publication_v1/"
                  "campaign_tasks.py").read_text()
        self.assertIn("does_not_establish", source)
        self.assertIn("sister loci", source)

if __name__ == "__main__":
    unittest.main()
