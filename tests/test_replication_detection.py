import unittest

from pipeline.replication_detection import (
    build_replichores,
    classify_replication,
)


class CoherenceMetricTest(unittest.TestCase):
    """max_consecutive_confirmed_run vs max_ori_anchored_run."""

    def setUp(self):
        self.arm_a, self.arm_b = build_replichores(17, 7, 2, 21)

    def _ann(self, confirmed):
        return classify_replication(
            confirmed, set(range(2, 22)), set(),
            self.arm_a, self.arm_b, 17, 7)

    def test_isolated_call_has_run_of_one_and_no_ori_anchor(self):
        ann = self._ann({12})
        self.assertEqual(ann['max_consecutive_confirmed_run'], 1)
        self.assertEqual(ann['max_ori_anchored_run'], 0)

    def test_off_ori_run_counts_consecutive_but_not_ori_anchored(self):
        ann = self._ann({18, 19, 20})
        self.assertEqual(ann['max_consecutive_confirmed_run'], 3)
        self.assertEqual(ann['max_ori_anchored_run'], 0)

    def test_ori_anchored_run_counts_ori_itself(self):
        ann = self._ann({17, 18, 19})
        self.assertEqual(ann['max_consecutive_confirmed_run'], 3)
        self.assertEqual(ann['max_ori_anchored_run'], 3)

    def test_gap_breaks_both_metrics(self):
        ann = self._ann({17, 18, 20})
        self.assertEqual(ann['max_consecutive_confirmed_run'], 2)
        self.assertEqual(ann['max_ori_anchored_run'], 2)

    def test_arms_are_maxed_not_summed(self):
        # symmetric bubble: two of Ori's neighbours, one on each arm
        ann = self._ann({17, 18, 16})
        self.assertEqual(ann['max_ori_anchored_run'], 2)
        self.assertEqual(ann['max_consecutive_confirmed_run'], 2)

    def test_unevaluable_locus_breaks_the_run(self):
        ann = classify_replication(
            {17, 18, 20, 21}, set(range(2, 22)) - {19}, {19},
            self.arm_a, self.arm_b, 17, 7)
        self.assertEqual(ann['max_ori_anchored_run'], 2)
        self.assertEqual(ann['max_consecutive_confirmed_run'], 2)
        self.assertIn('TP019', ann['unevaluable_tps'])


class ReplicationDetectionTest(unittest.TestCase):
    def setUp(self):
        self.arm_a, self.arm_b = build_replichores(17, 7, 2, 21)
        self.expected = set(self.arm_a) | set(self.arm_b)

    def classify(self, confirmed=(), evaluated=None, failed=()):
        if evaluated is None:
            evaluated = self.expected
        return classify_replication(
            confirmed, evaluated, failed,
            self.arm_a, self.arm_b, 17, 7)

    def test_default_biological_order(self):
        self.assertEqual(
            self.arm_a, [17, 18, 19, 20, 21, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            self.arm_b, [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7])

    def test_confocal_range_wraps_from_tp019_to_tp000(self):
        arm_a, arm_b = build_replichores(17, 7, 0, 19)
        self.assertEqual(
            arm_a, [17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            arm_b, [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7])

    def test_no_two_spots_is_non_replicating(self):
        result = self.classify()
        self.assertEqual(result['replication_state'], 'non_replicating')
        self.assertEqual(result['MultiSpot'], 0)
        self.assertEqual(result['Replicating'], 0)

    def test_multispot_without_ori_is_not_replicating(self):
        result = self.classify({10})
        self.assertEqual(result['replication_state'], 'multispot_only')
        self.assertEqual(result['MultiSpot'], 1)
        self.assertEqual(result['Replicating'], 0)

    def test_ori_two_and_ter_one_is_active_from_initiation(self):
        result = self.classify({17})
        self.assertEqual(result['replication_state'], 'active_replicating')
        self.assertEqual(result['Replicating'], 1)
        self.assertEqual(result['replication_progress'], 1 / 20)
        self.assertEqual(result['replication_front_arm_a_tp'], 'TP017')
        self.assertEqual(result['replication_front_arm_b_tp'], 'TP017')

    def test_active_fronts_are_independent_and_gap_limited(self):
        # Arm A reaches TP020; TP021 is a gap. TP002 beyond the gap must not
        # extend the front. Arm B reaches only TP016.
        result = self.classify({17, 18, 19, 20, 2, 16})
        self.assertEqual(result['replication_front_arm_a_tp'], 'TP020')
        self.assertEqual(result['replication_front_arm_b_tp'], 'TP016')
        self.assertAlmostEqual(result['replication_progress_arm_a'], 0.3)
        self.assertAlmostEqual(result['replication_progress_arm_b'], 0.1)
        self.assertAlmostEqual(result['replication_asymmetry'], 0.2)
        self.assertAlmostEqual(result['replication_progress'], 5 / 20)

    def test_all_loci_two_is_fully_replicated(self):
        result = self.classify(self.expected)
        self.assertEqual(result['replication_state'], 'fully_replicated')
        self.assertEqual(result['FullyReplicated'], 1)
        self.assertEqual(result['FullReplicationCandidate'], 1)
        self.assertEqual(result['replication_progress'], 1.0)

    def test_ori_and_ter_with_internal_negative_is_incomplete_candidate(self):
        result = self.classify({17, 7})
        self.assertEqual(
            result['replication_state'], 'full_candidate_incomplete')
        self.assertEqual(result['Replicating'], 1)
        self.assertEqual(result['FullyReplicated'], 0)
        self.assertEqual(result['replication_progress'], '')

    def test_missing_ori_is_not_treated_as_one_spot(self):
        evaluated = self.expected - {17}
        result = self.classify({10}, evaluated=evaluated)
        self.assertEqual(result['replication_state'], 'indeterminate_ori')
        self.assertEqual(result['OriEvaluated'], 0)

    def test_failed_ter_makes_completion_indeterminate(self):
        evaluated = self.expected - {7}
        result = self.classify(
            {17, 18}, evaluated=evaluated, failed={7})
        self.assertEqual(
            result['replication_state'], 'indeterminate_completion')
        self.assertEqual(result['TerEvaluated'], 0)
        self.assertEqual(result['failed_tps'], 'TP007')

    def test_front_reports_censoring_at_unevaluable_gap(self):
        evaluated = self.expected - {19}
        result = self.classify(
            {17, 18}, evaluated=evaluated, failed={19})
        self.assertEqual(result['replication_front_arm_a_tp'], 'TP018')
        self.assertEqual(result['replication_front_arm_a_censored'], 1)


if __name__ == '__main__':
    unittest.main()
