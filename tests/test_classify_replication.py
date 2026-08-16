import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.classify_replication import (
    _sets_for_walk,
    active_metric_filters,
    apply_coherence,
    main,
)
from pipeline.replication_detection import build_replichores

ARM_A, ARM_B = build_replichores(17, 7, 2, 21)


def _row(tp, confirmed=1, status='confirmed', **metrics):
    row = {
        'tp': f'TP{tp:03d}',
        'status': status,
        'replication_confirmed': str(int(confirmed)),
        'pinned_double_fit_suspect': '0',
        'merged_mask_suspect': '0',
    }
    row.update({k: str(v) for k, v in metrics.items()})
    return row


class PostHocFilterTest(unittest.TestCase):
    """The candidate confirmation policies must only ever demote."""

    def test_demoted_confirmation_stays_evaluated_not_failed(self):
        rows = [_row(17, delta_BIC=100)]
        confirmed, evaluated, failed, _, _ = _sets_for_walk(
            rows, [('delta_BIC', 2000.0)])
        self.assertEqual(confirmed, set())      # demoted
        self.assertEqual(evaluated, {17})       # still measured
        self.assertEqual(failed, set())         # never unanswerable

    def test_value_exactly_at_threshold_passes(self):
        rows = [_row(17, delta_BIC=2000)]
        confirmed, _, _, _, _ = _sets_for_walk(rows, [('delta_BIC', 2000.0)])
        self.assertEqual(confirmed, {17})

    def test_missing_metric_fails_only_when_filter_enabled(self):
        rows = [_row(17)]                       # no delta_BIC column at all
        confirmed, _, _, _, _ = _sets_for_walk(rows, [])
        self.assertEqual(confirmed, {17}, 'disabled filter must not demote')
        confirmed, evaluated, failed, _, _ = _sets_for_walk(
            rows, [('delta_BIC', 2000.0)])
        self.assertEqual(confirmed, set())
        self.assertEqual(evaluated, {17})
        self.assertEqual(failed, set())

    def test_non_finite_metric_fails_only_when_filter_enabled(self):
        rows = [_row(17, delta_BIC='nan')]
        self.assertEqual(_sets_for_walk(rows, [])[0], {17})
        self.assertEqual(_sets_for_walk(rows, [('delta_BIC', 1.0)])[0], set())

    def test_filters_can_never_promote_an_unconfirmed_locus(self):
        rows = [_row(9, confirmed=0, status='tested_not_confirmed',
                     delta_BIC=99999, spot_sep_xy_lcb_nm=9999, amp_ratio=1.0)]
        for active in ([], [('delta_BIC', 1.0)],
                       [('spot_sep_xy_lcb_nm', 1.0)], [('amp_ratio', 0.1)]):
            self.assertEqual(_sets_for_walk(rows, active)[0], set())

    def test_suspect_flags_survive_filtering(self):
        rows = [dict(_row(17, delta_BIC=10),
                     pinned_double_fit_suspect='1', merged_mask_suspect='1')]
        confirmed, _, _, pinned, merged = _sets_for_walk(
            rows, [('delta_BIC', 2000.0)])
        self.assertEqual(confirmed, set())
        self.assertEqual((pinned, merged), (1, 1))

    def test_active_metric_filters_only_reports_enabled(self):
        args = SimpleNamespace(min_spot_sep_xy_lcb_nm=500.0, min_delta_bic=0.0,
                               min_amp_ratio=0.0)
        self.assertEqual(active_metric_filters(args),
                         [('spot_sep_xy_lcb_nm', 500.0)])


class CoherenceFilterTest(unittest.TestCase):
    def test_isolated_call_removed_but_run_kept(self):
        confirmed = {17, 18, 19, 12}            # run of 3 on arm A, isolated 12
        kept = apply_coherence(confirmed, ARM_A, ARM_B, 3, 0)
        self.assertEqual(kept, {17, 18, 19})

    def test_gap_breaks_a_run(self):
        confirmed = {17, 18, 20, 21}            # 19 missing -> two runs of 2
        self.assertEqual(apply_coherence(confirmed, ARM_A, ARM_B, 3, 0), set())
        self.assertEqual(apply_coherence(confirmed, ARM_A, ARM_B, 2, 0),
                         {17, 18, 20, 21})

    def test_ori_anchored_requires_ori(self):
        off_ori = {18, 19, 20}                  # coherent but not at Ori
        self.assertEqual(apply_coherence(off_ori, ARM_A, ARM_B, 3, 0), off_ori)
        self.assertEqual(apply_coherence(off_ori, ARM_A, ARM_B, 0, 3), set())

    def test_shared_endpoints_qualify_on_either_arm(self):
        # Ori(17) is arm[0] of both arms; a run down arm B must be retained.
        confirmed = {17, 16, 15}
        self.assertEqual(apply_coherence(confirmed, ARM_A, ARM_B, 3, 0),
                         confirmed)
        self.assertEqual(apply_coherence(confirmed, ARM_A, ARM_B, 0, 3),
                         confirmed)
        # Ter(7) is the shared last element of both arms.
        self.assertIn(7, ARM_A)
        self.assertIn(7, ARM_B)

    def test_disabled_coherence_is_identity(self):
        confirmed = {17, 12}
        self.assertEqual(apply_coherence(confirmed, ARM_A, ARM_B, 0, 0),
                         confirmed)
        self.assertEqual(apply_coherence(confirmed, ARM_A, ARM_B, 1, 0),
                         confirmed)


class ClassifyReplicationTest(unittest.TestCase):
    def test_failed_two_spot_fit_is_unevaluable_not_negative(self):
        rows = [{
            'tp': 'TP007',
            'status': '2spot_fail:optimizer',
            'replication_confirmed': '0',
            'pinned_double_fit_suspect': '0',
            'merged_mask_suspect': '0',
        }]
        confirmed, evaluated, failed, pinned, merged = _sets_for_walk(rows)
        self.assertEqual(confirmed, set())
        self.assertEqual(evaluated, set())
        self.assertEqual(failed, {7})
        self.assertEqual((pinned, merged), (0, 0))

    def test_end_to_end_writes_and_annotates_hierarchy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / 'localizations_640_replication_qc.csv'
            drift = root / 'localizations_640_drift_corrected.csv'

            with sidecar.open('w', newline='') as handle:
                fields = [
                    'WalkID', 'tp', 'status', 'replication_confirmed',
                    'pinned_double_fit_suspect', 'merged_mask_suspect',
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for tp in range(2, 22):
                    writer.writerow({
                        'WalkID': 3,
                        'tp': f'TP{tp:03d}',
                        'status': (
                            'confirmed' if tp in {17, 18} else
                            'tested_not_confirmed'
                        ),
                        'replication_confirmed': int(tp in {17, 18}),
                        'pinned_double_fit_suspect': 0,
                        'merged_mask_suspect': 0,
                    })

            with drift.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=['WalkID', 'tp', 'x'])
                writer.writeheader()
                writer.writerow({'WalkID': 3, 'tp': 'TP017', 'x': '1.2300'})

            import sys
            from unittest.mock import patch

            argv = [
                'classify_replication.py',
                '--replication_qc_csv', str(sidecar),
                '--outdir', str(root),
            ]
            with patch.object(sys, 'argv', argv):
                main()

            summary = (
                root / 'localizations_640_replication_classification.csv')
            with summary.open(newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row['replication_state'], 'active_replicating')
            self.assertEqual(row['replication_front_arm_a_tp'], 'TP018')
            self.assertEqual(row['replication_front_arm_b_tp'], 'TP017')
            self.assertEqual(row['ReplicatingComplexID'], '1')

            with drift.open(newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row['x'], '1.2300')
            self.assertEqual(row['Replicating'], '1')
            self.assertEqual(row['ReplicatingComplexID'], '1')
            self.assertEqual(row['replication_state'], 'active_replicating')


class PolicyEndToEndTest(unittest.TestCase):
    """Default run unchanged; a policy run demotes and records what it did."""

    def _build(self, root, confirmed_tps, **metrics):
        sidecar = root / 'localizations_640_replication_qc.csv'
        fields = ['WalkID', 'tp', 'status', 'replication_confirmed',
                  'pinned_double_fit_suspect', 'merged_mask_suspect',
                  'delta_BIC', 'spot_sep_xy_lcb_nm', 'amp_ratio']
        with sidecar.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for tp in range(2, 22):
                conf = tp in confirmed_tps
                writer.writerow({
                    'WalkID': 3, 'tp': f'TP{tp:03d}',
                    'status': 'confirmed' if conf else 'tested_not_confirmed',
                    'replication_confirmed': int(conf),
                    'pinned_double_fit_suspect': 0, 'merged_mask_suspect': 0,
                    'delta_BIC': metrics.get('delta_BIC', 5000),
                    'spot_sep_xy_lcb_nm': metrics.get('spot_sep_xy_lcb_nm', 900),
                    'amp_ratio': metrics.get('amp_ratio', 0.9),
                })
        return sidecar

    def _run(self, sidecar, root, extra=()):
        argv = ['classify_replication.py', '--replication_qc_csv', str(sidecar),
                '--outdir', str(root), '--no_annotate', *extra]
        with patch.object(sys, 'argv', argv):
            main()
        with (root / 'localizations_640_replication_classification.csv').open(
                newline='') as handle:
            row = next(csv.DictReader(handle))
        policy = json.loads(
            (root / 'localizations_640_replication_policy.json').read_text())
        return row, policy

    def test_default_run_is_unchanged_and_records_default_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._build(root, {17, 18}, delta_BIC=100)
            row, policy = self._run(sidecar, root)
            # a weak dBIC of 100 survives because no filter is enabled
            self.assertEqual(row['replication_state'], 'active_replicating')
            self.assertEqual(row['n_confirmed_tps'], '2')
            self.assertTrue(policy['is_default_policy'])
            self.assertEqual(policy['metric_filters'], [])
            self.assertEqual(policy['counts']['confirmations_demoted'], 0)

    def test_policy_run_demotes_and_reclassifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._build(root, {17, 18}, delta_BIC=100)
            row, policy = self._run(
                sidecar, root, ['--min_delta_bic', '2000'])
            self.assertEqual(row['replication_state'], 'non_replicating')
            self.assertEqual(row['MultiSpot'], '0')
            self.assertFalse(policy['is_default_policy'])
            self.assertEqual(policy['counts']['confirmations_demoted'], 2)
            self.assertEqual(policy['metric_filters'],
                             [{'column': 'delta_BIC', 'min': 2000.0}])

    def test_policy_json_documents_the_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._build(root, {17, 18})
            _, policy = self._run(sidecar, root, [
                '--min_spot_sep_xy_lcb_nm', '500', '--min_delta_bic', '2000',
                '--min_consecutive_run', '3', '--min_ori_anchored_run', '2'])
            self.assertTrue(policy['requires_original_confirmation'])
            self.assertIn('>=', policy['comparison'])
            self.assertEqual(policy['hierarchy']['ori_tp'], 17)
            self.assertEqual(policy['hierarchy']['ter_tp'], 7)
            self.assertEqual(policy['hierarchy']['replication_tp_start'], 2)
            self.assertEqual(policy['hierarchy']['replication_tp_end'], 21)
            self.assertEqual(policy['coherence']['min_consecutive_run'], 3)
            self.assertEqual(policy['coherence']['min_ori_anchored_run'], 2)
            self.assertIn('EITHER arm', policy['coherence']['locus_retained_if'])
            self.assertIn('spot_sep_xy_lcb_nm',
                          [f['column'] for f in policy['metric_filters']])
            for untouched in ('PinnedDoubleFitSuspect', 'MergedMaskSuspect'):
                self.assertIn(untouched, policy['unchanged_by_this_step'])

    def test_coherence_demotes_isolated_call_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._build(root, {12})     # one isolated locus, no Ori
            row, _ = self._run(sidecar, root)
            self.assertEqual(row['replication_state'], 'multispot_only')
            row, policy = self._run(
                sidecar, root, ['--min_consecutive_run', '3'])
            self.assertEqual(row['replication_state'], 'non_replicating')
            self.assertEqual(row['MultiSpot'], '0')
            self.assertEqual(policy['counts']['confirmations_demoted'], 1)


LOCI = list(range(2, 22))
ORI, TER = 17, 7

SIDECAR_FIELDS = ['WalkID', 'tp', 'status', 'replication_confirmed',
                  'pinned_double_fit_suspect', 'merged_mask_suspect',
                  'ownership_unevaluable', 'ownership_failure_reason']
LEDGER_FIELDS = ['WalkID', 'tp', 'fit_ret_code', 'localization_written',
                 'CenterOwnershipValid', 'CenterOwnershipFailureReason',
                 'unevaluable_reason']


class AttemptedWalkUniverseTest(unittest.TestCase):
    """Every ATTEMPTED walk must stay in the classification denominator.

    A walk whose every localization was ownership-rejected or failed has no
    replication side-car rows at all. Classifying only side-car walks silently
    dropped it from the totals, which is a denominator bug: it neither appeared
    nor was counted, exactly as if it had never been imaged.
    """

    def _write(self, root, *, sidecar_rows, ledger_rows):
        sidecar = root / 'localizations_640_replication_qc.csv'
        with sidecar.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(sidecar_rows)
        ledger = root / 'localizations_640_z_gate_status.csv'
        with ledger.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ledger_rows)
        return sidecar

    @staticmethod
    def _sidecar(walk, tp, *, confirmed=0, status='tested_not_confirmed'):
        return {'WalkID': walk, 'tp': f'TP{tp:03d}', 'status': status,
                'replication_confirmed': int(confirmed),
                'pinned_double_fit_suspect': 0, 'merged_mask_suspect': 0,
                'ownership_unevaluable': 0, 'ownership_failure_reason': ''}

    @staticmethod
    def _attempt(walk, tp, outcome):
        """One ledger row. outcome in {accepted, rejected, failed}."""
        return {
            'WalkID': walk, 'tp': f'TP{tp:03d}',
            'fit_ret_code': -1 if outcome == 'failed' else 0,
            'localization_written': int(outcome == 'accepted'),
            'CenterOwnershipValid': (
                '' if outcome == 'failed' else int(outcome == 'accepted')),
            'CenterOwnershipFailureReason': (
                'center_background' if outcome == 'rejected' else ''),
            'unevaluable_reason': {
                'accepted': '', 'rejected': 'center_background',
                'failed': 'fit_failed:-1'}[outcome],
        }

    def _run(self, root, sidecar, extra=()):
        argv = ['classify_replication.py', '--replication_qc_csv', str(sidecar),
                '--outdir', str(root), '--no_annotate',
                '--ori_tp', str(ORI), '--ter_tp', str(TER),
                '--replication_tp_start', str(LOCI[0]),
                '--replication_tp_end', str(LOCI[-1]), *extra]
        with patch.object(sys, 'argv', argv):
            main()
        with (root / 'localizations_640_replication_classification.csv').open(
                newline='') as handle:
            rows = list(csv.DictReader(handle))
        policy = json.loads(
            (root / 'localizations_640_replication_policy.json').read_text())
        return {int(r['WalkID']): r for r in rows}, rows, policy

    ZERO_FLAGS = ('MultiSpot', 'Replicating', 'ActiveReplicating',
                  'FullyReplicated', 'FullReplicationCandidate',
                  'replication_pattern_complete', 'OriEvaluated', 'OriTwoSpot')

    def _assert_wholly_indeterminate(self, row, reason):
        self.assertEqual(row['replication_state'], 'indeterminate_ori')
        self.assertNotEqual(row['replication_state'], 'non_replicating')
        for field in self.ZERO_FLAGS:
            self.assertEqual(row[field], '0', field)
        self.assertEqual(row['n_confirmed_tps'], '0')
        self.assertEqual(row['n_evaluated_tps'], '0')
        self.assertEqual(row['ReplicationClassificationReason'], reason)

    # -- 1. entirely ownership-rejected -----------------------------------
    def test_an_all_rejected_walk_is_classified_indeterminate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar_rows = [self._sidecar(1, tp) for tp in LOCI]
            ledger = [self._attempt(1, tp, 'accepted') for tp in LOCI]
            ledger += [self._attempt(2, tp, 'rejected') for tp in LOCI]
            path = self._write(root, sidecar_rows=sidecar_rows,
                               ledger_rows=ledger)
            by_walk, rows, policy = self._run(root, path)
            self.assertEqual(len(rows), 2, 'one row per attempted walk')
            self.assertEqual(sorted(by_walk), [1, 2])
            self.assertEqual(
                sum(1 for r in rows if int(r['WalkID']) == 2), 1)
            self._assert_wholly_indeterminate(
                by_walk[2], 'all_localizations_ownership_unevaluable')
            self.assertEqual(by_walk[2]['n_failed_tps'], str(len(LOCI)),
                             'the loci WERE attempted, so they are not missing')
            self.assertEqual(by_walk[2]['n_missing_tps'], '0')
            # The walk that could be classified is untouched.
            self.assertEqual(by_walk[1]['replication_state'], 'non_replicating')
            self.assertEqual(by_walk[1]['replication_pattern_complete'], '1')
            self.assertEqual(policy['walk_universe']['attempted_walks'], 2)
            self.assertEqual(policy['walk_universe']['classified_walks'], 2)
            self.assertEqual(
                policy['walk_universe'][
                    'walks_without_replication_measurement'], 1)

    # -- 2. entirely fit-failed -------------------------------------------
    def test_an_all_failed_walk_is_indeterminate_and_distinguishable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = [self._attempt(1, tp, 'accepted') for tp in LOCI]
            ledger += [self._attempt(3, tp, 'failed') for tp in LOCI]
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=ledger)
            by_walk, rows, _ = self._run(root, path)
            self.assertEqual(len(rows), 2)
            self._assert_wholly_indeterminate(
                by_walk[3], 'all_localization_fits_failed')
            self.assertNotEqual(
                by_walk[3]['ReplicationClassificationReason'],
                'all_localizations_ownership_unevaluable',
                'a fitter failure is not an ownership rejection')

    def test_a_mixed_failure_walk_keeps_the_generic_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = [self._attempt(1, tp, 'accepted') for tp in LOCI]
            ledger += [self._attempt(4, tp, 'rejected' if tp % 2 else 'failed')
                       for tp in LOCI]
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=ledger)
            by_walk, _, _ = self._run(root, path)
            self._assert_wholly_indeterminate(
                by_walk[4], 'no_evaluable_localization')

    # -- 3. mixed valid and invalid ---------------------------------------
    def test_the_full_mixture_keeps_every_walk_and_every_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar_rows, ledger = [], []
            # 1: ordinary non-replicating
            sidecar_rows += [self._sidecar(1, tp) for tp in LOCI]
            ledger += [self._attempt(1, tp, 'accepted') for tp in LOCI]
            # 2: active replicating (Ori confirmed, Ter not)
            for tp in LOCI:
                conf = tp in (ORI, ORI + 1)
                sidecar_rows.append(self._sidecar(
                    2, tp, confirmed=conf,
                    status='confirmed' if conf else 'tested_not_confirmed'))
            ledger += [self._attempt(2, tp, 'accepted') for tp in LOCI]
            # 3: Ori rejected, everything else fine
            sidecar_rows += [self._sidecar(3, tp) for tp in LOCI if tp != ORI]
            ledger += [self._attempt(3, tp,
                                     'rejected' if tp == ORI else 'accepted')
                       for tp in LOCI]
            # 4: every locus rejected      5: every fit failed
            ledger += [self._attempt(4, tp, 'rejected') for tp in LOCI]
            ledger += [self._attempt(5, tp, 'failed') for tp in LOCI]
            path = self._write(root, sidecar_rows=sidecar_rows,
                               ledger_rows=ledger)
            by_walk, rows, policy = self._run(root, path)

            self.assertEqual(len(rows), 5)
            self.assertEqual(sorted(by_walk), [1, 2, 3, 4, 5])
            self.assertEqual(by_walk[1]['replication_state'],
                             'non_replicating')
            self.assertEqual(by_walk[2]['replication_state'],
                             'active_replicating')
            self.assertEqual(by_walk[2]['Replicating'], '1')
            self.assertEqual(by_walk[3]['replication_state'],
                             'indeterminate_ori')
            self.assertEqual(by_walk[3]['OriEvaluated'], '0')
            self._assert_wholly_indeterminate(
                by_walk[4], 'all_localizations_ownership_unevaluable')
            self._assert_wholly_indeterminate(
                by_walk[5], 'all_localization_fits_failed')
            # Category counts sum to the attempted-walk universe.
            states = {}
            for row in rows:
                states[row['replication_state']] = (
                    states.get(row['replication_state'], 0) + 1)
            self.assertEqual(sum(states.values()), 5)
            self.assertEqual(states.get('non_replicating'), 1)
            self.assertEqual(states.get('indeterminate_ori'), 3)
            self.assertEqual(policy['walk_universe']['attempted_walks'], 5)

    # -- 4. no usable replication side-car at all -------------------------
    def test_a_location_with_no_evaluable_walk_still_reports_every_walk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = [self._attempt(walk, tp, 'rejected')
                      for walk in (7, 8, 9) for tp in LOCI]
            path = self._write(root, sidecar_rows=[], ledger_rows=ledger)
            by_walk, rows, policy = self._run(root, path)
            self.assertEqual(len(rows), 3, 'an empty side-car is not an empty '
                                           'walk universe')
            for walk in (7, 8, 9):
                self._assert_wholly_indeterminate(
                    by_walk[walk], 'all_localizations_ownership_unevaluable')
            self.assertEqual(policy['walk_universe']['sidecar_walks'], 0)
            self.assertEqual(policy['walk_universe']['classified_walks'], 3)

    # -- integrity and opt-out --------------------------------------------
    def test_a_duplicated_attempt_key_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = [self._attempt(1, 2, 'rejected'),
                      self._attempt(1, 2, 'rejected')]
            path = self._write(root, sidecar_rows=[], ledger_rows=ledger)
            with self.assertRaises(SystemExit):
                self._run(root, path)

    def test_a_sidecar_walk_absent_from_the_ledger_is_still_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=[self._attempt(2, tp, 'rejected') for tp in LOCI])
            by_walk, rows, _ = self._run(root, path)
            self.assertEqual(sorted(by_walk), [1, 2],
                             'the universe is the UNION, not the ledger alone')
            self.assertEqual(by_walk[1]['replication_state'],
                             'non_replicating')

    def test_the_historical_sidecar_only_universe_stays_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=[self._attempt(1, tp, 'accepted') for tp in LOCI]
                + [self._attempt(2, tp, 'rejected') for tp in LOCI])
            by_walk, rows, policy = self._run(
                root, path, extra=('--no_attempt_ledger',))
            self.assertEqual(sorted(by_walk), [1])
            self.assertEqual(policy['walk_universe']['source'],
                             'replication_sidecar_only')

    def _provenance(self, root, policy, name='localizations_640'):
        """Write ownership provenance exactly where the stages write it."""
        path = root / f'{name}_chromatic_registration.json'
        path.write_text(json.dumps({
            'schema_version': 'chromatic_provenance_v1',
            'stage': 'localize_probe0', 'mode': 'metadata',
            'ownership_policy': policy,
        }))
        return path

    # -- 1. strict + required ledger succeeds ------------------------------
    def test_strict_with_an_explicit_required_ledger_classifies_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._provenance(root, 'strict')
            ledger = [self._attempt(1, tp, 'accepted') for tp in LOCI]
            ledger += [self._attempt(375, tp, 'rejected') for tp in LOCI]
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=ledger)
            by_walk, rows, policy = self._run(root, path, extra=(
                '--require_attempt_ledger', '--attempt_ledger_csv',
                str(root / 'localizations_640_z_gate_status.csv')))
            self.assertEqual(sorted(by_walk), [1, 375])
            self.assertEqual(len(rows), 2)
            self._assert_wholly_indeterminate(
                by_walk[375], 'all_localizations_ownership_unevaluable')
            self.assertEqual(policy['walk_universe']['ownership_policy'],
                             'strict')
            self.assertTrue(policy['walk_universe']['ledger_required'])
            self.assertFalse(policy['walk_universe']['legacy_opt_out'])

    # -- 2./3./4. every way a required ledger can be absent ----------------
    def _assert_fails_without_output(self, root, sidecar, extra=(),
                                     expect_in_message=()):
        summary = root / 'localizations_640_replication_classification.csv'
        sentinel = 'SENTINEL — must not be overwritten\n'
        summary.write_text(sentinel)
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as caught, \
                contextlib.redirect_stderr(stderr):
            self._run(root, sidecar, extra=extra)
        self.assertNotEqual(caught.exception.code, 0)
        message = stderr.getvalue()
        for fragment in expect_in_message:
            self.assertIn(fragment, message)
        self.assertEqual(summary.read_text(), sentinel,
                         'a failed ledger validation must not touch the output')
        self.assertFalse(
            (root / 'localizations_640_replication_policy.json').exists(),
            'no policy record may be written on a ledger failure')
        return message

    def test_required_ledger_missing_fails_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / 'localizations_640_replication_qc.csv'
            with sidecar.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
                writer.writeheader()
                writer.writerows([self._sidecar(1, tp) for tp in LOCI])
            message = self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('attempt ledger unusable',
                                   'auto-discovery',
                                   'localizations_640_z_gate_status.csv',
                                   '--no_attempt_ledger'))
            self.assertIn('required            : yes', message)

    def test_an_explicit_missing_path_fails_even_without_require(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A perfectly good ledger exists under the discoverable name; the
            # explicit path must still not fall back to it.
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=[self._attempt(1, tp, 'accepted') for tp in LOCI])
            bogus = root / 'nowhere' / 'ledger.csv'
            message = self._assert_fails_without_output(
                root, path, extra=('--attempt_ledger_csv', str(bogus)),
                expect_in_message=(str(bogus),
                                   'explicitly supplied path does not exist'))
            self.assertNotIn('localizations_640_z_gate_status.csv',
                             message.split('deliberate legacy')[0])

    def test_any_enforcing_policy_makes_the_ledger_mandatory(self):
        """constrained_experimental also drops loci, so it must require it too."""
        for policy in ('strict', 'constrained_experimental'):
            with self.subTest(policy):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._provenance(root, policy)
                    sidecar = root / 'localizations_640_replication_qc.csv'
                    with sidecar.open('w', newline='') as handle:
                        writer = csv.DictWriter(handle,
                                                fieldnames=SIDECAR_FIELDS)
                        writer.writeheader()
                        writer.writerows(
                            [self._sidecar(1, tp) for tp in LOCI])
                    self._assert_fails_without_output(
                        root, sidecar,
                        expect_in_message=(f"ownership policy    : {policy!r}",))

    def test_an_allow_policy_keeps_the_historical_behaviour(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._provenance(root, 'allow')
            sidecar = root / 'localizations_640_replication_qc.csv'
            with sidecar.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
                writer.writeheader()
                writer.writerows([self._sidecar(1, tp) for tp in LOCI])
            by_walk, rows, policy = self._run(root, sidecar)
            self.assertEqual(sorted(by_walk), [1])
            self.assertFalse(policy['walk_universe']['ledger_required'])

    def test_strict_provenance_with_no_ledger_argument_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._provenance(root, 'strict')
            sidecar = root / 'localizations_640_replication_qc.csv'
            with sidecar.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
                writer.writeheader()
                writer.writerows([self._sidecar(1, tp) for tp in LOCI])
            message = self._assert_fails_without_output(
                root, sidecar,
                expect_in_message=("ownership policy    : 'strict'",
                                   'strict ownership makes it mandatory'))
            self.assertIn('localizations_640_chromatic_registration.json',
                          message)

    # -- 5. invalid ledgers, each failing distinctly -----------------------
    def _invalid_case(self, root, ledger_rows, fields=None):
        sidecar = root / 'localizations_640_replication_qc.csv'
        with sidecar.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
            writer.writeheader()
            writer.writerows([self._sidecar(1, tp) for tp in LOCI])
        ledger = root / 'localizations_640_z_gate_status.csv'
        with ledger.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or LEDGER_FIELDS,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ledger_rows)
        return sidecar

    def test_an_empty_ledger_is_not_an_empty_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._invalid_case(root, [])
            self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('contains no attempt rows',
                                   'a missing denominator is not an empty one'))

    def test_missing_required_columns_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._invalid_case(
                root, [{'WalkID': 1, 'tp': 'TP002'}],
                fields=['WalkID', 'tp'])
            self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('missing required column',
                                   'fit_ret_code'))

    def test_a_duplicate_attempt_key_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._invalid_case(
                root, [self._attempt(1, 2, 'rejected'),
                       self._attempt(1, 2, 'rejected')])
            self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('duplicates attempt',))

    def test_a_malformed_walkid_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._attempt(1, 2, 'accepted')
            row['WalkID'] = 'not-a-number'
            sidecar = self._invalid_case(root, [row])
            self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('unusable WalkID',))

    def test_a_blank_tp_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._attempt(1, 2, 'accepted')
            row['tp'] = ''
            sidecar = self._invalid_case(root, [row])
            self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('blank tp',))

    def test_ambiguous_auto_discovery_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = self._invalid_case(
                root, [self._attempt(1, tp, 'accepted') for tp in LOCI])
            # A second candidate, and no exactly-prefixed name to prefer.
            (root / 'localizations_640_z_gate_status.csv').rename(
                root / 'probe0_z_gate_status.csv')
            (root / 'other_z_gate_status.csv').write_text(
                (root / 'probe0_z_gate_status.csv').read_text())
            self._assert_fails_without_output(
                root, sidecar, extra=('--require_attempt_ledger',),
                expect_in_message=('auto-discovery is ambiguous',
                                   'probe0_z_gate_status.csv',
                                   'other_z_gate_status.csv'))

    def test_disagreeing_ownership_provenance_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._provenance(root, 'strict', name='localizations_640')
            self._provenance(root, 'allow', name='replication_qc')
            sidecar = self._invalid_case(
                root, [self._attempt(1, tp, 'accepted') for tp in LOCI])
            self._assert_fails_without_output(
                root, sidecar,
                expect_in_message=('ownership-policy provenance disagrees',
                                   "localizations_640_chromatic_registration"
                                   ".json='strict'",
                                   "replication_qc_chromatic_registration"
                                   ".json='allow'"))

    # -- 6. conflicting options -------------------------------------------
    def test_conflicting_ledger_flags_are_rejected(self):
        for extra in (('--require_attempt_ledger', '--no_attempt_ledger'),
                      ('--no_attempt_ledger', '--attempt_ledger_csv', 'x.csv')):
            with self.subTest(extra):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = self._write(
                        root,
                        sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                        ledger_rows=[self._attempt(1, tp, 'accepted')
                                     for tp in LOCI])
                    stderr = io.StringIO()
                    with self.assertRaises(SystemExit) as caught, \
                            contextlib.redirect_stderr(stderr):
                        self._run(root, path, extra=extra)
                    self.assertNotEqual(caught.exception.code, 0)
                    self.assertIn('--no_attempt_ledger',
                                  stderr.getvalue())

    # -- 7. deliberate legacy opt-out --------------------------------------
    def test_the_legacy_opt_out_warns_and_loses_ledger_only_walks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._provenance(root, 'strict')
            path = self._write(
                root, sidecar_rows=[self._sidecar(1, tp) for tp in LOCI],
                ledger_rows=[self._attempt(1, tp, 'accepted') for tp in LOCI]
                + [self._attempt(375, tp, 'rejected') for tp in LOCI])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                by_walk, rows, policy = self._run(
                    root, path, extra=('--no_attempt_ledger',))
            printed = stdout.getvalue()
            self.assertEqual(sorted(by_walk), [1],
                             'the ledger-only walk is NOT represented')
            self.assertNotIn(375, by_walk)
            self.assertIn('WARNING', printed)
            self.assertIn('denominator is ABANDONED', printed)
            self.assertIn('STRICT centre ownership', printed)
            self.assertEqual(policy['walk_universe']['source'],
                             'replication_sidecar_only')
            self.assertTrue(policy['walk_universe']['legacy_opt_out'])

    # -- 8. non-strict legacy compatibility --------------------------------
    def test_no_ledger_present_behaves_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / 'localizations_640_replication_qc.csv'
            with sidecar.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDS)
                writer.writeheader()
                writer.writerows([self._sidecar(1, tp) for tp in LOCI])
            by_walk, rows, policy = self._run(root, sidecar)
            self.assertEqual(sorted(by_walk), [1])
            self.assertEqual(policy['walk_universe']['source'],
                             'replication_sidecar_only')
            self.assertEqual(
                by_walk[1]['ReplicationClassificationReason'], '')


if __name__ == '__main__':
    unittest.main()
