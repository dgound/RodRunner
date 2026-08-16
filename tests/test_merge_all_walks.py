import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from drivers.merge_all_walks import _tp_order, main
from pipeline.trace_salvage import load_config


class MergeAllWalksComplexIDTest(unittest.TestCase):
    def test_trace_chromosome_order_parser_is_explicit_and_unique(self):
        self.assertEqual(_tp_order('TP002, 3, TP004'), [2, 3, 4])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, 'duplicate'):
            _tp_order('2,3,2')

    def test_trace_salvage_requires_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'base').mkdir()
            argv = [
                'merge_all_walks.py', '--base', str(root / 'base'),
                '--outdir', str(root / 'merged'), '--trace_salvage',
            ]
            with patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(
                        ValueError, 'trace_salvage_dataset is required'):
                    main()

    def test_canonical_fit_detects_an_entirely_absent_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'base' / 'Location_1').mkdir(parents=True)
            argv = [
                'merge_all_walks.py', '--base', str(root / 'base'),
                '--outdir', str(root / 'merged'), '--trace_salvage',
                '--trace_salvage_dataset', 'confocal',
            ]
            with patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(
                        FileNotFoundError, 'Location_2'):
                    main()

    def test_full_and_filtered_outputs_get_independent_contiguous_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            outdir = root / 'merged'

            fixtures = {
                'Location-01': [
                    # Replicating and retained in the filtered subset.
                    (2, 1, 0, 0, True),
                    # Non-replicating sentinel.
                    (5, 0, 0, 0, True),
                ],
                'Location-02': [
                    # Replicating but excluded from the filtered subset.
                    (1, 1, 1, 0, True),
                    # Replicating and retained; must become ID 2, not 3, in
                    # the independently renumbered filtered CSV.
                    (9, 1, 0, 0, True),
                ],
            }
            for location, walks in fixtures.items():
                locdir = base / location / 'probe_0' / 'localizations'
                locdir.mkdir(parents=True)
                rows = []
                movement_rows = []
                for walk_id, replicating, pinned, merged, movement in walks:
                    for tp in ('TP000', 'TP001'):
                        rows.append({
                            'tp': tp,
                            'WalkID': walk_id,
                            'Replicating': replicating,
                            'ReplicatingComplexID': 99,
                            'PinnedDoubleFitSuspect': pinned,
                            'MergedMaskSuspect': merged,
                        })
                    if movement:
                        movement_rows.append({'WalkID': walk_id})
                pd.DataFrame(rows).to_csv(
                    locdir / 'localizations_640_drift_corrected_full_coverage.csv',
                    index=False)
                pd.DataFrame(movement_rows).to_csv(
                    locdir / 'localizations_640_drift_corrected_filtered.csv',
                    index=False)

            argv = [
                'merge_all_walks.py',
                '--base', str(base),
                '--outdir', str(outdir),
            ]
            with patch.object(sys, 'argv', argv):
                main()

            full = pd.read_csv(outdir / 'all_locations_640_valid_walks.csv')
            filtered = pd.read_csv(
                outdir / 'all_locations_640_valid_walks_nosuspect_movement.csv')

            self.assertEqual(
                sorted(full.loc[full.Replicating == 1,
                                'ReplicatingComplexID'].unique()),
                [1, 2, 3])
            self.assertTrue(
                (full.loc[full.Replicating == 0,
                          'ReplicatingComplexID'] == -1).all())
            self.assertEqual(
                sorted(filtered.loc[filtered.Replicating == 1,
                                    'ReplicatingComplexID'].unique()),
                [1, 2])
            self.assertFalse((full.ReplicatingComplexID == 0).any())
            self.assertFalse((filtered.ReplicatingComplexID == 0).any())
            self.assertFalse(any(outdir.glob('*trace_salvage*')),
                             'legacy/default finalization must not run salvage')

    def test_explicit_trace_salvage_builds_auditable_v2_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            outdir = root / 'merged'
            for location in ('Location_1', 'Location_2'):
                locdir = base / location / 'probe_0' / 'localizations'
                locdir.mkdir(parents=True)
                pd.DataFrame([{
                    'tp': 'TP000', 'WalkID': 1,
                    'Replicating': 0,
                    'PinnedDoubleFitSuspect': 0,
                    'MergedMaskSuspect': 0,
                }]).to_csv(
                    locdir / 'localizations_640_drift_corrected_full_coverage.csv',
                    index=False)
                pd.DataFrame([{'WalkID': 1}]).to_csv(
                    locdir / 'localizations_640_drift_corrected_filtered.csv',
                    index=False)
                pd.DataFrame([{
                    'tp': 'TP000', 'WalkID': 1, 'status': 'not_suspect',
                    'replication_confirmed': 0,
                }]).to_csv(
                    locdir / 'localizations_640_replication_qc.csv', index=False)
                pd.DataFrame([{
                    'WalkID': 1, 'replication_state': 'non_replicating',
                }]).to_csv(
                    locdir / 'localizations_640_replication_classification.csv',
                    index=False)

            fake_run = Mock(return_value={
                'trace_csv': outdir / 'all_locations_640_trace_salvaged.csv',
            })
            expected_path = root / 'expected_locations.json'
            expected_path.write_text(json.dumps(['Location_1', 'Location_2']))
            argv = [
                'merge_all_walks.py', '--base', str(base),
                '--outdir', str(outdir),
                '--trace_salvage', '--trace_salvage_dataset', 'confocal',
                '--trace_salvage_expected_locations', str(expected_path),
            ]
            with patch.object(sys, 'argv', argv), patch(
                    'pipeline.trace_salvage.run_dataset', fake_run):
                main()

            config_path = (
                outdir / 'all_locations_640_trace_salvage_config.json')
            cfg_payload = json.loads(config_path.read_text())
            cfg = load_config(config_path)
            self.assertEqual(cfg.dataset, 'confocal')
            self.assertEqual(cfg.evidence_schema, 'v2')
            self.assertEqual(cfg.chromosome_order, tuple(
                f'TP{index:03d}' for index in range(0, 20)))
            self.assertEqual((cfg.ori_tp, cfg.ter_tp), ('TP017', 'TP007'))
            self.assertEqual(cfg.xy_pixel_nm, 81.25)
            self.assertEqual(cfg.z_step_nm, 180.0)
            self.assertEqual(cfg.min_calibration_links, 100)
            self.assertEqual(cfg.clean_link_quantile, 0.95)
            self.assertEqual(cfg.winner_ratio, 0.8)
            self.assertEqual(
                [item['location'] for item in cfg_payload['locations']],
                ['Location_1', 'Location_2'])
            for item in cfg_payload['locations']:
                self.assertTrue(Path(item['evidence_csv']).is_absolute())
                self.assertTrue(Path(item['coordinates_csv']).is_absolute())
                self.assertTrue(Path(item['walk_summary_csv']).is_absolute())
            fake_run.assert_called_once()
            _, kwargs = fake_run.call_args
            self.assertEqual(Path(kwargs['outdir']), outdir.resolve())
            self.assertEqual(kwargs['output_prefix'], 'all_locations_640')

            # Existing, authoritative products are still present and retain
            # their schema; salvage uses distinct filenames only.
            merged = pd.read_csv(outdir / 'all_locations_640_valid_walks.csv')
            self.assertNotIn('TraceSalvageStatus', merged.columns)
            self.assertTrue(config_path.is_file())

    def test_requested_salvage_fails_closed_if_classification_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locdir = (root / 'base' / 'Location-01' / 'probe_0' /
                      'localizations')
            locdir.mkdir(parents=True)
            pd.DataFrame([{
                'tp': 'TP002', 'WalkID': 1,
                'Replicating': 0,
                'PinnedDoubleFitSuspect': 0, 'MergedMaskSuspect': 0,
            }]).to_csv(
                locdir / 'localizations_640_drift_corrected_full_coverage.csv',
                index=False)
            pd.DataFrame([{
                'tp': 'TP002', 'WalkID': 1, 'status': 'tested_not_confirmed',
                'replication_confirmed': 0,
            }]).to_csv(
                locdir / 'localizations_640_replication_qc.csv', index=False)
            outdir = root / 'merged'
            argv = ['merge_all_walks.py', '--base', str(root / 'base'),
                    '--outdir', str(outdir), '--trace_salvage',
                    '--trace_salvage_dataset', 'rep3',
                    '--trace_salvage_policy', str(root / 'unused_policy.json')]
            with patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(
                        FileNotFoundError, 'replication_classification'):
                    main()

    def test_single_location_requires_and_can_apply_precomputed_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locdir = (root / 'base' / 'Location-01' / 'probe_0' /
                      'localizations')
            locdir.mkdir(parents=True)
            pd.DataFrame([{
                'tp': 'TP002', 'WalkID': 1, 'Replicating': 0,
                'PinnedDoubleFitSuspect': 0, 'MergedMaskSuspect': 0,
            }]).to_csv(
                locdir / 'localizations_640_drift_corrected_full_coverage.csv',
                index=False)
            pd.DataFrame([{'WalkID': 1}]).to_csv(
                locdir / 'localizations_640_drift_corrected_filtered.csv',
                index=False)
            pd.DataFrame([{
                'tp': 'TP002', 'WalkID': 1,
                'status': 'tested_not_confirmed',
                'replication_confirmed': 0,
            }]).to_csv(
                locdir / 'localizations_640_replication_qc.csv', index=False)
            pd.DataFrame([{
                'WalkID': 1, 'replication_state': 'non_replicating',
            }]).to_csv(
                locdir / 'localizations_640_replication_classification.csv',
                index=False)
            policy_path = root / 'held_policy.json'
            policy_path.write_text('{"schema": "fixture-policy"}\n')
            expected_path = root / 'one_location.json'
            expected_path.write_text('["Location-01"]\n')
            outdir = root / 'merged'

            fake_module = types.ModuleType('pipeline.trace_salvage')
            fake_module.load_config = Mock(return_value='CONFIG')
            fake_module.run_dataset = Mock()
            fake_module.apply_policy = Mock(return_value={})
            no_policy_argv = [
                'merge_all_walks.py', '--base', str(root / 'base'),
                '--outdir', str(outdir), '--trace_salvage',
                '--trace_salvage_dataset', 'rep3',
                '--trace_salvage_expected_locations', str(expected_path),
            ]
            with patch.object(sys, 'argv', no_policy_argv), patch.dict(
                    sys.modules,
                    {'pipeline.trace_salvage': fake_module}):
                with self.assertRaisesRegex(
                        ValueError, 'leave-one-location-out'):
                    main()
            fake_module.run_dataset.assert_not_called()
            fake_module.apply_policy.assert_not_called()

            argv = [
                'merge_all_walks.py', '--base', str(root / 'base'),
                '--outdir', str(outdir), '--trace_salvage',
                '--trace_salvage_dataset', 'rep3',
                '--trace_salvage_policy', str(policy_path),
            ]
            with patch.object(sys, 'argv', argv), patch.dict(
                    sys.modules,
                    {'pipeline.trace_salvage': fake_module}):
                main()

            fake_module.run_dataset.assert_not_called()
            fake_module.apply_policy.assert_called_once()
            call_args, call_kwargs = fake_module.apply_policy.call_args
            self.assertEqual(call_args[0], 'CONFIG')
            self.assertEqual(call_args[1], {'schema': 'fixture-policy'})
            self.assertEqual(Path(call_kwargs['outdir']), outdir.resolve())


if __name__ == '__main__':
    unittest.main()
