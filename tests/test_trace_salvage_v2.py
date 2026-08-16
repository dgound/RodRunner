import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from pipeline.trace_salvage import (
    LocationInputs,
    TraceSalvageConfig,
    apply_policy,
    fit_policy,
    run_dataset,
)


ORDER = tuple(f'TP{index:03d}' for index in range(7))
EVIDENCE_FIELDS = [
    'tp', 'WalkID', 'status', 'replication_confirmed',
    'm2_c1_z', 'm2_c1_y', 'm2_c1_x',
    'm2_c2_z', 'm2_c2_y', 'm2_c2_x',
]
COORDINATE_FIELDS = [
    'tp', 'WalkID', 'z_nm', 'y_nm', 'x_nm',
    'drift_z', 'drift_y', 'drift_x', 'RawMarker',
]
SUMMARY_FIELDS = ['WalkID', 'replication_state', 'LegacyMarker']


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


class V2Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.evidence = {}
        self.coordinates = {}
        self.summaries = {}

    def _lists(self, location):
        self.evidence.setdefault(location, [])
        self.coordinates.setdefault(location, [])
        self.summaries.setdefault(location, [])
        return (self.evidence[location], self.coordinates[location],
                self.summaries[location])

    def add_clean_walk(self, location, walk_id, *, step_nm=100.0):
        evidence, coordinates, summaries = self._lists(location)
        for index, tp in enumerate(ORDER):
            evidence.append({
                'tp': tp, 'WalkID': walk_id,
                'status': 'tested_not_confirmed',
                'replication_confirmed': 0,
            })
            coordinates.append({
                'tp': tp, 'WalkID': walk_id,
                'z_nm': 50.0, 'y_nm': walk_id * 10.0,
                'x_nm': index * step_nm,
                'drift_z': 0.0, 'drift_y': 0.0, 'drift_x': 0.0,
                'RawMarker': f'raw-{location}-{walk_id}-{tp}',
            })
        summaries.append({
            'WalkID': walk_id, 'replication_state': 'non_replicating',
            'LegacyMarker': f'legacy-{location}-{walk_id}',
        })

    def add_walk(self, location, walk_id, *, calls=None,
                 coordinate_overrides=None, missing_evidence=(),
                 missing_coordinates=(), status_overrides=None):
        """Add a complete synthetic walk except for explicitly missing loci.

        ``calls`` maps TP to either two component XYZ-pixel tuples or ``None``
        (a confirmed call whose component coordinates are unavailable).
        """
        calls = calls or {}
        coordinate_overrides = coordinate_overrides or {}
        status_overrides = status_overrides or {}
        missing_evidence = set(missing_evidence)
        missing_coordinates = set(missing_coordinates)
        evidence, coordinates, summaries = self._lists(location)
        for index, tp in enumerate(ORDER):
            if tp not in missing_evidence:
                call = tp in calls
                row = {
                    'tp': tp, 'WalkID': walk_id,
                    'status': status_overrides.get(
                        tp, 'confirmed' if call else 'tested_not_confirmed'),
                    'replication_confirmed': int(call),
                }
                components = calls.get(tp)
                if components is not None:
                    for component_index, component in enumerate(components, 1):
                        for axis, value in zip(('z', 'y', 'x'), component):
                            row[f'm2_c{component_index}_{axis}'] = value
                evidence.append(row)
            if tp not in missing_coordinates:
                row = {
                    'tp': tp, 'WalkID': walk_id,
                    'z_nm': 0.0, 'y_nm': 400.0, 'x_nm': index * 100.0,
                    'drift_z': 0.0, 'drift_y': 0.0, 'drift_x': 0.0,
                    'RawMarker': f'raw-{location}-{walk_id}-{tp}',
                }
                row.update(coordinate_overrides.get(tp, {}))
                coordinates.append(row)
        summaries.append({
            'WalkID': walk_id, 'replication_state': 'non_replicating',
            'LegacyMarker': f'legacy-{location}-{walk_id}',
        })

    @staticmethod
    def _write(path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    def finish(self):
        inputs = []
        for location in sorted(self.evidence):
            directory = self.root / location
            evidence_path = directory / 'replication_qc.csv'
            coordinate_path = directory / 'drift_corrected.csv'
            summary_path = directory / 'classification.csv'
            self._write(evidence_path, EVIDENCE_FIELDS,
                        self.evidence[location])
            self._write(coordinate_path, COORDINATE_FIELDS,
                        self.coordinates[location])
            self._write(summary_path, SUMMARY_FIELDS,
                        self.summaries[location])
            inputs.append(LocationInputs(
                location=location,
                evidence_csv=str(evidence_path),
                coordinates_csv=str(coordinate_path),
                walk_summary_csv=str(summary_path),
            ))
        return tuple(inputs)


def _config(inputs, *, min_links=100):
    return TraceSalvageConfig(
        dataset='fixture', evidence_schema='v2', chromosome_order=ORDER,
        ori_tp='TP000', ter_tp='TP006',
        xy_pixel_nm=100.0, z_step_nm=100.0,
        locations=inputs, min_calibration_links=min_links,
        clean_link_quantile=0.95, winner_ratio=0.8,
    )


class TraceSalvageV2BehaviorTest(unittest.TestCase):
    def _calibrated_fixture(self, root):
        fixture = V2Fixture(root)
        for location in ('A', 'B', 'C'):
            for walk_id in range(1, 21):
                fixture.add_clean_walk(location, walk_id)
        return fixture

    def test_strict_acceptance_rejections_endpoints_and_complete_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._calibrated_fixture(Path(directory) / 'inputs')

            # Accepted: selection is XY-only. Component 1 is close in XY but
            # deliberately 400 nm away in z; component 2 has z=0 but bad XY.
            fixture.add_walk(
                'C', 100,
                calls={'TP002': (((5, 6, 4), (1, 10, 13)))},
                coordinate_overrides={
                    'TP001': {'x_nm': 0.0, 'y_nm': 400.0},
                    'TP002': {
                        'x_nm': 999.0, 'y_nm': 999.0, 'z_nm': 999.0,
                        'drift_z': 1.0, 'drift_y': 2.0, 'drift_x': 3.0,
                    },
                    'TP003': {'x_nm': 200.0, 'y_nm': 400.0},
                })
            # Winner path 200, runner-up 240: only a 16.7% advantage.
            fixture.add_walk(
                'C', 101,
                calls={'TP002': (((0, 4, 11), (0, 4, 9.8)))},
                coordinate_overrides={
                    'TP001': {'x_nm': 1000.0},
                    'TP003': {'x_nm': 1200.0},
                })
            # Clear winner, but one winning link is 150 > calibrated p95=100.
            fixture.add_walk(
                'C', 102,
                calls={'TP002': (((0, 4, 1.5), (0, 4, -2)))},
                coordinate_overrides={
                    'TP001': {'x_nm': 0.0},
                    'TP003': {'x_nm': 200.0},
                })
            # Exact tie must fail the 20% winner requirement.
            fixture.add_walk(
                'C', 103,
                calls={'TP002': (((0, 4, 1), (0, 4, 1)))},
                coordinate_overrides={
                    'TP001': {'x_nm': 0.0},
                    'TP003': {'x_nm': 200.0},
                })
            fixture.add_walk('C', 104, calls={'TP002': None})
            fixture.add_walk('C', 105, calls={
                'TP002': (((0, 4, 2), (0, 4, 8))),
                'TP003': (((0, 4, 3), (0, 4, 9))),
            })
            fixture.add_walk('C', 106, calls={
                'TP006': (((0, 4, 6), (0, 4, 10))),
            })
            fixture.add_walk('C', 107, calls={
                'TP000': (((0, 4, 0), (0, 4, 5))),
            })
            fixture.add_walk('C', 108, missing_evidence={'TP004'})
            fixture.add_walk('C', 109)
            fixture.add_walk(
                'C', 110,
                calls={'TP002': (((0, 4, 2), (0, 4, 8)))},
                missing_coordinates={'TP003'})
            # Two separated raw doublets can both be resolved; completion is
            # true only because every raw positive was accepted.
            fixture.add_walk('C', 111, calls={
                'TP002': (((0, 4, 2), (0, 4, 8))),
                'TP004': (((0, 4, 4), (0, 4, 10))),
            })

            inputs = fixture.finish()
            raw_hashes = {
                path: _sha256(path)
                for item in inputs
                for path in (item.evidence_csv, item.coordinates_csv,
                             item.walk_summary_csv)
            }
            outdir = Path(directory) / 'outputs'
            run_dataset(_config(inputs), outdir, 'v2', diagnostics=True)

            # The resolver is side-car only: all evidence, coordinates and
            # original classifications are byte-for-byte unchanged.
            self.assertEqual(
                raw_hashes,
                {path: _sha256(path) for path in raw_hashes})

            locus = {
                (row['Location'], int(row['OriginalWalkID']), row['tp']): row
                for row in _read_rows(
                    outdir / 'v2_trace_salvage_full_schema.csv')
            }
            audit = {
                (row['Location'], int(row['WalkID']), row['TP']): row
                for row in _read_rows(outdir / 'v2_trace_salvage_audit.csv')
            }
            walks = {
                (row['Location'], int(row['OriginalWalkID'])): row
                for row in _read_rows(
                    outdir / 'v2_trace_salvage_walk_summary.csv')
            }

            accepted = locus[('C', 100, 'TP002')]
            self.assertEqual(accepted['TraceSalvaged'], '1')
            self.assertEqual(accepted['TraceSelectedComponent'], '1')
            self.assertAlmostEqual(float(accepted['TraceXNm']), 100.0)
            self.assertAlmostEqual(float(accepted['TraceYNm']), 400.0)
            self.assertAlmostEqual(float(accepted['TraceZNm']), 400.0)
            self.assertEqual(accepted['OriginalXNm'], '999')
            self.assertEqual(
                accepted['TraceCoordinateSource'],
                'm2_component_1_drift_corrected')

            self.assertEqual(
                audit[('C', 101, 'TP002')]['TraceSalvageReason'],
                'winner_not_clear')
            self.assertEqual(
                audit[('C', 102, 'TP002')]['TraceSalvageReason'],
                'winning_link_exceeds_calibration')
            self.assertEqual(
                audit[('C', 103, 'TP002')]['TraceSalvageReason'],
                'winner_not_clear')
            self.assertEqual(
                audit[('C', 104, 'TP002')]['TraceSalvageReason'],
                'missing_component_xyz')
            self.assertEqual(
                audit[('C', 110, 'TP002')]['TraceSalvageReason'],
                'missing_required_coordinates')
            for tp in ('TP002', 'TP003'):
                self.assertEqual(
                    audit[('C', 105, tp)]['TraceSalvageReason'],
                    'not_isolated_between_confident_singles')
            self.assertEqual(
                audit[('C', 106, 'TP006')]['TraceSalvageReason'],
                'endpoint_never_salvaged')
            self.assertEqual(
                audit[('C', 107, 'TP000')]['TraceSalvageReason'],
                'endpoint_never_salvaged')

            accepted_walk = walks[('C', 100)]
            self.assertEqual(
                accepted_walk['TraceSalvageStatus'],
                'complete_after_salvage')
            self.assertEqual(accepted_walk['TraceSalvagedMultiSpot'], '0')
            self.assertEqual(
                accepted_walk['TraceSalvagedSingleSpotComplete'], '1')
            self.assertEqual(accepted_walk['TraceRawDoubletLociCount'], '1')
            self.assertEqual(accepted_walk['TraceSalvagedLociCount'], '1')
            self.assertEqual(
                accepted_walk['TraceUnresolvedDoubletLociCount'], '0')
            self.assertEqual(accepted_walk['LegacyMarker'], 'legacy-C-100')
            self.assertEqual(
                walks[('C', 108)]['TraceSalvageStatus'],
                'incomplete_or_unevaluable')
            self.assertEqual(
                walks[('C', 109)]['TraceSalvageStatus'], 'complete_clean')
            self.assertEqual(
                walks[('C', 107)]['TraceSalvageStatus'],
                'ori_not_confident_single')
            twice = walks[('C', 111)]
            self.assertEqual(twice['TraceSalvageStatus'],
                             'complete_after_salvage')
            self.assertEqual(twice['TraceSalvagedLociCount'], '2')
            self.assertEqual(twice['TraceUnresolvedDoubletLociCount'], '0')
            self.assertEqual(twice['TraceSalvagedTPs'], 'TP002;TP004')

    def test_leave_one_location_out_calibration_has_no_target_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = V2Fixture(Path(directory) / 'inputs')
            for location in ('A', 'B'):
                for walk_id in range(1, 11):
                    fixture.add_clean_walk(location, walk_id, step_nm=100.0)
            # Target-location clean traces are deliberately much looser. They
            # would raise p95 to 1000 nm if the target leaked into calibration.
            for walk_id in range(1, 21):
                fixture.add_clean_walk('C', walk_id, step_nm=1000.0)
            fixture.add_walk(
                'C', 100,
                calls={'TP002': (((0, 4, 1.5), (0, 4, -2)))},
                coordinate_overrides={
                    'TP001': {'x_nm': 0.0},
                    'TP003': {'x_nm': 200.0},
                })
            config = _config(fixture.finish())
            policy = fit_policy(config)
            calibration = policy['calibration']['locations']['C']
            self.assertEqual(calibration['source_locations'], ['A', 'B'])
            self.assertEqual(calibration['n_calibration_links'], 120)
            self.assertAlmostEqual(calibration['max_link_xy_nm'], 100.0)

            outdir = Path(directory) / 'outputs'
            apply_policy(config, policy, outdir, 'loo', diagnostics=True)
            audit = {
                (row['Location'], int(row['WalkID']), row['TP']): row
                for row in _read_rows(outdir / 'loo_trace_salvage_audit.csv')
            }
            candidate = audit[('C', 100, 'TP002')]
            self.assertEqual(candidate['TraceSalvaged'], '0')
            self.assertEqual(
                candidate['TraceSalvageReason'],
                'winning_link_exceeds_calibration')

    def test_fewer_than_100_training_links_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = V2Fixture(Path(directory) / 'inputs')
            for walk_id in range(1, 17):  # 16 walks * 6 links = 96
                fixture.add_clean_walk('B', walk_id)
            fixture.add_walk(
                'A', 100,
                calls={'TP002': (((0, 4, 1), (0, 4, 8)))},
                coordinate_overrides={
                    'TP001': {'x_nm': 0.0},
                    'TP003': {'x_nm': 200.0},
                })
            config = _config(fixture.finish())
            policy = fit_policy(config)
            calibration = policy['calibration']['locations']['A']
            self.assertEqual(
                calibration['status'], 'insufficient_calibration_links')
            self.assertEqual(calibration['n_calibration_links'], 96)
            self.assertIsNone(calibration['max_link_xy_nm'])

            outdir = Path(directory) / 'outputs'
            apply_policy(config, policy, outdir, 'few', diagnostics=True)
            audit = {
                (row['Location'], int(row['WalkID']), row['TP']): row
                for row in _read_rows(outdir / 'few_trace_salvage_audit.csv')
            }
            self.assertEqual(
                audit[('A', 100, 'TP002')]['TraceSalvageReason'],
                'calibration_unavailable')
            self.assertEqual(
                audit[('A', 100, 'TP002')]['TraceSalvaged'], '0')


if __name__ == '__main__':
    unittest.main()
