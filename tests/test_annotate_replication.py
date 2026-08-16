import csv
import tempfile
import unittest
from pathlib import Path

from pipeline.annotate_replication import annotate_csv
from pipeline.replication_detection import WALK_ANNOTATION_FIELDS


class AnnotateReplicationTest(unittest.TestCase):
    def test_annotation_preserves_coordinate_text_and_replaces_old_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / 'summary.csv'
            localizations = root / 'localizations.csv'

            annotation = {field: '' for field in WALK_ANNOTATION_FIELDS}
            annotation.update({
                'MultiSpot': '1',
                'Replicating': '1',
                'ActiveReplicating': '1',
                'FullyReplicated': '0',
                'FullReplicationCandidate': '0',
                'replication_state': 'active_replicating',
                'replication_progress': '0.25',
                'PinnedDoubleFitSuspect': '0',
                'MergedMaskSuspect': '0',
            })
            with summary.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=['WalkID', *WALK_ANNOTATION_FIELDS])
                writer.writeheader()
                writer.writerow({'WalkID': 4, **annotation})

            with localizations.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        'tp', 'WalkID', 'x_nm', 'y_nm', 'z_nm',
                        'Replicating',
                    ])
                writer.writeheader()
                writer.writerow({
                    'tp': 'TP017', 'WalkID': 4,
                    'x_nm': '123.4000', 'y_nm': '55.000',
                    'z_nm': '-7.50', 'Replicating': 0,
                })

            result = annotate_csv(
                str(localizations), str(summary), str(localizations))

            self.assertEqual(result['rows'], 1)
            with localizations.open(newline='') as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames.count('Replicating'), 1)
            self.assertEqual(rows[0]['x_nm'], '123.4000')
            self.assertEqual(rows[0]['y_nm'], '55.000')
            self.assertEqual(rows[0]['z_nm'], '-7.50')
            self.assertEqual(rows[0]['Replicating'], '1')
            self.assertEqual(rows[0]['ReplicatingComplexID'], '1')
            self.assertEqual(
                rows[0]['replication_state'], 'active_replicating')

    def test_missing_walk_gets_explicit_missing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / 'summary.csv'
            localizations = root / 'localizations.csv'
            with summary.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=['WalkID', *WALK_ANNOTATION_FIELDS])
                writer.writeheader()
            with localizations.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=['tp', 'WalkID', 'x'])
                writer.writeheader()
                writer.writerow({'tp': 'TP002', 'WalkID': 9, 'x': '1.0'})

            result = annotate_csv(
                str(localizations), str(summary), str(localizations))

            self.assertEqual(result['walks_without_summary'], 1)
            with localizations.open(newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row['replication_state'], 'missing_replication_summary')
            self.assertEqual(row['Replicating'], '0')
            self.assertEqual(row['ReplicatingComplexID'], '-1')

    def test_filtered_output_restarts_positive_ids_at_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / 'summary.csv'
            localizations = root / 'filtered.csv'

            annotation = {field: '' for field in WALK_ANNOTATION_FIELDS}
            annotation.update({
                'MultiSpot': '1',
                'Replicating': '1',
                'ActiveReplicating': '1',
                'FullyReplicated': '0',
                'FullReplicationCandidate': '0',
                'replication_state': 'active_replicating',
            })
            with summary.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=['WalkID', *WALK_ANNOTATION_FIELDS])
                writer.writeheader()
                writer.writerow({'WalkID': 1, **annotation})
                writer.writerow({'WalkID': 8, **annotation})

            with localizations.open('w', newline='') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        'tp', 'WalkID', 'ReplicatingComplexID',
                    ])
                writer.writeheader()
                writer.writerow({
                    'tp': 'TP000', 'WalkID': 8,
                    'ReplicatingComplexID': 99,
                })
                writer.writerow({
                    'tp': 'TP001', 'WalkID': 8,
                    'ReplicatingComplexID': 99,
                })

            annotate_csv(str(localizations), str(summary), str(localizations))

            with localizations.open(newline='') as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    reader.fieldnames.count('ReplicatingComplexID'), 1)
            self.assertEqual(
                [row['ReplicatingComplexID'] for row in rows], ['1', '1'])


if __name__ == '__main__':
    unittest.main()
