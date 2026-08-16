import unittest

import pandas as pd

from pipeline.replicating_complex_ids import (
    NON_REPLICATING_COMPLEX_ID,
    REPLICATING_COMPLEX_ID_FIELD,
    annotate_dataframe_complex_ids,
    annotate_record_complex_ids,
    build_replicating_complex_mapping,
    complex_id_field_order,
)


class ReplicatingComplexIDsTest(unittest.TestCase):
    def test_record_ids_are_one_based_sorted_and_repeated_per_walk(self):
        rows = [
            {'WalkID': 9, 'Replicating': 1, 'tp': 'TP000'},
            {'WalkID': 2, 'Replicating': 0, 'tp': 'TP000'},
            {'WalkID': 5, 'Replicating': 1, 'tp': 'TP000'},
            {'WalkID': 9, 'Replicating': 1, 'tp': 'TP001'},
            {'WalkID': 5, 'Replicating': 1, 'tp': 'TP001'},
        ]

        mapping = annotate_record_complex_ids(rows)

        self.assertEqual(mapping, {5: 1, 9: 2})
        self.assertEqual(
            [row[REPLICATING_COMPLEX_ID_FIELD] for row in rows],
            [2, NON_REPLICATING_COMPLEX_ID, 1, 2, 1],
        )
        self.assertNotIn(0, {
            row[REPLICATING_COMPLEX_ID_FIELD] for row in rows
        })

    def test_dataframe_recomputes_stale_local_ids_for_final_output(self):
        frame = pd.DataFrame([
            {'WalkID': 20, 'Replicating': 1,
             REPLICATING_COMPLEX_ID_FIELD: 99, 'tp': 'TP000'},
            {'WalkID': 10, 'Replicating': 0,
             REPLICATING_COMPLEX_ID_FIELD: 18, 'tp': 'TP000'},
            {'WalkID': 30, 'Replicating': 1,
             REPLICATING_COMPLEX_ID_FIELD: 7, 'tp': 'TP000'},
        ])

        result = annotate_dataframe_complex_ids(frame)

        self.assertEqual(
            result[REPLICATING_COMPLEX_ID_FIELD].tolist(), [1, -1, 2])
        self.assertEqual(
            result.columns.get_loc(REPLICATING_COMPLEX_ID_FIELD),
            result.columns.get_loc('Replicating') + 1)
        self.assertEqual(frame[REPLICATING_COMPLEX_ID_FIELD].tolist(),
                         [99, 18, 7], 'input frame must not be mutated')

    def test_conflicting_replicating_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent Replicating'):
            build_replicating_complex_mapping([
                (4, 0),
                (4, 1),
            ])

    def test_field_order_is_idempotent_and_places_id_after_replicating(self):
        fields = complex_id_field_order([
            'WalkID', REPLICATING_COMPLEX_ID_FIELD, 'Replicating', 'tp'])
        self.assertEqual(fields, [
            'WalkID', 'Replicating', REPLICATING_COMPLEX_ID_FIELD, 'tp'])


if __name__ == '__main__':
    unittest.main()
