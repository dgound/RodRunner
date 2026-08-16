#!/usr/bin/env python3
"""Join walk-level replication classifications onto localization CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile

try:
    from .replicating_complex_ids import (
        NON_REPLICATING_COMPLEX_ID,
        REPLICATING_COMPLEX_ID_FIELD,
        build_replicating_complex_mapping,
        complex_id_field_order,
    )
    from .replication_detection import (
        CLASSIFICATION_REASON_FIELD,
        WALK_ANNOTATION_FIELDS,
    )
except ImportError:  # direct execution: python pipeline/annotate_replication.py
    from replicating_complex_ids import (
        NON_REPLICATING_COMPLEX_ID,
        REPLICATING_COMPLEX_ID_FIELD,
        build_replicating_complex_mapping,
        complex_id_field_order,
    )
    from replication_detection import (
        CLASSIFICATION_REASON_FIELD,
        WALK_ANNOTATION_FIELDS,
    )


#: Annotation columns a pre-existing summary CSV may legitimately lack.
_OPTIONAL_ANNOTATION_FIELDS = {CLASSIFICATION_REASON_FIELD}

_ZERO_DEFAULT_FIELDS = {
    'MultiSpot',
    'Replicating',
    'ActiveReplicating',
    'FullyReplicated',
    'FullReplicationCandidate',
    'OriEvaluated',
    'OriTwoSpot',
    'TerEvaluated',
    'TerTwoSpot',
    'replication_pattern_complete',
    'PinnedDoubleFitSuspect',
    'MergedMaskSuspect',
    'max_consecutive_confirmed_run',
    'max_ori_anchored_run',
}


def load_walk_annotations(summary_csv: str) -> dict[int, dict[str, str]]:
    annotations = {}
    with open(summary_csv, newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or 'WalkID' not in reader.fieldnames:
            raise ValueError(f'{summary_csv} has no WalkID column')
        # The reason column is newer than the rest, so an archived summary that
        # predates it stays annotatable; it simply comes through blank.
        missing = [
            field for field in WALK_ANNOTATION_FIELDS
            if field not in reader.fieldnames
            and field not in _OPTIONAL_ANNOTATION_FIELDS
        ]
        if missing:
            raise ValueError(
                f'{summary_csv} is missing replication annotation columns: '
                f'{missing}')
        for row in reader:
            try:
                walk_id = int(row['WalkID'])
            except (TypeError, ValueError):
                raise ValueError(
                    f'{summary_csv} has invalid WalkID={row.get("WalkID")!r}')
            if walk_id in annotations:
                raise ValueError(
                    f'{summary_csv} has duplicate WalkID={walk_id}')
            annotations[walk_id] = {
                field: row.get(field, '')
                for field in WALK_ANNOTATION_FIELDS
            }
    return annotations


def _missing_annotation() -> dict[str, str]:
    annotation = {
        field: ('0' if field in _ZERO_DEFAULT_FIELDS else '')
        for field in WALK_ANNOTATION_FIELDS
    }
    annotation['replication_state'] = 'missing_replication_summary'
    annotation[REPLICATING_COMPLEX_ID_FIELD] = str(
        NON_REPLICATING_COMPLEX_ID)
    return annotation


def annotate_csv(input_csv: str, summary_csv: str, output_csv: str) -> dict:
    annotations = load_walk_annotations(summary_csv)
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    temp_path = None
    n_rows = 0
    missing_walks = set()

    # IDs are local to this output CSV, not to the broader classification
    # summary. A filtered subset therefore still receives contiguous 1..N
    # positive IDs instead of inheriting gaps from excluded walks.
    input_walk_ids = set()
    with open(input_csv, newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or 'WalkID' not in reader.fieldnames:
            raise ValueError(f'{input_csv} has no WalkID column')
        for row in reader:
            try:
                input_walk_ids.add(int(row['WalkID']))
            except (TypeError, ValueError):
                raise ValueError(
                    f'{input_csv} has invalid '
                    f'WalkID={row.get("WalkID")!r}') from None
    complex_mapping = build_replicating_complex_mapping(
        (
            walk_id,
            annotations.get(walk_id, {'Replicating': '0'})['Replicating'],
        )
        for walk_id in input_walk_ids)

    try:
        with open(input_csv, newline='') as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or 'WalkID' not in reader.fieldnames:
                raise ValueError(f'{input_csv} has no WalkID column')
            annotation_fields = complex_id_field_order(
                WALK_ANNOTATION_FIELDS)
            fields = [
                field for field in reader.fieldnames
                if field not in annotation_fields
            ] + annotation_fields

            with tempfile.NamedTemporaryFile(
                    mode='w', newline='', dir=output_dir,
                    prefix='.replication_annotation_', suffix='.csv',
                    delete=False) as destination:
                temp_path = destination.name
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                for row in reader:
                    try:
                        walk_id = int(row['WalkID'])
                    except (TypeError, ValueError):
                        raise ValueError(
                            f'{input_csv} has invalid '
                            f'WalkID={row.get("WalkID")!r}')
                    annotation = annotations.get(walk_id)
                    if annotation is None:
                        annotation = _missing_annotation()
                        missing_walks.add(walk_id)
                    else:
                        annotation = dict(annotation)
                    annotation[REPLICATING_COMPLEX_ID_FIELD] = str(
                        complex_mapping.get(
                            walk_id, NON_REPLICATING_COMPLEX_ID))
                    row.update(annotation)
                    writer.writerow(row)
                    n_rows += 1

        os.replace(temp_path, output_csv)
        temp_path = None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)

    return {
        'input_csv': input_csv,
        'output_csv': output_csv,
        'rows': n_rows,
        'walks_without_summary': len(missing_walks),
    }


def _default_output(input_csv: str) -> str:
    stem, extension = os.path.splitext(input_csv)
    return f'{stem}_replication_annotated{extension or ".csv"}'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Join walk-level replication hierarchy annotations onto '
                    'one or more localization CSVs.')
    parser.add_argument('--summary_csv', required=True)
    parser.add_argument(
        '--input_csv', action='append', required=True,
        help='Localization CSV to annotate. Repeat for multiple files.')
    parser.add_argument(
        '--in_place', action='store_true',
        help='Atomically replace each input CSV. Without this flag, write '
             '<input>_replication_annotated.csv.')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for input_csv in args.input_csv:
        output_csv = input_csv if args.in_place else _default_output(input_csv)
        result = annotate_csv(input_csv, args.summary_csv, output_csv)
        print(
            f'Annotated {result["rows"]} rows -> {result["output_csv"]}; '
            f'walks_without_summary={result["walks_without_summary"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
