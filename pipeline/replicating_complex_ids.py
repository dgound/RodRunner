#!/usr/bin/env python3
"""Deterministic integer identifiers for biologically replicating walks.

``ReplicatingComplexID`` is an output identifier, not a classifier input:

* walks with ``Replicating == 1`` receive IDs ``1..N`` in ascending WalkID
  order;
* every row belonging to a walk receives the same ID; and
* every other walk receives the explicit sentinel ``-1``.  Zero is never
  emitted, so the sentinel cannot be confused with a zero-based identifier.

Merged deliverables must call the assignment *after* WalkID has been made
globally unique.  This keeps the positive IDs unique and contiguous within
each final CSV even when per-location inputs already carry local IDs.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

import numpy as np


REPLICATING_COMPLEX_ID_FIELD = 'ReplicatingComplexID'
NON_REPLICATING_COMPLEX_ID = -1


def _walk_id(value: Any) -> int:
    """Parse one integer WalkID without silently truncating a float."""
    if isinstance(value, bool):
        raise ValueError(f'invalid WalkID={value!r}')
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if value != value or value != int(value):
            raise ValueError(f'invalid WalkID={value!r}')
        return int(value)
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        pass
    else:
        return parsed
    try:
        f = float(text)
    except (TypeError, ValueError):
        raise ValueError(f'invalid WalkID={value!r}') from None
    if f != f or f != int(f):
        raise ValueError(f'invalid WalkID={value!r}')
    return int(f)


def _replicating_flag(value: Any) -> int:
    """Parse the serialized binary Replicating field."""
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise ValueError(f'invalid Replicating={value!r}; expected 0 or 1') from None
    if number not in (0.0, 1.0):
        raise ValueError(f'invalid Replicating={value!r}; expected 0 or 1')
    return int(number)


def build_replicating_complex_mapping(
        walk_replication: Iterable[tuple[Any, Any]],
) -> dict[int, int]:
    """Return ``WalkID -> 1-based complex ID`` for replicating walks.

    Repeated rows are expected in localization CSVs.  A walk whose
    ``Replicating`` value changes between rows is rejected rather than being
    assigned an ambiguous identifier.
    """
    states: dict[int, int] = {}
    for raw_walk_id, raw_replicating in walk_replication:
        walk_id = _walk_id(raw_walk_id)
        replicating = _replicating_flag(raw_replicating)
        previous = states.setdefault(walk_id, replicating)
        if previous != replicating:
            raise ValueError(
                f'WalkID={walk_id} has inconsistent Replicating values: '
                f'{previous} and {replicating}')

    replicating_walks = sorted(
        walk_id for walk_id, replicating in states.items() if replicating == 1)
    return {
        walk_id: complex_id
        for complex_id, walk_id in enumerate(replicating_walks, start=1)
    }


def annotate_record_complex_ids(
        records: Iterable[MutableMapping[str, Any]],
        *,
        walk_field: str = 'WalkID',
        replicating_field: str = 'Replicating',
) -> dict[int, int]:
    """Add ``ReplicatingComplexID`` to mutable row dictionaries in place."""
    rows = list(records)
    for field in (walk_field, replicating_field):
        if any(field not in row for row in rows):
            raise ValueError(f'cannot assign complex IDs: missing {field} column')

    mapping = build_replicating_complex_mapping(
        (row[walk_field], row[replicating_field]) for row in rows)
    for row in rows:
        walk_id = _walk_id(row[walk_field])
        row[REPLICATING_COMPLEX_ID_FIELD] = mapping.get(
            walk_id, NON_REPLICATING_COMPLEX_ID)
    return mapping


def annotate_dataframe_complex_ids(
        dataframe,
        *,
        walk_field: str = 'WalkID',
        replicating_field: str = 'Replicating',
):
    """Return a copy of a pandas DataFrame with deterministic complex IDs.

    Pandas is intentionally not imported here; callers already use it and the
    duck-typed implementation keeps this small helper usable by the standard
    library CSV writers too.
    """
    missing = [
        field for field in (walk_field, replicating_field)
        if field not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            'cannot assign complex IDs: missing column(s): '
            + ', '.join(missing))

    result = dataframe.copy()
    mapping = build_replicating_complex_mapping(zip(
        result[walk_field].tolist(), result[replicating_field].tolist()))
    values = [
        mapping.get(_walk_id(walk_id), NON_REPLICATING_COMPLEX_ID)
        for walk_id in result[walk_field].tolist()
    ]

    if REPLICATING_COMPLEX_ID_FIELD in result.columns:
        result = result.drop(columns=[REPLICATING_COMPLEX_ID_FIELD])
    insert_at = result.columns.get_loc(replicating_field) + 1
    result.insert(insert_at, REPLICATING_COMPLEX_ID_FIELD, values)
    return result


def complex_id_field_order(fields: Iterable[str]) -> list[str]:
    """Place the ID directly after ``Replicating`` without duplicating it."""
    ordered = [field for field in fields
               if field != REPLICATING_COMPLEX_ID_FIELD]
    try:
        insert_at = ordered.index('Replicating') + 1
    except ValueError:
        insert_at = len(ordered)
    ordered.insert(insert_at, REPLICATING_COMPLEX_ID_FIELD)
    return ordered
