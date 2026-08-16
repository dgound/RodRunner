#!/usr/bin/env python3
"""Biological-order replication classification from per-locus two-spot calls.

The image-level detector reports whether sister copies of one genomic locus are
spatially resolved as two spots. This module aggregates those observations into
an operational cell state. A two-spot call is evidence of sister-locus
separation; it is not a direct measurement of DNA synthesis.
"""

from __future__ import annotations

from typing import Iterable


REPLICATION_TP_START_DEFAULT = 2
REPLICATION_TP_END_DEFAULT = 21
ORI_TP_DEFAULT = 17
TER_TP_DEFAULT = 7


REPLICATION_CLASSIFICATION_FIELDS = [
    'MultiSpot',
    'Replicating',
    'ActiveReplicating',
    'FullyReplicated',
    'FullReplicationCandidate',
    'replication_state',
    'replication_progress',
    'replication_progress_arm_a',
    'replication_progress_arm_b',
    'replication_asymmetry',
    'replication_front_arm_a_tp',
    'replication_front_arm_b_tp',
    'replication_front_arm_a_censored',
    'replication_front_arm_b_censored',
    'OriEvaluated',
    'OriTwoSpot',
    'TerEvaluated',
    'TerTwoSpot',
    'replication_pattern_complete',
    'n_expected_tps',
    'n_evaluated_tps',
    'n_confirmed_tps',
    'n_failed_tps',
    'n_missing_tps',
    'unconfirmed_tps',
    'failed_tps',
    'missing_tps',
    'unevaluable_tps',
    'max_consecutive_confirmed_run',
    'max_ori_anchored_run',
]

#: Why a walk carries the classification it does, when that needs saying. It is
#: NOT a status column and never encodes a biological call: `replication_state`
#: remains the only classification. It exists because zero positive flags is
#: ambiguous on its own -- a walk with `MultiSpot=0` may have been measured and
#: found single, or may have had nothing measurable at all -- and a reader must be
#: able to tell those apart from the walk row alone.
CLASSIFICATION_REASON_FIELD = 'ReplicationClassificationReason'

#: No locus in the configured universe could be evaluated in either direction.
#: The classifier can see this much on its own; the CLI refines it using the
#: localization-attempt ledger (see the REASON_* constants in
#: classify_replication.py), which is the only place ownership rejections and
#: fitter failures are distinguishable.
REASON_NO_EVALUABLE_LOCALIZATION = 'no_evaluable_localization'

WALK_ANNOTATION_FIELDS = (
    REPLICATION_CLASSIFICATION_FIELDS
    + ['PinnedDoubleFitSuspect', 'MergedMaskSuspect',
       CLASSIFICATION_REASON_FIELD]
)


def tp_label(tp_index: int) -> str:
    return f'TP{int(tp_index):03d}'


def build_replichores(
        ori_tp: int,
        ter_tp: int,
        tp_start: int = REPLICATION_TP_START_DEFAULT,
        tp_end: int = REPLICATION_TP_END_DEFAULT,
) -> tuple[list[int], list[int]]:
    """Return the two circular genomic paths from Ori to Ter.

    Bounds are inclusive 0-based TP filename indices. Numeric TP order within
    the selected range must match genomic order.
    """
    if tp_start < 0:
        raise ValueError(f'tp_start={tp_start} must be >= 0')
    if tp_end < tp_start:
        raise ValueError(
            f'tp_end={tp_end} must be >= tp_start={tp_start}')
    order = list(range(tp_start, tp_end + 1))
    for name, value in (('ori_tp', ori_tp), ('ter_tp', ter_tp)):
        if value not in order:
            raise ValueError(
                f'{name}={value} is outside the replication TP range '
                f'[{tp_start}, {tp_end}]')
    if ori_tp == ter_tp:
        raise ValueError('ori_tp and ter_tp must be different')

    n_tps = len(order)
    ori_index = order.index(ori_tp)
    ter_index = order.index(ter_tp)

    arm_a = []
    index = ori_index
    while True:
        arm_a.append(order[index])
        if index == ter_index:
            break
        index = (index + 1) % n_tps

    arm_b = []
    index = ori_index
    while True:
        arm_b.append(order[index])
        if index == ter_index:
            break
        index = (index - 1) % n_tps

    return arm_a, arm_b


def _format_tps(values: Iterable[int]) -> str:
    return ';'.join(tp_label(tp) for tp in sorted(set(values)))


def _front(arm: list[int], confirmed: set[int],
           unevaluable: set[int]) -> tuple[int, int, int]:
    """Return (steps beyond Ori, front TP, censored-at-first-gap)."""
    steps = 0
    front_tp = arm[0]
    censored = 0
    for tp in arm[1:]:
        if tp not in confirmed:
            censored = int(tp in unevaluable)
            break
        steps += 1
        front_tp = tp
    return steps, front_tp, censored


def longest_consecutive_run(arm: list[int], confirmed: set[int]) -> int:
    """Longest run of consecutive confirmed loci anywhere in the arm's order.

    Ori and Ter are the shared endpoints of both arms, so each is counted once
    within a given arm's sequence. Any locus that is not confirmed terminates a
    run, whatever the reason; whether the break was censoring rather than a
    measured single spot stays visible through the front_*_censored fields.
    """
    best = current = 0
    for tp in arm:
        current = current + 1 if tp in confirmed else 0
        best = max(best, current)
    return best


def ori_anchored_run(arm: list[int], confirmed: set[int]) -> int:
    """Run length starting AT Ori (arm[0]) and following replication order.

    Zero when Ori itself is not confirmed. This is the biologically stronger
    coherence measure: replication initiates at Ori, so a run that does not
    include it can still be structured false-positive signal. The two-sided
    extent of a replication bubble is `replication_progress`, not this field.
    """
    if not arm or arm[0] not in confirmed:
        return 0
    length = 0
    for tp in arm:
        if tp not in confirmed:
            break
        length += 1
    return length


def classify_replication(
        confirmed_tps: Iterable[int],
        evaluated_tps: Iterable[int],
        failed_tps: Iterable[int],
        arm_a: list[int],
        arm_b: list[int],
        ori_tp: int,
        ter_tp: int,
) -> dict:
    """Classify one WalkID using the Ori/Ter hierarchy.

    The two replication fronts are measured independently. Exact left/right
    symmetry is not required, because fork progression and visible sister-locus
    separation can differ between replichores.
    """
    if not arm_a or not arm_b:
        raise ValueError('replichore arms must be non-empty')
    if arm_a[0] != ori_tp or arm_b[0] != ori_tp:
        raise ValueError('both replichore arms must start at ori_tp')
    if arm_a[-1] != ter_tp or arm_b[-1] != ter_tp:
        raise ValueError('both replichore arms must end at ter_tp')

    expected = set(arm_a) | set(arm_b)
    confirmed = set(confirmed_tps) & expected
    evaluated = (set(evaluated_tps) & expected) | confirmed
    failed = (set(failed_tps) & expected) - evaluated
    missing = expected - evaluated - failed
    unevaluable = failed | missing
    unconfirmed = evaluated - confirmed

    ori_evaluated = ori_tp in evaluated
    ori_two = ori_tp in confirmed
    ter_evaluated = ter_tp in evaluated
    ter_two = ter_tp in confirmed
    multispot = bool(confirmed)

    state = 'non_replicating'
    replicating = 0
    active = 0
    fully = 0
    full_candidate = 0
    progress = ''
    progress_a = ''
    progress_b = ''
    asymmetry = ''
    front_a_tp = ''
    front_b_tp = ''
    front_a_censored = ''
    front_b_censored = ''

    if not ori_evaluated:
        state = 'indeterminate_ori'
    elif not ori_two:
        state = 'multispot_only' if multispot else 'non_replicating'
    elif not ter_evaluated:
        state = 'indeterminate_completion'
    elif not ter_two:
        state = 'active_replicating'
        replicating = 1
        active = 1
        steps_a, front_a, censored_a = _front(
            arm_a, confirmed, unevaluable)
        steps_b, front_b, censored_b = _front(
            arm_b, confirmed, unevaluable)
        progress_a = steps_a / max(len(arm_a) - 1, 1)
        progress_b = steps_b / max(len(arm_b) - 1, 1)
        progress = (1 + steps_a + steps_b) / max(len(expected), 1)
        asymmetry = abs(progress_a - progress_b)
        front_a_tp = tp_label(front_a)
        front_b_tp = tp_label(front_b)
        front_a_censored = censored_a
        front_b_censored = censored_b
    else:
        full_candidate = 1
        replicating = 1
        if confirmed == expected:
            state = 'fully_replicated'
            fully = 1
            progress = 1.0
            progress_a = 1.0
            progress_b = 1.0
            asymmetry = 0.0
            front_a_tp = tp_label(ter_tp)
            front_b_tp = tp_label(ter_tp)
            front_a_censored = 0
            front_b_censored = 0
        else:
            state = 'full_candidate_incomplete'

    return {
        CLASSIFICATION_REASON_FIELD: (
            '' if evaluated else REASON_NO_EVALUABLE_LOCALIZATION),
        'MultiSpot': int(multispot),
        'Replicating': replicating,
        'ActiveReplicating': active,
        'FullyReplicated': fully,
        'FullReplicationCandidate': full_candidate,
        'replication_state': state,
        'replication_progress': progress,
        'replication_progress_arm_a': progress_a,
        'replication_progress_arm_b': progress_b,
        'replication_asymmetry': asymmetry,
        'replication_front_arm_a_tp': front_a_tp,
        'replication_front_arm_b_tp': front_b_tp,
        'replication_front_arm_a_censored': front_a_censored,
        'replication_front_arm_b_censored': front_b_censored,
        'OriEvaluated': int(ori_evaluated),
        'OriTwoSpot': int(ori_two),
        'TerEvaluated': int(ter_evaluated),
        'TerTwoSpot': int(ter_two),
        'replication_pattern_complete': int(not unevaluable),
        'n_expected_tps': len(expected),
        'n_evaluated_tps': len(evaluated),
        'n_confirmed_tps': len(confirmed),
        'n_failed_tps': len(failed),
        'n_missing_tps': len(missing),
        'unconfirmed_tps': _format_tps(unconfirmed),
        'failed_tps': _format_tps(failed),
        'missing_tps': _format_tps(missing),
        'unevaluable_tps': _format_tps(unevaluable),
        'max_consecutive_confirmed_run': max(
            longest_consecutive_run(arm_a, confirmed),
            longest_consecutive_run(arm_b, confirmed)),
        'max_ori_anchored_run': max(
            ori_anchored_run(arm_a, confirmed),
            ori_anchored_run(arm_b, confirmed)),
    }
