#!/usr/bin/env python3
"""Standalone replication-classification step.

Reads the per-(WalkID, TP) two-spot side-car produced by replication_qc.py,
aggregates it per walk into an operational replication state via
replication_detection.classify_replication (Ori/Ter hierarchy, per-replichore
fronts, evaluability-aware), and writes:

  <prefix>_replication_classification.csv   one row per WalkID (the authoritative
                                             replication call + all annotations)

and, unless --no_annotate, adds the per-walk replication columns onto every
<prefix>_drift_corrected*.csv found in --outdir (join on WalkID), so downstream
consumers that read `Replicating`/`MultiSpot` off the drift-corrected tables keep
working.

This is deliberately independent of drift_correct.py: drift correction registers
coordinates; replication classification interprets the two-spot pattern. They
share only the WalkID key.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from . import center_ownership as _own
    from . import replication_detection as _detection
    from .annotate_replication import annotate_csv
    from .replicating_complex_ids import (
        annotate_record_complex_ids,
        complex_id_field_order,
    )
    from .replication_detection import (
        ORI_TP_DEFAULT,
        TER_TP_DEFAULT,
        REPLICATION_TP_START_DEFAULT,
        REPLICATION_TP_END_DEFAULT,
        WALK_ANNOTATION_FIELDS,
        build_replichores,
        classify_replication,
        ori_anchored_run,
    )
except ImportError:  # direct execution
    import center_ownership as _own
    import replication_detection as _detection
    from annotate_replication import annotate_csv
    from replicating_complex_ids import (
        annotate_record_complex_ids,
        complex_id_field_order,
    )
    from replication_detection import (
        ORI_TP_DEFAULT,
        TER_TP_DEFAULT,
        REPLICATION_TP_START_DEFAULT,
        REPLICATION_TP_END_DEFAULT,
        WALK_ANNOTATION_FIELDS,
        build_replichores,
        classify_replication,
        ori_anchored_run,
    )

_TP_RE = re.compile(r'(\d+)')


def _tp_index(tp: str) -> int:
    m = _TP_RE.search(str(tp))
    if not m:
        raise ValueError(f'cannot parse tp index from {tp!r}')
    return int(m.group(1))


#: Optional post-hoc metric filters, as (CLI attribute, side-car column).
#: Each demotes an EXISTING confirmation whose value is below the threshold.
#: A threshold of 0 disables that filter entirely.
METRIC_FILTERS = (
    ('min_spot_sep_xy_lcb_nm', 'spot_sep_xy_lcb_nm'),
    ('min_delta_bic', 'delta_BIC'),
    ('min_amp_ratio', 'amp_ratio'),
)


def active_metric_filters(args) -> list[tuple[str, float]]:
    """The (column, threshold) pairs that are switched on."""
    out = []
    for attr, column in METRIC_FILTERS:
        threshold = float(getattr(args, attr, 0.0) or 0.0)
        if threshold > 0:
            out.append((column, threshold))
    return out


def _passes_metric_filters(row, active) -> bool:
    """True when the row clears every ENABLED metric filter (>= comparison).

    A missing or non-finite value fails, but only for a filter that is actually
    switched on: a disabled filter can never demote a confirmation.
    """
    for column, threshold in active:
        try:
            value = float(row.get(column, ''))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value) or value < threshold:
            return False
    return True


def apply_coherence(confirmed, arm_a, arm_b, min_consecutive, min_ori_anchored):
    """Demote confirmations that do not sit inside a long enough run.

    Runs are read along each replichore's genomic order. A locus is retained
    when it qualifies on EITHER arm. `min_consecutive` accepts a run anywhere in
    the arm; `min_ori_anchored` accepts only a run that starts at Ori, which is
    the stricter biological reading -- if Ori is unconfirmed nothing qualifies
    and the whole walk is demoted. Both are applied when both are set.
    """
    keep = set(confirmed)
    if min_consecutive > 1:
        allowed = set()
        for arm in (arm_a, arm_b):
            run = []
            for tp in arm:
                if tp in confirmed:
                    run.append(tp)
                    continue
                if len(run) >= min_consecutive:
                    allowed.update(run)
                run = []
            if len(run) >= min_consecutive:
                allowed.update(run)
        keep &= allowed
    if min_ori_anchored > 0:
        allowed = set()
        for arm in (arm_a, arm_b):
            if ori_anchored_run(arm, confirmed) >= min_ori_anchored:
                for tp in arm:
                    if tp not in confirmed:
                        break
                    allowed.add(tp)
        keep &= allowed
    return keep


#: Status strings for which the one-vs-two-spot question was actually ANSWERED.
#: IMPORTED, not mirrored: a verbatim copy is how a status added on one side
#: silently fails to be honoured on the other. `ownership_unevaluable` is
#: deliberately absent from it, as are the fit failures.
try:
    from .replication_status import EVALUABLE_STATUSES as _EVALUABLE_STATUSES
except ImportError:
    from replication_status import EVALUABLE_STATUSES as _EVALUABLE_STATUSES


#: Default filename of the localization-attempt ledger, relative to --outdir.
#: This is the authoritative record of what was ATTEMPTED, written by
#: localize_probe0.py for every (WalkID, tp) submitted to the fitter, including
#: failed fits and ownership rejections.
ATTEMPT_LEDGER_SUFFIX = '_z_gate_status.csv'

#: Columns the ledger must have before it can define the walk denominator.
LEDGER_REQUIRED_COLUMNS = ('WalkID', 'fit_ret_code', 'localization_written')

#: Provenance files that record which ownership policy actually ran, written by
#: the stages themselves. Searched in --outdir; never invented here, and never
#: replaced by a new policy file of our own.
OWNERSHIP_PROVENANCE_GLOBS = ('*_chromatic_registration.json',)

#: The CLI option that deliberately restores the historical, under-counting
#: universe. Named in every failure message so the operator is never left
#: guessing which switch would make the run proceed.
LEGACY_OPT_OUT_FLAG = '--no_attempt_ledger'


class LedgerValidationError(Exception):
    """A required attempt ledger is missing, ambiguous, or unusable.

    Deliberately its own type: it must never be swallowed by a handler meant for
    a malformed row, and it must never be converted into an empty denominator.
    """


def _ownership_policies(outdir) -> dict:
    """{provenance path: ownership policy} for every record in ``outdir``.

    Reads only what the stages already wrote. A file without an
    ``ownership_policy`` key predates the field and is ignored rather than
    treated as evidence of a policy.
    """
    found = {}
    for pattern in OWNERSHIP_PROVENANCE_GLOBS:
        for path in sorted(glob.glob(os.path.join(outdir, pattern))):
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            policy = record.get('ownership_policy')
            if policy:
                found[path] = str(policy)
    return found


def _resolve_ownership_policy(outdir):
    """(policy, sources) from provenance. Disagreement is fatal.

    ``policy`` is None when no provenance records one, which is a legacy tree --
    NOT permission to skip the ledger. Permission comes only from an explicit
    CLI flag.
    """
    sources = _ownership_policies(outdir)
    values = set(sources.values())
    if len(values) > 1:
        listing = '; '.join(f'{os.path.basename(path)}={value!r}'
                            for path, value in sorted(sources.items()))
        raise LedgerValidationError(
            'ownership-policy provenance disagrees, so it cannot be trusted to '
            f'decide whether the attempt ledger is mandatory: {listing}. '
            'Re-run the stages consistently, or state the intent explicitly '
            'with --attempt_ledger_csv / --require_attempt_ledger / '
            f'{LEGACY_OPT_OUT_FLAG}.')
    return (next(iter(values)) if values else None), sources


def _discover_attempt_ledger(outdir, prefix):
    """Candidate ledgers in ``outdir`` only. Never searches other directories.

    Returns the single candidate, or raises when discovery is ambiguous. Nothing
    outside ``outdir`` is considered: copying a ledger in from another Location
    would silently classify one cell population against another's denominator.
    """
    exact = os.path.join(outdir, f'{prefix}{ATTEMPT_LEDGER_SUFFIX}')
    if os.path.isfile(exact):
        return exact
    candidates = sorted(glob.glob(
        os.path.join(outdir, f'*{ATTEMPT_LEDGER_SUFFIX}')))
    if len(candidates) > 1:
        raise LedgerValidationError(
            f'attempt-ledger auto-discovery is ambiguous in {outdir}: '
            + ', '.join(os.path.basename(path) for path in candidates)
            + f'. Name the intended file with --attempt_ledger_csv, or use '
              f'{LEGACY_OPT_OUT_FLAG} to classify the side-car universe only.')
    return candidates[0] if candidates else None


def _ledger_failure(reason, *, path, required, policy, sources):
    """One consistent, actionable message for every ledger failure."""
    lines = [
        f'attempt ledger unusable: {reason}',
        f'  required            : {"yes" if required else "no"}',
        f'  path                : {path if path else "<none resolved>"}',
        f'  ownership policy    : {policy!r}'
        + (' from ' + ', '.join(os.path.basename(p) for p in sorted(sources))
           if sources else ' (no provenance records one)'),
        f'  deliberate legacy   : {LEGACY_OPT_OUT_FLAG} classifies only walks '
        'present in the replication side-car, which EXCLUDES every walk whose '
        'localizations were all rejected or all failed',
    ]
    return '\n'.join(lines)

#: Refinements of replication_detection.REASON_NO_EVALUABLE_LOCALIZATION, which
#: only the ledger can distinguish. A rejected centre and a failed fit are both
#: unevaluable, but they are different failures and must not be conflated.
REASON_ALL_OWNERSHIP_UNEVALUABLE = 'all_localizations_ownership_unevaluable'
REASON_ALL_FITS_FAILED = 'all_localization_fits_failed'
REASON_NO_ATTEMPTS_IN_RANGE = 'no_localization_attempt_in_configured_range'


def _read_attempt_ledger(path, tp_start, tp_end):
    """Attempted walks and their per-locus outcome, from the ledger.

    Returns ``{walk_id: {'rejected': {tp}, 'failed': {tp}, 'accepted': {tp}}}``
    restricted to the configured genomic range. Every walk in the ledger appears,
    including one whose every attempt was rejected -- that walk is exactly the one
    that used to vanish from the classification denominator.

    A duplicated ``(WalkID, tp)`` is a corrupt ledger, not a duplicate to be
    tolerated, and raises: silently collapsing it would make the attempted-walk
    universe depend on row order.
    """
    universe = {}
    seen = set()
    try:
        handle = open(path, newline='')
    except OSError as exc:
        raise LedgerValidationError(
            f'cannot read {path}: {exc.strerror or exc}') from None
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LedgerValidationError(f'{path} has no CSV header')
        absent = [name for name in LEDGER_REQUIRED_COLUMNS
                  if name not in reader.fieldnames]
        if 'tp' not in reader.fieldnames and 'TP' not in reader.fieldnames:
            absent.append('tp')
        if absent:
            raise LedgerValidationError(
                f'{path} is missing required column(s) {absent}; it cannot '
                'define the attempted-walk universe')
        for number, row in enumerate(reader, start=2):
            try:
                wid = int(str(row['WalkID']).strip())
            except (TypeError, ValueError, KeyError):
                raise LedgerValidationError(
                    f'{path}:{number} has an unusable WalkID '
                    f'{row.get("WalkID")!r}') from None
            label = str(row.get('tp', row.get('TP', ''))).strip()
            if not label:
                raise LedgerValidationError(
                    f'{path}:{number} has a blank tp for WalkID={wid}; an '
                    'attempt with no timepoint cannot be counted')
            key = (wid, label)
            if key in seen:
                raise LedgerValidationError(
                    f'{path}:{number} duplicates attempt {key!r}; the attempt '
                    'ledger must have one row per (WalkID, tp)')
            seen.add(key)
            entry = universe.setdefault(
                wid, {'rejected': set(), 'failed': set(), 'accepted': set()})
            try:
                tp = _tp_index(label)
            except ValueError:
                continue
            if not (tp_start <= tp <= tp_end):
                continue
            # One authoritative reading (pipeline/center_ownership.py). A blank
            # fit_ret_code used to fall through to 'accepted' here, which put an
            # attempt that may never have been fitted into the denominator as a
            # tested single spot.
            try:
                outcome = _own.attempt_outcome(row)
            except _own.LedgerRowError as exc:
                raise LedgerValidationError(f'{path}: {exc}') from None
            if outcome == _own.ATTEMPT_FIT_FAILED:
                entry['failed'].add(tp)
            elif outcome == _own.ATTEMPT_ACCEPTED:
                entry['accepted'].add(tp)
            else:
                entry['rejected'].add(tp)
    if not universe:
        raise LedgerValidationError(
            f'{path} contains no attempt rows; a missing denominator is not an '
            'empty one, and an empty ledger cannot be the walk universe')
    return universe


def _reason_for_unevaluable(entry) -> str:
    """Refine the coarse reason using the ledger's own outcome counts."""
    if entry is None:
        return _detection.REASON_NO_EVALUABLE_LOCALIZATION
    rejected, failed = len(entry['rejected']), len(entry['failed'])
    if not rejected and not failed:
        return REASON_NO_ATTEMPTS_IN_RANGE
    if rejected and not failed:
        return REASON_ALL_OWNERSHIP_UNEVALUABLE
    if failed and not rejected:
        return REASON_ALL_FITS_FAILED
    return _detection.REASON_NO_EVALUABLE_LOCALIZATION


def _ownership_rejected(row) -> bool:
    """True when strict centre ownership made this locus unevaluable.

    Reads the explicit side-car column rather than inferring it from the status
    string, so the two cannot drift apart, and defaults to False so an older
    side-car written before the column existed behaves exactly as before.
    """
    try:
        return bool(int(float(row.get('ownership_unevaluable', 0) or 0)))
    except (TypeError, ValueError):
        return False


def _sets_for_walk(rows, active_filters=()):
    """(confirmed, evaluated, failed) tp-index sets + suspect flags for one walk.

    evaluated = the two-spot question was successfully answered
                (not_suspect / tested_not_confirmed / confirmed).
    failed    = the two-spot question was not answerable. A failed M2 fit is
                not converted into negative one-spot evidence.

    `active_filters` may demote a confirmation, never create one, and a demoted
    locus stays EVALUATED -- it was measured, the answer is simply now read as
    "not a confirmed two-spot" rather than as unanswerable.

    Strict centre ownership is different from a metric filter and is applied
    BEFORE any of this: an ownership-rejected locus is unevaluable in both
    directions, so it goes to `failed`, cannot contribute a confirmation, and is
    never read as evidence of a single spot.
    """
    confirmed, evaluated, failed = set(), set(), set()
    pinned = merged = 0
    for r in rows:
        try:
            tp = _tp_index(r['tp'])
        except (ValueError, KeyError):
            continue
        status = str(r.get('status', ''))
        conf = int(float(r.get('replication_confirmed', 0) or 0))
        if _ownership_rejected(r):
            failed.add(tp)
            pinned = pinned or int(float(
                r.get('pinned_double_fit_suspect', 0) or 0))
            merged = merged or int(float(
                r.get('merged_mask_suspect', 0) or 0))
            continue
        if conf and _passes_metric_filters(r, active_filters):
            confirmed.add(tp)
        # NOTE: keyed on the ORIGINAL confirmation, so filtering can never push
        # a measured locus into `failed`.
        if conf or status in _EVALUABLE_STATUSES:
            evaluated.add(tp)
        else:
            failed.add(tp)
        pinned = pinned or int(float(r.get('pinned_double_fit_suspect', 0) or 0))
        merged = merged or int(float(r.get('merged_mask_suspect', 0) or 0))
    return confirmed, evaluated, failed, pinned, merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replication_qc_csv', required=True,
                   help='Per-(WalkID, TP) side-car from replication_qc.py.')
    p.add_argument('--outdir', default=None,
                   help='Output dir (default: dir of --replication_qc_csv).')
    p.add_argument('--output_prefix', default='localizations_640')
    p.add_argument('--ori_tp', type=int, default=ORI_TP_DEFAULT)
    p.add_argument('--ter_tp', type=int, default=TER_TP_DEFAULT)
    p.add_argument('--replication_tp_start', type=int,
                   default=REPLICATION_TP_START_DEFAULT)
    p.add_argument('--replication_tp_end', type=int,
                   default=REPLICATION_TP_END_DEFAULT)
    p.add_argument('--no_annotate', action='store_true',
                   help='Only write the classification CSV; do not annotate the '
                        'drift-corrected CSVs.')
    ledger = p.add_argument_group(
        'attempted-walk denominator',
        'A walk whose every localization was ownership-rejected or failed has no '
        'replication side-car rows at all, so a side-car-only universe silently '
        'loses it. Under strict ownership the ledger is therefore MANDATORY, and '
        'an absent ledger is fatal rather than permission to fall back.')
    ledger.add_argument('--attempt_ledger_csv', default=None,
                        help='Explicit ledger path. If supplied and unusable the '
                             'run fails; no other path is ever tried.')
    ledger.add_argument('--require_attempt_ledger', action='store_true',
                        help='Require a ledger, discovering it in --outdir when '
                             '--attempt_ledger_csv is not given. Nothing found, '
                             'or more than one candidate, is fatal. Strict '
                             'production drivers pass this together with the '
                             'explicit path.')
    ledger.add_argument('--no_attempt_ledger', action='store_true',
                        help='DELIBERATE LEGACY OPT-OUT. Classify only walks '
                             'present in the side-car, reproducing the '
                             'historical under-counting universe. Mutually '
                             'exclusive with the two options above, and never '
                             'used by a strict production driver.')

    g = p.add_argument_group(
        'candidate confirmation policies (all OFF by default)',
        'Post-hoc filters that DEMOTE existing confirmations before the '
        'Ori/Ter hierarchy is recomputed. They never promote a locus, never '
        'move one into "failed", and never touch the side-car, the fits, the '
        'bootstrap or the suspect flags. With every value at 0 the '
        'classification is identical to the unfiltered run. See '
        'docs/replication_policy_candidates.md -- do not redefine these after '
        'looking at validation labels.')
    g.add_argument('--min_spot_sep_xy_lcb_nm', type=float, default=0.0,
                   help='Demote confirmations whose 95%% lower confidence bound '
                        'on LATERAL separation (side-car spot_sep_xy_lcb_nm) is '
                        'below this. Same conservative quantity the '
                        'replication_qc --min_spot_sep_xy_nm gate uses, NOT the '
                        'raw fitted spot_sep_xy_nm. (default: %(default)s = off)')
    g.add_argument('--min_delta_bic', type=float, default=0.0,
                   help='Demote confirmations with delta_BIC below this. '
                        'Predictive score, not an evidence scale: it grows with '
                        'fitted voxel count and sits in a singular '
                        'model-selection setting. (default: %(default)s = off)')
    g.add_argument('--min_amp_ratio', type=float, default=0.0,
                   help='Demote confirmations with amp_ratio below this. NOT a '
                        'free refinement -- it acts mostly on the Replicating '
                        'population and can preferentially remove real sisters '
                        'with unequal hybridisation, bleaching or focus. '
                        '(default: %(default)s = off)')
    g.add_argument('--min_consecutive_run', type=int, default=0,
                   help='Demote confirmations not inside a run of at least N '
                        'consecutive confirmed loci along either replichore. '
                        '(default: %(default)s = off)')
    g.add_argument('--min_ori_anchored_run', type=int, default=0,
                   help='Stricter coherence: demote confirmations unless they '
                        'sit in a run of at least N starting AT Ori. If Ori is '
                        'unconfirmed the whole walk is demoted, so this can '
                        'remove genuine very-early replication where only Ori '
                        'is resolved. (default: %(default)s = off)')
    args = p.parse_args()
    for flag, value in (('--min_consecutive_run', args.min_consecutive_run),
                        ('--min_ori_anchored_run', args.min_ori_anchored_run)):
        if value < 0:
            p.error(f'{flag} must be >= 0')

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.replication_qc_csv))
    try:
        arm_a, arm_b = build_replichores(
            args.ori_tp, args.ter_tp,
            args.replication_tp_start, args.replication_tp_end)
    except ValueError as exc:
        p.error(str(exc))

    rows_by_walk = defaultdict(list)
    with open(args.replication_qc_csv, newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                rows_by_walk[int(row['WalkID'])].append(row)
            except (TypeError, ValueError, KeyError):
                continue

    # --- the attempted-walk denominator ------------------------------------
    # The side-car only contains walks that produced at least one accepted
    # localization, so it is NOT the walk universe: a walk whose every centre was
    # ownership-rejected has no side-car rows and used to be absent from the
    # totals entirely -- neither counted nor reported, which reads as if it had
    # never been imaged. The attempt ledger is authoritative for what was
    # attempted; the universe is the union, so a side-car walk missing from the
    # ledger is still classified rather than dropped.
    #
    # Resolution is FAIL-CLOSED and happens before any output file is opened.
    # The one thing it must never do is treat an absent ledger as permission to
    # use the old universe: that is how a strict run silently loses every walk
    # whose localizations were all rejected.
    if args.no_attempt_ledger and (args.require_attempt_ledger
                                   or args.attempt_ledger_csv):
        conflicting = ' and '.join(filter(None, [
            '--require_attempt_ledger' if args.require_attempt_ledger else '',
            '--attempt_ledger_csv' if args.attempt_ledger_csv else '']))
        p.error(f'{LEGACY_OPT_OUT_FLAG} contradicts {conflicting}: decide '
                'whether the attempted-walk denominator is required or '
                'deliberately abandoned')
    attempted = {}
    ledger_path = None
    try:
        policy, policy_sources = _resolve_ownership_policy(outdir)
        # Any ENFORCING policy can leave a walk with no accepted localization, so
        # any of them makes the ledger mandatory -- not just `strict`.
        # `constrained_experimental` drops the loci whose refit did not land
        # inside, so it can lose a walk exactly the same way. A tree with no
        # recorded policy is legacy and keeps its historical behaviour; that is
        # the one case where absence is not treated as enforcement.
        strict = policy is not None and policy != _own.POLICY_ALLOW
        if args.no_attempt_ledger:
            # Explicit, and loud: the operator is choosing a denominator that
            # cannot represent a wholly unevaluable walk.
            print(f'[classify_replication] WARNING: {LEGACY_OPT_OUT_FLAG} -- the '
                  'attempted-walk denominator is ABANDONED. Walks whose '
                  'localizations were all rejected or all failed are absent from '
                  'these totals entirely.')
            if strict:
                print('[classify_replication] WARNING: this dataset ran under '
                      'STRICT centre ownership '
                      f'({", ".join(sorted(os.path.basename(s) for s in policy_sources))}), '
                      'where that denominator is normally mandatory. Overriding '
                      'it is a legacy-reproduction escape hatch, not a '
                      'production configuration.')
        else:
            required = args.require_attempt_ledger or strict
            if args.attempt_ledger_csv:
                ledger_path = args.attempt_ledger_csv
                if not os.path.isfile(ledger_path):
                    # Explicit path: never silently try another one, whether or
                    # not the ledger was otherwise required.
                    raise LedgerValidationError(_ledger_failure(
                        'the explicitly supplied path does not exist',
                        path=ledger_path, required=True, policy=policy,
                        sources=policy_sources))
            else:
                ledger_path = _discover_attempt_ledger(
                    outdir, args.output_prefix)
                if ledger_path is None and required:
                    raise LedgerValidationError(_ledger_failure(
                        'no ledger was found by auto-discovery in the output '
                        'directory' + (
                            ' and strict ownership makes it mandatory'
                            if strict and not args.require_attempt_ledger
                            else ''),
                        path=os.path.join(
                            outdir,
                            f'{args.output_prefix}{ATTEMPT_LEDGER_SUFFIX}'),
                        required=True, policy=policy, sources=policy_sources))
            if ledger_path is not None:
                try:
                    attempted = _read_attempt_ledger(
                        ledger_path, args.replication_tp_start,
                        args.replication_tp_end)
                except LedgerValidationError as exc:
                    raise LedgerValidationError(_ledger_failure(
                        str(exc), path=ledger_path, required=required,
                        policy=policy, sources=policy_sources)) from None
            else:
                print('[classify_replication] no attempt ledger found and none '
                      'required (ownership policy '
                      f'{policy!r}); classifying the side-car universe only')
    except LedgerValidationError as exc:
        # Nothing has been written at this point: no walk summary, no policy
        # JSON, no annotation. Exit nonzero so a shell or SLURM dependency stops
        # the downstream merge and trace-salvage stages.
        print(f'[classify_replication] FATAL: {exc}', file=sys.stderr)
        raise SystemExit(2)
    universe = sorted(set(rows_by_walk) | set(attempted))
    n_ledger_only = len(set(attempted) - set(rows_by_walk))
    n_sidecar_only = len(set(rows_by_walk) - set(attempted)) if attempted else 0

    active = active_metric_filters(args)
    records = []
    n_conf_before = n_conf_after = 0
    for wid in universe:
        rows = rows_by_walk.get(wid, [])
        # 1. original confirmations only -> 2. per-row metric filters
        confirmed, evaluated, failed, pinned, merged = _sets_for_walk(rows, active)
        if not rows:
            # No replication measurement exists for this walk at all. Nothing is
            # manufactured: the loci the ledger records as attempted-and-
            # unevaluable go to `failed`, the rest stay missing, and the ordinary
            # hierarchy then classifies the walk as indeterminate_ori with every
            # positive flag zero. Walks that DO have side-car rows are untouched,
            # so no existing classification changes.
            entry = attempted.get(wid)
            if entry is not None:
                failed = set(entry['rejected']) | set(entry['failed'])
        n_conf_before += sum(
            1 for r in rows
            if int(float(r.get('replication_confirmed', 0) or 0)))
        # 3./4. runs are built from the survivors, then coherence is applied at
        # walk level -- it needs the whole set plus the replichore arms.
        confirmed = apply_coherence(
            confirmed, arm_a, arm_b,
            args.min_consecutive_run, args.min_ori_anchored_run)
        n_conf_after += len(confirmed)
        # 5. hand the final confirmation set to the hierarchy
        ann = classify_replication(
            confirmed, evaluated, failed, arm_a, arm_b,
            args.ori_tp, args.ter_tp)
        ann['PinnedDoubleFitSuspect'] = int(pinned)
        ann['MergedMaskSuspect'] = int(merged)
        if ann[_detection.CLASSIFICATION_REASON_FIELD]:
            # Refine "nothing was evaluable" into WHY, which only the ledger
            # knows. Left exactly as the classifier set it when there is no
            # ledger, so behaviour without one is unchanged.
            ann[_detection.CLASSIFICATION_REASON_FIELD] = (
                _reason_for_unevaluable(attempted.get(wid))
                if attempted else ann[_detection.CLASSIFICATION_REASON_FIELD])
        rec = {'WalkID': wid}
        rec.update({k: ann[k] for k in WALK_ANNOTATION_FIELDS})
        records.append(rec)

    # This per-location classification uses ascending WalkID order. Final
    # merged deliverables recompute the field after global WalkID assignment,
    # making IDs unique and contiguous within each output CSV.
    complex_mapping = annotate_record_complex_ids(records)

    out_csv = os.path.join(outdir, f'{args.output_prefix}_replication_classification.csv')
    os.makedirs(outdir, exist_ok=True)
    with open(out_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['WalkID', *complex_id_field_order(
                WALK_ANNOTATION_FIELDS)])
        writer.writeheader()
        writer.writerows(records)
    n_rep = sum(int(row['Replicating']) for row in records)
    n_active = sum(int(row['ActiveReplicating']) for row in records)
    n_full = sum(int(row['FullyReplicated']) for row in records)
    state_counts = Counter(row['replication_state'] for row in records)
    n_strict_clean = sum(
        row['replication_state'] == 'non_replicating'
        and int(row['replication_pattern_complete']) == 1
        and int(row['PinnedDoubleFitSuspect']) == 0
        and int(row['MergedMaskSuspect']) == 0
        for row in records
    )
    print(f'[classify_replication] Ori=TP{args.ori_tp:03d} Ter=TP{args.ter_tp:03d} '
          f'range=TP{args.replication_tp_start:03d}..TP{args.replication_tp_end:03d} | '
          f'{len(records)} walks: Replicating={n_rep} '
          f'(active={n_active}, fully={n_full})')
    if attempted:
        print(f'[classify_replication] attempted-walk universe: {len(attempted)} '
              f'from {os.path.basename(ledger_path)}; side-car walks '
              f'{len(rows_by_walk)}; classified {len(records)} '
              f'(+{n_ledger_only} with no replication measurement at all'
              + (f', {n_sidecar_only} side-car-only' if n_sidecar_only else '')
              + ')')
        if len(records) != len(set(attempted) | set(rows_by_walk)):
            raise AssertionError(
                'one classification row per attempted walk is required')
    else:
        print('[classify_replication] attempted-walk ledger: NOT USED -- the '
              'universe is the side-car alone, so a wholly unevaluable walk is '
              'absent from these totals')
    print('[classify_replication] states: ' + ', '.join(
        f'{state}={count}' for state, count in sorted(state_counts.items())))
    reason_counts = Counter(
        row[_detection.CLASSIFICATION_REASON_FIELD] for row in records
        if row[_detection.CLASSIFICATION_REASON_FIELD])
    if reason_counts:
        print('[classify_replication] unevaluable reasons: ' + ', '.join(
            f'{reason}={count}'
            for reason, count in sorted(reason_counts.items())))
    print(
        '[classify_replication] strict clean non-replicating='
        f'{n_strict_clean} (complete non_replicating state; suspect flags clear)')
    print(f'[classify_replication] wrote {out_csv}')
    print(
        '[classify_replication] ReplicatingComplexID: '
        f'{len(complex_mapping)} positive 1-based IDs; '
        'non-replicating/other walks=-1')

    policy = {
        'requires_original_confirmation': True,
        'comparison': '>= threshold (a value exactly at the threshold passes)',
        'missing_or_non_finite_metric': 'fails, but only for an enabled filter',
        'filters_may': 'demote an existing confirmation only; never promote',
        'demoted_locus_remains': 'evaluated (never failed, never missing)',
        'metric_filters': [
            {'column': column, 'min': threshold} for column, threshold in active
        ],
        'coherence': {
            'min_consecutive_run': args.min_consecutive_run,
            'min_ori_anchored_run': args.min_ori_anchored_run,
            'runs_read_along': 'each replichore arm in genomic order',
            'locus_retained_if': 'it qualifies on EITHER arm',
            'run_broken_by': 'any locus that is not confirmed, whatever the '
                             'reason; censoring stays visible via '
                             'replication_front_arm_*_censored',
            'shared_endpoints': 'Ori and Ter belong to both arms and are '
                                'counted once within each arm',
        },
        'hierarchy': {
            'ori_tp': args.ori_tp, 'ter_tp': args.ter_tp,
            'replication_tp_start': args.replication_tp_start,
            'replication_tp_end': args.replication_tp_end,
            'arm_a': arm_a, 'arm_b': arm_b,
        },
        'unchanged_by_this_step': [
            'side-car metrics', 'M1/M2 fits', 'bootstrap FDR calibration',
            'PinnedDoubleFitSuspect', 'MergedMaskSuspect',
        ],
        'counts': {
            'walks': len(records),
            'confirmations_before_filtering': n_conf_before,
            'confirmations_after_filtering': n_conf_after,
            'confirmations_demoted': n_conf_before - n_conf_after,
        },
        'walk_universe': {
            'source': ('attempt_ledger_union_sidecar' if attempted
                       else 'replication_sidecar_only'),
            'ownership_policy': policy,
            'ownership_policy_sources': {
                os.path.basename(path): value
                for path, value in sorted(policy_sources.items())},
            'ledger_required': bool(args.require_attempt_ledger
                                    or policy == 'strict'),
            'legacy_opt_out': bool(args.no_attempt_ledger),
            'attempt_ledger_csv': (
                os.path.abspath(ledger_path) if attempted else None),
            'attempted_walks': len(attempted),
            'sidecar_walks': len(rows_by_walk),
            'classified_walks': len(records),
            'walks_without_replication_measurement': n_ledger_only,
            'rule': ('every attempted walk gets exactly one classification row; '
                     'a walk with no evaluable locus is indeterminate_ori with '
                     'every positive flag zero, never non_replicating'),
        },
        'is_default_policy': not active and not args.min_consecutive_run
                             and not args.min_ori_anchored_run,
    }
    policy_json = os.path.join(
        outdir, f'{args.output_prefix}_replication_policy.json')
    with open(policy_json, 'w') as handle:
        json.dump(policy, handle, indent=2)
    if policy['is_default_policy']:
        print('[classify_replication] policy: DEFAULT (no filters); '
              f'{n_conf_before} confirmations unchanged')
    else:
        print(f'[classify_replication] policy: demoted '
              f'{n_conf_before - n_conf_after} of {n_conf_before} confirmations '
              f'({100.0 * (n_conf_before - n_conf_after) / max(n_conf_before, 1):.1f}%)')
    print(f'[classify_replication] wrote {policy_json}')

    if args.no_annotate:
        return
    for path in sorted(glob.glob(os.path.join(
            outdir, f'{args.output_prefix}_drift_corrected*.csv'))):
        result = annotate_csv(path, out_csv, path)
        print(
            f'[classify_replication] annotated {os.path.basename(path)} '
            f'({result["rows"]} rows, '
            f'walks_without_summary={result["walks_without_summary"]})')


if __name__ == '__main__':
    main()
