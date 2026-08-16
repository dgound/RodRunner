#!/usr/bin/env python3
"""Explicit centre-ownership classification for probe-0 one-spot fits.

A one-spot fit is only interpretable as *this cell's* locus if the fitted centre
actually falls inside that cell's support.  Historically the fitter's support was
the DAPI mask in its own (unregistered) frame, so a chromatic offset between 405
and 640 could place a genuine 640 centre in background or inside a neighbour
without anything recording it.

This module states that classification explicitly.  It does not correct, clamp,
dilate, or salvage anything -- it labels, and the caller decides.

Mask label convention: ``WalkID`` is zero-based, the mask label is
``WalkID + 1``.
"""

from __future__ import annotations

import ctypes
import re

import numpy as np

OWN_MASK = "own_mask"
BACKGROUND = "background"
OTHER_ACTIVE_CELL = "other_active_cell"
OUT_OF_BOUNDS = "out_of_bounds"

UNCLASSIFIABLE = "unclassifiable"

#: Ordered for stable reporting.
CATEGORIES = (OWN_MASK, BACKGROUND, OTHER_ACTIVE_CELL, OUT_OF_BOUNDS,
              UNCLASSIFIABLE)

#: Only this category is a pass.
PASSING = (OWN_MASK,)

# --------------------------------------------------------------------------
# Ownership policies
# --------------------------------------------------------------------------
#: Record diagnostics, keep every unconstrained fit. Backward compatible: with
#: chromatic registration off this reproduces legacy numerical output.
POLICY_ALLOW = "allow"

#: FAIL-CLOSED FILTER. Run the ordinary unconstrained fit, classify its centre
#: against the registered effective mask, and accept ONLY if that original centre
#: lies in the cell's own label. Anything else -- background, another active
#: label, off-grid, non-finite -- makes the locus UNEVALUABLE. The coordinate is
#: never moved, projected, clamped, snapped, or re-fitted.
POLICY_STRICT = "strict"

#: EXPERIMENTAL, NOT FOR ANALYSIS. Retains the constrained refit that pushes an
#: out-of-support centre back inside the owned label. Measured on real data this
#: behaves as boundary clamping through optimization: on rep3 Location-01 all
#: 2,399 attempted refits "succeeded", none were rejected, ~47% still rounded
#: outside the mask after CSV serialization, and ~92% of those landing inside sat
#: exactly on an XY mask boundary. Downstream that removed 53-86% of complete
#: traces, overwhelmingly by making loci UNEVALUABLE rather than by detecting
#: doublets -- i.e. it destroys fits rather than reinterpreting them. Never
#: selected by `strict`, never used by production drivers or documentation.
POLICY_CONSTRAINED_EXPERIMENTAL = "constrained_experimental"

POLICIES = (POLICY_ALLOW, POLICY_STRICT, POLICY_CONSTRAINED_EXPERIMENTAL)

#: Per-locus ownership columns carried on EVERY localization row, so a forced or
#: rejected coordinate is identifiable from the comprehensive CSV alone rather
#: than only from a sidecar.
OWNERSHIP_ROW_FIELDS = (
    "CenterOwnershipPolicy",
    "CenterOwnershipInitial",
    "CenterOwnershipFinal",
    "CenterOwnershipValid",
    "CenterOwnershipFailureReason",
    "CenterRefitAttempted",
    "CenterRefitAccepted",
    "OwnershipCoordinateFrame",
    "ChromaticRegistrationApplied",
    "ChromaticRegistrationReferenceWavelengthNm",
    "ChromaticRegistrationTargetWavelengthNm",
)

#: Walk-level ownership columns added to merged / full-schema outputs.
OWNERSHIP_WALK_FIELDS = (
    "CenterOwnershipComplete",
    "CenterOwnershipValidCount",
    "CenterOwnershipExpectedCount",
    "CenterOwnershipRejectedCount",
)

#: Ownership is decided in the RAW target-channel image frame, before drift
#: correction: the mask is registered into that frame and the fitted centre is
#: still in raw pixels there. Recorded on every row so a consumer cannot mistake
#: it for a drift-corrected judgement.
FRAME_RAW_TARGET_CHANNEL = "raw_target_channel_pixels"


def failure_reason(category: str) -> str:
    """Stable, CSV-safe reason string for a non-passing category."""
    if category in PASSING:
        return ""
    return f"center_{category}"


def label_for_walk(walk_id: int) -> int:
    """Mask label of a zero-based ``WalkID``."""
    return int(walk_id) + 1


#: Decimal places used when a coordinate is written to CSV. The ownership
#: decision is made on the QUANTIZED value so the verdict recorded in the file
#: and the verdict a consumer recomputes from the file can never disagree.
COORD_DECIMALS = 4


def quantize_for_output(value: float) -> float:
    """Round a coordinate exactly as the localizer will serialize it.

    Without this, a centre lying within 5e-5 px of a half-voxel tie can be
    accepted at full precision and then re-round into the neighbouring voxel when
    read back -- measured at 12 of 14,332 accepted centres on rep3 Location-01.
    Deciding on the serialized value removes the discrepancy by construction
    rather than hiding it behind a tolerance.
    """
    v = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(v)):
        return value
    return np.round(v, COORD_DECIMALS)


def nearest_voxel(value: float) -> float:
    """Round a coordinate to its nearest voxel index, C-compatibly.

    ``gauss3d_fit.c`` resolves the nearest voxel with ``lround``, which rounds a
    half away from zero; ``np.rint`` rounds a half to EVEN.  They disagree at
    exactly ``*.5``, which is not hypothetical -- rep3 Location-01 TP010 has a
    fitted centre at x = 980.50 sitting on a label boundary, where ``np.rint``
    gives 980 (the neighbour) and ``lround`` gives 981 (its own cell).

    A disagreement here would let the constrained refit accept a centre that
    this classifier then reports as out-of-support, so the two must use one rule.
    """
    v = np.asarray(value, dtype=float)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def classify_center(effective_labels: np.ndarray, walk_id: int,
                    z: float, y: float, x: float) -> str:
    """Classify one fitted centre against the effective, per-timepoint support.

    ``effective_labels`` must be the *same* label volume the fitter used for this
    timepoint -- registered into the probe's grid, and in 2.5D z-gated mode
    already restricted to that timepoint's axial gate.

    Lookup is nearest-voxel and categorical: label IDs are never interpolated,
    and the rounding rule matches the C constraint exactly (see
    :func:`nearest_voxel`).

    Call this on the FITTED value, not on a value round-tripped through a CSV.  A
    constrained refit converges against its owned label's boundary, so the
    localizer's 4-decimal output can land on the half-voxel tie and re-round into
    the neighbouring voxel.  The verdict computed here is recorded in the
    per-attempt status table and that record is authoritative; recomputing it
    from the written coordinate can disagree by one voxel at a boundary.
    """
    labels = np.asarray(effective_labels)
    if labels.ndim != 3:
        raise ValueError(f"effective_labels must be 3D, got {labels.shape}")
    nz, ny, nx = labels.shape

    if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
        # Distinguished from OUT_OF_BOUNDS: an off-grid centre is a real position
        # outside the field, a non-finite one is not a position at all.
        return UNCLASSIFIABLE

    # Decide on the SERIALIZED coordinate (see quantize_for_output): the verdict
    # must survive a CSV round-trip exactly.
    zi = int(nearest_voxel(quantize_for_output(z)))
    yi = int(nearest_voxel(quantize_for_output(y)))
    xi = int(nearest_voxel(quantize_for_output(x)))
    if not (0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx):
        return OUT_OF_BOUNDS

    found = int(labels[zi, yi, xi])
    if found == label_for_walk(walk_id):
        return OWN_MASK
    if found == 0:
        return BACKGROUND
    return OTHER_ACTIVE_CELL


def owned_voxel_seed(image: np.ndarray, effective_labels: np.ndarray,
                     walk_id: int) -> tuple[int, int, int] | None:
    """Brightest voxel owned by this cell, as a constrained-refit start.

    Returns ``(z, y, x)`` or ``None`` when the cell has no owned voxel at this
    timepoint (e.g. a z gate that excluded every plane).
    """
    labels = np.asarray(effective_labels)
    image = np.asarray(image)
    if labels.shape != image.shape:
        raise ValueError(
            f"image {image.shape} and labels {labels.shape} shapes differ")
    owned = labels == label_for_walk(walk_id)
    if not owned.any():
        return None
    masked = np.where(owned, image, -np.inf)
    flat = int(np.argmax(masked))
    return tuple(int(v) for v in np.unravel_index(flat, labels.shape))


def row_fields(policy: str, initial: str, *, final: str | None = None,
               refit_attempted: int = 0, refit_accepted: int = 0,
               registration_applied: bool = False,
               reference_nm: str | float | None = None,
               target_nm: str | float | None = None,
               frame: str = FRAME_RAW_TARGET_CHANNEL) -> dict:
    """Build the per-locus ownership columns for one localization row.

    Single source of truth so the localizer, drift correction, replication QC and
    every merge produce byte-identical column names and value conventions.

    ``final`` defaults to ``initial``: under the corrected ``strict`` policy no
    refit happens, so the final category IS the initial one.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown ownership policy {policy!r}; expected one of "
                         f"{POLICIES}")
    final = initial if final is None else final
    valid = int(final in PASSING)
    return {
        "CenterOwnershipPolicy": policy,
        "CenterOwnershipInitial": initial,
        "CenterOwnershipFinal": final,
        "CenterOwnershipValid": valid,
        "CenterOwnershipFailureReason": "" if valid else failure_reason(final),
        "CenterRefitAttempted": int(refit_attempted),
        "CenterRefitAccepted": int(refit_accepted),
        "OwnershipCoordinateFrame": frame,
        "ChromaticRegistrationApplied": int(bool(registration_applied)),
        "ChromaticRegistrationReferenceWavelengthNm":
            "" if reference_nm is None else str(reference_nm),
        "ChromaticRegistrationTargetWavelengthNm":
            "" if target_nm is None else str(target_nm),
    }


def summarize(categories) -> dict:
    """Count categories, including zeros, for reporting."""
    counts = {name: 0 for name in CATEGORIES}
    for value in categories:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


# --------------------------------------------------------------------------
# The single constrained refit
# --------------------------------------------------------------------------

#: ``gauss3d_fit.c`` return code for a strict centre-ownership refusal.
STRICT_FAILURE_CODE = -3


def declare_batch_fit_ex(lib) -> None:
    """Declare the ``batch_fit_ex`` ctypes signature on an already-loaded lib."""
    lib.batch_fit_ex.restype = None
    lib.batch_fit_ex.argtypes = [
        ctypes.POINTER(ctypes.c_double),   # data
        ctypes.POINTER(ctypes.c_int),      # mask_labels
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),      # bboxes
        ctypes.c_int,                      # n entries
        ctypes.POINTER(ctypes.c_int),      # cell_ids (may be NULL)
        ctypes.c_int,                      # strict_own
        ctypes.POINTER(ctypes.c_double),   # results
        ctypes.POINTER(ctypes.c_int),      # success
    ]


def constrained_refit(lib, image: np.ndarray, effective_labels: np.ndarray,
                      bboxes: np.ndarray, walk_ids):
    """Re-fit the listed walks with the centre constrained to their own support.

    This is the *one* refit the ownership contract allows.  It is a genuine
    constrained optimization in ``gauss3d_fit.c``: the seed is the brightest
    owned voxel and any proposed step that leaves the owned label is rejected
    with increased damping.  Nothing is clamped, projected, dilated, or shelled,
    and no alternative salvage is attempted -- a walk whose constrained fit does
    not converge inside its own support comes back as ``STRICT_FAILURE_CODE``.

    ``walk_ids`` are zero-based; the mask label used is ``walk_id + 1``.
    Returns ``(results, success)`` with one row per entry of ``walk_ids``.
    """
    walk_ids = [int(w) for w in walk_ids]
    m = len(walk_ids)
    results = np.zeros((m, 11), dtype=np.float64)
    success = np.zeros(m, dtype=np.int32)
    if m == 0:
        return results, success

    image = np.ascontiguousarray(image, dtype=np.float64)
    labels = np.ascontiguousarray(effective_labels, dtype=np.int32)
    if image.shape != labels.shape:
        raise ValueError(
            f"image {image.shape} and labels {labels.shape} shapes differ")
    nz, ny, nx = labels.shape

    sub_bboxes = np.ascontiguousarray(
        np.asarray(bboxes, dtype=np.int32)[walk_ids], dtype=np.int32)
    cell_ids = np.ascontiguousarray(
        np.array([label_for_walk(w) for w in walk_ids], dtype=np.int32))

    lib.batch_fit_ex(
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nz, ny, nx,
        sub_bboxes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        m,
        cell_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        1,
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        success.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return results, success


# --------------------------------------------------------------------------
# Walk-level completeness (CenterOwnershipComplete)
# --------------------------------------------------------------------------

WALK_SUMMARY_FIELDS = [
    "WalkID", "n_loci_expected", "n_loci_attempted", "n_loci_own_mask",
    "CenterOwnershipComplete",
]

#: Per-locus validity columns, in preference order. The first is the canonical
#: name carried on localization rows; the second is the per-attempt audit
#: table's column, which is the only place a STRICT-rejected locus still exists
#: (strict drops the coordinate row, so the coordinate CSV cannot be used to
#: count rejections).
VALID_COLUMN_CANDIDATES = ("CenterOwnershipValid", "center_ownership_pass")


class OwnershipSchemaMissing(LookupError):
    """No per-locus ownership column is present in the rows at all.

    Raised rather than defaulted: filling absent ownership with "valid" would
    manufacture a clean result, and filling it with "invalid" would silently
    zero every walk in a dataset that simply predates the ownership schema. The
    caller must decide, visibly.
    """


def detect_valid_column(rows) -> str:
    """Name of the per-locus validity column present in ``rows``."""
    seen: set[str] = set()
    for row in rows:
        seen.update(row.keys())
        break
    for name in VALID_COLUMN_CANDIDATES:
        if name in seen:
            return name
    raise OwnershipSchemaMissing(
        "none of " + ", ".join(VALID_COLUMN_CANDIDATES) + " is present; "
        "these rows predate the centre-ownership schema")


def is_valid_flag(value) -> bool:
    """Truth of a serialized ownership flag. Blank/absent is NOT valid."""
    text = str(value if value is not None else "").strip()
    if text == "" or text.lower() in ("na", "nan", "none"):
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return text.lower() in ("true", "yes", "y", "t")


def _blank(value) -> bool:
    text = str(value if value is not None else "").strip()
    return text == "" or text.lower() in ("na", "nan", "none")


def verdict_of(row, valid_column: str) -> bool | None:
    """Three-way ownership verdict for one attempt row.

    ``True`` accepted, ``False`` explicitly rejected, ``None`` no verdict was
    ever reached -- the fit failed, so there is no centre to classify.

    The distinction matters: a failed fit is a MISSING locus, not an ownership
    rejection.  Collapsing the two would blame the ownership filter for fitter
    failures in ``CenterOwnershipRejectedCount`` and would censor a locus in the
    resolver as ``ownership_invalid`` when its real status is a failed fit.
    """
    if _blank(row.get(valid_column)):
        return None
    return is_valid_flag(row.get(valid_column))


# --------------------------------------------------------------------------
# The attempt ledger: one authoritative reader.
#
# This vocabulary was previously re-derived inline in five places, each with its
# own reading of the same two columns, and three of them treated a BLANK
# `fit_ret_code` as success. Blank is not success: it means the column was never
# written for that row, so whether a fit ran is UNKNOWN. Calling it success
# promotes an attempt that may never have been fitted into the accepted set,
# where it can then act as negative biological evidence -- a locus counted as
# "tested, single spot" that was never actually tested.
# --------------------------------------------------------------------------
LEDGER_RET_CODE = "fit_ret_code"
LEDGER_WRITTEN = "localization_written"

#: Exhaustive, mutually exclusive outcomes of one localization attempt. These
#: describe the ATTEMPT only. They are deliberately distinct from the ownership
#: categories above (where the centre landed) and from replication evidence
#: status (what the two-spot test concluded): conflating any two of the three has
#: already produced real misclassifications.
ATTEMPT_ACCEPTED = "accepted"
ATTEMPT_OWNERSHIP_REJECTED = "ownership_rejected"
ATTEMPT_FIT_FAILED = "fit_failed"
ATTEMPT_INDETERMINATE = "indeterminate"
ATTEMPT_OUTCOMES = (ATTEMPT_ACCEPTED, ATTEMPT_OWNERSHIP_REJECTED,
                    ATTEMPT_FIT_FAILED, ATTEMPT_INDETERMINATE)


def emits_coordinate_row(policy: str, fit_converged: bool,
                         ownership_passed: bool) -> bool:
    """Whether this attempt yields a usable coordinate row. ONE definition.

    The localizer used to decide this twice, differently:

    * the emission loop skipped a failing centre when
      ``policy != POLICY_ALLOW``;
    * ``localization_written`` was computed as
      ``policy != 'strict' or passed``.

    Those agree for ``allow`` and ``strict`` but NOT for
    ``constrained_experimental``, where the row was skipped while the ledger
    recorded ``localization_written=1``. The ledger then claimed a coordinate row
    that does not exist, which breaks the attempt-ledger invariant that every
    accepted attempt has exactly one row -- and an accepted attempt with no
    coordinates is precisely the shape that becomes phantom negative evidence.
    """
    if not fit_converged:
        return False
    return policy == POLICY_ALLOW or ownership_passed


class LedgerRowError(ValueError):
    """A ledger row whose outcome cannot be determined."""


def attempt_outcome(row, *, strict=True):
    """The outcome of ONE ledger row: exactly one of ATTEMPT_OUTCOMES.

    ``fit_ret_code`` nonzero  -> the fit failed; there is no centre to own.
    ``fit_ret_code`` zero     -> a fit exists, so ``localization_written``
                                 decides accepted vs ownership-rejected.

    A blank or unparseable ``fit_ret_code`` is NOT zero. Under ``strict`` it
    raises, so a malformed ledger stops the run instead of quietly inflating the
    accepted set; with ``strict=False`` (historical inputs only, at an explicit
    input boundary) it yields ATTEMPT_INDETERMINATE, which is never accepted and
    never counts as evidence either way.
    """
    raw = row.get(LEDGER_RET_CODE)
    if _blank(raw):
        if strict:
            raise LedgerRowError(
                f"{LEDGER_RET_CODE} is blank, so it is unknown whether a fit "
                "ran. Blank is not success: treating it as 0 would put an "
                "unfitted attempt into the accepted set, where it could act as "
                "negative biological evidence.")
        return ATTEMPT_INDETERMINATE
    try:
        ret = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        if strict:
            raise LedgerRowError(
                f"{LEDGER_RET_CODE}={raw!r} is not an integer return code")
        return ATTEMPT_INDETERMINATE
    if ret != 0:
        return ATTEMPT_FIT_FAILED
    written = row.get(LEDGER_WRITTEN)
    if _blank(written):
        if strict:
            raise LedgerRowError(
                f"{LEDGER_WRITTEN} is blank for a converged fit "
                f"({LEDGER_RET_CODE}=0), so it is unknown whether a coordinate "
                "row was emitted")
        return ATTEMPT_INDETERMINATE
    return (ATTEMPT_ACCEPTED if is_valid_flag(written)
            else ATTEMPT_OWNERSHIP_REJECTED)


def walk_ownership_summary(rows, locus_order, *, walk_key="WalkID",
                           tp_key="tp", valid_column=None) -> dict:
    """Walk-level ownership fields over an EXPLICIT configured locus universe.

    ``locus_order`` is the pipeline's configured genomic locus set (0-based TP
    indices) -- e.g. ``chromosome_order`` or ``range(tp_start, tp_end + 1)``. It
    is never derived from whichever rows survived filtering, because a filter
    that removed a locus would otherwise shrink the denominator and make an
    incomplete trace look complete.

    Returns ``{walk_id: {field: value}}`` over ``OWNERSHIP_WALK_FIELDS``:

    * ``CenterOwnershipExpectedCount`` -- size of the configured universe.
    * ``CenterOwnershipValidCount``    -- required loci present AND valid.
    * ``CenterOwnershipRejectedCount`` -- required loci with an EXPLICIT
      rejection verdict. A locus that is entirely absent, or that has a row but
      no verdict because its fit failed, is neither valid nor rejected; it is
      missing, and ``ExpectedCount - ValidCount - RejectedCount`` recovers it.
      No fifth column is added for that: the four fields already separate the
      three outcomes arithmetically.
    * ``CenterOwnershipComplete``      -- 1 only when every required locus is
      present and valid. Rejected and missing loci both make it 0.

    Duplicate ``(walk, locus)`` rows are counted once. Two rows for the same key
    that DISAGREE about validity are a corrupt join rather than a duplicate, and
    raise instead of letting either verdict win by row order.
    """
    expected = sorted({int(index) for index in locus_order})
    if not expected:
        raise ValueError("configured locus universe is empty")
    expected_set = set(expected)

    rows = list(rows)
    if not rows:
        return {}
    column = valid_column or detect_valid_column(rows)

    # (walk, locus) -> validity, so a duplicated row cannot be double counted.
    verdicts: dict[tuple[int, int], bool] = {}
    walks: set[int] = set()
    n_unparsed = 0
    n_rows = 0
    for row in rows:
        n_rows += 1
        try:
            walk = int(str(row[walk_key]).strip())
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"ownership rows have no usable {walk_key!r} column") from None
        walks.add(walk)
        try:
            locus = tp_index(row[tp_key])
        except (KeyError, ValueError):
            n_unparsed += 1
            continue
        if locus not in expected_set:
            continue
        valid = verdict_of(row, column)
        if valid is None:
            continue
        key = (walk, locus)
        if key in verdicts and verdicts[key] != valid:
            raise ValueError(
                f"contradictory ownership verdicts for WalkID={walk} "
                f"TP{locus:03d}: both valid and invalid rows present")
        verdicts[key] = valid

    # Same silent-empty guard as walk_completeness: a label form the parser
    # rejects must not be reported as "no loci".
    if n_rows and n_unparsed == n_rows:
        raise ValueError(
            f"none of the {n_rows} ownership rows had a parseable "
            f"{tp_key!r} label (expected TP### or TP###_CYC###)")

    summary = {}
    for walk in sorted(walks):
        valid_loci = {locus for (w, locus), ok in verdicts.items()
                      if w == walk and ok}
        rejected_loci = {locus for (w, locus), ok in verdicts.items()
                         if w == walk and not ok}
        summary[walk] = {
            "CenterOwnershipComplete": int(valid_loci >= expected_set),
            "CenterOwnershipValidCount": len(valid_loci),
            "CenterOwnershipExpectedCount": len(expected_set),
            "CenterOwnershipRejectedCount": len(rejected_loci),
        }
    return summary


#: Same pattern as ``replication_qc._TP_NAME_RE``. The ``_CYC###`` suffix is NOT
#: optional decoration -- the TIFF-era replicate-1 tree labels every acquisition
#: ``TP000_CYC000``, so a parser that accepts only bare ``TP###`` silently drops
#: every rep1 row.
_TP_NAME_RE = re.compile(r"^TP(\d+)(?:_CYC\d+)?$", re.IGNORECASE)


def tp_index(tp_name: str) -> int:
    """0-based acquisition index of a ``TP###`` or ``TP###_CYC###`` label.

    Parses the number rather than matching a fixed list: the locus interval is a
    configured range, and TP names are never hardcoded.
    """
    match = _TP_NAME_RE.match(str(tp_name).strip())
    if match is None:
        raise ValueError(f"unrecognized TP name: {tp_name!r}")
    return int(match.group(1))


def walk_completeness(rows, tp_start: int, tp_end: int) -> list[dict]:
    """Aggregate per-attempt ownership rows into walk-level completeness.

    ``rows`` is any iterable of mappings with ``tp``, ``WalkID`` and
    ``center_ownership_pass`` -- i.e. the per-attempt status table the localizer
    already writes, so no new reduced CSV is needed upstream.

    A walk is ``CenterOwnershipComplete`` when EVERY acquisition in the inclusive
    ``[tp_start, tp_end]`` interval has a centre that passed ownership.  A locus
    that was never attempted, whose fit failed, or whose centre landed outside
    its own support all count identically against completeness -- the field is a
    statement about the trace, not about why a locus is missing.
    """
    if tp_end < tp_start:
        raise ValueError(f"empty locus interval [{tp_start}, {tp_end}]")
    expected = set(range(int(tp_start), int(tp_end) + 1))

    attempted: dict[int, set[int]] = {}
    passed: dict[int, set[int]] = {}
    n_rows = 0
    n_unparsed = 0
    for row in rows:
        n_rows += 1
        try:
            label = row["tp"]
        except KeyError:
            raise ValueError(
                "per-attempt status rows have no 'tp' column") from None
        try:
            index = tp_index(label)
        except ValueError:
            n_unparsed += 1
            continue
        if index not in expected:
            continue
        walk = int(row["WalkID"])
        attempted.setdefault(walk, set()).add(index)
        flag = str(row.get("center_ownership_pass", "") or "0")
        if flag not in ("", "0"):
            passed.setdefault(walk, set()).add(index)

    # Guard against the silent-empty failure mode. A parser that rejects the
    # dataset's label form -- rep1 labels every acquisition TP000_CYC000 -- would
    # otherwise drop every row and report "0 walks" while exiting successfully.
    if n_rows and n_unparsed == n_rows:
        raise ValueError(
            f"none of the {n_rows} status rows had a parseable TP label "
            "(expected TP### or TP###_CYC###); refusing to report an empty "
            "completeness result")

    out = []
    for walk in sorted(attempted):
        hits = passed.get(walk, set())
        out.append({
            "WalkID": walk,
            "n_loci_expected": len(expected),
            "n_loci_attempted": len(attempted[walk]),
            "n_loci_own_mask": len(hits),
            "CenterOwnershipComplete": int(hits >= expected),
        })
    return out


# =========================================================================
# Usability classification: XY cell identity + Z diagnostic, separate from
# the exact voxel-level ownership retained for audit and legacy compat.
# =========================================================================

DECISION_EXACT_OWN_MASK = "exact_own_mask"
DECISION_NEAR_BACKGROUND = "near_background"
DECISION_UNREGISTERED_AUDIT_ONLY = "unregistered_audit_only"
DECISION_HARD_REJECT = "hard_reject"

USABLE_DECISIONS = (DECISION_EXACT_OWN_MASK, DECISION_NEAR_BACKGROUND,
                    DECISION_UNREGISTERED_AUDIT_ONLY)

REASON_OTHER_ACTIVE_CELL = "other_active_cell"
REASON_FAR_BACKGROUND = "far_background"
REASON_NONFINITE = "nonfinite_coordinates"
REASON_OUT_OF_BOUNDS_XY = "out_of_bounds_xy"
REASON_OUT_OF_BOUNDS_Z = "out_of_bounds_z"
REASON_NO_REASON = ""

XY_EXACT_OWN = "exact_own"
XY_NEAR_BACKGROUND = "near_background"
XY_FAR_BACKGROUND = "far_background"
XY_OTHER_CELL = "other_active_cell"
XY_OUT_OF_BOUNDS = "out_of_bounds"
XY_NONFINITE = "nonfinite"

Z_INSIDE_GATE = "inside_gate"
Z_BELOW_GATE = "below_gate"
Z_ABOVE_GATE = "above_gate"
Z_OUT_OF_BOUNDS = "out_of_bounds"
Z_NONFINITE = "nonfinite"
Z_NO_SUPPORT = "no_z_support"

USABILITY_ROW_FIELDS = (
    "CenterOwnershipUsable",
    "CenterOwnershipDecision",
    "CenterOwnershipHardRejectReason",
    "CenterOwnershipXYCategory",
    "CenterOwnershipXYDistanceNm",
    "CenterOwnershipZCategory",
    "CenterOwnershipZDistanceNm",
)

USABILITY_WALK_FIELDS = (
    "CenterOwnershipUsableComplete",
    "CenterOwnershipUsableCount",
    "CenterOwnershipHardRejectedCount",
    "TraceUsableLocusCount",
)

#: Ownership usability methods for analysis policy.
USABILITY_DISTANCE_AWARE = "distance_aware"
USABILITY_STRICT_VOXEL = "strict_voxel"

#: Salvage policies.
SALVAGE_SIMPLE_NEIGHBOR = "simple_neighbor_minimum"
SALVAGE_CALIBRATED_LOO = "calibrated_loo"


def _xy_projection_owned(effective_labels: np.ndarray, label: int,
                         yi: int, xi: int) -> bool:
    """Whether *label* appears at (yi, xi) for ANY z slice."""
    return bool(np.any(effective_labels[:, yi, xi] == label))


def _nearest_own_pixel_dist_2d(y: float, x: float,
                               effective_labels: np.ndarray,
                               label: int, search_radius: int = 3
                               ) -> float:
    """Min Euclidean pixel distance from (y, x) to the nearest own-label
    pixel in the 2D XY projection, searching a square window."""
    nz, ny, nx = effective_labels.shape
    yi = int(nearest_voxel(quantize_for_output(y)))
    xi = int(nearest_voxel(quantize_for_output(x)))
    yq = quantize_for_output(y)
    xq = quantize_for_output(x)
    best = float("inf")
    for dy in range(-search_radius, search_radius + 1):
        py = yi + dy
        if py < 0 or py >= ny:
            continue
        for dx in range(-search_radius, search_radius + 1):
            px = xi + dx
            if px < 0 or px >= nx:
                continue
            if np.any(effective_labels[:, py, px] == label):
                dist = ((float(yq) - py) ** 2 + (float(xq) - px) ** 2) ** 0.5
                if dist < best:
                    best = dist
    return best


def _z_gate_bounds(effective_labels: np.ndarray, label: int,
                   yi: int, xi: int, search_radius: int = 3
                   ) -> tuple[int, int] | None:
    """Z-gate bounds for *label* near (yi, xi).

    Uses the neighbourhood to be robust to boundary pixels.
    """
    nz, ny, nx = effective_labels.shape
    z_lo, z_hi = nz, -1
    for dy in range(-search_radius, search_radius + 1):
        py = yi + dy
        if py < 0 or py >= ny:
            continue
        for dx in range(-search_radius, search_radius + 1):
            px = xi + dx
            if px < 0 or px >= nx:
                continue
            col = effective_labels[:, py, px]
            hits = np.where(col == label)[0]
            if len(hits):
                z_lo = min(z_lo, int(hits[0]))
                z_hi = max(z_hi, int(hits[-1]))
    if z_hi < 0:
        return None
    return z_lo, z_hi


def classify_center_usable(
        effective_labels: np.ndarray, walk_id: int,
        z: float, y: float, x: float,
        xy_pixel_nm: float, z_step_nm: float,
        registered: bool,
        tolerance_px: float = 1.0,
) -> dict:
    """Canonical centre usability classification.

    Returns a dict with keys matching :data:`USABILITY_ROW_FIELDS`.

    For registered datasets: XY cell identity is the production decision.
    Exact own-mask and near-background (within *tolerance_px*) are usable.
    Z-gate position is diagnostic only.

    For unregistered datasets: every finite, in-bounds centre is usable
    with ``decision=unregistered_audit_only``.
    """
    labels = np.asarray(effective_labels)
    if labels.ndim != 3:
        raise ValueError(f"effective_labels must be 3D, got {labels.shape}")
    nz, ny, nx = labels.shape
    label = label_for_walk(walk_id)

    if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
        return usability_row_fields(
            DECISION_HARD_REJECT, XY_NONFINITE, float("nan"),
            Z_NONFINITE, float("nan"), reason=REASON_NONFINITE)

    zq = quantize_for_output(z)
    yq = quantize_for_output(y)
    xq = quantize_for_output(x)
    zi = int(nearest_voxel(zq))
    yi = int(nearest_voxel(yq))
    xi = int(nearest_voxel(xq))

    if not (0 <= yi < ny and 0 <= xi < nx):
        return usability_row_fields(
            DECISION_HARD_REJECT, XY_OUT_OF_BOUNDS, float("nan"),
            Z_OUT_OF_BOUNDS if not (0 <= zi < nz) else Z_INSIDE_GATE,
            float("nan"), reason=REASON_OUT_OF_BOUNDS_XY)

    if not (0 <= zi < nz):
        z_cat = Z_OUT_OF_BOUNDS
        z_dist_nm = float("nan")
    else:
        z_cat = Z_INSIDE_GATE
        z_dist_nm = 0.0

    # --- unregistered: audit-only, no hard exclusion ----------------------
    if not registered:
        xy_owned = _xy_projection_owned(labels, label, yi, xi)
        xy_cat = XY_EXACT_OWN if xy_owned else XY_NEAR_BACKGROUND
        xy_dist = 0.0 if xy_owned else (
            _nearest_own_pixel_dist_2d(y, x, labels, label) * xy_pixel_nm)
        bounds = _z_gate_bounds(labels, label, yi, xi)
        if bounds is not None:
            z_lo, z_hi = bounds
            if zi < z_lo:
                z_cat = Z_BELOW_GATE
                z_dist_nm = (z_lo - float(zq)) * z_step_nm
            elif zi > z_hi:
                z_cat = Z_ABOVE_GATE
                z_dist_nm = (float(zq) - z_hi) * z_step_nm
            else:
                z_cat = Z_INSIDE_GATE
                z_dist_nm = 0.0
        elif z_cat != Z_OUT_OF_BOUNDS:
            z_cat = Z_NO_SUPPORT
            z_dist_nm = float("nan")
        return usability_row_fields(
            DECISION_UNREGISTERED_AUDIT_ONLY, xy_cat, xy_dist,
            z_cat, z_dist_nm)

    # --- registered: XY is the cell-identity decision ---------------------
    xy_owned = _xy_projection_owned(labels, label, yi, xi)

    if xy_owned:
        xy_cat = XY_EXACT_OWN
        xy_dist_nm = 0.0
    else:
        found_at_xy = labels[:, yi, xi]
        other_present = np.any((found_at_xy != 0) & (found_at_xy != label))
        if other_present:
            xy_cat = XY_OTHER_CELL
            xy_dist_nm = _nearest_own_pixel_dist_2d(
                y, x, labels, label) * xy_pixel_nm
            return usability_row_fields(
                DECISION_HARD_REJECT, xy_cat, xy_dist_nm,
                z_cat, z_dist_nm, reason=REASON_OTHER_ACTIVE_CELL)
        pixel_dist = _nearest_own_pixel_dist_2d(y, x, labels, label)
        xy_dist_nm = pixel_dist * xy_pixel_nm
        if pixel_dist <= tolerance_px:
            xy_cat = XY_NEAR_BACKGROUND
        else:
            xy_cat = XY_FAR_BACKGROUND
            return usability_row_fields(
                DECISION_HARD_REJECT, xy_cat, xy_dist_nm,
                z_cat, z_dist_nm, reason=REASON_FAR_BACKGROUND)

    # XY is usable. Classify z separately (diagnostic, not a rejection).
    bounds = _z_gate_bounds(labels, label, yi, xi)
    if bounds is not None:
        z_lo, z_hi = bounds
        if zi < z_lo:
            z_cat = Z_BELOW_GATE
            z_dist_nm = (z_lo - float(zq)) * z_step_nm
        elif zi > z_hi:
            z_cat = Z_ABOVE_GATE
            z_dist_nm = (float(zq) - z_hi) * z_step_nm
        else:
            z_cat = Z_INSIDE_GATE
            z_dist_nm = 0.0
    elif z_cat != Z_OUT_OF_BOUNDS:
        z_cat = Z_NO_SUPPORT
        z_dist_nm = float("nan")

    decision = (DECISION_EXACT_OWN_MASK if xy_cat == XY_EXACT_OWN
                else DECISION_NEAR_BACKGROUND)
    return usability_row_fields(decision, xy_cat, xy_dist_nm,
                                z_cat, z_dist_nm)


def usability_row_fields(decision: str, xy_category: str,
                         xy_distance_nm: float, z_category: str,
                         z_distance_nm: float,
                         reason: str = "") -> dict:
    """Build the per-locus usability columns for one localization row."""
    import math as _math
    usable = int(decision in USABLE_DECISIONS)
    return {
        "CenterOwnershipUsable": usable,
        "CenterOwnershipDecision": decision,
        "CenterOwnershipHardRejectReason": reason if not usable else "",
        "CenterOwnershipXYCategory": xy_category,
        "CenterOwnershipXYDistanceNm": (
            "" if (_math.isnan(xy_distance_nm) if isinstance(xy_distance_nm, float)
                   else False)
            else format(xy_distance_nm, ".4f")),
        "CenterOwnershipZCategory": z_category,
        "CenterOwnershipZDistanceNm": (
            "" if (_math.isnan(z_distance_nm) if isinstance(z_distance_nm, float)
                   else False)
            else format(z_distance_nm, ".4f")),
    }


def blank_usability_row_fields() -> dict:
    """Blank usability fields for a failed fit (no centre to classify)."""
    return {field: "" for field in USABILITY_ROW_FIELDS}


def is_usable_flag(value) -> bool:
    """Truth of a serialized usability flag. Blank/absent is NOT usable."""
    text = str(value if value is not None else "").strip()
    if text == "" or text.lower() in ("na", "nan", "none"):
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return text.lower() in ("true", "yes", "y", "t")


USABLE_COLUMN = "CenterOwnershipUsable"
USABLE_COLUMN_CANDIDATES = ("CenterOwnershipUsable",)


def usability_verdict_of(row, usable_column: str = USABLE_COLUMN
                         ) -> bool | None:
    """Three-way usability verdict for one attempt row.

    True = usable, False = hard-rejected, None = no verdict (fit failed).
    """
    if _blank(row.get(usable_column)):
        return None
    return is_usable_flag(row.get(usable_column))


def walk_usability_summary(rows, locus_order, *, walk_key="WalkID",
                           tp_key="tp") -> dict:
    """Walk-level usability fields over the configured locus universe.

    Returns ``{walk_id: {field: value}}`` over :data:`USABILITY_WALK_FIELDS`.
    """
    expected = sorted({int(index) for index in locus_order})
    if not expected:
        raise ValueError("configured locus universe is empty")
    expected_set = set(expected)
    n_expected = len(expected_set)

    rows = list(rows)
    if not rows:
        return {}

    has_usable = any(USABLE_COLUMN in row for row in rows[:1])
    has_valid = any(
        name in row for row in rows[:1]
        for name in VALID_COLUMN_CANDIDATES)
    if not has_usable and not has_valid:
        return {}

    valid_col = None
    if has_valid:
        try:
            valid_col = detect_valid_column(rows)
        except OwnershipSchemaMissing:
            pass

    # (walk, locus) -> (usable, exact_valid)
    verdicts: dict[tuple[int, int], tuple[bool, bool]] = {}
    walks: set[int] = set()
    n_unparsed = 0
    n_rows = 0
    for row in rows:
        n_rows += 1
        try:
            walk = int(str(row[walk_key]).strip())
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"usability rows have no usable {walk_key!r} column") from None
        walks.add(walk)
        try:
            locus = tp_index(row[tp_key])
        except (KeyError, ValueError):
            n_unparsed += 1
            continue
        if locus not in expected_set:
            continue
        usable = usability_verdict_of(row) if has_usable else None
        exact = verdict_of(row, valid_col) if valid_col else None
        if usable is None and exact is None:
            continue
        if usable is None:
            usable = exact
        if exact is None:
            exact = False
        key = (walk, locus)
        if key in verdicts:
            prev_u, prev_e = verdicts[key]
            if prev_u != usable:
                raise ValueError(
                    f"contradictory usability verdicts for WalkID={walk} "
                    f"TP{locus:03d}")
        verdicts[key] = (usable, exact)

    if n_rows and n_unparsed == n_rows:
        raise ValueError(
            f"none of the {n_rows} usability rows had a parseable "
            f"{tp_key!r} label")

    summary = {}
    for walk in sorted(walks):
        usable_loci = {locus for (w, locus), (u, _) in verdicts.items()
                       if w == walk and u}
        exact_loci = {locus for (w, locus), (_, e) in verdicts.items()
                      if w == walk and e}
        rejected_loci = {locus for (w, locus), (u, _) in verdicts.items()
                         if w == walk and not u}
        usable_count = len(usable_loci & expected_set)
        exact_count = len(exact_loci & expected_set)
        rejected_count = len(rejected_loci & expected_set)
        summary[walk] = {
            "CenterOwnershipUsableComplete": int(usable_loci >= expected_set),
            "CenterOwnershipUsableCount": usable_count,
            "CenterOwnershipHardRejectedCount": rejected_count,
            "TraceUsableLocusCount": usable_count,
        }
    return summary


def _main() -> None:
    import argparse
    import csv
    import os

    ap = argparse.ArgumentParser(
        description="Aggregate the per-attempt centre-ownership status table "
                    "into walk-level CenterOwnershipComplete.")
    ap.add_argument("--status_csv", required=True,
                    help="localizations_640_z_gate_status.csv from localize_probe0.")
    ap.add_argument("--out_csv", default=None,
                    help="Default: <status_csv dir>/"
                         "localizations_640_center_ownership_walks.csv")
    ap.add_argument("--replication_tp_start", type=int, default=2,
                    help="First 0-based genomic locus TP index (inclusive).")
    ap.add_argument("--replication_tp_end", type=int, default=21,
                    help="Last 0-based genomic locus TP index (inclusive).")
    args = ap.parse_args()

    with open(args.status_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = walk_completeness(
        rows, args.replication_tp_start, args.replication_tp_end)

    out_csv = args.out_csv or os.path.join(
        os.path.dirname(os.path.abspath(args.status_csv)),
        "localizations_640_center_ownership_walks.csv")
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WALK_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    n_complete = sum(r["CenterOwnershipComplete"] for r in summary)
    print(f"Loci interval: TP{args.replication_tp_start:03d}.."
          f"TP{args.replication_tp_end:03d} "
          f"({args.replication_tp_end - args.replication_tp_start + 1} loci)")
    print(f"Walks: {len(summary):,}   CenterOwnershipComplete: {n_complete:,}"
          + (f" ({n_complete / len(summary) * 100:.2f}%)" if summary else ""))
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    _main()
