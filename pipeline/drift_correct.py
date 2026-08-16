#!/usr/bin/env python
"""Drift correction for 640 localizations using 488 as fiducial reference.

Pipeline overview
-----------------
1. Load matched 488 (probe_1) and 640 (probe_0) localizations per WalkID
   (identity defined by the shared DAPI mask).
2. Use the intersection of valid timepoints between the two probes.
3. Sample ~N_FIDUCIALS 488 ROIs spread uniformly across the FOV via
   farthest-point sampling. Eligibility requires the ROI to be fit in the
   reference TP, present in >= half the TPs, and not within EDGE_PAD of the
   FOV edge.
4. PASS 1 — estimate drift with all sampled fiducials:
     a. For each TP, compute per-ROI displacements relative to the reference TP.
     b. MAD-based outlier rejection per axis.
     c. Robust median (optionally weighted by 1/err^2) gives the per-TP drift.
5. Compute per-ROI stability metrics from pass-1 residuals (tp_coverage,
   outlier_frac, med_abs_res_{x,y,z}, mad_res_{x,y,z}).
6. Apply explicit stability criteria (STABLE_* constants; tp_coverage is
   controllable via the --tp_coverage CLI flag) to filter fiducials. If
   fewer than STABLE_MIN_FIDUCIALS pass, thresholds are multiplicatively
   relaxed; worst case the best-scoring ROIs are kept as a fallback.
7. PASS 2 — re-estimate drift with the stable fiducial subset (a, b, c).
8. Flag per-axis low-confidence TPs using retained-count and robust-spread
   thresholds.
9. Apply axis-specific smoothing to the robust per-TP drift:
     - X/Y: raw robust per-TP drift. Low-confidence TPs are linearly
       interpolated from neighboring valid TPs. No global smoothing.
     - Z:   degree-Z_POLY_DEGREE polynomial fit through valid per-TP drift
       (low-confidence TPs excluded from the fit). Simpler than a cubic
       spline and harder to overfit — matters because Z has only 11 slices
       and noisier fits.
10. Apply the final (dz, dy, dx) per TP to every 640 localization.

Conventions
-----------
- drift[TP_t] = current_488 - reference_488   (per ROI, then robust aggregate)
- corrected_640 = raw_640 - drift[TP_t]
- Units: XY in pixels (1 px = XY_PIXEL_NM nm).  Z in z-slice index (1 = Z_STEP_NM nm).
- Timepoint ordering is the integer `###` in TP###_CYC###, not lexical.
- Axis-triple ordering is (Z, Y, X) throughout — see AXIS_Z / AXIS_Y / AXIS_X.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:                                        # canonical ownership column names
    from center_ownership import OWNERSHIP_ROW_FIELDS as _OWNERSHIP_ROW_FIELDS
    from center_ownership import USABILITY_ROW_FIELDS as _USABILITY_ROW_FIELDS
except ImportError:                         # pragma: no cover - standalone use
    _OWNERSHIP_ROW_FIELDS = (
        'CenterOwnershipPolicy', 'CenterOwnershipInitial', 'CenterOwnershipFinal',
        'CenterOwnershipValid', 'CenterOwnershipFailureReason',
        'CenterRefitAttempted', 'CenterRefitAccepted', 'OwnershipCoordinateFrame',
        'ChromaticRegistrationApplied',
        'ChromaticRegistrationReferenceWavelengthNm',
        'ChromaticRegistrationTargetWavelengthNm')
    _USABILITY_ROW_FIELDS = (
        'CenterOwnershipUsable', 'CenterOwnershipDecision',
        'CenterOwnershipHardRejectReason',
        'CenterOwnershipXYCategory', 'CenterOwnershipXYDistanceNm',
        'CenterOwnershipZCategory', 'CenterOwnershipZDistanceNm')


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Output root. Env-overridable (PIPELINE_OUT) so independent pipeline runs can
# proceed concurrently in their own trees; unset reproduces the historical
# hardcoded path byte-for-byte. Set by run_pipeline_dat.sh /
# run_pipeline_confocal.sh, which no longer symlink $SALMO/output.
PIPELINE_OUT = os.environ.get('PIPELINE_OUT', '/mnt/altnas/work/dimos/Salmo_WF/output')
PROBE0_LOC = os.path.join(PIPELINE_OUT, 'probe_0', 'localizations',
                          'localizations_640.csv')
PROBE1_LOC = os.path.join(PIPELINE_OUT, 'probe_1', 'localizations',
                          'localizations_488.csv')
# Replication classification is a SEPARATE step (classify_replication.py, using
# replication_detection.py). Drift correction registers coordinates only; it no
# longer reads the replication side-car or emits replication columns.
# Default output directory. Overridable at the CLI via --outdir. All three
# CSVs (drift-corrected, full-coverage, movement-filtered), the QC PNGs, and
# the follow-up Moran's I / parabola / distance-matrix outputs land here.
OUTDIR_DEFAULT = os.path.join(PIPELINE_OUT, 'probe_0', 'localizations')
OUTDIR         = OUTDIR_DEFAULT

# Axis-triple ordering used throughout:
#   per_roi[tp] columns after wid are (dz, dy, dx, ez, ey, ex)
#   drift[tp] / drift_final[tp] / low_conf[tp] / spread[tp]  are (z, y, x)
# Use these constants at call sites that would otherwise hardcode 0/1/2.
AXIS_Z, AXIS_Y, AXIS_X = 0, 1, 2

# Fiducial selection
N_FIDUCIALS   = 200
Z_STEP_NM     = 100.0     # nm per z-slice
XY_PIXEL_NM   = 103.28    # nm per xy pixel (Vutara config PixelSizeX/Y 102.97/103.59, mean; was 108)
EDGE_PAD      = 100       # px margin from FOV edges for fiducial pool
FOV_SIZE      = 1866      # image is 1866 x 1866 px

# Robust drift estimation
MAD_K               = 5.0   # MAD-based outlier rejection threshold
MIN_KEPT_ROIS       = 10    # minimum retained ROIs per axis for a TP to be usable
USE_FIT_ERR_WEIGHTS = True  # if True: use weighted median (1/err^2) after MAD filter

# ---- Stable-ROI criteria (all in nm or fractions) ----
# An ROI passes the stability filter iff ALL criteria are satisfied.
# Changing these defaults is the intended tuning knob; they should not be
# duplicated elsewhere in the code. TP coverage is the CLI default for
# --tp_coverage and is the single source of truth for that threshold.

STABLE_TP_COVERAGE_MIN       = 1.0     # fraction: present in >= this fraction of TPs (CLI default)
STABLE_OUTLIER_FRAC_MAX      = 0.15    # at most this fraction of TPs where ROI was MAD-rejected
STABLE_MED_ABS_RES_XY_NM     = 70.0    # median |corrected XY residual| <= this (nm)
STABLE_MED_ABS_RES_Z_NM      = 140.0   # median |corrected Z residual| <= this (nm)
STABLE_MAD_RES_XY_NM         = 80.0    # MAD-sigma of corrected XY residual <= this (nm)
STABLE_MAD_RES_Z_NM          = 150.0   # MAD-sigma of corrected Z residual <= this (nm)
STABLE_MIN_FIDUCIALS          = 50      # hard floor: if fewer pass, relax thresholds (see below)
STABLE_RELAX_FACTOR           = 1.5     # multiplicative relaxation of nm thresholds per round

# Low-confidence TP flagging (per axis)
LOW_CONF_NKEPT_FRAC       = 0.5    # low conf if n_kept[axis] < frac * median(n_kept[axis])
LOW_CONF_SPREAD_MULT      = 3.0    # low conf if drift MAD-sigma > mult * median(drift MAD-sigma)

# Axis-specific smoothing
# X/Y use the raw robust per-TP drift; low-confidence TPs are linearly interpolated.
# Z uses a low-degree polynomial fit to avoid overfitting its noisy per-TP estimates.
Z_POLY_DEGREE = 3

# ---- Final movement filter on the full-coverage WalkIDs ----
# Drop any WalkID whose drift-corrected 3D Euclidean displacement between
# MOVEMENT_TP_FROM_1BASED and MOVEMENT_TP_TO_1BASED exceeds the movement
# threshold. The threshold is supplied at the CLI as --movement_max_nm
# (default MOVEMENT_MAX_NM_DEFAULT). TP numbering here is 1-based (e.g.
# 3 == the 3rd TP by numeric index, which maps to filename TP002).
# Physically: removes cells that moved too much between the two reference
# TPs.
MOVEMENT_TP_FROM_1BASED  = 3
MOVEMENT_TP_TO_1BASED    = 23
MOVEMENT_MAX_NM_DEFAULT  = 100.0

# Repeatability QC is a POLICY. It used to be expressed as a threshold, and
# "not applicable" was expressed as --movement_max_nm 1e9, which made every walk
# compare favourably against an absurd bound and then let the output be
# described as having PASSED movement QC. It had not: nothing was measured. The
# three policies below are exhaustive and mutually exclusive.
MOVEMENT_POLICY_XY = 'xy_repeatability'
MOVEMENT_POLICY_XYZ_LEGACY = 'xyz_repeatability_legacy'
MOVEMENT_POLICY_DISABLED = 'disabled'
#: The axes each measuring policy uses. `disabled` measures nothing, so it maps
#: to no axes rather than to a default set.
MOVEMENT_POLICY_AXES = {
    MOVEMENT_POLICY_XY: 'xy',
    MOVEMENT_POLICY_XYZ_LEGACY: 'xyz',
    MOVEMENT_POLICY_DISABLED: '',
}
#: Values of the `movement_qc_status` column. `not_evaluated` is not a pass.
MOVEMENT_QC_EVALUATED = 'evaluated'
MOVEMENT_QC_NOT_EVALUATED = 'not_evaluated'


def resolve_movement_policy(policy, axes, max_nm):
    """The policy, its axes and its threshold, or a hard error.

    Returns ``(policy, axes, max_nm)`` where ``max_nm`` is None exactly when the
    policy is `disabled`. Inconsistent combinations are refused rather than
    silently resolved, because each of them has a plausible-but-wrong reading:
    a threshold supplied alongside `disabled` invites the 1e9 trick back, and
    axes contradicting the policy would make `movement_def` describe something
    other than what was computed.
    """
    if policy is None:
        # Legacy invocation: no policy given, so derive one from the axes. This
        # is the ONLY place the old flag still decides anything.
        policy = (MOVEMENT_POLICY_XY if axes == 'xy'
                  else MOVEMENT_POLICY_XYZ_LEGACY)
    expected_axes = MOVEMENT_POLICY_AXES[policy]
    if policy == MOVEMENT_POLICY_DISABLED:
        if max_nm is not None:
            raise ValueError(
                f'--movement_max_nm {max_nm} was given with --movement_policy '
                f'{MOVEMENT_POLICY_DISABLED}. A disabled test has no '
                'threshold: nothing is compared, so no bound can apply. Pass '
                'no threshold, or choose a measuring policy.')
        return policy, expected_axes, None
    if axes is not None and axes != expected_axes:
        raise ValueError(
            f'--movement_axes {axes} contradicts --movement_policy {policy}, '
            f'which measures {expected_axes}. Remove --movement_axes, or '
            'select the policy that matches the axes you intend.')
    if max_nm is None:
        max_nm = MOVEMENT_MAX_NM_DEFAULT
    return policy, expected_axes, max_nm

# ---- Follow-up post-filter QC / analysis range (1-based, inclusive) ----
# These control the TP range used by the parabola QC and distance-matrix
# follow-up analysis that runs after the final filtering. They do NOT affect
# drift correction or the full-coverage / movement filters themselves.
ANALYSIS_TP_START_1BASED_DEFAULT = 3
ANALYSIS_TP_END_1BASED_DEFAULT   = 23

OUT_CSV = os.path.join(OUTDIR, 'localizations_640_drift_corrected.csv')



# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_localizations(path: str) -> list[dict]:
    """Load a localizations CSV and normalize WalkID + numeric columns."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if 'WalkID' in row:
                row['WalkID'] = int(row['WalkID'])
            elif 'cell_id' in row:
                row['WalkID'] = int(row['cell_id']) - 1
                del row['cell_id']
            else:
                raise KeyError(f"No WalkID / cell_id column in {path}")
            for k in ('z', 'y', 'x'):
                try:
                    row[k] = float(row[k])
                except (ValueError, TypeError):
                    raise ValueError(f"{path}: non-numeric {k}={row.get(k)!r} "
                                     f"(tp={row.get('tp')}, WalkID={row.get('WalkID')})")
            for k in ('z_err', 'y_err', 'x_err'):
                if k in row and row[k] not in ('', None):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = np.nan
            rows.append(row)
    return rows


# Accept both TP### and TP###_CYC### — CYC defaults to 0 when absent.
TP_RE = re.compile(r'^TP(\d+)(?:_CYC(\d+))?$')

def tp_index(tp_name: str) -> int:
    """Extract integer TP index from a name like 'TP007_CYC000'."""
    m = TP_RE.match(tp_name)
    if m is None:
        raise ValueError(f"Unrecognized TP name: {tp_name!r}")
    return int(m.group(1))


def ordered_tps(tps: list[str]) -> list[str]:
    """Return TPs sorted numerically. Warn if numeric vs lexical sort disagree."""
    numeric = sorted(tps, key=tp_index)
    lexical = sorted(tps)
    if numeric != lexical:
        print("WARNING: numeric and lexical TP orderings differ; using numeric.",
              file=sys.stderr)
    return numeric


# ----------------------------------------------------------------------------
# Robust helpers
# ----------------------------------------------------------------------------
def _mad_sigma(values: np.ndarray) -> float:
    """Robust sigma estimate = 1.4826 * MAD. Returns nan for empty input."""
    if len(values) == 0:
        return float('nan')
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return 1.4826 * mad


def robust_median_mad(values: np.ndarray, k: float) -> tuple[float, np.ndarray]:
    """(median, keep_mask) with MAD-based outlier rejection."""
    if len(values) == 0:
        return np.nan, np.zeros(0, dtype=bool)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad == 0:
        return med, np.ones(len(values), dtype=bool)
    sigma_est = 1.4826 * mad
    keep = np.abs(values - med) <= k * sigma_est
    return med, keep


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted 1-D median; falls back to unweighted median if weights unusable.

    When the cumulative weight lands exactly at total/2 (a tie between two
    adjacent sorted values), return the average of those two values — matches
    numpy's convention for even-length unweighted medians.
    """
    if len(values) == 0:
        return np.nan
    w = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(w)) or np.sum(w) <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    half = cw[-1] / 2.0
    idx = int(np.searchsorted(cw, half))
    idx = min(idx, len(v_sorted) - 1)
    # Tie case: if cumulative weight equals half exactly and there is a next
    # value, average the two straddling values.
    if idx + 1 < len(v_sorted) and np.isclose(cw[idx], half):
        return 0.5 * (float(v_sorted[idx]) + float(v_sorted[idx + 1]))
    return float(v_sorted[idx])


# ----------------------------------------------------------------------------
# Fiducial selection
# ----------------------------------------------------------------------------
def select_fiducials(pos_488: dict, all_tps: list[str], ref_tp: str,
                     n_fiducials: int, edge_pad: int, fov_size: int) -> set:
    """Pick ~n_fiducials WalkIDs spread uniformly across the FOV (XY).

    Eligibility:
      - has 488 at the reference TP
      - has 488 in >= half of all TPs
      - >= edge_pad pixels from any FOV edge (in both X and Y)
    Selection: farthest-point sampling starting from the FOV center.
    """
    tp_count = defaultdict(int)
    for (wid, tp) in pos_488:
        tp_count[wid] += 1

    at_ref = {wid for (wid, tp) in pos_488 if tp == ref_tp}
    print(f'Cells with 488 at {ref_tp}: {len(at_ref)}')

    min_tps = max(1, len(all_tps) // 2)
    enough_tps = {wid for wid in at_ref if tp_count[wid] >= min_tps}
    print(f'  ... present in >= {min_tps} TPs: {len(enough_tps)}')

    inside = set()
    for wid in enough_tps:
        z, y, x, *_ = pos_488[(wid, ref_tp)]
        if edge_pad <= x <= fov_size - edge_pad and edge_pad <= y <= fov_size - edge_pad:
            inside.add(wid)
    print(f'  ... >= {edge_pad}px from FOV edge: {len(inside)}')

    if len(inside) == 0:
        raise RuntimeError("No fiducials eligible after filters — cannot proceed.")

    wids = np.array(sorted(inside))
    positions = np.array([(pos_488[(w, ref_tp)][2], pos_488[(w, ref_tp)][1]) for w in wids])

    fov_center = positions.mean(axis=0)
    start_idx = int(np.argmin(np.sum((positions - fov_center) ** 2, axis=1)))
    selected = [start_idx]
    min_dist = np.sum((positions - positions[start_idx]) ** 2, axis=1)

    target = min(n_fiducials, len(wids))
    while len(selected) < target:
        next_idx = int(np.argmax(min_dist))
        # Belt-and-suspenders: with floating-point ROI positions this never
        # fires in practice, but it guards against coincident/duplicate points
        # that would otherwise let us "pick" the same ROI twice.
        if min_dist[next_idx] <= 0:
            break
        selected.append(next_idx)
        new_dist = np.sum((positions - positions[next_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    fiducial_wids = {int(wids[i]) for i in selected}
    print(f'Selected {len(fiducial_wids)} fiducials (farthest-point sampling)')
    return fiducial_wids


# ----------------------------------------------------------------------------
# Robust per-TP drift estimation
# ----------------------------------------------------------------------------
def _errs_from_entry(entry: tuple, axis_idx: int) -> float | None:
    """Return the fit error for (z=0 / y=1 / x=2) if present in the tuple.

    pos_488 entry layout: (z, y, x[, z_err, y_err, x_err])
    Returns None if the error is missing or non-finite.
    """
    if len(entry) < 6:
        return None
    err = entry[3 + axis_idx]
    return None if err is None or not np.isfinite(err) or err <= 0 else float(err)


def _err_or_nan(entry: tuple, axis_idx: int) -> float:
    """Thin wrapper: float fit error if available, else np.nan.

    Avoids the `_errs_from_entry(...) or np.nan` footgun (which would also
    map a legitimate 0.0 to nan — harmless today but a bug-in-waiting).
    """
    e = _errs_from_entry(entry, axis_idx)
    return np.nan if e is None else e


def _robust_axis_estimate(vals: np.ndarray, errs: np.ndarray,
                          keep: np.ndarray, use_err_weights: bool) -> float:
    """Robust per-axis estimator used by estimate_drift_pass.

    Applies *keep* (post-MAD mask) then either:
      - returns the unweighted median of retained values, or
      - returns the weighted median with weights = 1/err^2, falling back to
        unweighted median when no finite weights are available.
    """
    v = vals[keep]
    if v.size == 0:
        return np.nan
    if not use_err_weights:
        return float(np.median(v))
    e = errs[keep]
    w = np.where(np.isfinite(e) & (e > 0), 1.0 / (e * e), np.nan)
    if np.all(~np.isfinite(w)):
        return float(np.median(v))
    mask = np.isfinite(w)
    return weighted_median(v[mask], w[mask])


def estimate_drift_pass(pos_488: dict, fiducials: set, all_tps: list[str],
                        ref_tp: str, mad_k: float, min_kept: int,
                        use_err_weights: bool):
    """One pass of robust per-TP drift estimation.

    Returns:
        drift_raw[tp] = (dz, dy, dx) estimator or (nan,) if unreliable
        n_kept[tp]    = (n_z, n_y, n_x) retained after MAD filter
        spread[tp]    = (sigma_z, sigma_y, sigma_x) MAD-sigma of retained displacements (px)
        per_roi[tp]   = np.ndarray shape (N, 7) columns: wid, dz, dy, dx, ez, ey, ex
                        (errors are nan if unavailable)
        per_tp_keep[tp] = (keep_z, keep_y, keep_x) bool arrays aligned with per_roi[tp]
                        — True means the row survived the per-TP MAD filter on that axis
    """
    drift_raw   = {}
    n_kept      = {}
    spread      = {}
    per_roi     = {}
    per_tp_keep = {}

    for tp in all_tps:
        wid_list = []
        dz_list, dy_list, dx_list = [], [], []
        ez_list, ey_list, ex_list = [], [], []
        for wid in fiducials:
            ref_entry = pos_488.get((wid, ref_tp))
            cur_entry = pos_488.get((wid, tp))
            if ref_entry is None or cur_entry is None:
                continue
            rz, ry, rx = ref_entry[:3]
            cz, cy, cx = cur_entry[:3]
            wid_list.append(wid)
            dz_list.append(cz - rz); dy_list.append(cy - ry); dx_list.append(cx - rx)
            # Use the current-TP fit errors as the weighting source
            ez_list.append(_err_or_nan(cur_entry, AXIS_Z))
            ey_list.append(_err_or_nan(cur_entry, AXIS_Y))
            ex_list.append(_err_or_nan(cur_entry, AXIS_X))

        dz_arr = np.asarray(dz_list, dtype=float)
        dy_arr = np.asarray(dy_list, dtype=float)
        dx_arr = np.asarray(dx_list, dtype=float)
        ez_arr = np.asarray(ez_list, dtype=float)
        ey_arr = np.asarray(ey_list, dtype=float)
        ex_arr = np.asarray(ex_list, dtype=float)

        if len(wid_list):
            per_roi[tp] = np.column_stack([wid_list, dz_arr, dy_arr, dx_arr,
                                           ez_arr, ey_arr, ex_arr])
        else:
            per_roi[tp] = np.zeros((0, 7))

        # MAD filter per axis
        _, keep_z = robust_median_mad(dz_arr, mad_k)
        _, keep_y = robust_median_mad(dy_arr, mad_k)
        _, keep_x = robust_median_mad(dx_arr, mad_k)
        n_kept[tp] = (int(keep_z.sum()), int(keep_y.sum()), int(keep_x.sum()))
        per_tp_keep[tp] = (keep_z, keep_y, keep_x)

        # Robust estimator on retained values (hoisted helper)
        z_est = _robust_axis_estimate(dz_arr, ez_arr, keep_z, use_err_weights)
        y_est = _robust_axis_estimate(dy_arr, ey_arr, keep_y, use_err_weights)
        x_est = _robust_axis_estimate(dx_arr, ex_arr, keep_x, use_err_weights)

        # Robust spread of retained displacements (MAD-sigma, pixel units)
        s_z = _mad_sigma(dz_arr[keep_z]) if keep_z.any() else np.nan
        s_y = _mad_sigma(dy_arr[keep_y]) if keep_y.any() else np.nan
        s_x = _mad_sigma(dx_arr[keep_x]) if keep_x.any() else np.nan
        spread[tp] = (s_z, s_y, s_x)

        if min(n_kept[tp]) < min_kept:
            drift_raw[tp] = (np.nan, np.nan, np.nan)
        else:
            drift_raw[tp] = (z_est, y_est, x_est)

    return drift_raw, n_kept, spread, per_roi, per_tp_keep


# ----------------------------------------------------------------------------
# Stable-ROI scoring and filtering
# ----------------------------------------------------------------------------
def compute_roi_stability_metrics(per_roi: dict, drift: dict, all_tps: list[str],
                                  roi_pool: set) -> dict[int, dict]:
    """Compute per-ROI stability metrics from pass-1 corrected residuals.

    For each ROI in *roi_pool*, the returned dict maps WalkID → metric dict:
      tp_coverage      : fraction of TPs where this ROI had a displacement
      outlier_frac_xy  : fraction of TPs where it was MAD-rejected on X or Y
      outlier_frac_z   : fraction of TPs where it was MAD-rejected on Z
      med_abs_res_x    : median |corrected X residual| (px)
      med_abs_res_y    : median |corrected Y residual| (px)
      med_abs_res_z    : median |corrected Z residual| (px)
      mad_res_x        : MAD-sigma of corrected X residual (px)
      mad_res_y        : MAD-sigma of corrected Y residual (px)
      mad_res_z        : MAD-sigma of corrected Z residual (px)

    "Corrected residual" for ROI *i* at TP *t* is defined as:
        residual_i(t) = [pos_i(t) - pos_i(ref)] - drift(t)
    i.e. the per-ROI displacement minus the estimated global drift.

    Outlier tracking is now split XY vs Z so that an ROI that is noisy only
    in Z can still be considered XY-stable.
    """
    n_tps = len(all_tps)

    res_by_roi = defaultdict(lambda: {
        'rz': [], 'ry': [], 'rx': [],
        'n_seen': 0, 'n_outlier_xy': 0, 'n_outlier_z': 0,
    })

    for tp in all_tps:
        arr = per_roi[tp]
        dz, dy, dx = drift.get(tp, (np.nan, np.nan, np.nan))
        if arr.shape[0] == 0 or not (np.isfinite(dz) and np.isfinite(dy) and np.isfinite(dx)):
            continue

        # Per-TP MAD keep masks (same as estimate_drift_pass)
        dz_arr = arr[:, 1]; dy_arr = arr[:, 2]; dx_arr = arr[:, 3]
        _, keep_z = robust_median_mad(dz_arr, MAD_K)
        _, keep_y = robust_median_mad(dy_arr, MAD_K)
        _, keep_x = robust_median_mad(dx_arr, MAD_K)

        for i in range(arr.shape[0]):
            wid = int(arr[i, 0])
            if wid not in roi_pool:
                continue
            entry = res_by_roi[wid]
            entry['n_seen'] += 1
            entry['rz'].append(arr[i, 1] - dz)
            entry['ry'].append(arr[i, 2] - dy)
            entry['rx'].append(arr[i, 3] - dx)
            if not (keep_y[i] and keep_x[i]):
                entry['n_outlier_xy'] += 1
            if not keep_z[i]:
                entry['n_outlier_z'] += 1

    metrics = {}
    for wid in roi_pool:
        entry = res_by_roi.get(wid)
        if entry is None or entry['n_seen'] == 0:
            metrics[wid] = {
                'tp_coverage': 0.0,
                'outlier_frac_xy': 1.0, 'outlier_frac_z': 1.0,
                'med_abs_res_x': np.inf, 'med_abs_res_y': np.inf, 'med_abs_res_z': np.inf,
                'mad_res_x': np.inf, 'mad_res_y': np.inf, 'mad_res_z': np.inf,
            }
            continue

        rz = np.asarray(entry['rz']); ry = np.asarray(entry['ry']); rx = np.asarray(entry['rx'])
        metrics[wid] = {
            'tp_coverage':      entry['n_seen'] / n_tps,
            'outlier_frac_xy':  entry['n_outlier_xy'] / entry['n_seen'],
            'outlier_frac_z':   entry['n_outlier_z']  / entry['n_seen'],
            'med_abs_res_x':    float(np.median(np.abs(rx))),
            'med_abs_res_y':    float(np.median(np.abs(ry))),
            'med_abs_res_z':    float(np.median(np.abs(rz))),
            'mad_res_x':        _mad_sigma(rx),
            'mad_res_y':        _mad_sigma(ry),
            'mad_res_z':        _mad_sigma(rz),
        }
    return metrics


def filter_stable_rois(metrics: dict[int, dict], *, axis_group: str,
                       tp_cov_min, outlier_frac_max,
                       med_nm, mad_nm,
                       min_stable, relax_factor) -> tuple[set, dict[str, int]]:
    """Return WalkIDs that pass stability criteria for ONE axis group.

    axis_group: 'xy' or 'z'.  XY criteria use max(x, y) metrics;
                Z criteria use the Z-only metrics.

    Common criteria (applied first):
      tp_coverage    >= tp_cov_min
      outlier_frac_* <= outlier_frac_max   (XY or Z version, selected by group)

    Axis-group-specific criteria:
      med_abs_res_*  <= med_nm / unit
      mad_res_*      <= mad_nm / unit

    Graceful degradation: if fewer than *min_stable* pass, nm thresholds
    (NOT tp_coverage or outlier_frac) are relaxed by *relax_factor* per
    round, up to 10 rounds. Final fallback keeps the best *min_stable*
    ROIs by the axis-group composite residual score.

    Returns (stable_wids, rejection_breakdown).
    """
    if axis_group == 'xy':
        unit_nm = XY_PIXEL_NM
        outlier_key = 'outlier_frac_xy'
        def residual_score(m):
            return max(m['med_abs_res_x'], m['med_abs_res_y'])
        def mad_score(m):
            return max(m['mad_res_x'], m['mad_res_y'])
    elif axis_group == 'z':
        unit_nm = Z_STEP_NM
        outlier_key = 'outlier_frac_z'
        def residual_score(m):
            return m['med_abs_res_z']
        def mad_score(m):
            return m['mad_res_z']
    else:
        raise ValueError(f"axis_group must be 'xy' or 'z', got {axis_group!r}")

    med_thresh = med_nm / unit_nm
    mad_thresh = mad_nm / unit_nm

    round_num = 0
    max_rounds = 10

    while round_num < max_rounds:
        pool = set(metrics.keys())
        breakdown = {}

        def _apply(name, pred):
            nonlocal pool
            before = len(pool)
            pool = {w for w in pool if pred(metrics[w])}
            breakdown[name] = before - len(pool)

        _apply('tp_coverage',  lambda m: m['tp_coverage'] >= tp_cov_min)
        _apply('outlier_frac', lambda m: m[outlier_key] <= outlier_frac_max)
        _apply('med_abs_res',  lambda m: residual_score(m) <= med_thresh)
        _apply('mad_res',      lambda m: mad_score(m) <= mad_thresh)

        if len(pool) >= min_stable:
            return pool, breakdown

        round_num += 1
        med_thresh *= relax_factor
        mad_thresh *= relax_factor
        print(f'  [stable-ROI/{axis_group}] Only {len(pool)} pass (min {min_stable}); '
              f'relaxing nm thresholds x{relax_factor} (round {round_num})')

    print(f'  [stable-ROI/{axis_group}] WARNING: still <{min_stable} after {max_rounds} rounds; '
          f'FALLING BACK to best {min_stable} by composite residual score — the tp_coverage and '
          f'outlier_frac guarantees are ABANDONED for this axis (drift estimated from possibly-poor fiducials)')
    scores = {w: residual_score(m) for w, m in metrics.items()}
    ranked = sorted(scores, key=scores.get)
    pool = set(ranked[:min_stable])
    return pool, breakdown


# ----------------------------------------------------------------------------
# Low-confidence TP flags
# ----------------------------------------------------------------------------
def flag_low_confidence(n_kept: dict, spread: dict, all_tps: list[str],
                        nkept_frac: float, spread_mult: float) -> dict:
    """Return {tp: (low_z, low_y, low_x)} boolean per-axis flags.

    A TP/axis is flagged low-confidence if:
      n_kept[axis] < nkept_frac * median(n_kept[axis] across TPs)
      OR
      spread[axis] > spread_mult * median(spread[axis] across TPs).
    """
    nk = np.array([n_kept[tp] for tp in all_tps], dtype=float)    # shape (T, 3) z,y,x
    sp = np.array([spread[tp] for tp in all_tps], dtype=float)    # shape (T, 3)

    med_nk = np.array([np.median(nk[:, a]) for a in range(3)])
    med_sp = np.array([np.nanmedian(sp[:, a]) for a in range(3)])

    low = {}
    for i, tp in enumerate(all_tps):
        flags = []
        for a in range(3):
            bad_n = nk[i, a] < nkept_frac * med_nk[a]
            bad_s = np.isfinite(sp[i, a]) and sp[i, a] > spread_mult * med_sp[a]
            flags.append(bool(bad_n or bad_s))
        # returned order is (z, y, x)
        low[tp] = tuple(flags)
    return low


# ----------------------------------------------------------------------------
# Axis-specific smoothing
# ----------------------------------------------------------------------------
def smooth_xy_with_interp(all_tps: list[str], drift_raw: dict, low_conf: dict,
                          ref_tp: str, axis_idx: int) -> tuple[dict, dict]:
    """Use raw robust per-TP drift for X/Y.

    Replaces:
      - NaN values (TP with < MIN_KEPT_ROIS retained)
      - Low-confidence TPs
    with linear interpolation across the valid (TP_index, drift) grid. The
    reference TP is forced to 0.

    Returns (smooth_dict[tp] = value_px, source_dict[tp] = 'raw'|'interpolated'|'reference').
    """
    t = np.array([tp_index(tp) for tp in all_tps])
    y = np.array([drift_raw[tp][axis_idx] for tp in all_tps], dtype=float)
    low = np.array([low_conf[tp][axis_idx] for tp in all_tps], dtype=bool)

    # Mark invalid: NaN or low-confidence (except the reference TP which is 0 by def)
    ref_t = tp_index(ref_tp)
    valid = np.isfinite(y) & (~low)
    valid[t == ref_t] = True  # always include reference
    if (t == ref_t).any():
        y[t == ref_t] = 0.0

    if valid.sum() < 2:
        raise RuntimeError(
            f"Axis {axis_idx}: only {valid.sum()} valid TPs for interpolation. "
            "Check upstream 488 fit quality / MAD filtering.")

    # Sort by numeric TP index
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    v_sorted = valid[order]
    tps_sorted = [all_tps[i] for i in order]
    low_sorted = low[order]

    y_interp = np.interp(t_sorted, t_sorted[v_sorted], y_sorted[v_sorted])

    # Shift so curve passes through 0 at the reference TP
    ref_pos = np.where(t_sorted == ref_t)[0]
    if len(ref_pos):
        y_interp = y_interp - y_interp[int(ref_pos[0])]

    smooth = {}
    source = {}
    for i, tp in enumerate(tps_sorted):
        smooth[tp] = float(y_interp[i])
        if not v_sorted[i]:
            source[tp] = 'interpolated'
        elif tp == ref_tp:
            source[tp] = 'reference'
        elif low_sorted[i]:
            source[tp] = 'interpolated'
        else:
            source[tp] = 'raw'
    return smooth, source


def smooth_z_polynomial(all_tps: list[str], drift_raw: dict, low_conf: dict,
                        ref_tp: str, degree: int) -> tuple[dict, dict, np.ndarray]:
    """Fit a degree-N polynomial trend through valid per-TP Z drift.

    Low-confidence Z TPs and TPs with NaN Z drift are excluded from the fit,
    then the polynomial is evaluated at every TP. The polynomial is shifted
    so it passes through 0 at the reference TP.

    Returns (smooth[tp] = z_px, source[tp], coeffs_array).
    """
    t = np.array([tp_index(tp) for tp in all_tps])
    y = np.array([drift_raw[tp][0] for tp in all_tps], dtype=float)
    low = np.array([low_conf[tp][0] for tp in all_tps], dtype=bool)

    ref_t = tp_index(ref_tp)
    # Always pin the reference to 0
    if (t == ref_t).any():
        y[t == ref_t] = 0.0

    valid = np.isfinite(y) & (~low)
    valid[t == ref_t] = True

    if valid.sum() < degree + 1:
        degree_effective = max(1, int(valid.sum()) - 1)
        print(f'WARNING: Z polynomial degree reduced from {degree} to {degree_effective} '
              f'(only {valid.sum()} valid TPs)', file=sys.stderr)
    else:
        degree_effective = degree

    if valid.sum() < 2:
        raise RuntimeError(
            f"Z axis: only {valid.sum()} valid TPs. Cannot fit polynomial.")

    coeffs = np.polyfit(t[valid], y[valid], degree_effective)
    y_fit = np.polyval(coeffs, t)

    # Shift so the polynomial passes through (ref_t, 0)
    ref_pos = np.where(t == ref_t)[0]
    if len(ref_pos):
        y_fit = y_fit - y_fit[int(ref_pos[0])]

    # Use a stable label 'poly' regardless of the effective degree so that
    # downstream filters/groupings on drift_source_z are not degree-sensitive.
    # The effective degree is reported separately via summary_metrics.
    smooth = {}
    source = {}
    for i, tp in enumerate(all_tps):
        smooth[tp] = float(y_fit[i])
        if tp == ref_tp:
            source[tp] = 'reference'
        elif not valid[i]:
            source[tp] = 'poly_interp'
        else:
            source[tp] = 'poly'
    return smooth, source, coeffs, degree_effective


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _tp_coverage_arg(s: str) -> float:
    """argparse type: float in [0, 1]."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--tp_coverage must be a float, got {s!r}")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(
            f"--tp_coverage must be in [0.0, 1.0], got {v}")
    return v


def _positive_int_arg(s: str) -> int:
    """argparse type: strictly positive int (>= 1)."""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer >= 1, got {s!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(
            f"expected an integer >= 1, got {v}")
    return v


def _positive_float_arg(s: str) -> float:
    """argparse type: strictly positive float (> 0)."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive float, got {s!r}")
    if not v > 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive float, got {v}")
    return v


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='488-fiducial-based drift correction for 640 localizations.')
    p.add_argument(
        '--tp_coverage', type=_tp_coverage_arg, default=STABLE_TP_COVERAGE_MIN,
        help=f'Minimum TP coverage required of a stable fiducial (0.0-1.0). '
             f'Default {STABLE_TP_COVERAGE_MIN} matches STABLE_TP_COVERAGE_MIN.')
    p.add_argument(
        '--outdir', type=str, default=OUTDIR_DEFAULT,
        help='Output directory for the drift-corrected CSVs, the full-coverage '
             'and movement-filtered CSVs, the QC PNGs, and the follow-up QC '
             f'outputs. Default: {OUTDIR_DEFAULT}')
    p.add_argument(
        '--movement_policy',
        choices=(MOVEMENT_POLICY_XY, MOVEMENT_POLICY_XYZ_LEGACY,
                 MOVEMENT_POLICY_DISABLED),
        default=None,
        help='Repeatability-QC policy, which is a first-class decision rather '
             'than a threshold value. "%s" measures lateral repeatability, the '
             'robust choice for this thin-stack geometry. "%s" reproduces the '
             'original 3D displacement, whose 100 nm threshold sits BELOW z\'s '
             'own ~173 nm repeatability and so culls walks for axial '
             'imprecision. "%s" means the test is NOT EVALUATED: no pass/fail '
             'comparison is made and no threshold is invented, which is the '
             'only honest option for a layout with no same-locus re-image '
             'pair. Omit to derive the policy from --movement_axes (legacy).'
             % (MOVEMENT_POLICY_XY, MOVEMENT_POLICY_XYZ_LEGACY,
                MOVEMENT_POLICY_DISABLED))
    p.add_argument(
        '--movement_max_nm', type=_positive_float_arg, default=None,
        help=f'Threshold (nm) on the displacement between the two reference '
             f'TPs. Required unless --movement_policy '
             f'{MOVEMENT_POLICY_DISABLED}, and REFUSED with it -- a disabled '
             f'test has no threshold. Default when a measuring policy is in '
             f'effect: {MOVEMENT_MAX_NM_DEFAULT}.')
    p.add_argument(
        '--analysis_tp_start_1based', type=_positive_int_arg,
        default=ANALYSIS_TP_START_1BASED_DEFAULT,
        help=f'Follow-up QC TP range start (1-based, inclusive). '
             f'Default {ANALYSIS_TP_START_1BASED_DEFAULT}.')
    p.add_argument(
        '--analysis_tp_end_1based', type=_positive_int_arg,
        default=ANALYSIS_TP_END_1BASED_DEFAULT,
        help=f'Follow-up QC TP range end (1-based, inclusive). '
             f'Default {ANALYSIS_TP_END_1BASED_DEFAULT}.')
    p.add_argument(
        '--coverage_scope', choices=('all', 'analysis'),
        default='all',
        help='Which TPs the full-coverage filter requires. '
             '"all" (default) requires every TP in the 488/640 intersection. '
             '"analysis" requires only the analysis-range TPs '
             '(--analysis_tp_start_1based to --analysis_tp_end_1based).')
    p.add_argument(
        '--movement_tp_from_1based', type=_positive_int_arg,
        default=MOVEMENT_TP_FROM_1BASED,
        help=f'Reference TP (1-based) for the FROM side of the movement '
             f'filter. Default {MOVEMENT_TP_FROM_1BASED}.')
    p.add_argument(
        '--movement_tp_to_1based', type=_positive_int_arg,
        default=MOVEMENT_TP_TO_1BASED,
        help=f'Reference TP (1-based) for the TO side of the movement '
             f'filter. Default {MOVEMENT_TP_TO_1BASED}.')
    p.add_argument(
        '--movement_axes', choices=('xyz', 'xy'), default=None,
        help='Axes used for the movement-filter displacement. Normally implied '
             'by --movement_policy; supplying both is allowed only if they '
             'agree. Unset with no policy means "xyz", the historical default. '
             '"xyz" '
             'uses the 3D displacement and preserves existing behaviour. "xy" '
             'uses only the lateral displacement (sqrt(dy^2+dx^2)), which is '
             'the robust choice for this thin-stack (~10-14 z-slice) geometry '
             'where z is frequently unidentifiable: the 3D filter rejects '
             '~28%% of walks purely on z displacement that the XY filter '
             'keeps. See flag_z_unidentifiable.py.')
    p.add_argument('--xy_pixel_nm', type=float, default=XY_PIXEL_NM,
                   help=f'nm per xy pixel (default {XY_PIXEL_NM} = Vutara widefield; '
                        f'confocal 2x2-binned = 81.25).')
    p.add_argument('--z_step_nm', type=float, default=Z_STEP_NM,
                   help=f'nm per z-slice (default {Z_STEP_NM}).')
    p.add_argument('--fov_size', type=int, default=FOV_SIZE,
                   help=f'FOV width/height in px (default {FOV_SIZE} = rep1-3 '
                        f'Vutara widefield; the Novo tree is 2304). Only used to '
                        f'exclude edge fiducials, so a value smaller than the '
                        f'real field silently drops every fiducial in the outer '
                        f'band and biases drift to one corner.')
    args = p.parse_args()
    if args.movement_tp_to_1based < args.movement_tp_from_1based:
        p.error(f'--movement_tp_to_1based ({args.movement_tp_to_1based}) '
                f'must be >= --movement_tp_from_1based '
                f'({args.movement_tp_from_1based}).')
    if args.analysis_tp_end_1based < args.analysis_tp_start_1based:
        p.error(
            f'--analysis_tp_end_1based ({args.analysis_tp_end_1based}) '
            f'must be >= --analysis_tp_start_1based '
            f'({args.analysis_tp_start_1based}).')
    # Resolve the movement policy HERE, not deep in main(): an incoherent
    # combination must be refused before any drift correction runs, not after.
    try:
        (args.resolved_movement_policy, args.resolved_movement_axes,
         args.resolved_movement_max_nm) = resolve_movement_policy(
            args.movement_policy, args.movement_axes, args.movement_max_nm)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main():
    args = _parse_args()

    global XY_PIXEL_NM, Z_STEP_NM, FOV_SIZE
    XY_PIXEL_NM = args.xy_pixel_nm
    Z_STEP_NM = args.z_step_nm
    FOV_SIZE = args.fov_size

    # Apply the CLI --outdir globally so that OUT_CSV and the QC helpers,
    # which currently reference these module-level constants, all land in
    # the same user-chosen directory.
    global OUTDIR, OUT_CSV
    OUTDIR = args.outdir
    OUT_CSV = os.path.join(OUTDIR, 'localizations_640_drift_corrected.csv')
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'Output directory: {OUTDIR}')

    print('Loading localizations...')
    probe0 = load_localizations(PROBE0_LOC)

    probe1 = load_localizations(PROBE1_LOC)
    print(f'  640: {len(probe0)} localizations')
    print(f'  488: {len(probe1)} localizations')

    # pos_488[(WalkID, tp)] = (z, y, x, z_err, y_err, x_err)
    pos_488 = {}
    for r in probe1:
        pos_488[(r['WalkID'], r['tp'])] = (
            r['z'], r['y'], r['x'],
            r.get('z_err', np.nan), r.get('y_err', np.nan), r.get('x_err', np.nan),
        )

    tps_640 = {r['tp'] for r in probe0}
    tps_488 = {tp for (_, tp) in pos_488}
    common_tps = tps_640 & tps_488
    missing_from_488 = tps_640 - tps_488
    if missing_from_488:
        print(f'WARNING: these 640 TPs have no 488 data and cannot be corrected: '
              f'{sorted(missing_from_488, key=tp_index)}', file=sys.stderr)
    all_tps = ordered_tps(list(common_tps))
    ref_tp = all_tps[0]
    print(f'Reference TP: {ref_tp}')
    print(f'Intersection of 488/640 TPs: {len(all_tps)}')

    # ---- Fiducials ----
    fiducials_initial = select_fiducials(
        pos_488, all_tps, ref_tp,
        n_fiducials=N_FIDUCIALS, edge_pad=EDGE_PAD, fov_size=FOV_SIZE)

    # ---- Pass 1: initial drift estimate with all sampled fiducials ----
    print(f'\n[Pass 1] Estimating drift with {len(fiducials_initial)} fiducials '
          f'(MAD k={MAD_K}, min kept={MIN_KEPT_ROIS}, weighted={USE_FIT_ERR_WEIGHTS})...')
    drift_raw_p1, n_kept_p1, spread_p1, per_roi_p1, _keep_p1 = estimate_drift_pass(
        pos_488, fiducials_initial, all_tps, ref_tp,
        mad_k=MAD_K, min_kept=MIN_KEPT_ROIS, use_err_weights=USE_FIT_ERR_WEIGHTS)

    # ---- Compute stability metrics from pass-1 residuals ----
    # We deliberately pass drift_raw_p1 (the unsmoothed per-TP robust estimate)
    # rather than any smoothed drift. Smoothing would propagate signal between
    # TPs and inflate per-TP residuals, which would bias the stability scores
    # against ROIs that happen to be far from the spline on a noisy TP.
    print('\n[Stability scoring] Computing per-ROI metrics from pass-1 residuals...')
    roi_metrics = compute_roi_stability_metrics(
        per_roi_p1, drift_raw_p1, all_tps, fiducials_initial)

    # ---- Stability filtering — DECOUPLED per axis group ----
    # An ROI that is noisy only in Z can still be XY-stable (and vice versa).
    # We build two separate stable pools and run pass-2 twice.
    tp_cov_min = args.tp_coverage
    stable_xy, breakdown_xy = filter_stable_rois(
        roi_metrics, axis_group='xy',
        tp_cov_min=tp_cov_min,
        outlier_frac_max=STABLE_OUTLIER_FRAC_MAX,
        med_nm=STABLE_MED_ABS_RES_XY_NM,
        mad_nm=STABLE_MAD_RES_XY_NM,
        min_stable=STABLE_MIN_FIDUCIALS,
        relax_factor=STABLE_RELAX_FACTOR,
    )
    stable_z, breakdown_z = filter_stable_rois(
        roi_metrics, axis_group='z',
        tp_cov_min=tp_cov_min,
        outlier_frac_max=STABLE_OUTLIER_FRAC_MAX,
        med_nm=STABLE_MED_ABS_RES_Z_NM,
        mad_nm=STABLE_MAD_RES_Z_NM,
        min_stable=STABLE_MIN_FIDUCIALS,
        relax_factor=STABLE_RELAX_FACTOR,
    )

    unstable_xy = fiducials_initial - stable_xy
    unstable_z  = fiducials_initial - stable_z

    print(f'  Stable XY ROIs: {len(stable_xy)} / {len(fiducials_initial)} sampled  '
          f'(tp_cov_min={tp_cov_min:.2f})')
    print(f'  Rejection breakdown (XY, sequential):')
    for crit, n_rej in breakdown_xy.items():
        print(f'    {crit}: {n_rej} removed')
    print(f'  Stable Z  ROIs: {len(stable_z)} / {len(fiducials_initial)} sampled')
    print(f'  Rejection breakdown (Z, sequential):')
    for crit, n_rej in breakdown_z.items():
        print(f'    {crit}: {n_rej} removed')

    overlap = len(stable_xy & stable_z)
    print(f'  Overlap XY ∩ Z: {overlap} ROIs  '
          f'(XY-only: {len(stable_xy - stable_z)}, '
          f'Z-only: {len(stable_z - stable_xy)})')

    # ---- Pass 2a: XY drift estimate using XY-stable fiducials ----
    print(f'\n[Pass 2a / XY] Estimating drift with {len(stable_xy)} XY-stable fiducials...')
    drift_raw_xy, n_kept_xy, spread_xy, per_roi_xy, keep_xy = estimate_drift_pass(
        pos_488, stable_xy, all_tps, ref_tp,
        mad_k=MAD_K, min_kept=MIN_KEPT_ROIS, use_err_weights=USE_FIT_ERR_WEIGHTS)

    # ---- Pass 2b: Z drift estimate using Z-stable fiducials ----
    print(f'[Pass 2b / Z ] Estimating drift with {len(stable_z)} Z-stable fiducials...')
    drift_raw_z, n_kept_z, spread_z, per_roi_z, keep_z_mask = estimate_drift_pass(
        pos_488, stable_z, all_tps, ref_tp,
        mad_k=MAD_K, min_kept=MIN_KEPT_ROIS, use_err_weights=USE_FIT_ERR_WEIGHTS)

    # ---- Combine: drift_raw is per-axis from its own pool ----
    drift_raw = {
        tp: (drift_raw_z[tp][AXIS_Z], drift_raw_xy[tp][AXIS_Y], drift_raw_xy[tp][AXIS_X])
        for tp in all_tps
    }
    # n_kept / spread for QC are per axis, but each axis's values come from
    # its own pass. We build a composite (z_z, y_xy, x_xy) for each TP.
    n_kept = {
        tp: (n_kept_z[tp][AXIS_Z], n_kept_xy[tp][AXIS_Y], n_kept_xy[tp][AXIS_X])
        for tp in all_tps
    }
    spread = {
        tp: (spread_z[tp][AXIS_Z], spread_xy[tp][AXIS_Y], spread_xy[tp][AXIS_X])
        for tp in all_tps
    }

    # ---- Low-confidence flagging ----
    low_conf = flag_low_confidence(
        n_kept, spread, all_tps,
        nkept_frac=LOW_CONF_NKEPT_FRAC, spread_mult=LOW_CONF_SPREAD_MULT)

    low_z = [tp for tp in all_tps if low_conf[tp][0]]
    low_y = [tp for tp in all_tps if low_conf[tp][1]]
    low_x = [tp for tp in all_tps if low_conf[tp][2]]
    print('\nLow-confidence TPs per axis:')
    print(f'  X: {[tp_index(t) for t in low_x]}')
    print(f'  Y: {[tp_index(t) for t in low_y]}')
    print(f'  Z: {[tp_index(t) for t in low_z]}')

    # ---- Axis-specific smoothing ----
    print('\nAxis-specific smoothing:')
    print(f'  X: raw robust + linear interp of low-conf/missing TPs')
    print(f'  Y: raw robust + linear interp of low-conf/missing TPs')
    print(f'  Z: degree-{Z_POLY_DEGREE} polynomial trend through valid TPs')

    y_smooth, y_src = smooth_xy_with_interp(all_tps, drift_raw, low_conf, ref_tp, axis_idx=AXIS_Y)
    x_smooth, x_src = smooth_xy_with_interp(all_tps, drift_raw, low_conf, ref_tp, axis_idx=AXIS_X)
    z_smooth, z_src, z_coeffs, z_poly_degree_effective = smooth_z_polynomial(
        all_tps, drift_raw, low_conf, ref_tp, degree=Z_POLY_DEGREE)
    # Reproducibility breadcrumb: Z polynomial coefficients (high-order first)
    print(f'  Z polynomial: degree {z_poly_degree_effective}, coeffs '
          f'[{", ".join(f"{c:+.4e}" for c in z_coeffs)}] (px, high-order first)',
          file=sys.stderr)

    drift_final = {tp: (z_smooth[tp], y_smooth[tp], x_smooth[tp]) for tp in all_tps}
    drift_source = {tp: (z_src[tp], y_src[tp], x_src[tp]) for tp in all_tps}

    # Report
    print('\nFinal drift per TP (px = nm-converted in parens):')
    print(f'{"TP":>16s}  {"dx_raw(nm)":>10s}  {"dy_raw(nm)":>10s}  {"dz_raw(nm)":>10s}  '
          f'{"dx(nm)":>8s}  {"dy(nm)":>8s}  {"dz(nm)":>8s}  source(z,y,x)')
    for tp in all_tps:
        dz_r, dy_r, dx_r = drift_raw[tp]
        dz_f, dy_f, dx_f = drift_final[tp]
        sz, sy, sx = drift_source[tp]
        fmt_r = lambda v, unit: f'{v * unit:+10.1f}' if np.isfinite(v) else f'{"nan":>10s}'
        fmt_f = lambda v, unit: f'{v * unit:+8.1f}'  if np.isfinite(v) else f'{"nan":>8s}'
        print(f'{tp:>16s}  {fmt_r(dx_r, XY_PIXEL_NM)}  {fmt_r(dy_r, XY_PIXEL_NM)}  '
              f'{fmt_r(dz_r, Z_STEP_NM)}  '
              f'{fmt_f(dx_f, XY_PIXEL_NM)}  {fmt_f(dy_f, XY_PIXEL_NM)}  '
              f'{fmt_f(dz_f, Z_STEP_NM)}  ({sz},{sy},{sx})')

    # ---- Apply correction ----
    print(f'\nApplying correction to 640 localizations...')
    fields = ['tp', 'WalkID',
              'z_nm', 'y_nm', 'x_nm',
              'z', 'y', 'x',
              'z_raw', 'y_raw', 'x_raw',
              'drift_z', 'drift_y', 'drift_x',
              'drift_z_raw', 'drift_y_raw', 'drift_x_raw',
              'drift_source_z', 'drift_source_y', 'drift_source_x',
              'tp_low_conf_z', 'tp_low_conf_y', 'tp_low_conf_x',
              'n_fid_kept_z', 'n_fid_kept_y', 'n_fid_kept_x',
              'sigma_z', 'sigma_y', 'sigma_x', 'amplitude', 'background',
              'z_err', 'y_err', 'x_err']

    # Carry the per-locus ownership columns through drift correction. The header
    # here is hardcoded, so without this they were silently DROPPED and the
    # comprehensive downstream CSV could not identify a rejected coordinate.
    # Only columns actually present upstream are added, so a legacy input still
    # produces a legacy header.
    _own_cols = [c for c in _OWNERSHIP_ROW_FIELDS
                 if probe0 and c in probe0[0]]
    _usab_cols = [c for c in _USABILITY_ROW_FIELDS
                  if probe0 and c in probe0[0]]
    fields = fields + _own_cols + _usab_cols

    n_written, n_skipped = 0, 0
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in probe0:
            tp = row['tp']
            if tp not in drift_final:
                n_skipped += 1
                continue
            dz, dy, dx = drift_final[tp]
            dz_r, dy_r, dx_r = drift_raw[tp]
            sz, sy, sx = drift_source[tp]
            lcz, lcy, lcx = low_conf[tp]
            nz, ny, nx = n_kept[tp]

            z_raw, y_raw, x_raw = row['z'], row['y'], row['x']
            z_corr = z_raw - dz
            y_corr = y_raw - dy
            x_corr = x_raw - dx

            writer.writerow({
                'tp': tp, 'WalkID': row['WalkID'],
                'z_nm': f'{z_corr * Z_STEP_NM:.1f}',
                'y_nm': f'{y_corr * XY_PIXEL_NM:.1f}',
                'x_nm': f'{x_corr * XY_PIXEL_NM:.1f}',
                'z': f'{z_corr:.4f}', 'y': f'{y_corr:.4f}', 'x': f'{x_corr:.4f}',
                'z_raw': f'{z_raw:.4f}', 'y_raw': f'{y_raw:.4f}', 'x_raw': f'{x_raw:.4f}',
                'drift_z': f'{dz:.4f}', 'drift_y': f'{dy:.4f}', 'drift_x': f'{dx:.4f}',
                'drift_z_raw': '' if not np.isfinite(dz_r) else f'{dz_r:.4f}',
                'drift_y_raw': '' if not np.isfinite(dy_r) else f'{dy_r:.4f}',
                'drift_x_raw': '' if not np.isfinite(dx_r) else f'{dx_r:.4f}',
                'drift_source_z': sz, 'drift_source_y': sy, 'drift_source_x': sx,
                'tp_low_conf_z': int(lcz), 'tp_low_conf_y': int(lcy), 'tp_low_conf_x': int(lcx),
                'n_fid_kept_z': nz, 'n_fid_kept_y': ny, 'n_fid_kept_x': nx,
                **{c: row.get(c, '') for c in _own_cols},
                'sigma_z': row.get('sigma_z', ''), 'sigma_y': row.get('sigma_y', ''),
                'sigma_x': row.get('sigma_x', ''),
                'amplitude': row.get('amplitude', ''), 'background': row.get('background', ''),
                'z_err': row.get('z_err', ''), 'y_err': row.get('y_err', ''),
                'x_err': row.get('x_err', ''),
            })
            n_written += 1

    print(f'Wrote {n_written} corrected 640 localizations to {OUT_CSV}')
    if n_skipped:
        print(f'Skipped {n_skipped} 640 rows (no 488 data for their TP)')

    # ---- Residuals (using final drift) ----
    # Two views are maintained so QC can compare them:
    #   residuals_all      — all stable ROIs present at this TP (XY from per_roi_xy,
    #                         Z from per_roi_z). Used to flag general goodness of fit.
    #   residuals_retained — same populations but only rows that passed the per-TP
    #                         MAD keep-mask on that axis, i.e. the ROIs that actually
    #                         contributed to the drift estimator for that TP/axis.
    #
    # XY residuals (rows: wid, res_y, res_x) come from per_roi_xy / keep_xy.
    # Z  residuals (rows: wid, res_z)         come from per_roi_z / keep_z_mask.
    residuals_xy_all       = {}
    residuals_xy_retained  = {}
    residuals_z_all        = {}
    residuals_z_retained   = {}
    for tp in all_tps:
        dz_f, dy_f, dx_f = drift_final[tp]

        arr_xy = per_roi_xy[tp]
        if arr_xy.shape[0]:
            ry = arr_xy[:, AXIS_Y + 1] - dy_f
            rx = arr_xy[:, AXIS_X + 1] - dx_f
            residuals_xy_all[tp] = np.column_stack([arr_xy[:, 0], ry, rx])
            # Retained on both X and Y:
            kz_xy, ky_xy, kx_xy = keep_xy[tp]
            mask = ky_xy & kx_xy
            residuals_xy_retained[tp] = residuals_xy_all[tp][mask]
        else:
            residuals_xy_all[tp] = np.zeros((0, 3))
            residuals_xy_retained[tp] = np.zeros((0, 3))

        arr_z = per_roi_z[tp]
        if arr_z.shape[0]:
            rz = arr_z[:, AXIS_Z + 1] - dz_f
            residuals_z_all[tp] = np.column_stack([arr_z[:, 0], rz])
            kz_z, _, _ = keep_z_mask[tp]
            residuals_z_all[tp]
            residuals_z_retained[tp] = residuals_z_all[tp][kz_z]
        else:
            residuals_z_all[tp] = np.zeros((0, 2))
            residuals_z_retained[tp] = np.zeros((0, 2))

    # ---- Summary metrics for both residual populations ----
    def _mad_abs_nm(vals_px, unit_nm):
        if len(vals_px) == 0:
            return (np.nan, np.nan)
        abs_nm = np.abs(vals_px) * unit_nm
        return float(np.median(abs_nm)), float(np.percentile(abs_nm, 95))

    def _pool_xy(res_dict, col):
        arrs = [res_dict[tp][:, col] for tp in all_tps if res_dict[tp].size]
        return np.concatenate(arrs) if arrs else np.zeros(0)

    def _pool_z(res_dict):
        arrs = [res_dict[tp][:, 1] for tp in all_tps if res_dict[tp].size]
        return np.concatenate(arrs) if arrs else np.zeros(0)

    arx_all = _pool_xy(residuals_xy_all, 2)   # col 2 = rx
    ary_all = _pool_xy(residuals_xy_all, 1)   # col 1 = ry
    arz_all = _pool_z(residuals_z_all)
    arx_ret = _pool_xy(residuals_xy_retained, 2)
    ary_ret = _pool_xy(residuals_xy_retained, 1)
    arz_ret = _pool_z(residuals_z_retained)

    med_x_all, p95_x_all = _mad_abs_nm(arx_all, XY_PIXEL_NM)
    med_y_all, p95_y_all = _mad_abs_nm(ary_all, XY_PIXEL_NM)
    med_z_all, p95_z_all = _mad_abs_nm(arz_all, Z_STEP_NM)
    med_x_ret, p95_x_ret = _mad_abs_nm(arx_ret, XY_PIXEL_NM)
    med_y_ret, p95_y_ret = _mad_abs_nm(ary_ret, XY_PIXEL_NM)
    med_z_ret, p95_z_ret = _mad_abs_nm(arz_ret, Z_STEP_NM)

    # Rejection rate per TP (fraction of stable ROIs that failed per-TP MAD on
    # the worst axis of their axis-group). Kept for console summary only.
    reject_fracs = []
    for tp in all_tps:
        # XY: of per_roi_xy, fraction where ky & kx is False
        arr_xy = per_roi_xy[tp]; arr_z = per_roi_z[tp]
        if arr_xy.shape[0] == 0 and arr_z.shape[0] == 0:
            reject_fracs.append(0.0); continue
        total = arr_xy.shape[0] + arr_z.shape[0]
        _, ky, kx = keep_xy[tp]
        retained_xy = int((ky & kx).sum())
        kz, _, _ = keep_z_mask[tp]
        retained_z = int(kz.sum())
        reject_fracs.append(1.0 - (retained_xy + retained_z) / max(total, 1))

    print('\nResidual summary — pooled over all TPs')
    print(f'  [all stable ROIs]     median |res|  X={med_x_all:6.1f}  Y={med_y_all:6.1f}  Z={med_z_all:6.1f} nm')
    print(f'                        P95    |res|  X={p95_x_all:6.1f}  Y={p95_y_all:6.1f}  Z={p95_z_all:6.1f} nm')
    print(f'  [per-TP retained]     median |res|  X={med_x_ret:6.1f}  Y={med_y_ret:6.1f}  Z={med_z_ret:6.1f} nm')
    print(f'                        P95    |res|  X={p95_x_ret:6.1f}  Y={p95_y_ret:6.1f}  Z={p95_z_ret:6.1f} nm')
    print(f'  ROI rejection rate (worst axis): min={min(reject_fracs):.1%}, '
          f'median={np.median(reject_fracs):.1%}, max={max(reject_fracs):.1%}')
    print(f'  Stable XY ROIs: {len(stable_xy)} / {len(fiducials_initial)} sampled')
    print(f'  Stable Z  ROIs: {len(stable_z)} / {len(fiducials_initial)} sampled')
    print(f'  Low-confidence TPs: X={len(low_x)}, Y={len(low_y)}, Z={len(low_z)}')

    summary_metrics = {
        # All-stable-ROI view (default labelling kept for backwards-compat in QC)
        'median_abs_x_nm': med_x_all, 'p95_abs_x_nm': p95_x_all,
        'median_abs_y_nm': med_y_all, 'p95_abs_y_nm': p95_y_all,
        'median_abs_z_nm': med_z_all, 'p95_abs_z_nm': p95_z_all,
        # Retained-only view
        'median_abs_x_nm_ret': med_x_ret, 'p95_abs_x_nm_ret': p95_x_ret,
        'median_abs_y_nm_ret': med_y_ret, 'p95_abs_y_nm_ret': p95_y_ret,
        'median_abs_z_nm_ret': med_z_ret, 'p95_abs_z_nm_ret': p95_z_ret,
        # Bookkeeping
        'reject_frac_med': float(np.median(reject_fracs)),
        'reject_frac_max': float(max(reject_fracs)),
        'n_sampled': len(fiducials_initial),
        'n_stable_xy': len(stable_xy),
        'n_stable_z':  len(stable_z),
        'n_low_conf_x': len(low_x), 'n_low_conf_y': len(low_y), 'n_low_conf_z': len(low_z),
        'z_poly_degree': int(z_poly_degree_effective),
        'breakdown_xy': breakdown_xy,
        'breakdown_z':  breakdown_z,
        'tp_cov_min': tp_cov_min,
    }

    # ---- QC ----
    make_drift_model_qc(all_tps, ref_tp, pos_488, stable_xy, stable_z,
                        drift_raw, drift_final, drift_source,
                        n_kept, spread, low_conf,
                        residuals_xy_all, residuals_xy_retained,
                        residuals_z_all,  residuals_z_retained,
                        summary_metrics, tp_cov_min)
    make_roi_diagnostics_qc(all_tps, ref_tp, pos_488, stable_xy, stable_z,
                            per_roi_xy, per_roi_z,
                            drift_raw, drift_final, roi_metrics, tp_cov_min)

    # ---- Final post-filter: keep only WalkIDs with full TP coverage ----
    if args.coverage_scope == 'analysis':
        coverage_tps = {f'TP{i:03d}' for i in range(
            args.analysis_tp_start_1based - 1, args.analysis_tp_end_1based)}
        coverage_tps = [tp for tp in all_tps if tp in coverage_tps]
        scope_label = (f'analysis range '
                       f'TP{args.analysis_tp_start_1based-1:03d}..'
                       f'TP{args.analysis_tp_end_1based-1:03d}')
    else:
        coverage_tps = list(all_tps)
        scope_label = 'all TPs'
    n_coverage_tps = len(coverage_tps)
    coverage_set = set(coverage_tps)

    tps_per_wid: dict[int, set] = defaultdict(set)
    for row in probe0:
        if row['tp'] in drift_final and row['tp'] in coverage_set:
            tps_per_wid[row['WalkID']].add(row['tp'])
    full_cov_wids = {w for w, tps in tps_per_wid.items()
                     if len(tps) == n_coverage_tps}

    filtered_csv = OUT_CSV.replace('.csv', '_full_coverage.csv')
    n_kept_rows = 0
    with open(OUT_CSV) as src, open(filtered_csv, 'w', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row['WalkID']) in full_cov_wids:
                writer.writerow(row)
                n_kept_rows += 1

    n_candidate_wids = len(tps_per_wid)
    print(f'\n[Full-coverage filter] WalkIDs with all {n_coverage_tps} TPs '
          f'({scope_label}) in 640: '
          f'{len(full_cov_wids)} / {n_candidate_wids} ({len(full_cov_wids)/max(n_candidate_wids,1):.1%})')
    print(f'  Kept {n_kept_rows} rows ({n_kept_rows/max(n_written,1):.1%} of drift-corrected output)')
    print(f'  Wrote {filtered_csv}')

    # ---- Movement filter: drop WalkIDs whose 3D displacement between the two
    # reference TPs (1-based MOVEMENT_TP_FROM_1BASED / MOVEMENT_TP_TO_1BASED)
    # exceeds the CLI-configured threshold --movement_max_nm. Operates on
    # the drift-corrected physical nm coordinates (x_nm, y_nm, z_nm) written
    # above.
    def _tp_by_1based(n_1based: int) -> str | None:
        target_0 = n_1based - 1
        for tp in all_tps:
            if tp_index(tp) == target_0:
                return tp
        return None

    def _write_movement_columns(path, disp_by_wid, definition,
                                status, reason):
        """Append movement_nm / movement_def to a drift CSV, atomically.

        The displacement is a per-walk property, so its value repeats across
        that walk's rows. Written into the FULL-COVERAGE csv, not only the
        filtered one, so the threshold can be revisited later without
        recomputing anything: the rejected walks carry their value too.

        movement_def records which reference pair and which axes produced the
        number (e.g. "TP002->TP022:xy"). A bare displacement column would not
        say whether z is included, and that ambiguity has already caused one
        real error in this pipeline.

        movement_qc_status is `evaluated` or `not_evaluated`, and
        movement_qc_reason says why when it is the latter. Without these a
        consumer cannot tell a walk that PASSED repeatability from one that was
        never tested -- the two used to be indistinguishable, because a
        disabled test was expressed as a threshold nothing could exceed.
        """
        import tempfile
        _MOVE_COLS = ('movement_nm', 'movement_def',
                      'movement_qc_status', 'movement_qc_reason')
        out_dir = os.path.dirname(os.path.abspath(path))
        tmp = None
        try:
            with open(path, newline='') as src:
                reader = csv.DictReader(src)
                fields = [f for f in reader.fieldnames if f not in _MOVE_COLS]
                fields += list(_MOVE_COLS)
                with tempfile.NamedTemporaryFile(
                        mode='w', newline='', dir=out_dir, delete=False,
                        prefix='.movement_', suffix='.csv') as dst:
                    tmp = dst.name
                    writer = csv.DictWriter(dst, fieldnames=fields)
                    writer.writeheader()
                    for row in reader:
                        d = disp_by_wid.get(int(row['WalkID']))
                        row['movement_nm'] = '' if d is None else f'{d:.3f}'
                        row['movement_def'] = definition
                        row['movement_qc_status'] = status
                        row['movement_qc_reason'] = reason
                        writer.writerow(row)
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

    # Already resolved and validated at parse time.
    movement_policy = args.resolved_movement_policy
    movement_axes = args.resolved_movement_axes
    movement_max_nm = args.resolved_movement_max_nm
    movement_from_1b = args.movement_tp_from_1based
    movement_to_1b   = args.movement_tp_to_1based
    tp_from = _tp_by_1based(movement_from_1b)
    tp_to   = _tp_by_1based(movement_to_1b)
    final_csv_for_qc = filtered_csv
    movement_qc = {
        'movement_policy': movement_policy,
        'movement_axes': movement_axes,
        'movement_max_nm': movement_max_nm,
        'reference_pair_1based': [movement_from_1b, movement_to_1b],
    }
    if movement_policy == MOVEMENT_POLICY_DISABLED:
        # NOTHING is measured and NOTHING is compared. No threshold is invented,
        # no walk is dropped, and the output is explicitly NOT described as
        # having passed repeatability QC -- it was never tested.
        reason = ('movement_policy=disabled: this acquisition layout has no '
                  'same-locus re-image pair, so repeatability is not '
                  'measurable and no pass/fail comparison was made')
        move_csv = OUT_CSV.replace('.csv', '_filtered.csv')
        _write_movement_columns(filtered_csv, {}, 'not_evaluated:disabled',
                                MOVEMENT_QC_NOT_EVALUATED, reason)
        import shutil as _shutil
        _shutil.copyfile(filtered_csv, move_csv)
        movement_qc.update(status=MOVEMENT_QC_NOT_EVALUATED, reason=reason,
                           walks_dropped=0, threshold_applied=False)
        print(f'\n[Movement QC] NOT EVALUATED (policy='
              f'{MOVEMENT_POLICY_DISABLED}).')
        print(f'  {reason}.')
        print(f'  Wrote {move_csv}: identical to the full-coverage set. These '
              'walks have NOT passed movement QC; they were never tested.')
        final_csv_for_qc = move_csv
    elif tp_from is None or tp_to is None:
        print(f'\n[Movement filter] Skipped: TP{movement_from_1b} (1-based) '
              f'or TP{movement_to_1b} (1-based) not present in the '
              f'drift-corrected TPs; cannot compute displacement.')
        # Always write _filtered.csv so downstream consumers never silently read
        # a stale file: when the movement filter cannot run, emit an identity
        # copy of the full-coverage set (no walks dropped) and flag the status.
        move_csv = OUT_CSV.replace('.csv', '_filtered.csv')
        # Keep the schema stable even when the filter cannot run: the columns
        # are present but empty, and movement_def says why.
        _endpoint_reason = (
            f'reference pair TP{movement_from_1b}/TP{movement_to_1b} (1-based) '
            'is not present in the drift-corrected TPs, so the repeatability '
            'displacement could not be computed')
        _write_movement_columns(filtered_csv, {}, 'skipped:missing_endpoint',
                                MOVEMENT_QC_NOT_EVALUATED, _endpoint_reason)
        movement_qc.update(status=MOVEMENT_QC_NOT_EVALUATED,
                           reason=_endpoint_reason, walks_dropped=0,
                           threshold_applied=False)
        import shutil as _shutil
        _shutil.copyfile(filtered_csv, move_csv)
        print(f'  WARNING movement_filter_status=skipped_missing_endpoint: '
              f'wrote identity {move_csv} (no walks dropped).')
        final_csv_for_qc = move_csv
    else:
        # Read positions at the two TPs from the full-coverage CSV
        positions: dict[int, dict[str, tuple[float, float, float]]] = defaultdict(dict)
        with open(filtered_csv) as f:
            for row in csv.DictReader(f):
                tp = row['tp']
                if tp == tp_from or tp == tp_to:
                    wid = int(row['WalkID'])
                    positions[wid][tp] = (float(row['z_nm']),
                                          float(row['y_nm']),
                                          float(row['x_nm']))

        kept_wids = set()
        dists_nm = []
        disp_by_wid: dict[int, float] = {}
        for wid, tp_pos in positions.items():
            if tp_from not in tp_pos or tp_to not in tp_pos:
                continue
            zf, yf, xf = tp_pos[tp_from]
            zt, yt, xt = tp_pos[tp_to]
            if movement_axes == 'xy':
                # Lateral-only displacement: robust for thin-stack data where
                # the axial fit is frequently unidentifiable (see
                # flag_z_unidentifiable.py). z_nm is excluded entirely.
                d = ((yf-yt)**2 + (xf-xt)**2) ** 0.5
            else:
                d = ((zf-zt)**2 + (yf-yt)**2 + (xf-xt)**2) ** 0.5
            dists_nm.append(d)
            disp_by_wid[wid] = d
            if d <= movement_max_nm:
                kept_wids.add(wid)

        # Persist the per-walk displacement before filtering, so the number the
        # decision was made on is auditable and re-thresholdable. Exactly the
        # configured reference pair is used -- no averaging over the other
        # re-image pairs present in the acquisition.
        _write_movement_columns(
            filtered_csv, disp_by_wid,
            f'{tp_from}->{tp_to}:{movement_axes}',
            MOVEMENT_QC_EVALUATED, '')

        move_csv = OUT_CSV.replace('.csv', '_filtered.csv')
        n_written_mov = 0
        with open(filtered_csv) as src, open(move_csv, 'w', newline='') as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if int(row['WalkID']) in kept_wids:
                    writer.writerow(row)
                    n_written_mov += 1

        n_cand = len(positions)
        import numpy as _np
        d_arr = _np.asarray(dists_nm) if dists_nm else _np.zeros(0)
        if d_arr.size:
            med_nm = float(_np.median(d_arr))
            p95_nm = float(_np.percentile(d_arr, 95))
        else:
            med_nm = p95_nm = float('nan')

        print(f'\n[Movement filter] TP{movement_from_1b} -> TP{movement_to_1b} '
              f'(1-based) = {tp_from} -> {tp_to}, '
              f'policy={movement_policy}, axes={movement_axes}, '
              f'threshold <= {movement_max_nm:.1f} nm (--movement_max_nm)')
        print(f'  Displacement over full-coverage set: median={med_nm:.1f} nm, '
              f'P95={p95_nm:.1f} nm')
        print(f'  WalkIDs kept: {len(kept_wids)} / {n_cand} '
              f'({len(kept_wids)/max(n_cand,1):.1%})')
        print(f'  Kept {n_written_mov} rows')
        print(f'  Wrote {move_csv}')
        movement_qc.update(status=MOVEMENT_QC_EVALUATED, reason='',
                           walks_evaluated=n_cand,
                           walks_dropped=n_cand - len(kept_wids),
                           threshold_applied=True,
                           displacement_median_nm=med_nm,
                           displacement_p95_nm=p95_nm)
        final_csv_for_qc = move_csv

    # The status belongs beside the data, not only in the log: a later consumer
    # must be able to tell "passed repeatability" from "never tested" without
    # reconstructing which flags this run was given.
    _movement_qc_json = os.path.join(
        os.path.dirname(os.path.abspath(OUT_CSV)), 'movement_qc.json')
    with open(_movement_qc_json, 'w') as _f:
        json.dump(movement_qc, _f, indent=2, sort_keys=True)
    print(f'  movement QC status -> {_movement_qc_json} '
          f'({movement_qc["status"]})')

    # ---- Follow-up QC / analysis on the final downstream-ready walks ----
    # Parabola QC and distance matrices, computed on the exact walk set that
    # will be used downstream. The analysis TP range is configured via
    # --analysis_tp_start_1based / --analysis_tp_end_1based (both 1-based and
    # inclusive). This project has one localization per walk per TP after
    # filtering, so no centroid aggregation is performed.
    # qc_post_filter.py lives at the repo root (the reorg kept it there, next to
    # report_cards_single_curve.py which also imports it). Now that this module
    # runs from pipeline/, put the repo root on sys.path so the import resolves.
    import os as _os, sys as _sys
    _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _repo_root not in _sys.path:
        _sys.path.append(_repo_root)
    from qc_post_filter import run_post_filter_analysis

    # This step is a DIAGNOSTIC: it writes QC figures (parabola fits, distance
    # matrices) and nothing outside this module reads its output. It needs every
    # timepoint of the analysis range to be present in the final walk set, and it
    # correctly refuses to clamp when they are not.
    #
    # It used to be called unguarded, and it is the LAST thing this stage does --
    # so on a dataset where few walks achieve coverage across the whole analysis
    # range, a missing FIGURE aborted the stage after every data product had
    # already been written, and the Location never reached
    # classify_replication. Declining to draw a figure must not decide whether a
    # Location is analysable.
    #
    # The precondition is therefore checked here and the outcome RECORDED, in the
    # same spirit as movement_qc_status: `not_computed` is not `passed`, and the
    # range is never silently narrowed to whatever happens to be present.
    _analysis_tps = set(range(args.analysis_tp_start_1based - 1,
                              args.analysis_tp_end_1based))
    _present = set()
    with open(final_csv_for_qc, newline='') as _fh:
        for _row in csv.DictReader(_fh):
            try:
                _present.add(tp_index(_row['tp']))
            except (KeyError, ValueError):
                continue
    _absent = sorted(_analysis_tps - _present)
    _post_qc = {'analysis_tp_range_1based': [args.analysis_tp_start_1based,
                                             args.analysis_tp_end_1based],
                'source_csv': os.path.basename(final_csv_for_qc)}
    if _absent:
        _post_qc.update(
            status='not_computed',
            missing_0based_tp=_absent,
            reason=('the final walk set contains no localization at '
                    f'{len(_absent)} timepoint(s) of the analysis range, so the '
                    'post-filter QC figures cannot be drawn. The range was NOT '
                    'narrowed to fit; this is a coverage limitation of the data '
                    'under the configured filters, recorded rather than hidden.'))
        print(f'\n[Post-filter QC] NOT COMPUTED: {len(_absent)} of '
              f'{len(_analysis_tps)} analysis timepoints have no localization in '
              f'{os.path.basename(final_csv_for_qc)}.')
        print('  Missing 0-based TP indices: '
              f'{_absent if len(_absent) <= 24 else str(_absent[:24]) + "..."}')
        print('  The QC FIGURES are skipped. Every data product of this stage '
              'was written, and classification proceeds.')
    else:
        run_post_filter_analysis(
            final_csv_for_qc, OUTDIR,
            tp_start_1based=args.analysis_tp_start_1based,
            tp_end_1based=args.analysis_tp_end_1based,
        )
        _post_qc.update(status='computed', missing_0based_tp=[], reason='')
    with open(os.path.join(OUTDIR, 'post_filter_qc.json'), 'w') as _fh:
        json.dump(_post_qc, _fh, indent=2, sort_keys=True)

    print('Done!')


# ----------------------------------------------------------------------------
# QC — focused on: (A) stability, (B) residual collapse, (C) spatial sufficiency,
# (D) reference quality.  Now also shows two-pass diagnostics.
# ----------------------------------------------------------------------------
def make_drift_model_qc(all_tps, ref_tp, pos_488, stable_xy, stable_z,
                        drift_raw, drift_final, drift_source,
                        n_kept, spread, low_conf,
                        residuals_xy_all, residuals_xy_retained,
                        residuals_z_all,  residuals_z_retained,
                        summary_metrics, tp_cov_min):
    """Main drift-model QC figure.

    Axis-group decoupling: per-axis n_kept / spread / residuals come from
    XY-stable pool for X and Y, and from Z-stable pool for Z.

    Two residual populations are shown on every relevant B-row panel:
      "all stable ROIs"     — every row in per_roi_{xy,z}[tp]
      "per-TP retained"     — rows that passed the per-TP MAD keep mask
                               on the relevant axis (== ROIs that actually
                               contributed to the drift estimator).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_idx = np.array([tp_index(tp) for tp in all_tps])
    order = np.argsort(t_idx)
    tps_sorted = [all_tps[i] for i in order]
    tp_labels = [f'{t:03d}' for t in t_idx[order]]
    x_pos = np.arange(len(tps_sorted))

    dz_raw = np.array([drift_raw[tp][AXIS_Z] for tp in tps_sorted]) * Z_STEP_NM
    dy_raw = np.array([drift_raw[tp][AXIS_Y] for tp in tps_sorted]) * XY_PIXEL_NM
    dx_raw = np.array([drift_raw[tp][AXIS_X] for tp in tps_sorted]) * XY_PIXEL_NM
    dz_f   = np.array([drift_final[tp][AXIS_Z] for tp in tps_sorted]) * Z_STEP_NM
    dy_f   = np.array([drift_final[tp][AXIS_Y] for tp in tps_sorted]) * XY_PIXEL_NM
    dx_f   = np.array([drift_final[tp][AXIS_X] for tp in tps_sorted]) * XY_PIXEL_NM
    nk_z = np.array([n_kept[tp][AXIS_Z] for tp in tps_sorted])
    nk_y = np.array([n_kept[tp][AXIS_Y] for tp in tps_sorted])
    nk_x = np.array([n_kept[tp][AXIS_X] for tp in tps_sorted])
    sp_z = np.array([spread[tp][AXIS_Z] for tp in tps_sorted], dtype=float) * Z_STEP_NM
    sp_y = np.array([spread[tp][AXIS_Y] for tp in tps_sorted], dtype=float) * XY_PIXEL_NM
    sp_x = np.array([spread[tp][AXIS_X] for tp in tps_sorted], dtype=float) * XY_PIXEL_NM

    lc_z = np.array([low_conf[tp][AXIS_Z] for tp in tps_sorted], dtype=bool)
    lc_y = np.array([low_conf[tp][AXIS_Y] for tp in tps_sorted], dtype=bool)
    lc_x = np.array([low_conf[tp][AXIS_X] for tp in tps_sorted], dtype=bool)

    # Per-TP residual summaries for both populations.
    # XY residuals are (wid, ry, rx) in cols 0,1,2 ; Z residuals are (wid, rz) in cols 0,1.
    def _per_tp_stat(res_dict, col, unit):
        med = np.full(len(tps_sorted), np.nan)
        mad = np.full(len(tps_sorted), np.nan)
        for i, tp in enumerate(tps_sorted):
            r = res_dict[tp]
            if r.shape[0]:
                med[i] = np.median(r[:, col]) * unit
                mad[i] = _mad_sigma(r[:, col]) * unit
        return med, mad

    res_med_y_all, res_mad_y_all = _per_tp_stat(residuals_xy_all, 1, XY_PIXEL_NM)
    res_med_x_all, res_mad_x_all = _per_tp_stat(residuals_xy_all, 2, XY_PIXEL_NM)
    res_med_z_all, res_mad_z_all = _per_tp_stat(residuals_z_all,  1, Z_STEP_NM)
    res_med_y_ret, res_mad_y_ret = _per_tp_stat(residuals_xy_retained, 1, XY_PIXEL_NM)
    res_med_x_ret, res_mad_x_ret = _per_tp_stat(residuals_xy_retained, 2, XY_PIXEL_NM)
    res_med_z_ret, res_mad_z_ret = _per_tp_stat(residuals_z_retained,  1, Z_STEP_NM)

    def _pool(res_dict, col, unit):
        arrs = [res_dict[tp][:, col] for tp in tps_sorted if res_dict[tp].size]
        return (np.concatenate(arrs) if arrs else np.zeros(0)) * unit

    all_rx_all = _pool(residuals_xy_all, 2, XY_PIXEL_NM)
    all_ry_all = _pool(residuals_xy_all, 1, XY_PIXEL_NM)
    all_rz_all = _pool(residuals_z_all,  1, Z_STEP_NM)
    all_rx_ret = _pool(residuals_xy_retained, 2, XY_PIXEL_NM)
    all_ry_ret = _pool(residuals_xy_retained, 1, XY_PIXEL_NM)
    all_rz_ret = _pool(residuals_z_retained,  1, Z_STEP_NM)

    # Worst TP by XY residual MAD (using retained view)
    score = np.nan_to_num(np.hypot(res_mad_y_ret, res_mad_x_ret), nan=-1)
    idx_worst = int(np.argmax(score))
    worst_tp = tps_sorted[idx_worst]

    fig, axes = plt.subplots(4, 3, figsize=(22, 22))

    # ----- Row 1: drift curves per axis (raw robust + final) -----
    def _plot_axis(ax, name, raw_v, final_v, lc_mask, color):
        ax.plot(x_pos, raw_v, 'o', color=color, alpha=0.55, label='Robust per-TP')
        ax.plot(x_pos, final_v, '-', color=color, linewidth=2, label='Final (applied)')
        if lc_mask.any():
            ax.scatter(x_pos[lc_mask], np.zeros(lc_mask.sum()),
                       marker='v', color='black', s=50, label='low-conf TP')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
        ax.set_ylabel(f'{name} drift (nm)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    _plot_axis(axes[0, 0], 'X', dx_raw, dx_f, lc_x, 'red')
    axes[0, 0].set_title(f'A. X drift — raw robust vs applied (XY-stable pool, n={len(stable_xy)})')
    _plot_axis(axes[0, 1], 'Y', dy_raw, dy_f, lc_y, 'blue')
    axes[0, 1].set_title(f'A. Y drift — raw robust vs applied (XY-stable pool, n={len(stable_xy)})')
    _plot_axis(axes[0, 2], 'Z', dz_raw, dz_f, lc_z, 'green')
    axes[0, 2].set_title(f'A. Z drift — raw robust vs applied (Z-stable pool, n={len(stable_z)}, poly{Z_POLY_DEGREE})')

    # ----- Row 2: retained counts, drift-estimate spread, decoupled fiducial distribution -----
    ax = axes[1, 0]
    ax.plot(x_pos, nk_x, 'r-o', markersize=4, label=f'X (XY pool, |stable|={len(stable_xy)})')
    ax.plot(x_pos, nk_y, 'b-o', markersize=4, label=f'Y (XY pool)')
    ax.plot(x_pos, nk_z, 'g-o', markersize=4, label=f'Z (Z pool, |stable|={len(stable_z)})')
    ax.axhline(MIN_KEPT_ROIS, color='k', linestyle='--', alpha=0.5,
               label=f'min required ({MIN_KEPT_ROIS})')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Retained ROIs (after MAD)')
    ax.set_title(f'A. # retained ROIs per TP (MAD k={MAD_K})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_pos, sp_x, 'r-o', markersize=4, label='σ_X')
    ax.plot(x_pos, sp_y, 'b-o', markersize=4, label='σ_Y')
    ax.plot(x_pos, sp_z, 'g-o', markersize=4, label='σ_Z')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Drift-estimate MAD-σ (nm)')
    ax.set_title('A. Per-TP drift-estimate uncertainty (retained ROI MAD-σ)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    all_wids_ref = sorted({wid for (wid, tp) in pos_488 if tp == ref_tp})
    all_xy = np.array([(pos_488[(w, ref_tp)][2], pos_488[(w, ref_tp)][1]) for w in all_wids_ref])
    xy_only = (stable_xy - stable_z)
    z_only  = (stable_z - stable_xy)
    both    = (stable_xy & stable_z)
    def _xy(wids):
        return np.array([(pos_488[(w, ref_tp)][2], pos_488[(w, ref_tp)][1])
                         for w in wids if (w, ref_tp) in pos_488])
    p_both, p_xy_only, p_z_only = _xy(both), _xy(xy_only), _xy(z_only)
    ax.scatter(all_xy[:, 0], all_xy[:, 1], s=2, c='gray', alpha=0.25,
               label=f'488 at ref ({len(all_xy)})')
    if len(p_both):
        ax.scatter(p_both[:, 0], p_both[:, 1], s=15, c='purple', marker='o',
                   label=f'XY ∩ Z stable ({len(p_both)})')
    if len(p_xy_only):
        ax.scatter(p_xy_only[:, 0], p_xy_only[:, 1], s=15, c='red', marker='o',
                   label=f'XY-only stable ({len(p_xy_only)})')
    if len(p_z_only):
        ax.scatter(p_z_only[:, 0], p_z_only[:, 1], s=15, c='green', marker='o',
                   label=f'Z-only stable ({len(p_z_only)})')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title(f'A. Stable fiducials per axis group (ref_tp={ref_tp})')
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.legend(fontsize=8, loc='lower right')

    # ----- Row 3: residual median / MAD / distribution XY — BOTH populations -----
    ax = axes[2, 0]
    ax.plot(x_pos, res_med_x_all, 'r-o', markersize=4, alpha=0.5, label='median X (all)')
    ax.plot(x_pos, res_med_y_all, 'b-o', markersize=4, alpha=0.5, label='median Y (all)')
    ax.plot(x_pos, res_med_z_all, 'g-o', markersize=4, alpha=0.5, label='median Z (all)')
    ax.plot(x_pos, res_med_x_ret, 'r--s', markersize=3, label='median X (retained)')
    ax.plot(x_pos, res_med_y_ret, 'b--s', markersize=3, label='median Y (retained)')
    ax.plot(x_pos, res_med_z_ret, 'g--s', markersize=3, label='median Z (retained)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Residual median (nm)')
    ax.set_title('B. Per-TP residual median (all stable vs per-TP retained)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(x_pos, res_mad_x_all, 'r-o', markersize=4, alpha=0.5, label='σ_X (all)')
    ax.plot(x_pos, res_mad_y_all, 'b-o', markersize=4, alpha=0.5, label='σ_Y (all)')
    ax.plot(x_pos, res_mad_z_all, 'g-o', markersize=4, alpha=0.5, label='σ_Z (all)')
    ax.plot(x_pos, res_mad_x_ret, 'r--s', markersize=3, label='σ_X (retained)')
    ax.plot(x_pos, res_mad_y_ret, 'b--s', markersize=3, label='σ_Y (retained)')
    ax.plot(x_pos, res_mad_z_ret, 'g--s', markersize=3, label='σ_Z (retained)')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Residual MAD-σ (nm)')
    ax.set_title('B. Per-TP residual robust spread (all vs retained)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    def _bounds(arr, p=99):
        if arr.size == 0: return (-1, 1)
        lo, hi = np.percentile(arr, [100 - p, p])
        w = max(abs(lo), abs(hi)) * 1.1
        return (-w, w)

    ax = axes[2, 2]
    bounds = _bounds(np.concatenate([all_rx_all, all_ry_all]), 99)
    ax.hist(all_rx_all, bins=60, alpha=0.35, color='red',
            label=f'X all (n={len(all_rx_all)})', range=bounds)
    ax.hist(all_ry_all, bins=60, alpha=0.35, color='blue',
            label=f'Y all (n={len(all_ry_all)})', range=bounds)
    ax.hist(all_rx_ret, bins=60, alpha=0.8, color='darkred', histtype='step', linewidth=1.5,
            label=f'X retained (n={len(all_rx_ret)})', range=bounds)
    ax.hist(all_ry_ret, bins=60, alpha=0.8, color='darkblue', histtype='step', linewidth=1.5,
            label=f'Y retained (n={len(all_ry_ret)})', range=bounds)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Residual (nm)'); ax.set_title('B. Pooled residual distribution (XY)')
    ax.legend(fontsize=7)

    # ----- Row 4: Z residual dist + spatial + radius -----
    ax = axes[3, 0]
    bz = _bounds(all_rz_all, 99) if all_rz_all.size else (-1, 1)
    ax.hist(all_rz_all, bins=60, alpha=0.5, color='green',
            label=f'Z all (n={len(all_rz_all)})', range=bz)
    ax.hist(all_rz_ret, bins=60, alpha=0.9, color='darkgreen', histtype='step', linewidth=1.5,
            label=f'Z retained (n={len(all_rz_ret)})', range=bz)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Residual (nm)'); ax.set_title('B. Pooled residual distribution (Z)')
    ax.legend(fontsize=7)

    ax = axes[3, 1]
    # Spatial residuals at worst TP — use retained XY residuals
    res_worst = residuals_xy_retained[worst_tp]
    if res_worst.shape[0]:
        wids_here = res_worst[:, 0].astype(int)
        xy, mag = [], []
        for i, w in enumerate(wids_here):
            if (int(w), ref_tp) not in pos_488:
                continue
            x_ref = pos_488[(int(w), ref_tp)][2]
            y_ref = pos_488[(int(w), ref_tp)][1]
            xy.append((x_ref, y_ref))
            ry = res_worst[i, 1] * XY_PIXEL_NM
            rx = res_worst[i, 2] * XY_PIXEL_NM
            mag.append(np.hypot(rx, ry))
        xy = np.array(xy); mag = np.array(mag)
        if len(xy):
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=mag, s=36, cmap='viridis',
                            vmin=0, vmax=np.percentile(mag, 95) if len(mag) else 1)
            plt.colorbar(sc, ax=ax, label='|XY residual| (nm)')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title(f'C. Spatial XY residuals @ worst TP ({worst_tp.replace("_CYC000","")}, retained)')
    ax.set_aspect('equal'); ax.invert_yaxis()

    ax = axes[3, 2]
    cx_px = cy_px = FOV_SIZE / 2
    radii, mags_xy, mags_z = [], [], []
    for tp in tps_sorted:
        r_xy = residuals_xy_retained[tp]
        for i in range(r_xy.shape[0]):
            wid = int(r_xy[i, 0])
            if (wid, ref_tp) not in pos_488: continue
            x_ref = pos_488[(wid, ref_tp)][2]; y_ref = pos_488[(wid, ref_tp)][1]
            radii.append(np.hypot(x_ref - cx_px, y_ref - cy_px) * XY_PIXEL_NM)
            mags_xy.append(np.hypot(r_xy[i, 1] * XY_PIXEL_NM, r_xy[i, 2] * XY_PIXEL_NM))
        r_z = residuals_z_retained[tp]
        for i in range(r_z.shape[0]):
            wid = int(r_z[i, 0])
            if (wid, ref_tp) not in pos_488: continue
            x_ref = pos_488[(wid, ref_tp)][2]; y_ref = pos_488[(wid, ref_tp)][1]
            # Radius for Z points is computed the same way but kept separate
            mags_z.append(abs(r_z[i, 1]) * Z_STEP_NM)
    radii = np.asarray(radii); mags_xy = np.asarray(mags_xy); mags_z = np.asarray(mags_z)
    if radii.size:
        nbins = 10
        edges = np.linspace(0, radii.max(), nbins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        med_xy, lo_xy, hi_xy = [], [], []
        for j in range(nbins):
            sel = (radii >= edges[j]) & (radii < edges[j + 1])
            if sel.sum() < 3:
                med_xy.append(np.nan); lo_xy.append(np.nan); hi_xy.append(np.nan); continue
            med_xy.append(np.median(mags_xy[sel]))
            lo_xy.append(np.percentile(mags_xy[sel], 25))
            hi_xy.append(np.percentile(mags_xy[sel], 75))
        ax.plot(centers, med_xy, 'b-o', label='|XY| median (retained)')
        ax.fill_between(centers, lo_xy, hi_xy, alpha=0.2, color='blue')
    ax.set_xlabel('Radius from FOV center (nm)'); ax.set_ylabel('|residual| (nm)')
    ax.set_title('C. XY residual magnitude vs radius (pooled TPs, retained)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    m = summary_metrics
    fig.suptitle(
        f'Drift Correction QC — decoupled XY/Z stability, axis-specific (X/Y raw+interp, Z poly{Z_POLY_DEGREE})\n'
        f'tp_coverage={tp_cov_min:.2f}  |  XY-stable={m["n_stable_xy"]} / Z-stable={m["n_stable_z"]} / sampled={m["n_sampled"]}  |  '
        f'[all] median |res| X={m["median_abs_x_nm"]:.1f}, Y={m["median_abs_y_nm"]:.1f}, Z={m["median_abs_z_nm"]:.1f} nm  |  '
        f'[retained] X={m["median_abs_x_nm_ret"]:.1f}, Y={m["median_abs_y_nm_ret"]:.1f}, Z={m["median_abs_z_nm_ret"]:.1f} nm',
        fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    out_path = os.path.join(OUTDIR, 'qc_drift_model.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Drift-model QC saved to {out_path}')


def make_roi_diagnostics_qc(all_tps, ref_tp, pos_488, stable_xy, stable_z,
                            per_roi_xy, per_roi_z,
                            drift_raw, drift_final, roi_metrics, tp_cov_min):
    """ROI reference-quality QC with decoupled XY vs Z stability views.

    Row 1: per-axis median |residual| distributions (stable vs rejected),
           where stability membership is per-axis-group.
    Row 2: TP coverage (vs effective tp_cov_min), per-axis outlier fractions,
           and the decoupled spatial map (XY-only / Z-only / XY ∩ Z).
    Row 3: example fiducial trajectories drawn from the relevant stable pool
           per axis (X/Y from stable_xy; Z from stable_z).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_idx = np.array([tp_index(tp) for tp in all_tps])
    order = np.argsort(t_idx)
    tps_sorted = [all_tps[i] for i in order]
    tp_labels = [f'{t:03d}' for t in t_idx[order]]
    x_pos = np.arange(len(tps_sorted))

    rng = np.random.default_rng(42)
    sample_xy = sorted(stable_xy)
    if len(sample_xy) > 15:
        sample_xy = list(rng.choice(sample_xy, size=15, replace=False))
    sample_z = sorted(stable_z)
    if len(sample_z) > 15:
        sample_z = list(rng.choice(sample_z, size=15, replace=False))

    def _traj(per_roi_dict, wid, col):
        vals = np.full(len(tps_sorted), np.nan)
        for i, tp in enumerate(tps_sorted):
            arr = per_roi_dict[tp]
            if arr.shape[0] == 0: continue
            m = arr[:, 0] == wid
            if m.any():
                vals[i] = arr[m, col][0]
        return vals

    dz_rw = np.array([drift_raw[tp][AXIS_Z] for tp in tps_sorted]) * Z_STEP_NM
    dy_rw = np.array([drift_raw[tp][AXIS_Y] for tp in tps_sorted]) * XY_PIXEL_NM
    dx_rw = np.array([drift_raw[tp][AXIS_X] for tp in tps_sorted]) * XY_PIXEL_NM
    dz_fn = np.array([drift_final[tp][AXIS_Z] for tp in tps_sorted]) * Z_STEP_NM
    dy_fn = np.array([drift_final[tp][AXIS_Y] for tp in tps_sorted]) * XY_PIXEL_NM
    dx_fn = np.array([drift_final[tp][AXIS_X] for tp in tps_sorted]) * XY_PIXEL_NM

    all_wids = sorted(roi_metrics.keys())

    fig, axes = plt.subplots(3, 3, figsize=(22, 18))

    # ---- Row 1: per-axis residual distributions (stable-vs-rejected per axis group) ----
    def _hist_metric(ax, name, key, unit_nm, thresh_nm, stable_set):
        vals_stable = np.array([roi_metrics[w][key] * unit_nm
                                for w in all_wids if w in stable_set])
        vals_reject = np.array([roi_metrics[w][key] * unit_nm
                                for w in all_wids if w not in stable_set])
        finite = np.concatenate([vals_stable, vals_reject])
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        hi = min(np.percentile(finite, 99) * 1.2, thresh_nm * 4)
        ax.hist(vals_stable, bins=50, alpha=0.7, color='green', range=(0, hi),
                label=f'Stable ({len(vals_stable)})')
        ax.hist(vals_reject, bins=50, alpha=0.5, color='red', range=(0, hi),
                label=f'Rejected ({len(vals_reject)})')
        ax.axvline(thresh_nm, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold: {thresh_nm:.0f} nm')
        ax.set_xlabel(f'{name} (nm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _hist_metric(axes[0, 0], 'Median |res| X', 'med_abs_res_x', XY_PIXEL_NM,
                 STABLE_MED_ABS_RES_XY_NM, stable_xy)
    axes[0, 0].set_title('D. Median |res| X — XY-stable vs XY-rejected')

    _hist_metric(axes[0, 1], 'Median |res| Y', 'med_abs_res_y', XY_PIXEL_NM,
                 STABLE_MED_ABS_RES_XY_NM, stable_xy)
    axes[0, 1].set_title('D. Median |res| Y — XY-stable vs XY-rejected')

    _hist_metric(axes[0, 2], 'Median |res| Z', 'med_abs_res_z', Z_STEP_NM,
                 STABLE_MED_ABS_RES_Z_NM, stable_z)
    axes[0, 2].set_title('D. Median |res| Z — Z-stable vs Z-rejected')

    # ---- Row 2: tp_coverage + per-axis outlier fractions + decoupled spatial map ----
    ax = axes[1, 0]
    # Coverage uses the effective CLI value (Fix #1).
    cov_xy_stable = [roi_metrics[w]['tp_coverage'] for w in stable_xy]
    cov_z_stable  = [roi_metrics[w]['tp_coverage'] for w in stable_z]
    cov_reject    = [roi_metrics[w]['tp_coverage'] for w in all_wids
                     if w not in stable_xy and w not in stable_z]
    ax.hist(cov_reject, bins=30, alpha=0.4, color='red', range=(0, 1),
            label=f'Rejected by both ({len(cov_reject)})')
    ax.hist(cov_xy_stable, bins=30, alpha=0.6, color='purple', range=(0, 1),
            histtype='step', linewidth=1.5, label=f'XY-stable ({len(cov_xy_stable)})')
    ax.hist(cov_z_stable, bins=30, alpha=0.6, color='green', range=(0, 1),
            histtype='step', linewidth=1.5, label=f'Z-stable ({len(cov_z_stable)})')
    ax.axvline(tp_cov_min, color='black', linestyle='--',
               label=f'Threshold (effective): {tp_cov_min:.2f}')
    ax.set_xlabel('TP coverage fraction')
    ax.set_title('D. TP coverage (effective threshold)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    of_xy_stable = [roi_metrics[w]['outlier_frac_xy'] for w in stable_xy]
    of_xy_reject = [roi_metrics[w]['outlier_frac_xy'] for w in all_wids if w not in stable_xy]
    of_z_stable  = [roi_metrics[w]['outlier_frac_z']  for w in stable_z]
    of_z_reject  = [roi_metrics[w]['outlier_frac_z']  for w in all_wids if w not in stable_z]
    ax.hist(of_xy_reject, bins=30, alpha=0.3, color='red', range=(0, 1),
            label=f'XY rejected ({len(of_xy_reject)})')
    ax.hist(of_xy_stable, bins=30, alpha=0.6, color='purple', range=(0, 1),
            label=f'XY stable ({len(of_xy_stable)})')
    ax.hist(of_z_reject, bins=30, alpha=0.3, color='darkorange', range=(0, 1),
            histtype='step', linewidth=1.5, label=f'Z rejected ({len(of_z_reject)})')
    ax.hist(of_z_stable, bins=30, alpha=0.6, color='green', range=(0, 1),
            histtype='step', linewidth=1.5, label=f'Z stable ({len(of_z_stable)})')
    ax.axvline(STABLE_OUTLIER_FRAC_MAX, color='black', linestyle='--',
               label=f'Threshold: {STABLE_OUTLIER_FRAC_MAX:.2f}')
    ax.set_xlabel('Outlier fraction')
    ax.set_title('D. Per-axis outlier fraction')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    def _xy(wids):
        return np.array([(pos_488[(w, ref_tp)][2], pos_488[(w, ref_tp)][1])
                         for w in wids if (w, ref_tp) in pos_488])
    both    = stable_xy & stable_z
    xy_only = stable_xy - stable_z
    z_only  = stable_z  - stable_xy
    rejected = set(all_wids) - stable_xy - stable_z
    p_all_ref = _xy(all_wids)
    p_reject  = _xy(rejected)
    p_both    = _xy(both)
    p_xy_only = _xy(xy_only)
    p_z_only  = _xy(z_only)
    if len(p_all_ref):
        ax.scatter(p_all_ref[:, 0], p_all_ref[:, 1], s=2, c='gray', alpha=0.15,
                   label=f'all sampled ({len(p_all_ref)})')
    if len(p_reject):
        ax.scatter(p_reject[:, 0], p_reject[:, 1], s=12, c='red', marker='x',
                   alpha=0.7, label=f'Rejected both ({len(p_reject)})')
    if len(p_both):
        ax.scatter(p_both[:, 0], p_both[:, 1], s=15, c='purple', marker='o',
                   alpha=0.8, label=f'XY ∩ Z stable ({len(p_both)})')
    if len(p_xy_only):
        ax.scatter(p_xy_only[:, 0], p_xy_only[:, 1], s=15, c='red', marker='s',
                   alpha=0.8, label=f'XY-only stable ({len(p_xy_only)})')
    if len(p_z_only):
        ax.scatter(p_z_only[:, 0], p_z_only[:, 1], s=15, c='green', marker='D',
                   alpha=0.8, label=f'Z-only stable ({len(p_z_only)})')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title('D. Decoupled stability in FOV (XY vs Z)')
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower right')

    # ---- Row 3: example trajectories — sampled from the per-axis stable pools ----
    ax = axes[2, 0]
    for wid in sample_xy:
        ax.plot(x_pos, _traj(per_roi_xy, wid, AXIS_X + 1) * XY_PIXEL_NM,
                '-', color='gray', alpha=0.4, linewidth=0.8)
    ax.plot(x_pos, dx_rw, 'o', color='red', alpha=0.6, label='Robust per-TP')
    ax.plot(x_pos, dx_fn, '-', color='red', linewidth=2, label='Final (applied)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('X displacement (nm)')
    ax.set_title(f'D. XY-stable trajectories — X (n={len(sample_xy)})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for wid in sample_xy:
        ax.plot(x_pos, _traj(per_roi_xy, wid, AXIS_Y + 1) * XY_PIXEL_NM,
                '-', color='gray', alpha=0.4, linewidth=0.8)
    ax.plot(x_pos, dy_rw, 'o', color='blue', alpha=0.6, label='Robust per-TP')
    ax.plot(x_pos, dy_fn, '-', color='blue', linewidth=2, label='Final (applied)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Y displacement (nm)')
    ax.set_title(f'D. XY-stable trajectories — Y (n={len(sample_xy)})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    for wid in sample_z:
        ax.plot(x_pos, _traj(per_roi_z, wid, AXIS_Z + 1) * Z_STEP_NM,
                '-', color='gray', alpha=0.4, linewidth=0.8)
    ax.plot(x_pos, dz_rw, 'o', color='green', alpha=0.6, label='Robust per-TP')
    ax.plot(x_pos, dz_fn, '-', color='green', linewidth=2, label='Final (applied)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Z displacement (nm)')
    ax.set_title(f'D. Z-stable trajectories — Z (n={len(sample_z)})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Drift QC — ROI stability diagnostics (decoupled XY/Z)  '
        f'(tp_cov>={tp_cov_min:.2f}, outlier<={STABLE_OUTLIER_FRAC_MAX}, '
        f'med|res|XY<={STABLE_MED_ABS_RES_XY_NM:.0f}nm, '
        f'med|res|Z<={STABLE_MED_ABS_RES_Z_NM:.0f}nm, '
        f'MAD XY<={STABLE_MAD_RES_XY_NM:.0f}nm, MAD Z<={STABLE_MAD_RES_Z_NM:.0f}nm)',
        fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    out_path = os.path.join(OUTDIR, 'qc_drift_roi.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'ROI diagnostics QC saved to {out_path}')


if __name__ == '__main__':
    # main() returns a nonzero code on a fatal argument
    # conflict; a bare call would discard it and exit 0.
    raise SystemExit(main() or 0)
