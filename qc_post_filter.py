#!/usr/bin/env python
"""Post-filter QC / follow-up analysis for drift-corrected, fully filtered walks.

This module is a *follow-up QC / analysis stage* meant to run after
`drift_correct.py` finishes its drift correction + full-coverage filter +
movement filter. At that point each WalkID is expected to contribute one
drift-corrected point per timepoint. This is NOT a cloud / centroid
analysis, it is a one-point-per-TP analysis.

The user supplies an inclusive, 1-based timepoint range
(`tp_start_1based`, `tp_end_1based`). All matrices and parabolas are
computed on exactly that range.

Adapted from `~/dtm_cv_pipeline/walk_curator_lite.py`:
  - `_compute_diagonal_means_for_walk`  -> `_diagonal_means_from_matrix` here,
    taking a prebuilt (N, N) distance matrix for an already-ordered trace.
  - `_fit_parabola`                     -> kept verbatim in spirit, R2 only
                                           (confidence-interval bands dropped).
  - `plot_mean_parabola_xyz` 4-panel layout (3D / X / Y / Z, scatter + fit + R2).
  - Distance-matrix figure: 4-panel heatmap (3D / X / Y / Z), matrix per walk
    from the ordered point sequence (no sampling, no centroid step), then
    averaged across walks and normalized to [0, 1] per panel.

Intentional simplifications vs. the lite version:
  - No centroid aggregation. Each walk is a single ordered sequence of one
    point per TP, so pdist is applied directly.
  - No PCA / capsule-frame / PC123 variants.
  - No sampled mean-pairwise.
  - No individual-per-walk figures.
  - No approved / rejected bookkeeping (the caller is expected to hand over
    the final downstream-ready walk CSV).
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# TP naming: `TP###_CYC###`. The integer portion is 0-based (TP000 is the
# first timepoint). Duplicated here so this module is self-contained.
# ---------------------------------------------------------------------------
# Accept both TP### and TP###_CYC### — CYC defaults to absent.
TP_RE = re.compile(r'^TP(\d+)(?:_CYC(\d+))?$')


def _tp_index(tp_name):
    """Return the 0-based TP integer from a name like 'TP007_CYC000'."""
    m = TP_RE.match(tp_name)
    if m is None:
        raise ValueError(f"Unrecognized TP name: {tp_name!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Fit configuration (conservative defaults; tune here without a CLI surface).
#
# Terminal offsets are statistically weaker: a diagonal at offset d of an
# N x N distance matrix has only (N - d) pair-entries, so each walk's
# contribution at large d is a mean of very few numbers. On top of that,
# some late TPs add structured drift/noise. The constants below let us
# drop or downweight those offsets before the quadratic fit.
# ---------------------------------------------------------------------------
FIT_MIN_WALKS_PER_OFFSET        = 3     # absolute floor on walks per offset
FIT_MIN_FRAC_WALKS_PER_OFFSET   = 0.5   # also require at least this fraction of n_walks_used
FIT_MIN_PAIRS_PER_WALK          = 2     # drop offsets where each walk has < this many diagonal entries (i.e. N - d)
FIT_DROP_TERMINAL_OFFSETS       = 0     # optionally drop the top-k offsets before fitting


# ---------------------------------------------------------------------------
# Weighted parabola fit (adapted from walk_curator_lite._fit_parabola; CI
# dropped, weighting added). Returns both R2 variants so the caller can
# report the fit honestly.
# ---------------------------------------------------------------------------
def _fit_parabola(x, y, weights=None):
    """Fit y ~ a x^2 + b x + c, optionally weighted.

    Parameters
    ----------
    x, y : 1-D arrays of equal length.
    weights : optional 1-D array of non-negative weights, same length as x.
        Passed to `numpy.polyfit(..., w=weights)`. If None, the fit is
        unweighted.

    Returns
    -------
    dict with keys:
        'coeffs'         : (a, b, c) or None if fewer than 3 points.
        'x_smooth'       : dense x for plotting, or None.
        'y_smooth'       : dense fitted y, or None.
        'R2_unweighted'  : standard R^2 on the aggregated y.
        'R2_weighted'    : weight-scaled R^2 when weights is not None,
                           else equal to R2_unweighted.

    Notes on R^2
    ------------
    The weighted R^2 uses the weighted mean in the total sum of squares
    and weights both residuals, which is the consistent metric for a
    weighted least-squares fit. The unweighted R^2 is computed on the
    same aggregated points with equal weights, as a sanity check and for
    easy comparison against the prior unweighted fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return {'coeffs': None, 'x_smooth': None, 'y_smooth': None,
                'R2_unweighted': 0.0, 'R2_weighted': 0.0}

    if weights is None:
        coeffs = np.polyfit(x, y, 2)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != x.shape:
            raise ValueError("weights must have the same shape as x")
        coeffs = np.polyfit(x, y, 2, w=weights)

    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = poly(x_smooth)
    y_fit = poly(x)

    res = y - y_fit
    ss_res_u = float(np.sum(res ** 2))
    ss_tot_u = float(np.sum((y - y.mean()) ** 2))
    R2_u = 1.0 - ss_res_u / ss_tot_u if ss_tot_u > 0 else 0.0

    if weights is None:
        R2_w = R2_u
    else:
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            y_wmean = float(np.sum(weights * y) / w_sum)
            ss_res_w = float(np.sum(weights * res ** 2))
            ss_tot_w = float(np.sum(weights * (y - y_wmean) ** 2))
            R2_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0.0
        else:
            R2_w = 0.0

    return {'coeffs': coeffs, 'x_smooth': x_smooth, 'y_smooth': y_smooth,
            'R2_unweighted': R2_u, 'R2_weighted': R2_w}


# ---------------------------------------------------------------------------
# Per-offset aggregation stats across walks.
#
# The prior implementation computed a plain mean across walks at each
# offset. We now return n_walks, mean, median, std, mad_sigma, sem along
# with the per-walk diagonal support, so the fit can use a robust center
# (median) and a robust uncertainty (mad_sigma / sqrt(n_walks)).
# ---------------------------------------------------------------------------
def _offset_stats(offsets_pool, values_pool, n_tp_in_range):
    """Aggregate pooled (offset, per-walk diagonal mean) pairs by offset.

    Returns a dict { offset : stats_dict } where stats_dict contains:
        n_walks           : int, walks contributing at this offset
        n_pairs_per_walk  : int, = n_tp_in_range - offset (diagonal length)
        mean              : float, mean across walks
        median            : float, median across walks
        std               : float, sample std across walks (ddof=1 when n>=2)
        mad_sigma         : float, 1.4826 * MAD across walks
        sem               : float, std / sqrt(n_walks), NaN when n < 2
        sem_robust        : float, mad_sigma / sqrt(n_walks), NaN when n < 2
    """
    offsets_pool = np.asarray(offsets_pool)
    values_pool = np.asarray(values_pool, dtype=float)
    stats = {}
    for d in np.unique(offsets_pool):
        vals = values_pool[offsets_pool == d]
        n = int(vals.size)
        if n == 0:
            continue
        mean = float(vals.mean())
        median = float(np.median(vals))
        if n >= 2:
            std = float(vals.std(ddof=1))
            sem = std / np.sqrt(n)
        else:
            std = float('nan')
            sem = float('nan')
        abs_dev = np.abs(vals - median)
        mad = float(np.median(abs_dev))
        mad_sigma = 1.4826 * mad
        sem_robust = mad_sigma / np.sqrt(n) if n >= 2 else float('nan')
        stats[int(d)] = {
            'n_walks': n,
            'n_pairs_per_walk': int(n_tp_in_range - int(d)),
            'mean': mean,
            'median': median,
            'std': std,
            'mad_sigma': mad_sigma,
            'sem': sem,
            'sem_robust': sem_robust,
        }
    return stats


# ---------------------------------------------------------------------------
# Diagonal means of a distance matrix (adapted from
# walk_curator_lite._compute_diagonal_means_for_walk). The lite version
# re-derives the matrix from a centroid dict each call. Here each walk is
# already an ordered (N, 3) sequence, so we pass the (N, N) matrix directly.
# ---------------------------------------------------------------------------
def _diagonal_means_from_matrix(D):
    """Return (offsets, means) for a square distance matrix.

    For each genomic distance d in [1, N) compute the mean of D.diagonal(d).
    """
    n = D.shape[0]
    offsets = []
    means = []
    for d in range(1, n):
        diag = np.diagonal(D, offset=d)
        if diag.size:
            offsets.append(d)
            means.append(diag.mean())
    return np.array(offsets, dtype=int), np.array(means, dtype=float)


# ---------------------------------------------------------------------------
# Per-walk distance matrices for 3D and each axis (X / Y / Z).
# Each walk is an (N, 3) array in nm, already ordered by TP.
# ---------------------------------------------------------------------------
def _per_walk_matrices(points_nm):
    """Return (D_3d, D_x, D_y, D_z) for one walk.

    points_nm : (N, 3) ndarray of drift-corrected nm coordinates ordered
                by TP across the selected analysis range.
    """
    from scipy.spatial.distance import pdist, squareform

    D_3d = squareform(pdist(points_nm))
    x = points_nm[:, 0:1]
    y = points_nm[:, 1:2]
    z = points_nm[:, 2:3]
    D_x = np.abs(x - x.T)
    D_y = np.abs(y - y.T)
    D_z = np.abs(z - z.T)
    return D_3d, D_x, D_y, D_z


# ---------------------------------------------------------------------------
# Load + validate: build per-walk ordered point sequences over the
# user-specified analysis range.
# ---------------------------------------------------------------------------
def _load_ordered_walks(filtered_csv, tp_start_0, tp_end_0,
                        skip_incomplete=False):
    """Build per-walk (N, 3) arrays over the inclusive 0-based TP range.

    Enforces the "one point per TP" assumption for this project. A walk is
    skipped (and counted) if:
      - any required TP is missing for that walk, OR
      - any required TP has more than one row for that walk.

    Empty / non-finite coordinate cells (``x_nm``, ``y_nm``, ``z_nm``)
    are handled per the ``skip_incomplete`` flag:
      - False (default): raise a clear ValueError pointing at the
        offending WalkID + TP + column. Refuses to silently use bogus
        data.
      - True: treat the row as missing for that walk. The walk then
        falls into the existing ``skipped_missing_tp`` counter because
        the required TP no longer has a row.

    Returns:
        walks_ok : dict[int, np.ndarray]      {wid: (N, 3) array in nm}
        present_tp_names : list[str]          TP name per position (length N)
        present_tp_indices : list[int]        0-based TP index per position
        skipped_missing : int                 walks skipped due to missing TPs
        skipped_duplicate : int               walks skipped due to >1 rows / (wid, tp)
        n_walks_input : int                   total distinct WalkIDs seen in CSV
        skipped_empty_coord_rows : int        rows dropped because their
                                              xyz_nm cells were empty or
                                              non-finite (only when
                                              skip_incomplete=True)
    """
    required_tp0 = list(range(tp_start_0, tp_end_0 + 1))
    required_set = set(required_tp0)

    per_wid = defaultdict(lambda: defaultdict(list))
    tp_names_seen = {}
    skipped_empty_coord_rows = 0

    def _row_xyz(row, tp_name, wid):
        """Return (x, y, z) floats from a row, or raise / return None per
        the skip_incomplete policy. Handles empty strings, whitespace,
        non-numeric, and non-finite values uniformly."""
        out = []
        for k in ('x_nm', 'y_nm', 'z_nm'):
            raw = (row.get(k) or '').strip()
            if raw == '':
                if skip_incomplete:
                    return None
                raise ValueError(
                    f'{filtered_csv}: empty {k} for WalkID={wid} '
                    f'tp={tp_name!r}. Pass --skip_incomplete to drop '
                    'walks with incomplete coordinate rows, or fix the '
                    'upstream CSV.')
            try:
                v = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f'{filtered_csv}: non-numeric {k}={raw!r} for '
                    f'WalkID={wid} tp={tp_name!r}: {exc}') from exc
            if not np.isfinite(v):
                if skip_incomplete:
                    return None
                raise ValueError(
                    f'{filtered_csv}: non-finite {k}={raw!r} for '
                    f'WalkID={wid} tp={tp_name!r}. Pass '
                    '--skip_incomplete to drop such walks.')
            out.append(v)
        return out

    with open(filtered_csv) as f:
        reader = csv.DictReader(f)
        required_cols = {'WalkID', 'tp', 'x_nm', 'y_nm', 'z_nm'}
        missing_cols = required_cols - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(
                f"{filtered_csv}: missing required columns {sorted(missing_cols)}")
        for row in reader:
            tp_name = row['tp']
            tp0 = _tp_index(tp_name)
            if tp0 not in required_set:
                continue
            tp_names_seen[tp0] = tp_name
            wid = int(row['WalkID'])
            xyz = _row_xyz(row, tp_name, wid)
            if xyz is None:
                skipped_empty_coord_rows += 1
                continue
            per_wid[wid][tp0].append(np.array(xyz, dtype=float))

    available_tp0 = set(tp_names_seen.keys())
    missing_in_csv = [t for t in required_tp0 if t not in available_tp0]
    if missing_in_csv:
        raise ValueError(
            "Requested analysis TP range includes TP(s) that are not present "
            f"in {filtered_csv}: missing 0-based TP indices {missing_in_csv} "
            f"(1-based: {[t + 1 for t in missing_in_csv]}). "
            "Refusing to silently clamp."
        )

    present_tp_indices = required_tp0
    present_tp_names = [tp_names_seen[t] for t in present_tp_indices]

    walks_ok = {}
    skipped_missing = 0
    skipped_duplicate = 0
    n_walks_input = len(per_wid)
    for wid, tp_map in per_wid.items():
        has_all = all(t in tp_map for t in present_tp_indices)
        if not has_all:
            skipped_missing += 1
            continue
        has_dupe = any(len(tp_map[t]) != 1 for t in present_tp_indices)
        if has_dupe:
            skipped_duplicate += 1
            continue
        pts = np.vstack([tp_map[t][0] for t in present_tp_indices])
        walks_ok[wid] = pts

    return (walks_ok, present_tp_names, present_tp_indices,
            skipped_missing, skipped_duplicate, n_walks_input,
            skipped_empty_coord_rows)


# ---------------------------------------------------------------------------
# Plotting: 4-panel mean distance matrices (3D, X, Y, Z).
# ---------------------------------------------------------------------------
def compute_shared_matrix_display_limits(matrices):
    """Return (vmin, vmax) suitable as a shared colormap range for a list
    of square distance matrices. vmin is always 0 (distances are
    non-negative). vmax is the max over OFF-diagonal entries across every
    input matrix so that the zero diagonal does not dominate the scale.

    If no off-diagonal entries are available (e.g. every matrix is 1x1
    or empty) we fall back to the global raw max and, if that is still
    0 (pathological degenerate inputs), to 1.0 so `vmin == vmax` is
    never produced.

    The matrix values themselves are NOT modified. Only the display
    range is constrained.
    """
    off_diag_max = 0.0
    raw_max = 0.0
    for M in matrices:
        if M is None or M.size == 0:
            continue
        raw_max = max(raw_max, float(M.max()))
        if M.ndim == 2 and M.shape[0] >= 2:
            # Mask the diagonal without mutating the input.
            mask = ~np.eye(M.shape[0], dtype=bool)
            off_vals = M[mask]
            if off_vals.size:
                off_diag_max = max(off_diag_max, float(off_vals.max()))
    vmax = off_diag_max if off_diag_max > 0 else raw_max
    if vmax <= 0:
        vmax = 1.0
    return 0.0, vmax


def build_mean_distance_matrices_figure(walks_ok,
                                        tp_start_1based, tp_end_1based,
                                        axis_names=('X', 'Y', 'Z'),
                                        space_label='XYZ',
                                        global_colorbar=False,
                                        axis_caps_nm=None):
    """Build the 4-panel mean-distance-matrices figure.

    Values are the MEAN (across walks) of the per-walk distance matrices
    in nm. Raw matrix values are never normalized.

    Scaling is controlled by `axis_caps_nm` and `global_colorbar`:
      - axis_caps_nm: optional 4-tuple `(cap_3d, cap_a1, cap_a2, cap_a3)`
        in nm. When provided, the four panels use these as their `vmax`
        instead of an auto-vmax from the data. Pass `None` for the
        default off-diagonal-max behaviour.
      - global_colorbar=False (default): each panel uses its own scale.
        With axis_caps_nm set, that scale is the corresponding cap.
        Without caps, vmax is the off-diagonal max of that panel.
      - global_colorbar=True: all four panels share one colorbar. With
        caps set, the shared vmax is `cap_a3` (Z or PC3). Without caps,
        the shared vmax is the off-diagonal max of the third axis.

    The axis_names kwarg controls how the three per-axis panels are
    labeled. Default `('X', 'Y', 'Z')` preserves the current behavior
    for XYZ-space callers; pass `('PC1', 'PC2', 'PC3')` when the walks
    have already been transformed into the constrained PC basis.
    `space_label` appears in the figure suptitle.

    Returns (fig, n_walks). Does not save or close the figure; callers
    are responsible for `fig.savefig(...)` and `plt.close(fig)`. Use this
    when the figure needs to be embedded into a PDF in addition to / in
    place of the standalone PNG written by `_plot_mean_distance_matrices`.
    """
    if axis_caps_nm is not None:
        if len(axis_caps_nm) != 4:
            raise ValueError(
                f'axis_caps_nm must have 4 entries (3D, a1, a2, a3); '
                f'got {len(axis_caps_nm)}')
        axis_caps_nm = tuple(float(v) for v in axis_caps_nm)
        if any(c <= 0 for c in axis_caps_nm):
            raise ValueError(
                f'axis_caps_nm values must be > 0; got {axis_caps_nm}')
    br = LinearSegmentedColormap.from_list(
        'bright_red', [(1, 0, 0), (1, 1, 1)], N=256)

    D_3d_list, D_x_list, D_y_list, D_z_list = [], [], [], []
    for pts in walks_ok.values():
        D3, Dx, Dy, Dz = _per_walk_matrices(pts)
        D_3d_list.append(D3)
        D_x_list.append(Dx)
        D_y_list.append(Dy)
        D_z_list.append(Dz)

    n_walks = len(D_3d_list)
    if n_walks == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                f'No valid walks in TP range [{tp_start_1based}, {tp_end_1based}] (1-based).',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        return fig, 0

    D_3d_mean = np.mean(D_3d_list, axis=0)
    D_x_mean = np.mean(D_x_list, axis=0)
    D_y_mean = np.mean(D_y_list, axis=0)
    D_z_mean = np.mean(D_z_list, axis=0)

    a1, a2, a3 = axis_names
    panels = [
        (D_3d_mean, '3D distance'),
        (D_x_mean,  f'{a1} projection'),
        (D_y_mean,  f'{a2} projection'),
        (D_z_mean,  f'{a3} projection'),
    ]

    scale_axis = axis_names[-1]  # 'Z' or 'PC3'
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    if global_colorbar:
        # One shared range. Use the third-axis cap when provided, else
        # the third-axis off-diagonal max.
        if axis_caps_nm is not None:
            vmin, vmax = 0.0, axis_caps_nm[3]
        else:
            vmin, vmax = compute_shared_matrix_display_limits([D_z_mean])
        fig.suptitle(
            f'Distance Matrices {space_label}',
            fontsize=14, fontweight='bold')
        last_im = None
        for ax, (M, title) in zip(flat, panels):
            last_im = ax.imshow(M, cmap=br, aspect='auto',
                                interpolation='nearest',
                                vmin=vmin, vmax=vmax)
            ax.set_xlabel('TP position in range')
            ax.set_ylabel('TP position in range')
            ax.set_title(f'{title} (nm, mean across {n_walks} walks)')
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            shrink=0.85, pad=0.02)
        cbar.set_label('mean distance (nm)')
    else:
        # Per-panel scaling: each panel uses its own cap if provided,
        # otherwise its own off-diagonal max. Raw nm values throughout.
        fig.suptitle(
            f'Distance Matrices {space_label}',
            fontsize=14, fontweight='bold')
        for idx, (ax, (M, title)) in enumerate(zip(flat, panels)):
            if axis_caps_nm is not None:
                vmin_p, vmax_p = 0.0, axis_caps_nm[idx]
            else:
                vmin_p, vmax_p = compute_shared_matrix_display_limits([M])
            im = ax.imshow(M, cmap=br, aspect='auto',
                           interpolation='nearest',
                           vmin=vmin_p, vmax=vmax_p)
            ax.set_xlabel('TP position in range')
            ax.set_ylabel('TP position in range')
            ax.set_title(f'{title} (nm, mean across {n_walks} walks)')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('mean distance (nm)')
        plt.tight_layout()

    return fig, n_walks


def _plot_mean_distance_matrices(walks_ok, out_path,
                                 tp_start_1based, tp_end_1based,
                                 global_colorbar=False):
    """Write a 2x2 heatmap PNG averaged across walks.

    Thin wrapper around `build_mean_distance_matrices_figure` that saves +
    closes the figure. `global_colorbar` is forwarded unchanged.
    """
    fig, n_walks = build_mean_distance_matrices_figure(
        walks_ok, tp_start_1based, tp_end_1based,
        global_colorbar=global_colorbar)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n_walks


# ---------------------------------------------------------------------------
# Support-based offset selection for the fit.
#
# In this project the support per offset is (n_walks, n_pairs_per_walk).
# Because every walk in `walks_ok` has one point per TP across the full
# analysis range, n_walks is effectively constant across offsets, and
# "terminal noise" really comes from the pair-count dropping to 1 or 2.
# We still apply a walks-based minimum as a defensive floor, and honor
# an optional top-k terminal-drop knob.
# ---------------------------------------------------------------------------
def _select_fit_offsets(stats, n_walks_used):
    """Return (included_offsets, excluded_reasons).

    `included_offsets` is a sorted list of offsets that pass every rule.
    `excluded_reasons` is a dict { offset : reason_str } for the rest.
    """
    all_offsets = sorted(stats.keys())
    walks_floor = max(
        FIT_MIN_WALKS_PER_OFFSET,
        int(np.ceil(FIT_MIN_FRAC_WALKS_PER_OFFSET * n_walks_used)),
    )
    drop_top_k = max(0, int(FIT_DROP_TERMINAL_OFFSETS))
    terminal_drop_set = set(all_offsets[-drop_top_k:]) if drop_top_k else set()

    included = []
    excluded = {}
    for d in all_offsets:
        s = stats[d]
        reasons = []
        if s['n_walks'] < walks_floor:
            reasons.append(
                f'n_walks={s["n_walks"]} < min({FIT_MIN_WALKS_PER_OFFSET}, '
                f'{FIT_MIN_FRAC_WALKS_PER_OFFSET:.2f}*{n_walks_used}='
                f'{walks_floor})')
        if s['n_pairs_per_walk'] < FIT_MIN_PAIRS_PER_WALK:
            reasons.append(
                f'n_pairs_per_walk={s["n_pairs_per_walk"]} < '
                f'{FIT_MIN_PAIRS_PER_WALK}')
        if d in terminal_drop_set:
            reasons.append(f'in top-{drop_top_k} terminal offsets')
        if reasons:
            excluded[d] = '; '.join(reasons)
        else:
            included.append(d)

    return included, excluded


def _build_weights(stats, included):
    """Choose per-offset fit weights and the name of the scheme used.

    Primary scheme is `1 / sem_robust^2` where sem_robust =
    mad_sigma / sqrt(n_walks). If any included offset has a degenerate
    sem_robust (zero, NaN, or infinite weight), we fall back to
    `w = n_walks` (simple support weighting), which at least tracks the
    walks-level support. The name returned makes this explicit.
    """
    sem_r = np.array([stats[d]['sem_robust'] for d in included], dtype=float)
    bad = ~np.isfinite(sem_r) | (sem_r <= 0)
    if not bad.any():
        w = 1.0 / (sem_r ** 2)
        return w, 'inverse_sem_robust_sq (sem_robust = 1.4826*MAD/sqrt(n))'
    counts = np.array([stats[d]['n_walks'] for d in included], dtype=float)
    return counts, 'fallback:n_walks (primary weights had degenerate sem_robust)'


# ---------------------------------------------------------------------------
# Plotting: 1-panel 3D-only mean parabola. Reuses the same aggregation,
# support filter, weighting scheme, and fit as the 4-panel figure so the
# resulting fit / R^2 numbers are identical to the 3D panel of the
# 4-panel figure.
# ---------------------------------------------------------------------------
def build_mean_parabola_3d_only_figure(walks_ok,
                                       tp_start_1based, tp_end_1based,
                                       n_tp_in_range,
                                       space_label='XYZ'):
    """Build a single-axis 3D-only mean-parabola figure.

    Returns (fig, summary_3d, n_walks_used). The fit details match the
    `3D` entry of the 4-panel figure's summary dict exactly.
    """
    all_g, all_d = [], []
    for pts in walks_ok.values():
        D3, _Dx, _Dy, _Dz = _per_walk_matrices(pts)
        g, d = _diagonal_means_from_matrix(D3)
        all_g.extend(g)
        all_d.extend(d)

    n_walks_used = len(walks_ok)
    empty = {
        'coeffs': None, 'R2': 0.0,
        'R2_unweighted': 0.0, 'R2_weighted': 0.0,
        'aggregation': 'median across walks',
        'fit_method': 'weighted quadratic (y ~ a x^2 + b x + c)',
        'weighting': None,
        'support_per_offset': {},
        'offsets_included': [],
        'offsets_excluded': [],
        'excluded_reasons': {},
    }

    if n_walks_used == 0 or not all_g:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                f'No valid walks in TP range [{tp_start_1based}, {tp_end_1based}] (1-based).',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        return fig, empty, 0

    stats = _offset_stats(all_g, all_d, n_tp_in_range)
    included, excluded = _select_fit_offsets(stats, n_walks_used)

    x_all = np.array(sorted(stats.keys()), dtype=float)
    y_median = np.array([stats[int(d)]['median'] for d in x_all])
    sem_all = np.array([stats[int(d)]['sem_robust'] for d in x_all])
    n_all = np.array([stats[int(d)]['n_walks'] for d in x_all])

    if len(included) >= 3:
        x_fit = np.array(included, dtype=float)
        y_fit = np.array([stats[d]['median'] for d in included])
        weights, weighting_name = _build_weights(stats, included)
        fit = _fit_parabola(x_fit, y_fit, weights=weights)
    else:
        fit = {'coeffs': None, 'x_smooth': None, 'y_smooth': None,
               'R2_unweighted': 0.0, 'R2_weighted': 0.0}
        weighting_name = (f'insufficient included offsets '
                          f'(n={len(included)} < 3)')

    summary_3d = {
        'coeffs': None if fit['coeffs'] is None else fit['coeffs'].tolist(),
        'R2': fit['R2_weighted'],
        'R2_unweighted': fit['R2_unweighted'],
        'R2_weighted': fit['R2_weighted'],
        'aggregation': 'median across walks',
        'fit_method': 'weighted quadratic (y ~ a x^2 + b x + c)',
        'weighting': weighting_name,
        'offsets_included': included,
        'offsets_excluded': sorted(excluded.keys()),
        'excluded_reasons': {int(k): v for k, v in excluded.items()},
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    color = 'steelblue'
    incl_set = set(included)
    incl_mask = np.array([int(x) in incl_set for x in x_all], dtype=bool)
    max_n = max(1, int(n_all.max() if n_all.size else 1))
    sizes = 20 + 90 * (n_all / max_n)
    yerr = np.where(np.isfinite(sem_all), sem_all, 0.0)

    # Excluded offsets are intentionally NOT plotted: their support is
    # below the filter threshold (typically the trailing offset with
    # n_pairs_per_walk < FIT_MIN_PAIRS_PER_WALK) and they would clutter
    # the figure without contributing to the fit.
    if incl_mask.any():
        ax.errorbar(x_all[incl_mask], y_median[incl_mask],
                    yerr=yerr[incl_mask],
                    fmt='none', ecolor=color, alpha=0.8,
                    elinewidth=1, capsize=2, zorder=2)
        ax.scatter(x_all[incl_mask], y_median[incl_mask],
                   s=sizes[incl_mask], color=color, alpha=0.85,
                   edgecolor='k', linewidths=0.4,
                   label=f'included in fit (n={int(incl_mask.sum())})',
                   zorder=3)

    if fit['y_smooth'] is not None:
        ax.plot(fit['x_smooth'], fit['y_smooth'], 'r-', linewidth=2,
                label=(f"weighted fit, R2_w={fit['R2_weighted']:.3f}, "
                       f"R2_u={fit['R2_unweighted']:.3f}"))

    ax.set_xlabel('Contour Distance [s]', fontsize=11)
    ax.set_ylabel('Median 3D distance [nm]', fontsize=11)
    ax.set_xticks(np.arange(0, int(x_all.max()) + 1, 5))
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f'3D distance parabola {space_label}',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, summary_3d, n_walks_used


# ---------------------------------------------------------------------------
# Plotting: 4-panel mean parabola (3D, X, Y, Z).
# ---------------------------------------------------------------------------
def build_mean_parabola_xyz_figure(walks_ok,
                                   tp_start_1based, tp_end_1based,
                                   n_tp_in_range,
                                   axis_names=('X', 'Y', 'Z'),
                                   space_label='XYZ'):
    """Build the 4-panel mean-parabola QC figure.

    The axis_names kwarg renames the three per-axis panels (default
    `('X', 'Y', 'Z')` preserves XYZ behavior). Pass `('PC1','PC2','PC3')`
    for constrained PC-space walks. `space_label` appears in the figure
    suptitle.

    Returns (fig, summary, n_walks_used). Does not save or close. The
    `summary` dict has the per-axis fit details used by the JSON summary
    writer; callers can ignore it when they only need the figure.
    """
    all_g = {'3D': [], 'X': [], 'Y': [], 'Z': []}
    all_d = {'3D': [], 'X': [], 'Y': [], 'Z': []}

    for pts in walks_ok.values():
        D3, Dx, Dy, Dz = _per_walk_matrices(pts)
        for key, D in (('3D', D3), ('X', Dx), ('Y', Dy), ('Z', Dz)):
            g, d = _diagonal_means_from_matrix(D)
            all_g[key].extend(g)
            all_d[key].extend(d)

    n_walks_used = len(walks_ok)
    empty_axis_summary = {
        'coeffs': None, 'R2': 0.0,
        'R2_unweighted': 0.0, 'R2_weighted': 0.0,
        'aggregation': 'median across walks',
        'fit_method': 'weighted quadratic (y ~ a x^2 + b x + c)',
        'weighting': None,
        'support_per_offset': {},
        'offsets_included': [],
        'offsets_excluded': [],
        'excluded_reasons': {},
    }
    summary = {k: dict(empty_axis_summary) for k in ('3D', 'X', 'Y', 'Z')}

    if n_walks_used == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                f'No valid walks in TP range [{tp_start_1based}, {tp_end_1based}] (1-based).',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        return fig, summary, 0

    # Per-axis aggregation + fit. Internal dict keys stay '3D'/'X'/'Y'/'Z'
    # so the summary structure is stable across spaces; only the display
    # strings depend on `axis_names`.
    a1, a2, a3 = axis_names
    axis_colors = {'3D': 'steelblue', 'X': 'green', 'Y': 'orange', 'Z': 'purple'}
    axis_ylabels = {
        '3D': 'Median 3D distance [nm]',
        'X':  f'Median {a1} distance [nm]',
        'Y':  f'Median {a2} distance [nm]',
        'Z':  f'Median {a3} distance [nm]',
    }
    axis_titles = {
        '3D': '3D distance', 'X': f'{a1} projection',
        'Y': f'{a2} projection', 'Z': f'{a3} projection',
    }

    per_axis = {}
    for key in ('3D', 'X', 'Y', 'Z'):
        stats = _offset_stats(all_g[key], all_d[key], n_tp_in_range)
        included, excluded = _select_fit_offsets(stats, n_walks_used)

        x_all = np.array(sorted(stats.keys()), dtype=float)
        y_all_median = np.array([stats[int(d)]['median'] for d in x_all])
        sem_all = np.array([stats[int(d)]['sem_robust'] for d in x_all])
        n_all = np.array([stats[int(d)]['n_walks'] for d in x_all])

        if len(included) >= 3:
            x_fit = np.array(included, dtype=float)
            y_fit = np.array([stats[d]['median'] for d in included])
            weights, weighting_name = _build_weights(stats, included)
            fit = _fit_parabola(x_fit, y_fit, weights=weights)
        else:
            fit = {'coeffs': None, 'x_smooth': None, 'y_smooth': None,
                   'R2_unweighted': 0.0, 'R2_weighted': 0.0}
            weighting_name = ('insufficient included offsets '
                              f'(n={len(included)} < 3)')

        per_axis[key] = {
            'stats': stats,
            'included': included,
            'excluded_reasons': excluded,
            'fit': fit,
            'weighting': weighting_name,
            'x_all': x_all, 'y_all_median': y_all_median,
            'sem_all': sem_all, 'n_all': n_all,
        }

        summary[key] = {
            'coeffs': None if fit['coeffs'] is None else fit['coeffs'].tolist(),
            'R2': fit['R2_weighted'],  # alias for readability
            'R2_unweighted': fit['R2_unweighted'],
            'R2_weighted': fit['R2_weighted'],
            'aggregation': 'median across walks',
            'fit_method': 'weighted quadratic (y ~ a x^2 + b x + c)',
            'weighting': weighting_name,
            'support_per_offset': {
                int(d): {
                    'n_walks': stats[int(d)]['n_walks'],
                    'n_pairs_per_walk': stats[int(d)]['n_pairs_per_walk'],
                    'mean': stats[int(d)]['mean'],
                    'median': stats[int(d)]['median'],
                    'mad_sigma': stats[int(d)]['mad_sigma'],
                    'sem': stats[int(d)]['sem'],
                    'sem_robust': stats[int(d)]['sem_robust'],
                }
                for d in sorted(stats.keys())
            },
            'offsets_included': included,
            'offsets_excluded': sorted(excluded.keys()),
            'excluded_reasons': {int(k): v for k, v in excluded.items()},
        }

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Distance Parabolas {space_label}',
        fontsize=14, fontweight='bold')

    panel_axes = {'3D': axes[0, 0], 'X': axes[0, 1],
                  'Y':  axes[1, 0], 'Z': axes[1, 1]}
    for key, ax in panel_axes.items():
        data = per_axis[key]
        color = axis_colors[key]
        x_all = data['x_all']
        y_med = data['y_all_median']
        sem = data['sem_all']
        n_all = data['n_all']
        included = set(data['included'])

        incl_mask = np.array([int(x) in included for x in x_all], dtype=bool)

        max_n = max(1, int(n_all.max() if n_all.size else 1))
        sizes = 20 + 90 * (n_all / max_n)

        # Error bars: use sem_robust where finite, else 0 (hidden)
        yerr = np.where(np.isfinite(sem), sem, 0.0)

        # Excluded offsets are intentionally NOT plotted (their support
        # is below the filter threshold; they would clutter the panel
        # without contributing to the fit).
        # Included: axis color, size scaled by n_walks support
        if incl_mask.any():
            ax.errorbar(x_all[incl_mask], y_med[incl_mask],
                        yerr=yerr[incl_mask],
                        fmt='none', ecolor=color, alpha=0.8,
                        elinewidth=1, capsize=2, zorder=2)
            ax.scatter(x_all[incl_mask], y_med[incl_mask],
                       s=sizes[incl_mask], color=color, alpha=0.85,
                       edgecolor='k', linewidths=0.4,
                       label=f'included in fit (n={int(incl_mask.sum())})',
                       zorder=3)

        fit = data['fit']
        if fit['y_smooth'] is not None:
            ax.plot(fit['x_smooth'], fit['y_smooth'], 'r-', linewidth=2,
                    label=(f"weighted fit, R2_w={fit['R2_weighted']:.3f}, "
                           f"R2_u={fit['R2_unweighted']:.3f}"))
        ax.set_xlabel('Contour Distance [s]')
        ax.set_ylabel(axis_ylabels[key])
        ax.set_title(axis_titles[key])
        if x_all.size:
            ax.set_xticks(np.arange(0, int(x_all.max()) + 1, 5))
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, summary, n_walks_used


def _plot_mean_parabola_xyz(walks_ok, out_path,
                            tp_start_1based, tp_end_1based,
                            n_tp_in_range):
    """Write the mean-parabola QC PNG and return the per-axis fit summary.

    Thin wrapper around `build_mean_parabola_xyz_figure` that saves +
    closes the figure.
    """
    fig, summary, n_walks_used = build_mean_parabola_xyz_figure(
        walks_ok, tp_start_1based, tp_end_1based, n_tp_in_range)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary, n_walks_used


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_post_filter_analysis(filtered_csv, outdir,
                             tp_start_1based, tp_end_1based,
                             prefix='qc'):
    """Follow-up QC on the final filtered walk set over a user-supplied TP range.

    Args
    ----
    filtered_csv : str
        Path to the final downstream-ready walks CSV produced by
        `drift_correct.py` (after full-coverage + movement filters).
    outdir : str
        Directory to write outputs into.
    tp_start_1based, tp_end_1based : int
        Inclusive TP range, 1-based. Internally converted to 0-based indices
        that match the `TP###_CYC###` filename convention.
    prefix : str
        Output filename prefix.

    Outputs
    -------
    {outdir}/{prefix}_distance_matrices_xyz.png
    {outdir}/{prefix}_parabola_xyz.png
    {outdir}/{prefix}_summary.json
    """
    if not isinstance(tp_start_1based, int) or not isinstance(tp_end_1based, int):
        raise TypeError("tp_start_1based and tp_end_1based must be ints.")
    if tp_start_1based < 1:
        raise ValueError(
            f"tp_start_1based must be >= 1 (got {tp_start_1based}).")
    if tp_end_1based < tp_start_1based:
        raise ValueError(
            f"tp_end_1based ({tp_end_1based}) must be >= tp_start_1based "
            f"({tp_start_1based}).")

    tp_start_0 = tp_start_1based - 1
    tp_end_0 = tp_end_1based - 1

    os.makedirs(outdir, exist_ok=True)

    (walks_ok, tp_names, tp_indices_0,
     skipped_missing, skipped_dupe, n_input,
     _skipped_empty_coord_rows) = _load_ordered_walks(
        filtered_csv, tp_start_0, tp_end_0)

    dm_path = os.path.join(outdir, f'{prefix}_distance_matrices_xyz.png')
    pb_path = os.path.join(outdir, f'{prefix}_parabola_xyz.png')
    js_path = os.path.join(outdir, f'{prefix}_summary.json')

    print(f'\n[Post-filter QC] Input: {filtered_csv}')
    print(f'  Analysis TP range (1-based, inclusive): '
          f'[{tp_start_1based}, {tp_end_1based}]')
    print(f'  Internal 0-based TP indices: {tp_indices_0}')
    print(f'  Resolved TP names: {tp_names}')
    print(f'  WalkIDs seen in input CSV (restricted to range): {n_input}')
    print(f'  Walks skipped (missing one or more required TPs): {skipped_missing}')
    print(f'  Walks skipped (>1 row at some required TP): {skipped_dupe}')
    print(f'  Walks used: {len(walks_ok)}')

    n_dm = _plot_mean_distance_matrices(
        walks_ok, dm_path, tp_start_1based, tp_end_1based)
    parabola_summary, n_pb = _plot_mean_parabola_xyz(
        walks_ok, pb_path, tp_start_1based, tp_end_1based,
        n_tp_in_range=len(tp_indices_0))

    summary = {
        'input_csv': os.path.abspath(filtered_csv),
        'analysis_tp_range_1based': [tp_start_1based, tp_end_1based],
        'analysis_tp_indices_0based': tp_indices_0,
        'analysis_tp_names': tp_names,
        'n_timepoints_in_range': len(tp_indices_0),
        'n_walks_input_in_range': n_input,
        'n_walks_skipped_missing_tp': skipped_missing,
        'n_walks_skipped_duplicate_tp': skipped_dupe,
        'n_walks_used_distance_matrices': n_dm,
        'n_walks_used_parabola': n_pb,
        'parabola_fit_config': {
            'min_walks_per_offset': FIT_MIN_WALKS_PER_OFFSET,
            'min_frac_walks_per_offset': FIT_MIN_FRAC_WALKS_PER_OFFSET,
            'min_pairs_per_walk': FIT_MIN_PAIRS_PER_WALK,
            'drop_terminal_offsets': FIT_DROP_TERMINAL_OFFSETS,
        },
        'parabola_fit': parabola_summary,
        'outputs': {
            'distance_matrices_png': os.path.abspath(dm_path),
            'parabola_png': os.path.abspath(pb_path),
        },
    }
    with open(js_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'  Wrote {dm_path}')
    print(f'  Wrote {pb_path}')
    print(f'  Wrote {js_path}')
    for k in ('3D', 'X', 'Y', 'Z'):
        ax = parabola_summary[k]
        r2w = ax['R2_weighted']
        r2u = ax['R2_unweighted']
        n_in = len(ax['offsets_included'])
        n_ex = len(ax['offsets_excluded'])
        print(f'    Parabola {k}: R2_w={r2w:.3f}, R2_u={r2u:.3f}, '
              f'n_fit={n_in}, n_excl={n_ex}, weighting={ax["weighting"]}')
        if ax['offsets_excluded']:
            print(f'      Excluded offsets: {ax["offsets_excluded"]}')
    return summary


# ---------------------------------------------------------------------------
# Per-walk drawers: draw the 4-panel (3D/X/Y/Z) distance-matrix or parabola
# content of a SINGLE walk into an existing figure's SubplotSpec. Designed
# to be embedded into the per-walk report card so each WalkID's page shows
# its own per-walk QC directly, alongside the existing metric panels.
#
# `subplot_spec` is typically an element of a parent `GridSpec`, e.g.:
#     outer = fig.add_gridspec(4, 4)
#     draw_per_walk_parabola(fig, outer[2:4, 0:2], curve)
# We carve a 2x2 nested gridspec inside the given cell so each axis panel
# stays compact.
# ---------------------------------------------------------------------------
_PER_WALK_AXIS_KEYS  = ('3D', 'X', 'Y', 'Z')
_PER_WALK_AXIS_COLORS = {
    '3D': 'steelblue',
    'X':  'green',
    'Y':  'orange',
    'Z':  'purple',
}


def per_walk_matrices(curve):
    """Public alias for `_per_walk_matrices`; returns (D_3d, D_x, D_y, D_z)."""
    return _per_walk_matrices(curve)


def per_walk_diagonal_means(curve):
    """Return dict {axis_key -> (offsets, diagonal_means)} for a single walk.

    The diagonal_means are computed directly from the walk's own distance
    matrices (no aggregation across walks), matching the reference
    `_compute_diagonal_means_for_walk` behavior on a single centroid set.
    """
    mats = dict(zip(_PER_WALK_AXIS_KEYS, _per_walk_matrices(curve)))
    out = {}
    for key, D in mats.items():
        out[key] = _diagonal_means_from_matrix(D)
    return out


def draw_per_walk_distance_matrices(fig, subplot_spec, curve,
                                    title='Per-walk distance matrices (nm)',
                                    global_colorbar=False):
    """Draw the walk's own 4-panel distance matrices (3D/X/Y/Z) in nm.

    Values are the walk's own pairwise distances in nm (raw, not
    normalized). `global_colorbar` controls the display scale:
      - False (default): each panel uses its own off-diagonal max, so
        each panel's colorbar fills its own range. Four colorbars,
        all labeled in nm.
      - True: one shared colorbar scaled to the Z matrix's off-diagonal
        max, matching the global XYZ-page convention.
    """
    sub = subplot_spec.subgridspec(2, 2, hspace=0.45, wspace=0.30)
    br = LinearSegmentedColormap.from_list(
        'bright_red', [(1, 0, 0), (1, 1, 1)], N=256)
    D_3d, D_x, D_y, D_z = _per_walk_matrices(curve)
    mats = {'3D': D_3d, 'X': D_x, 'Y': D_y, 'Z': D_z}
    positions = {'3D': (0, 0), 'X': (0, 1), 'Y': (1, 0), 'Z': (1, 1)}
    panel_axes = []
    first_ax = None
    last_im = None
    if global_colorbar:
        vmin_shared, vmax_shared = compute_shared_matrix_display_limits([D_z])
    for key, (r, c) in positions.items():
        ax = fig.add_subplot(sub[r, c])
        panel_axes.append(ax)
        if first_ax is None:
            first_ax = ax
        M = mats[key]
        if global_colorbar:
            vmin, vmax = vmin_shared, vmax_shared
        else:
            vmin, vmax = compute_shared_matrix_display_limits([M])
        im = ax.imshow(M, cmap=br, aspect='auto',
                       interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'{key} (nm)', fontsize=9)
        ax.tick_params(labelsize=7)
        last_im = im
        if not global_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label('distance (nm)', fontsize=7)
            cbar.ax.tick_params(labelsize=7)
    if global_colorbar:
        cbar = fig.colorbar(last_im, ax=panel_axes,
                            shrink=0.75, pad=0.02)
        cbar.set_label('distance (nm)', fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    if first_ax is not None and title:
        first_ax.annotate(
            title, xy=(0.5, 1.18), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            annotation_clip=False)


def draw_per_walk_parabola(fig, subplot_spec, curve,
                            title='Per-walk parabola'):
    """Draw the walk's own 4-panel parabola scatter + quadratic fit (3D/X/Y/Z).

    The fit is unweighted (one walk -> one value per offset; no cross-walk
    aggregation, hence no robust-SEM weights to apply). R^2 is shown in
    each panel's legend.
    """
    sub = subplot_spec.subgridspec(2, 2, hspace=0.45, wspace=0.35)
    dmeans = per_walk_diagonal_means(curve)
    positions = {'3D': (0, 0), 'X': (0, 1), 'Y': (1, 0), 'Z': (1, 1)}
    first_ax = None
    for key, (r, c) in positions.items():
        ax = fig.add_subplot(sub[r, c])
        if first_ax is None:
            first_ax = ax
        color = _PER_WALK_AXIS_COLORS[key]
        offsets, means = dmeans[key]
        if offsets.size:
            ax.scatter(offsets, means, color=color, s=20, alpha=0.85,
                       edgecolor='black', linewidths=0.3)
            fit = _fit_parabola(offsets.astype(float), means)
            if fit['y_smooth'] is not None:
                ax.plot(fit['x_smooth'], fit['y_smooth'], 'r-',
                        linewidth=1.4,
                        label=f"R2={fit['R2_unweighted']:.2f}")
                ax.legend(loc='best', fontsize=7)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel('TP offset', fontsize=7)
        ax.set_ylabel('dist (nm)', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    if first_ax is not None and title:
        first_ax.annotate(
            title, xy=(0.5, 1.18), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            annotation_clip=False)
