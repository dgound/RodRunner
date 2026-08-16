#!/usr/bin/env python3
"""Replication QC for 640 localizations.

Detects likely replicating / split loci from raw 640 TP crops. The saved
one-spot localization is an initializer only. QC refits both candidate models
on the same owning-mask plus physical-shell support with identical nuisance
parameter bounds:

  - M1 = one Gaussian + background (8 parameters)
  - M2 = two Gaussians with shared sigma_{x,y,z} + background (12 parameters)
  - M2 uses four deterministic starts from two positive-residual peaks
  - Stage-A broadness/residual/lobe features remain diagnostic annotations;
    they do not decide whether M2 is attempted
  - compare the common-support fits with BIC:
          BIC = n ln(SSE / n) + k ln(n)
          delta_BIC = BIC_1 - BIC_2   (positive => 2-spot preferred)

Replication call is conservative. A (WalkID, TP) is marked
`replication_confirmed=True` only if ALL of:
  - delta_BIC >= --delta_bic_thresh
  - 95% lower confidence bound on spot separation >= --min_spot_sep_nm;
    use lateral separation when axial width is censored/unidentifiable
  - min(A1, A2) / max(A1, A2) >= --min_amp_ratio
  - both fitted centers stay in the owned support and away from crop bounds
  - neither width collapses to its lower numerical bound
  - both M1 and M2 converge with finite parameters and both M2 amplitudes > 0
  - when automatic null calibration is enabled, its per-TP pooled-null
    spatial wild-bootstrap q-value passes --auto_fdr_q

An intentional upper sigma_z cap is censoring, not a rejection condition.
Replicate 1 can therefore retain its 3.5-slice stabilizer, while replicate 3
uses --sigma_z_max 0 (no experimental cap; crop depth is the numerical guard).

Output is PURELY annotation. The main localization CSV is never edited;
this script emits a side-car CSV with one row per (WalkID, TP) plus an
optional per-walk summary and small QC figures. It does NOT introduce
duplicate rows into any downstream single-curve CSV.

The current 640 localizer (`localize_probe0.py` -> `gauss3d_fit.c`) fits
exactly one 3D Gaussian per mask per TP. When a cell actually contains
two nearby fluorophore foci (e.g. a replicated locus), the 1-spot fit
tends to collapse them into a single broader / higher-residual fit. This
script detects that case.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.optimize import least_squares  # kept as a fallback only

import dat_io
try:
    from .z_support_2p5d import load_z_support, materialize_labels
except ImportError:  # direct execution / legacy tests with pipeline on sys.path
    try:
        from z_support_2p5d import load_z_support, materialize_labels
    except ImportError:  # module is required only for --segmentation-mode 2p5d-zgated
        load_z_support = None
        materialize_labels = None
try:
    from . import center_ownership as _own
except ImportError:
    try:
        import center_ownership as _own
    except ImportError:      # required only for --center-ownership strict
        _own = None
try:
    from . import chromatic_registration as _chrom
except ImportError:
    try:
        import chromatic_registration as _chrom
    except ImportError:  # required only for --chromatic-registration metadata
        _chrom = None
try:
    from .replication_detection import (
        ORI_TP_DEFAULT,
        REPLICATION_TP_END_DEFAULT,
        REPLICATION_TP_START_DEFAULT,
        TER_TP_DEFAULT,
        WALK_ANNOTATION_FIELDS,
        build_replichores,
        classify_replication,
    )
except ImportError:  # direct execution / legacy tests with pipeline on sys.path
    from replication_detection import (
        ORI_TP_DEFAULT,
        REPLICATION_TP_END_DEFAULT,
        REPLICATION_TP_START_DEFAULT,
        TER_TP_DEFAULT,
        WALK_ANNOTATION_FIELDS,
        build_replichores,
        classify_replication,
    )

# --------------------------------------------------------------------------
# Compiled backend for the heavy per-crop computations.
# Auto-built on import if the .so is missing or stale, mirroring the
# gauss3d_fit.c pattern used by the localizers.
# --------------------------------------------------------------------------
_C_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'replication_qc_fit.c')
_C_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'replication_qc_fit.so')


def _build_backend() -> None:
    """Try to build with OpenMP first, fall back to serial if gcc rejects
    -fopenmp. A status line is printed so the caller can see which variant
    landed on disk."""
    base = (f'gcc -O3 -shared -fPIC '
            f'-o {_C_LIB} {_C_SRC} -lm')
    rc = os.system(f'gcc -O3 -fopenmp -shared -fPIC '
                   f'-o {_C_LIB} {_C_SRC} -lm 2>/dev/null')
    if rc == 0:
        print(f'[replication_qc] Built {os.path.basename(_C_LIB)} with OpenMP.')
        return
    print(f'[replication_qc] WARNING: -fopenmp build failed; falling back to serial.')
    rc = os.system(base)
    if rc != 0:
        raise RuntimeError(
            f'Failed to build {_C_LIB} from {_C_SRC}. Ensure gcc is on PATH.')
    print(f'[replication_qc] Built {os.path.basename(_C_LIB)} (serial).')


def _load_backend() -> ctypes.CDLL:
    if (not os.path.exists(_C_LIB) or
            os.path.getmtime(_C_LIB) < os.path.getmtime(_C_SRC)):
        _build_backend()
    lib = ctypes.CDLL(_C_LIB)

    # void stage_a_and_sse1(
    #     const double *obs, const double *Z, const double *Y, const double *X,
    #     int n_vox,
    #     double amp1, cz1, cy1, cx1, sz1, sy1, sx1, bg1,
    #     double *resid_score, *peak_rel_amp, *peak_offset_px,
    #     double *peak_z, *peak_y, *peak_x,
    #     double *sse1);
    lib.stage_a_and_sse1.restype = None
    lib.stage_a_and_sse1.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    # int fit_2spot(
    #     const double *obs, *Z, *Y, *X, int n_vox,
    #     const double *p0, *lo, *hi,
    #     int max_iter,
    #     double *out_params, *out_sse, int *out_pinned, *out_nfev);
    lib.fit_2spot.restype = ctypes.c_int
    lib.fit_2spot.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    lib.fit_1spot.restype = ctypes.c_int
    lib.fit_1spot.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]

    # void batch_replication_qc(
    #     image, mask_labels, nz, ny, nx,
    #     bboxes, has_fit1, fit1, n_cells,
    #     sigma_xy_broad_thr, sigma_z_broad_thr, resid_score_thr,
    #     min_lobe_sep_px, min_lobe_rel_amp,
    #     delta_bic_thr, min_spot_sep_nm, min_amp_ratio, min_amp_recovery,
    #     min_peak_support_c2, max_valley_frac, min_resid_core_frac,
    #     z_step_nm, xy_pixel_nm,
    #     min_n_voxels, max_iter_lm,
    #     out_floats, out_ints);
    lib.batch_replication_qc.restype = None
    lib.batch_replication_qc.argtypes = [
        ctypes.POINTER(ctypes.c_double),           # image
        ctypes.POINTER(ctypes.c_int),              # mask_labels
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # nz, ny, nx
        ctypes.POINTER(ctypes.c_int),              # bboxes
        ctypes.POINTER(ctypes.c_int),              # has_fit1
        ctypes.POINTER(ctypes.c_double),           # fit1
        ctypes.c_int,                              # n_cells
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double,                           # min_amp_recovery
        ctypes.c_double,                           # min_peak_support_c2
        ctypes.c_double,                           # max_valley_frac
        ctypes.c_double,                           # min_resid_core_frac
        ctypes.c_double,                           # sigma_z_max
        ctypes.c_double,                           # center_shell_nm
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),           # out_floats
        ctypes.POINTER(ctypes.c_int),              # out_ints
    ]

    # OpenMP diagnostic symbols (cheap runtime query from Python).
    lib.has_openmp.restype = ctypes.c_int
    lib.has_openmp.argtypes = []
    lib.openmp_max_threads.restype = ctypes.c_int
    lib.openmp_max_threads.argtypes = []
    if lib.has_openmp():
        print(f'[replication_qc] Backend: OpenMP enabled, '
              f'max threads = {lib.openmp_max_threads()}')
    else:
        print('[replication_qc] Backend: serial (no OpenMP in this .so)')
    return lib


_BACKEND = _load_backend()


def _as_double_ptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    """Return a ctypes pointer for a contiguous float64 array. Raises if not contiguous."""
    if not a.flags['C_CONTIGUOUS'] or a.dtype != np.float64:
        a = np.ascontiguousarray(a, dtype=np.float64)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# --------------------------------------------------------------------------
# Units. Same conventions as drift_correct.py so downstream comparisons
# line up in nm without further conversion.
# --------------------------------------------------------------------------
Z_STEP_NM = 100.0
XY_PIXEL_NM = 103.28  # Vutara config PixelSizeX/Y 102.97/103.59, mean; was 108

# --------------------------------------------------------------------------
# Defaults (all overridable on the CLI). Tuned conservatively: the point
# of Stage A is to flag cheap candidates for Stage B; real calls come from
# the combined Stage-A + Stage-B decision below.
# --------------------------------------------------------------------------
DEFAULT_PAD_XY = 3      # matches localize_probe0.py
DEFAULT_PAD_Z = 1       # matches localize_probe0.py

# Stage A thresholds.
# NOTE (2026-04-23): Stage A was over-firing on the genome WF data —
# sigma_z_broad was flagging nearly every cell (broad PSF in z), and
# resid / two-lobe flagged more loosely than needed. The C backend now
# drops sigma_z_broad from the suspect OR (kept as a diagnostic column
# only), and the residual + lobe thresholds here are tightened so Stage A
# actually discriminates. Re-run with `--sigma_z_broad 99` etc. to override.
DEFAULT_SIGMA_XY_BROAD = 2.5         # px  (unchanged; genuinely rare)
DEFAULT_SIGMA_Z_BROAD = 1.5          # z-slices (diagnostic only, no longer triggers Stage B)
DEFAULT_RESID_SCORE_THRESH = 0.05    # was 0.02 — sum(pos residual) / sum(model)
DEFAULT_MIN_LOBE_SEP_PX = 2.0        # voxel units inside crop
DEFAULT_MIN_LOBE_REL_AMP = 0.25      # was 0.15 — residual peak / 1-spot amplitude

# Stage B + call thresholds.  Tightened 2026-04-23 to reduce TP-level FPs:
#   delta_BIC  10 -> 15  (stronger model-comparison evidence required)
#   spot_sep  150 -> 250 nm (above typical PSF XY/Z; suppresses marginal splits)
#   amp_ratio 0.2 -> 0.3 (spots must be more balanced; kills single-spot + faint noise)
DEFAULT_DELTA_BIC_THRESH = 15.0
DEFAULT_MIN_SPOT_SEP_NM = 250.0
DEFAULT_MIN_AMP_RATIO = 0.3
# Stage B diagnostic gates. ALL default to "disabled" (0 / non-positive)
# so the side-car CSV is populated for review without changing the
# confirm decision. Production deployments opt in explicitly via the
# orchestrator (see run_all_locations.sh) — that keeps the existing
# production behaviour unchanged while leaving direct invocations of
# replication_qc.py purely diagnostic.
#
#   amp_recovery       = max(A1,A2) / amp_1spot
#   peak_support_c1/c2 = (raw at fitted center - bg) / (raw max in small ball - bg)
#   valley_frac        = (raw min along c1<->c2 - bg) / (min raw at centers - bg)
#   resid_core_frac    = positive-residual mass in core ball at Stage A peak
#                        / total positive-residual mass
#
# All five metrics are computed and written to the per-(WalkID, TP) CSV
# regardless of whether the gates are enabled.
DEFAULT_MIN_AMP_RECOVERY    = 0.0   # 0 = disabled
DEFAULT_MIN_PEAK_SUPPORT_C2 = 0.0   # 0 = disabled
DEFAULT_MAX_VALLEY_FRAC     = 0.0   # 0 = disabled (geometry/mask-sensitive; review first)
DEFAULT_MIN_RESID_CORE_FRAC = 0.0   # 0 = disabled

# Numerical lower bounds match the one-spot localizer. An upper sigma_z
# limit is replicate-specific; <=0 uses only the crop-depth guard.
MIN_SIGMA_XY = 0.3
MIN_SIGMA_Z = 0.2
REPLICATION_QC_SCHEMA = 'replication_qc_v2'
AUTO_CALIBRATION_SCHEMA = 'replication_qc_auto_calibration_v2'
AUTO_CALIBRATION_VERSION = 'auto-pooled-spatial-wild-bootstrap-v2'
NCOLS_FLOAT = 41
NCOLS_INT = 24

#: Status code for a locus whose two-spot answer is UNEVALUABLE because an M2
#: component centre failed registered centre ownership under `--center-ownership
#: strict`.  It must be its own code: the C backend's code 4 already means
#: `confirmed`, so reusing it wrote `status=confirmed` on a row with
#: `replication_confirmed=0`, and every consumer that reads evaluability off the
#: status string then counted the locus as measured-and-negative -- silently
#: converting "we cannot tell" into one-spot evidence.  Codes 0..9 are produced
#: by the backend, so this starts above them.
# The status vocabulary is centralized in replication_status (dependency-free),
# so a consumer needing only EVALUABLE_STATUSES does not have to import this
# module and compile the C backend to get it.
from replication_status import (  # noqa: E402
    EVALUABLE_STATUSES,
    STATUS_MAP,
    STATUS_OWNERSHIP_UNEVALUABLE,
    fit2_converged,
    status_label,
)

# TP filename pattern matching the rest of the pipeline.
_TP_FILE_RE = re.compile(r'^TP\d+(?:_CYC\d+)?\.tif$')
_TP_NAME_RE = re.compile(r'^TP(\d+)(?:_CYC\d+)?$')


def _tp_index_0based(tp_name):
    m = _TP_NAME_RE.match(str(tp_name))
    if m is None:
        raise ValueError(f'Unrecognized TP name: {tp_name!r}')
    return int(m.group(1))


# --------------------------------------------------------------------------
# Gaussian models + residual functions
# --------------------------------------------------------------------------
def _gaussian3d(amp, cz, cy, cx, sz, sy, sx, Z, Y, X):
    return amp * np.exp(
        -0.5 * ((Z - cz) / sz) ** 2
        - 0.5 * ((Y - cy) / sy) ** 2
        - 0.5 * ((X - cx) / sx) ** 2
    )


def _model_1spot(params, Z, Y, X):
    amp, cz, cy, cx, sz, sy, sx, bg = params
    return _gaussian3d(amp, cz, cy, cx, sz, sy, sx, Z, Y, X) + bg


def _model_2spot(params, Z, Y, X):
    (A1, A2, cz1, cy1, cx1, cz2, cy2, cx2, sz, sy, sx, bg) = params
    g1 = _gaussian3d(A1, cz1, cy1, cx1, sz, sy, sx, Z, Y, X)
    g2 = _gaussian3d(A2, cz2, cy2, cx2, sz, sy, sx, Z, Y, X)
    return g1 + g2 + bg


def _resid_2spot(params, Z, Y, X, obs):
    return _model_2spot(params, Z, Y, X) - obs


def _sse(residuals):
    return float(np.sum(residuals * residuals))


def _bic(sse, n, k):
    """Gaussian-residual BIC up to a constant: n ln(SSE/n) + k ln(n)."""
    if sse <= 0 or n <= 0:
        return float('inf')
    return n * np.log(sse / n) + k * np.log(n)


# --------------------------------------------------------------------------
# Crop + mask helpers (mirror localize_probe0.py's bounding-box policy)
# --------------------------------------------------------------------------
def compute_bboxes(masks, pad_xy, pad_z):
    """Return (n_cells, 6) int32 array of [z0, z1, y0, y1, x0, x1] inclusive."""
    nz, ny, nx = masks.shape
    n_cells = int(masks.max())
    slices = find_objects(masks)
    bboxes = np.zeros((n_cells, 6), dtype=np.int32)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        bboxes[i] = [
            max(0, sl[0].start - pad_z), min(nz - 1, sl[0].stop - 1 + pad_z),
            max(0, sl[1].start - pad_xy), min(ny - 1, sl[1].stop - 1 + pad_xy),
            max(0, sl[2].start - pad_xy), min(nx - 1, sl[2].stop - 1 + pad_xy),
        ]
    return bboxes


def _z_support_tp_arrays(z_support, tp_name):
    """Return one exact-name row from a validated z-support sidecar."""
    selected = z_support.select_tp(tp_name)
    return {
        'label_ids': np.asarray(selected.label_ids, dtype=np.int32),
        'z_min': np.asarray(selected.z_min),
        'z_max': np.asarray(selected.z_max),
        'status': np.asarray(selected.status),
        'fallback': np.asarray(selected.fallback),
        'quarantined': np.asarray(selected.quarantined, dtype=bool),
        'evaluable': np.asarray(selected.evaluable, dtype=bool),
        'fit_allowed': np.asarray(selected.fit_allowed, dtype=bool),
    }


def _protect_inactive_xy_ownership(effective_masks, labels_2d, n_cells):
    """Keep inactive labelled XY pixels out of another cell's QC shell.

    The C backend regards label 0 as background eligible for its physical
    shell.  A z-gated cell that is inactive on a plane must therefore not
    turn its authoritative XY ownership into background for a neighbouring
    cell.  A sentinel label above ``n_cells`` blocks shell use while remaining
    invisible to the per-cell fitter.
    """
    protected = np.asarray(effective_masks, dtype=np.int32).copy()
    owned_xy = np.asarray(labels_2d) > 0
    inactive = (protected == 0) & owned_xy[None, :, :]
    protected[inactive] = int(n_cells) + 1
    return protected


def _gate_bound_for_csv(value, nz):
    value = int(value)
    return value if 0 <= value < nz else ''


def _gate_status_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8')
    return str(value)


def _validate_fit_gate_provenance(fit, gate_tp, gate_axis, tp_name, walk_id):
    expected = {
        'z_gate_status': _gate_status_text(gate_tp['status'][gate_axis]),
        'z_gate_fallback': _gate_status_text(gate_tp['fallback'][gate_axis]),
        'z_gate_min': int(gate_tp['z_min'][gate_axis]),
        'z_gate_max': int(gate_tp['z_max'][gate_axis]),
        'z_gate_quarantined': bool(gate_tp['quarantined'][gate_axis]),
        'z_gate_evaluable': bool(gate_tp['evaluable'][gate_axis]),
        'z_gate_fit_allowed': bool(gate_tp['fit_allowed'][gate_axis]),
    }
    mismatches = {
        key: (fit.get(key), expected_value)
        for key, expected_value in expected.items()
        if fit.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f'Localization/z-support provenance mismatch at {tp_name}, '
            f'WalkID={walk_id}: {mismatches}')


def _crop_and_coords(img, masks, bbox, cell_label):
    """Return (obs, Z, Y, X, mask_in_crop) for the masked voxels of `cell_label`.

    Coordinates are returned in global pixel/slice units so they match the
    1-spot fit centers recorded in the CSV.
    """
    z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox]
    sub_img = img[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    sub_mask = (masks[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] == cell_label)
    if not np.any(sub_mask):
        return None
    Zg, Yg, Xg = np.meshgrid(
        np.arange(z0, z1 + 1),
        np.arange(y0, y1 + 1),
        np.arange(x0, x1 + 1),
        indexing='ij',
    )
    obs = sub_img[sub_mask].astype(np.float64)
    return (obs,
            Zg[sub_mask].astype(np.float64),
            Yg[sub_mask].astype(np.float64),
            Xg[sub_mask].astype(np.float64),
            sub_mask, (z0, y0, x0))


# --------------------------------------------------------------------------
# Stage A: suspicion screen
# --------------------------------------------------------------------------
def _stage_a_features_and_sse1(obs, Z, Y, X, fit1):
    """C-backed one-pass computation of:
      - Stage A residual features (resid_score, peak_rel_amp, peak_offset_px,
        peak_coords)
      - 1-spot SSE

    Returns (features_dict, sse1). The features_dict is the same shape that
    the previous Python implementation returned, minus a couple of fields
    that `_stage_a_flags` computes separately from fit1.
    """
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    Zc  = np.ascontiguousarray(Z, dtype=np.float64)
    Yc  = np.ascontiguousarray(Y, dtype=np.float64)
    Xc  = np.ascontiguousarray(X, dtype=np.float64)
    n   = obs.shape[0]

    resid_score   = ctypes.c_double(0.0)
    peak_rel_amp  = ctypes.c_double(0.0)
    peak_off_px   = ctypes.c_double(0.0)
    peak_z        = ctypes.c_double(0.0)
    peak_y        = ctypes.c_double(0.0)
    peak_x        = ctypes.c_double(0.0)
    sse1          = ctypes.c_double(0.0)

    _BACKEND.stage_a_and_sse1(
        _as_double_ptr(obs), _as_double_ptr(Zc),
        _as_double_ptr(Yc),  _as_double_ptr(Xc),
        ctypes.c_int(n),
        ctypes.c_double(fit1['amplitude']),
        ctypes.c_double(fit1['z']),
        ctypes.c_double(fit1['y']),
        ctypes.c_double(fit1['x']),
        ctypes.c_double(fit1['sigma_z']),
        ctypes.c_double(fit1['sigma_y']),
        ctypes.c_double(fit1['sigma_x']),
        ctypes.c_double(fit1['background']),
        ctypes.byref(resid_score),
        ctypes.byref(peak_rel_amp),
        ctypes.byref(peak_off_px),
        ctypes.byref(peak_z),
        ctypes.byref(peak_y),
        ctypes.byref(peak_x),
        ctypes.byref(sse1),
    )
    return {
        'resid_score':    resid_score.value,
        'peak_rel_amp':   peak_rel_amp.value,
        'peak_offset_px': peak_off_px.value,
        'peak_coords':    (peak_z.value, peak_y.value, peak_x.value),
    }, sse1.value


# Back-compat alias (unused now; kept so external callers or tests still work).
def _stage_a_features(obs, Z, Y, X, fit1):  # noqa: N802 - legacy name
    return _stage_a_features_and_sse1(obs, Z, Y, X, fit1)[0]


def _stage_a_flags(feats, fit1, cfg):
    sigma_xy_broad = (max(fit1['sigma_x'], fit1['sigma_y'])
                      > cfg['sigma_xy_broad'])
    sigma_z_broad = fit1['sigma_z'] > cfg['sigma_z_broad']
    resid_broad = feats['resid_score'] > cfg['resid_score_thresh']
    two_lobe = (feats['peak_offset_px'] >= cfg['min_lobe_sep_px']
                and feats['peak_rel_amp'] >= cfg['min_lobe_rel_amp'])
    suspect = bool(sigma_xy_broad or sigma_z_broad or resid_broad or two_lobe)
    return {
        'sigma_xy_broad': sigma_xy_broad,
        'sigma_z_broad': sigma_z_broad,
        'resid_broad': resid_broad,
        'two_lobe_suspect': two_lobe,
        'suspect': suspect,
    }


# --------------------------------------------------------------------------
# Stage B: 2-spot fit + BIC comparison
# --------------------------------------------------------------------------
def _fit_2spot(obs, Z, Y, X, fit1, peak_coords, bbox, sigma_z_max=0.0):
    """Fit A1, A2, (c1), (c2), shared sigma_{x,y,z}, shared bg using the
    C backend (replication_qc_fit.so :: fit_2spot).

    Returns (params, sse, info). info keys preserved for compatibility with
    the Python implementation:
        status     'ok' / 'non_finite_params' / '2spot_fail:...'
        converged  bool
        nfev       int
        cost       0.5 * sse  (matches scipy.least_squares.cost convention)
        pinned     bool (any finite-bounded param within 1% of endpoint)
    On fatal failure returns (None, None, {'status': '...'}).
    """
    z0, z1 = bbox[0], bbox[1]
    y0, y1 = bbox[2], bbox[3]
    x0, x1 = bbox[4], bbox[5]
    max_sigma_z = (min(float(sigma_z_max), z1 - z0 + 1)
                   if sigma_z_max > 0 else float(z1 - z0 + 1))
    max_sigma_y = max(MIN_SIGMA_XY, (y1 - y0 + 1) / 2.0)
    max_sigma_x = max(MIN_SIGMA_XY, (x1 - x0 + 1) / 2.0)

    # Guard: ensure the two initial centers are actually separated.
    c1 = (fit1['z'], fit1['y'], fit1['x'])
    c2 = peak_coords
    if np.hypot(np.hypot(c1[0] - c2[0], c1[1] - c2[1]),
                c1[2] - c2[2]) < 1.0:
        extents = [z1 - z0, y1 - y0, x1 - x0]
        axis = int(np.argmax(extents))
        shift = [0.0, 0.0, 0.0]
        shift[axis] = max(1.0, extents[axis] * 0.1)
        c2 = (c1[0] + shift[0], c1[1] + shift[1], c1[2] + shift[2])

    p0 = np.array([
        0.7 * fit1['amplitude'],       # A1
        0.3 * fit1['amplitude'],       # A2
        c1[0], c1[1], c1[2],           # cz1, cy1, cx1
        c2[0], c2[1], c2[2],           # cz2, cy2, cx2
        max(MIN_SIGMA_Z, min(max_sigma_z, fit1['sigma_z'])),
        max(MIN_SIGMA_XY, min(max_sigma_y, fit1['sigma_y'])),
        max(MIN_SIGMA_XY, min(max_sigma_x, fit1['sigma_x'])),
        max(0.0, fit1['background']),
    ], dtype=np.float64)
    lo = np.array([
        0.0, 0.0,
        z0, y0, x0,
        z0, y0, x0,
        MIN_SIGMA_Z, MIN_SIGMA_XY, MIN_SIGMA_XY,
        0.0,
    ], dtype=np.float64)
    hi = np.array([
        np.inf, np.inf,
        z1, y1, x1,
        z1, y1, x1,
        max_sigma_z, max_sigma_y, max_sigma_x,
        np.inf,
    ], dtype=np.float64)

    obs_c = np.ascontiguousarray(obs, dtype=np.float64)
    Zc    = np.ascontiguousarray(Z,   dtype=np.float64)
    Yc    = np.ascontiguousarray(Y,   dtype=np.float64)
    Xc    = np.ascontiguousarray(X,   dtype=np.float64)

    out_params = np.zeros(12, dtype=np.float64)
    out_sse    = ctypes.c_double(0.0)
    out_pinned = ctypes.c_int(0)
    out_nfev   = ctypes.c_int(0)

    rc = _BACKEND.fit_2spot(
        _as_double_ptr(obs_c), _as_double_ptr(Zc),
        _as_double_ptr(Yc),    _as_double_ptr(Xc),
        ctypes.c_int(int(obs_c.size)),
        _as_double_ptr(p0),
        _as_double_ptr(lo), _as_double_ptr(hi),
        ctypes.c_int(200),
        _as_double_ptr(out_params),
        ctypes.byref(out_sse),
        ctypes.byref(out_pinned),
        ctypes.byref(out_nfev),
    )

    if rc == -1:
        return None, None, {'status': '2spot_fail:invalid_input'}
    if rc == 2:
        return None, None, {'status': 'non_finite_params'}
    return out_params, float(out_sse.value), {
        'status':    'ok',
        'converged': rc == 0,
        'nfev':      int(out_nfev.value),
        'cost':      0.5 * float(out_sse.value),   # scipy-style
        'pinned':    bool(out_pinned.value),
    }


def _spot_sep_nm(params2):
    cz1, cy1, cx1, cz2, cy2, cx2 = params2[2:8]
    dz = (cz1 - cz2) * Z_STEP_NM
    dy = (cy1 - cy2) * XY_PIXEL_NM
    dx = (cx1 - cx2) * XY_PIXEL_NM
    return float(np.sqrt(dz * dz + dy * dy + dx * dx))


# --------------------------------------------------------------------------
# Existing-1-spot loading
# --------------------------------------------------------------------------
def _load_1spot_csv(path, require_z_gate=False):
    """Return dict {(tp, WalkID) -> fit1_dict}."""
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        if require_z_gate:
            required = {
                'z_gate_status', 'z_gate_min', 'z_gate_max',
                'z_gate_fallback',
                'z_gate_quarantined', 'z_gate_evaluable',
                'z_gate_fit_allowed',
            }
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(
                    '2p5d-zgated replication QC cannot reuse a localization '
                    f'CSV without gate provenance; missing columns: {missing}')
        for row in reader:
            key = (row['tp'], int(row['WalkID']))
            try:
                fit = {
                    'z': float(row['z']),
                    'y': float(row['y']),
                    'x': float(row['x']),
                    'sigma_z': float(row['sigma_z']),
                    'sigma_y': float(row['sigma_y']),
                    'sigma_x': float(row['sigma_x']),
                    'amplitude': float(row['amplitude']),
                    'background': float(row['background']),
                }
                if require_z_gate:
                    fit.update({
                        'z_gate_status': row['z_gate_status'],
                        'z_gate_fallback': row['z_gate_fallback'],
                        'z_gate_min': int(row['z_gate_min']),
                        'z_gate_max': int(row['z_gate_max']),
                        'z_gate_quarantined': bool(
                            int(row['z_gate_quarantined'])),
                        'z_gate_evaluable': bool(
                            int(row['z_gate_evaluable'])),
                        'z_gate_fit_allowed': bool(
                            int(row['z_gate_fit_allowed'])),
                    })
                out[key] = fit
            except (KeyError, ValueError) as exc:
                if require_z_gate:
                    raise ValueError(
                        f'Invalid 2p5d-zgated localization row for '
                        f'{row.get("tp", "<unknown>")}, '
                        f'WalkID={row.get("WalkID", "<unknown>")}') from exc
                continue
    return out


# --------------------------------------------------------------------------
# Automatic, model-based null calibration
# --------------------------------------------------------------------------
def _pooled_null_qvalues(observed_scores, null_scores):
    """Estimate monotone q-values from a pooled candidate-matched null.

    At every observed score threshold, ``D / B`` estimates the number of
    false target calls, where D is the number of pooled null exceedances and B
    is the number of null realizations per candidate. The +1 correction keeps
    the finite-simulation estimate conservative at an empty null tail.
    """
    observed = np.asarray(observed_scores, dtype=np.float64)
    null = np.asarray(null_scores, dtype=np.float64)
    if observed.ndim != 1 or null.ndim != 2:
        raise ValueError('observed_scores must be 1D and null_scores must be 2D')
    if null.shape[0] != observed.size or null.shape[1] < 1:
        raise ValueError('null_scores must have shape (n_candidates, n_replicates)')
    if not np.all(np.isfinite(observed)):
        raise ValueError('observed scores must be finite')

    order = np.argsort(-observed, kind='stable')
    sorted_observed = observed[order]
    pooled_null = null.ravel()
    null_exceedances = np.array(
        [np.count_nonzero(pooled_null >= threshold)
         for threshold in sorted_observed], dtype=np.int64)
    ranks = np.arange(1, observed.size + 1, dtype=np.float64)
    raw_fdr = (null_exceedances + 1.0) / (null.shape[1] * ranks)
    q_sorted = np.minimum.accumulate(raw_fdr[::-1])[::-1]

    q_values = np.empty_like(q_sorted)
    fdr_values = np.empty_like(raw_fdr)
    exceedance_values = np.empty_like(null_exceedances)
    q_values[order] = np.minimum(1.0, q_sorted)
    fdr_values[order] = raw_fdr
    exceedance_values[order] = null_exceedances
    return q_values, fdr_values, exceedance_values


def _owned_support(labels, cell_label, shell_nm, z_step_nm, xy_pixel_nm):
    """Mirror the C backend's owning-mask plus physical-shell support."""
    own = labels == cell_label
    if shell_nm <= 0:
        return own
    distance_nm = distance_transform_edt(
        ~own, sampling=(z_step_nm, xy_pixel_nm, xy_pixel_nm))
    return own | ((labels == 0) & (distance_nm <= shell_nm + 1e-9))


def _mammen_weights(rng, n_blocks):
    """Draw mean-zero, unit-variance Mammen multipliers for spatial blocks."""
    sqrt5 = math.sqrt(5.0)
    low = (1.0 - sqrt5) / 2.0
    high = (1.0 + sqrt5) / 2.0
    probability_high = (sqrt5 - 1.0) / (2.0 * sqrt5)
    return np.where(rng.random(n_blocks) < probability_high, high, low)


def _prepare_auto_null_candidate(image, masks, bbox, row, cfg, args, tp_index):
    """Prepare reusable arrays for one candidate's bootstrap null."""
    z0, z1, y0, y1, x0, x1 = [int(value) for value in bbox]
    crop_slices = np.s_[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    raw = np.asarray(image[crop_slices], dtype=np.float64)
    labels = np.asarray(masks[crop_slices], dtype=np.int32)
    cell_label = int(row['WalkID']) + 1
    own = labels == cell_label
    support = _owned_support(
        labels, cell_label, cfg['center_shell_nm'], Z_STEP_NM, XY_PIXEL_NM)
    if not np.any(own) or not np.any(support):
        raise RuntimeError(
            f'empty bootstrap support for {row["tp"]}, WalkID={row["WalkID"]}')

    zz, yy, xx = np.indices(raw.shape, dtype=np.float64)
    m1_params = np.array([
        float(row['m1_amplitude']), float(row['m1_z']) - z0,
        float(row['m1_y']) - y0, float(row['m1_x']) - x0,
        float(row['m1_sigma_z']), float(row['m1_sigma_y']),
        float(row['m1_sigma_x']), float(row['m1_background']),
    ], dtype=np.float64)
    model = _model_1spot(m1_params, zz, yy, xx)
    residual = raw - model
    residual = residual - float(np.mean(residual[support]))

    block_z = min(args.auto_null_block_z, raw.shape[0])
    block_xy = min(args.auto_null_block_xy, raw.shape[1], raw.shape[2])
    z_blocks = np.arange(raw.shape[0]) // block_z
    y_blocks = np.arange(raw.shape[1]) // block_xy
    x_blocks = np.arange(raw.shape[2]) // block_xy
    n_y_blocks = int(y_blocks[-1]) + 1
    n_x_blocks = int(x_blocks[-1]) + 1
    block_ids = ((z_blocks[:, None, None] * n_y_blocks
                  + y_blocks[None, :, None]) * n_x_blocks
                 + x_blocks[None, None, :])
    n_blocks = int(block_ids.max()) + 1

    return {
        'row': row,
        'model': model,
        'residual': residual,
        'own': own,
        'other': (labels != 0) & ~own,
        'm1_params': m1_params,
        'block_ids': block_ids,
        'n_blocks': n_blocks,
        'rng': np.random.default_rng(np.random.SeedSequence([
            int(args.auto_null_seed), int(tp_index), int(row['WalkID'])])),
    }


def _score_pooled_null_batch(cases, states, cfg, args):
    """Pack mixed candidate crops into one image and score them in C."""
    batch_size = len(cases)
    max_z = max(states[candidate]['model'].shape[0] for candidate, _ in cases)
    max_y = max(states[candidate]['model'].shape[1] for candidate, _ in cases)
    tiled_nx = sum(states[candidate]['model'].shape[2]
                   for candidate, _ in cases) + batch_size - 1
    sim_image = np.zeros((max_z, max_y, tiled_nx), dtype=np.float64)
    sim_labels = np.zeros(sim_image.shape, dtype=np.int32)
    sim_bboxes = np.zeros((batch_size, 6), dtype=np.int32)
    sim_has_fit1 = np.ones(batch_size, dtype=np.int32)
    sim_fit1 = np.zeros((batch_size, 8), dtype=np.float64)
    x_offset = 0

    for case_index, (candidate_index, weights) in enumerate(cases):
        state = states[candidate_index]
        shape = state['model'].shape
        x_slice = slice(x_offset, x_offset + shape[2])
        simulated = np.maximum(
            0.0, state['model']
            + state['residual'] * weights[state['block_ids']])
        sim_image[:shape[0], :shape[1], x_slice] = simulated
        case_labels = sim_labels[:shape[0], :shape[1], x_slice]
        case_labels[state['own']] = case_index + 1
        case_labels[state['other']] = batch_size + 1
        sim_bboxes[case_index] = [
            0, shape[0] - 1, 0, shape[1] - 1,
            x_offset, x_offset + shape[2] - 1,
        ]
        sim_fit1[case_index] = state['m1_params']
        sim_fit1[case_index, 3] += x_offset
        x_offset += shape[2] + 1

    sim_image = np.ascontiguousarray(sim_image, dtype=np.float64)
    sim_labels = np.ascontiguousarray(sim_labels, dtype=np.int32)
    sim_bboxes = np.ascontiguousarray(sim_bboxes, dtype=np.int32)
    sim_fit1 = np.ascontiguousarray(sim_fit1, dtype=np.float64)
    out_floats = np.zeros(batch_size * NCOLS_FLOAT, dtype=np.float64)
    out_ints = np.zeros(batch_size * NCOLS_INT, dtype=np.int32)

    _BACKEND.batch_replication_qc(
        _as_double_ptr(sim_image),
        sim_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(sim_image.shape[0]), ctypes.c_int(sim_image.shape[1]),
        ctypes.c_int(sim_image.shape[2]),
        sim_bboxes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        sim_has_fit1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _as_double_ptr(sim_fit1.ravel()), ctypes.c_int(batch_size),
        ctypes.c_double(cfg['sigma_xy_broad']),
        ctypes.c_double(cfg['sigma_z_broad']),
        ctypes.c_double(cfg['resid_score_thresh']),
        ctypes.c_double(cfg['min_lobe_sep_px']),
        ctypes.c_double(cfg['min_lobe_rel_amp']),
        ctypes.c_double(cfg['delta_bic_thresh']),
        ctypes.c_double(cfg['min_spot_sep_nm']),
        ctypes.c_double(cfg['min_amp_ratio']),
        ctypes.c_double(cfg['min_amp_recovery']),
        ctypes.c_double(cfg['min_peak_support_c2']),
        ctypes.c_double(cfg['max_valley_frac']),
        ctypes.c_double(cfg['min_resid_core_frac']),
        ctypes.c_double(cfg['sigma_z_max']),
        ctypes.c_double(cfg['center_shell_nm']),
        ctypes.c_double(Z_STEP_NM), ctypes.c_double(XY_PIXEL_NM),
        ctypes.c_int(20), ctypes.c_int(200),
        _as_double_ptr(out_floats),
        out_ints.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    results = []
    for case_index, (candidate_index, _) in enumerate(cases):
        ints = out_ints[
            case_index * NCOLS_INT:(case_index + 1) * NCOLS_INT]
        floats = out_floats[
            case_index * NCOLS_FLOAT:(case_index + 1) * NCOLS_FLOAT]
        passed = int(ints[3]) == 1
        if args.require_two_lobe_for_confirmation:
            passed = passed and int(ints[8]) == 1
        if cfg['min_spot_sep_xy_nm'] > 0:
            passed = (passed and np.isfinite(floats[34])
                      and floats[34] >= cfg['min_spot_sep_xy_nm'])
        if passed and cfg.get('se_correction_factor', 1.0) > 1.0:
            axial_ok = int(ints[19]) == 1
            sep = floats[3] if axial_ok else floats[19]
            se = floats[31] if axial_ok else floats[33]
            if not (np.isfinite(sep) and np.isfinite(se)
                    and sep - 1.96 * se * cfg['se_correction_factor']
                    >= cfg['min_spot_sep_nm']):
                passed = False
        score = float(floats[2]) if passed and np.isfinite(floats[2]) else -np.inf
        results.append((candidate_index, score,
                        int(ints[0] in (5, 6, 7, 8, 9))))
    return results


def _simulate_pooled_nulls(states, n_replicates, cfg, args):
    """Generate and score n additional null draws for every candidate."""
    scores = [[] for _ in states]
    failures = np.zeros(len(states), dtype=np.int64)
    cases = []
    max_z = max_y = total_x = 0
    n_batches = 0

    def flush():
        nonlocal cases, max_z, max_y, total_x, n_batches
        if not cases:
            return
        for candidate_index, score, failed in _score_pooled_null_batch(
                cases, states, cfg, args):
            scores[candidate_index].append(score)
            failures[candidate_index] += failed
        n_batches += 1
        cases = []
        max_z = max_y = total_x = 0

    # Interleave candidates within every replicate so each packed C call has
    # varied crop sizes and enough independent work for all OpenMP threads.
    for _ in range(n_replicates):
        for candidate_index, state in enumerate(states):
            shape = state['model'].shape
            candidate_max_z = max(max_z, shape[0])
            candidate_max_y = max(max_y, shape[1])
            candidate_total_x = total_x + shape[2] + (1 if cases else 0)
            packed_voxels = (candidate_max_z * candidate_max_y
                             * candidate_total_x)
            if cases and packed_voxels > args.auto_null_max_batch_voxels:
                flush()
                candidate_max_z, candidate_max_y = shape[:2]
                candidate_total_x = shape[2]
            cases.append((
                candidate_index,
                _mammen_weights(state['rng'], state['n_blocks'])))
            max_z, max_y, total_x = (
                candidate_max_z, candidate_max_y, candidate_total_x)
    flush()
    return np.asarray(scores, dtype=np.float64), failures, n_batches


def _auto_calibrate_candidates(image, masks, bboxes, candidates, cfg, args,
                               tp_index):
    """Run bounded pooled-null calibration for one candidate family."""
    states = [
        _prepare_auto_null_candidate(
            image, masks, bboxes[int(row['WalkID'])], row, cfg, args, tp_index)
        for row in candidates
    ]
    observed = np.asarray(
        [float(row['delta_BIC']) for row in candidates], dtype=np.float64)
    null_scores = np.empty((len(candidates), 0), dtype=np.float64)
    detector_failures = np.zeros(len(candidates), dtype=np.int64)
    target_replicates = min(
        args.auto_null_bootstrap_reps, args.auto_null_max_reps)
    n_batches = 0
    stages = []

    while True:
        additional = target_replicates - null_scores.shape[1]
        new_scores, new_failures, new_batches = _simulate_pooled_nulls(
            states, additional, cfg, args)
        null_scores = np.concatenate((null_scores, new_scores), axis=1)
        detector_failures += new_failures
        n_batches += new_batches
        q_values, raw_fdr, pooled_exceedances = _pooled_null_qvalues(
            observed, null_scores)
        stages.append({
            'replicates': int(null_scores.shape[1]),
            'retained': int(np.count_nonzero(q_values <= args.auto_fdr_q)),
            'minimum_q': float(np.min(q_values)),
        })
        if np.any(q_values <= args.auto_fdr_q):
            break

        # Extend only when finite-simulation resolution, rather than observed
        # null contamination, is blocking the best candidate prefix.
        best = int(np.argmin(raw_fdr))
        if (pooled_exceedances[best] != 0
                or null_scores.shape[1] >= args.auto_null_max_reps):
            break
        order = np.argsort(-observed, kind='stable')
        rank = int(np.flatnonzero(order == best)[0]) + 1
        needed = int(math.ceil(1.0 / (args.auto_fdr_q * rank)))
        if needed <= null_scores.shape[1]:
            break
        target_replicates = min(args.auto_null_max_reps, needed)

    own_exceedances = np.sum(null_scores >= observed[:, None], axis=1)
    marginal_p = (own_exceedances + 1.0) / (null_scores.shape[1] + 1.0)
    return {
        'q_values': q_values,
        'raw_fdr': raw_fdr,
        'pooled_exceedances': pooled_exceedances,
        'own_exceedances': own_exceedances,
        'marginal_p': marginal_p,
        'replicates': int(null_scores.shape[1]),
        'detector_failures': detector_failures,
        'n_blocks': np.asarray([state['n_blocks'] for state in states]),
        'batches': n_batches,
        'stages': stages,
    }


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def run(args):
    # Preserve the historical direct ``run(SimpleNamespace(...))`` API while
    # the CLI supplies these explicitly.
    if not hasattr(args, 'segmentation_mode'):
        args.segmentation_mode = 'stitched'
    if not hasattr(args, 'z_gate_path'):
        args.z_gate_path = None
    if not hasattr(args, 'z_gate_failure_policy'):
        args.z_gate_failure_policy = 'quarantine'
    cfg = dict(
        sigma_xy_broad=args.sigma_xy_broad,
        sigma_z_broad=args.sigma_z_broad,
        resid_score_thresh=args.resid_score_thresh,
        min_lobe_sep_px=args.min_lobe_sep_px,
        min_lobe_rel_amp=args.min_lobe_rel_amp,
        delta_bic_thresh=args.delta_bic_thresh,
        min_spot_sep_nm=args.min_spot_sep_nm,
        min_spot_sep_xy_nm=args.min_spot_sep_xy_nm,
        min_amp_ratio=args.min_amp_ratio,
        min_amp_recovery=args.min_amp_recovery,
        min_peak_support_c2=args.min_peak_support_c2,
        max_valley_frac=args.max_valley_frac,
        min_resid_core_frac=args.min_resid_core_frac,
        sigma_z_max=args.sigma_z_max,
        center_shell_nm=args.center_shell_nm,
        se_correction_factor=getattr(args, 'se_correction_factor', 1.0),
    )

    print('=' * 72)
    print(f'Replication QC (pad_xy={args.pad_xy}, pad_z={args.pad_z})')
    print('=' * 72)
    print(f'Stage A triggers (OR): sigma_xy>{cfg["sigma_xy_broad"]} px, '
          f'resid_score>{cfg["resid_score_thresh"]}, '
          f'lobe sep>={cfg["min_lobe_sep_px"]} px & '
          f'rel amp>={cfg["min_lobe_rel_amp"]}')
    print(f'Stage A diagnostic only (NOT a trigger): '
          f'sigma_z>{cfg["sigma_z_broad"]} z-slices')
    sigma_policy = (f'cap={cfg["sigma_z_max"]:g} slices'
                    if cfg['sigma_z_max'] > 0 else 'uncapped (crop-depth guard)')
    print(f'Model policy: refit M1/M2 on owned support + '
          f'{cfg["center_shell_nm"]:g} nm shell; sigma_z {sigma_policy}')
    stage_b_active = [
        f'delta_BIC>={cfg["delta_bic_thresh"]}',
        ('spot_sep_corrLCB>={:.0f} nm (SE*{:g})'.format(
            cfg['min_spot_sep_nm'], cfg.get('se_correction_factor', 1.0))
         if cfg.get('se_correction_factor', 1.0) > 1.0
         else f'spot_sep_95pct_LCB>={cfg["min_spot_sep_nm"]:.0f} nm'),
        f'amp_ratio>={cfg["min_amp_ratio"]}',
    ]
    if cfg['min_amp_recovery'] > 0:
        stage_b_active.append(f'amp_recovery>={cfg["min_amp_recovery"]}')
    if cfg['min_peak_support_c2'] > 0:
        stage_b_active.append(f'peak_support_c2>={cfg["min_peak_support_c2"]}')
    if cfg['max_valley_frac'] > 0:
        stage_b_active.append(f'valley_frac<={cfg["max_valley_frac"]}')
    if cfg['min_resid_core_frac'] > 0:
        stage_b_active.append(f'resid_core_frac>={cfg["min_resid_core_frac"]}')
    print(f'Stage B + call: {", ".join(stage_b_active)}'
          + (', require_two_lobe=ON'
             if args.require_two_lobe_for_confirmation else ''))
    diagnostic_only = []
    if cfg['min_amp_recovery'] <= 0:
        diagnostic_only.append('amp_recovery')
    if cfg['min_peak_support_c2'] <= 0:
        diagnostic_only.append('peak_support_c2')
    if cfg['max_valley_frac'] <= 0:
        diagnostic_only.append('valley_frac')
    if cfg['min_resid_core_frac'] <= 0:
        diagnostic_only.append('resid_core_frac')
    if diagnostic_only:
        print(f'Diagnostic-only metrics (computed and written to CSV but '
              f'not gating confirm): {", ".join(diagnostic_only)}')
    if args.auto_null_bootstrap_reps > 0:
        print(f'Automatic calibration: pooled spatial wild-bootstrap null, '
              f'{args.auto_null_bootstrap_reps} initial and '
              f'{args.auto_null_max_reps} maximum replicates/candidate, '
              f'FDR q<={args.auto_fdr_q:g} within each TP')

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load mask + 1-spot CSV ---
    masks = tifffile.imread(args.mask_path).astype(np.int32)
    if masks.ndim != 3:
        raise ValueError(
            f'Expected a 3D mask TIFF, got shape {masks.shape} at '
            f'{args.mask_path}')
    nz, ny, nx = masks.shape
    n_cells = int(masks.max())
    print(f'Mask: {masks.shape}, {n_cells} cells')

    # Resolve the source before loading a z-support sidecar: strict sidecar
    # validation is against the complete, exact set of probe-0 TP names, not
    # merely the optional --limit_tps processing prefix.
    src_dir = args.input_dir or args.probe0_dir

    # --- Chromatic registration of the DAPI mask into the 640 grid -----------
    # The owned support that M1/M2 are refit on (and the strict centre-ownership
    # test built from it via --center_shell_nm 0) must live in the SAME pixel
    # frame as the 640 image. Default 'off' preserves existing behaviour exactly.
    # Registration happens before compute_bboxes so crops follow the mask.
    chromatic_registration = getattr(args, 'chromatic_registration', 'off')
    chrom_calib = None
    if chromatic_registration == 'metadata':
        if _chrom is None:
            raise RuntimeError(
                'pipeline.chromatic_registration is required for '
                '--chromatic-registration metadata')
        meta_path = (getattr(args, 'chromatic_metadata', None)
                     or os.path.join(src_dir, 'widefield.json'))
        chrom_calib = _chrom.load_calibration(meta_path)   # raises if unusable
        ref_nm = (getattr(args, 'reference_wavelength', None)
                  or _chrom.modality_default(
                      getattr(args, 'modality', 'widefield'), 'reference'))
        target = (getattr(args, 'target_wavelength', None)
                  or _chrom.modality_default(
                      getattr(args, 'modality', 'widefield'), 'probe'))
        print(f'  wavelengths: reference {ref_nm} nm -> target {target} nm')
        print(f'Chromatic registration: {meta_path}')
        print(f'  channel map (wavelength -> transform index): '
              f'{dict(chrom_calib.transform_index)}')
        labels_before = set(np.unique(masks).tolist())
        masks = np.ascontiguousarray(
            _chrom.resample_labels_to_channel(masks, chrom_calib, target, ref_nm)
            .astype(np.int32, copy=False))
        if not set(np.unique(masks).tolist()).issubset(labels_before):
            raise RuntimeError('chromatic resample introduced new label values; '
                               'the lookup was not categorical')
        chrom_provenance = _chrom.provenance(
            chrom_calib, {'probe_0': target}, mode='metadata',
            stage='replication_qc', reference_wavelength=ref_nm,
            ownership_policy=getattr(args, 'center_ownership', 'allow'),
            cli_args={'modality': getattr(args, 'modality', 'widefield'),
                      'center_shell_nm': args.center_shell_nm,
                      'segmentation_mode': args.segmentation_mode,
                      'sigma_z_max': args.sigma_z_max},
            notes='DAPI label mask represented on the probe_0 (640) pixel grid '
                  'for crop bounding boxes, the owned M1/M2 refit support, and '
                  'the centre-ownership test derived from it.')
    elif _chrom is not None:
        chrom_provenance = _chrom.disabled_provenance(
            stage='replication_qc',
            ownership_policy=getattr(args, 'center_ownership', 'allow'),
            reason='Chromatic registration OFF: the 405-frame DAPI mask is used '
            'directly as the 640 owned support, so any 405<->640 chromatic '
            'offset is uncorrected in the M1/M2 refit support and in the '
            'centre-ownership test built from it.')
    else:
        chrom_provenance = None
    if chrom_provenance is not None:
        _chrom.write_provenance(
            os.path.join(args.outdir,
                         'replication_qc_chromatic_registration.json'),
            chrom_provenance)

    _dil_px = getattr(args, 'mask_dilation_xy_px', 0)
    if _dil_px > 0:
        from scipy.ndimage import maximum_filter as _mf
        _k = 2 * _dil_px + 1
        masks = _mf(masks, footprint=np.ones((1, _k, _k), dtype=bool)
                               ).astype(np.int32, copy=False)
        masks = np.ascontiguousarray(masks)
        print(f'Mask dilation: {_dil_px} px XY (kernel {_k}x{_k}), in memory only')

    bboxes = compute_bboxes(masks, args.pad_xy, args.pad_z)

    z_support = None
    labels_2d = None
    z_label_to_axis = None
    if args.segmentation_mode == '2p5d-zgated':
        source_tp_names = dat_io.timepoint_names(src_dir, 0, args.source)
        if load_z_support is None or materialize_labels is None:
            raise RuntimeError(
                'pipeline.z_support_2p5d is required for '
                '--segmentation-mode 2p5d-zgated')
        mask_dir = os.path.dirname(os.path.abspath(args.mask_path))
        labels_2d_path = os.path.join(mask_dir, 'masks_2p5d_2d.tif')
        z_gate_path = (args.z_gate_path or
                       os.path.join(mask_dir, 'z_support_2p5d.npz'))
        if not os.path.isfile(labels_2d_path):
            raise FileNotFoundError(
                f'2p5d-zgated requires the authoritative 2D mask: '
                f'{labels_2d_path}')
        labels_2d = tifffile.imread(labels_2d_path).astype(np.int32)
        if labels_2d.ndim != 2 or labels_2d.shape != (ny, nx):
            raise ValueError(
                f'2D ownership-mask shape mismatch: {labels_2d.shape} vs '
                f'expected {(ny, nx)}')
        z_support = load_z_support(
            z_gate_path,
            labels_2d,
            expected_tp_names=source_tp_names,
            expected_nz=nz,
            expected_failure_policy=args.z_gate_failure_policy,
        )
        label_ids = [int(value) for value in
                     np.asarray(z_support.label_ids).tolist()]
        z_label_to_axis = {label: index
                           for index, label in enumerate(label_ids)}
        expected_labels = list(range(1, n_cells + 1))
        if label_ids != expected_labels:
            raise ValueError(
                '2D z-support label IDs do not match the compatibility mask: '
                f'{label_ids} vs {expected_labels}')
        if tuple(z_support.mask_shape_yx) != (ny, nx):
            raise ValueError(
                f'z-support XY shape {tuple(z_support.mask_shape_yx)} does '
                f'not match mask shape {(ny, nx)}')
        if int(z_support.nz) != nz:
            raise ValueError(
                f'z-support depth {z_support.nz} does not match mask depth '
                f'{nz}')
        print(f'Z-gated support: {z_gate_path} '
              f'({len(source_tp_names)} exact TPs, '
              f'failure_policy={args.z_gate_failure_policy})')
        # The sidecar is validated against the UNREGISTERED 2D map, the frame it
        # was computed in; its per-label z intervals are frame-independent, so
        # per-TP materialization can use the registered ownership map.
        if chrom_calib is not None:
            labels_2d = np.ascontiguousarray(
                _chrom.resample_labels_to_channel(
                    labels_2d, chrom_calib, target, ref_nm)
                .astype(np.int32, copy=False))
            print('  registered the authoritative 2D ownership map into the '
                  '640 grid')
        if _dil_px > 0:
            from scipy.ndimage import maximum_filter as _mf2d
            labels_2d = _mf2d(labels_2d, size=2 * _dil_px + 1
                              ).astype(np.int32, copy=False)

    # Robust per-Location mask-size baseline. Used by the strict
    # merged_mask_suspect rule: a cell whose 3D voxel count is far above
    # the population median is a likely candidate for fused/merged-cell
    # masks. Computed once here because the mask is per-Location, not
    # per-TP. Threshold defaults to 1.5 * median; CLI override available.
    size_baseline_name = '3D voxel count'
    size_gate_name = 'n_voxels'
    size_by_cell = None
    if args.segmentation_mode == '2p5d-zgated':
        size_baseline_name = '2D ownership area'
        size_gate_name = size_baseline_name
        size_by_cell = np.bincount(
            labels_2d.ravel(), minlength=n_cells + 1)[1:n_cells + 1]
        _label_counts = size_by_cell
    elif n_cells > 0:
        _label_counts = np.bincount(masks.ravel(), minlength=n_cells + 1)[1:n_cells + 1]
        size_by_cell = _label_counts
    else:
        _label_counts = np.empty(0, dtype=np.int64)
        size_by_cell = _label_counts
    if n_cells > 0:
        _label_counts = _label_counts[_label_counts > 0]
        _median_voxels = float(np.median(_label_counts)) if _label_counts.size else 0.0
    else:
        _median_voxels = 0.0
    mask_size_threshold = (args.merged_mask_size_factor * _median_voxels
                           if _median_voxels > 0 else float('inf'))
    print(f'Mask-area baseline: median {size_baseline_name} = {_median_voxels:.0f}; '
          f'merged_mask_suspect requires {size_gate_name} >= '
          f'{mask_size_threshold:.0f}  (= {args.merged_mask_size_factor:g} x median).')

    fits1 = _load_1spot_csv(
        args.input_csv,
        require_z_gate=(args.segmentation_mode == '2p5d-zgated'))
    print(f'Loaded {len(fits1)} 1-spot rows from {args.input_csv}')

    # --- Iterate TPs (per-TP TIFFs, or de-interleaved from raw .dat) ---
    n_tp = dat_io.n_timepoints(src_dir, 0, args.source)
    n_use = min(n_tp, int(args.limit_tps)) if args.limit_tps else n_tp
    print(f'Processing {n_use} TPs ({args.source})')

    # TP range that contributes to the walk-level replication call. Stage A+B
    # still runs on every TP (useful for diagnostic review), but TPs outside
    # this range have replication_confirmed forced to 0 so they cannot push a
    # walk to "Replicating=1" in the downstream classifier output. Edge TPs
    # (typically QC TPs 1-2 and the control re-image TP) are excluded — they
    # have different SNR characteristics and can fire spurious 2-spot fits.
    call_start_0 = args.replication_tp_start
    call_end_0 = args.replication_tp_end
    arm_a, arm_b = build_replichores(
        args.ori_tp, args.ter_tp, call_start_0, call_end_0)
    print(f'Replication-call TP range (1-based, inclusive): '
          f'[{call_start_0 + 1}, {call_end_0 + 1}]  '
          f'(out-of-range TPs get replication_confirmed=0)')
    print(f'Replication hierarchy: Ori=TP{args.ori_tp:03d}, '
          f'Ter=TP{args.ter_tp:03d}; independent biological-order arms '
          f'{arm_a} and {arm_b}')

    out_rows = []
    gallery = []  # (score, obs, model1, model2, fit1, params2, meta)
    auto_reports = []
    t0 = time.time()

    # Status code -> CSV string mapping (batch backend uses the integer form).

    # Global status tallies across all TPs (printed at the end).
    status_totals = defaultdict(int)
    total_no_fit1 = 0
    total_too_few_vox = 0
    total_gate_unevaluable = 0

    # Pre-pack mask + bboxes once for the C-side buffer pointers.
    masks_c = np.ascontiguousarray(masks, dtype=np.int32)
    bboxes_c = np.ascontiguousarray(bboxes, dtype=np.int32)

    # Output buffers reused per TP. Shape matches the C contract.
    # Column layout — keep in sync with batch_replication_qc() in
    # replication_qc_fit.c:
    #   floats (41): BIC/model/fit diagnostics; see replication_qc_fit.c for
    #                the ABI index table. The first 20 retain the v1 layout.
    #                BIC_1, BIC_2, delta_BIC, spot_sep_nm, amp1, amp2,
    #                sigma_x, sigma_y, sigma_z, resid_score, peak_rel_amp,
    #                peak_offset_px, sse1, sse2, amp_recovery,
    #                peak_support_c1, peak_support_c2, valley_frac,
    #                resid_core_frac
    #   ints   (24): status, n_voxels, replication_suspect,
    #                replication_confirmed, n_spots_best, sigma_xy_broad,
    #                sigma_z_broad, resid_broad, two_lobe_suspect,
    #                fit2_pinned, fit2_center_pinned, fit2_sigma_pinned
    out_floats = np.zeros(n_cells * NCOLS_FLOAT, dtype=np.float64)
    out_ints   = np.zeros(n_cells * NCOLS_INT,   dtype=np.int32)

    for tp_idx, (tp_name, img) in enumerate(dat_io.iter_timepoints(src_dir, 0, args.source)):
        if tp_idx >= n_use:
            break
        img = img.astype(np.float64)
        if img.shape != masks.shape:
            raise ValueError(
                f'Shape mismatch at {tp_name}: img={img.shape} vs mask={masks.shape}. '
                'Refusing to turn a missing QC timepoint into negative evidence.')

        # Is this TP eligible to contribute to the walk-level replication call?
        m_tp = _TP_NAME_RE.match(tp_name)
        tp_idx_0 = int(m_tp.group(1)) if m_tp else tp_idx
        tp_in_call_range = (call_start_0 <= tp_idx_0 <= call_end_0)
        tp_row_start = len(out_rows)

        # Materialize the exact per-TP support used for M1/M2.  Axial bbox
        # padding is intentionally disabled: the C backend's physical shell
        # must not inspect a plane outside a cell's selected z interval.
        gate_tp = None
        if z_support is not None:
            gate_tp = _z_support_tp_arrays(z_support, tp_name)
            if gate_tp['label_ids'].tolist() != list(range(1, n_cells + 1)):
                raise ValueError(
                    f'z-support label order mismatch at {tp_name}: '
                    f'{gate_tp["label_ids"].tolist()}')
            effective_masks = materialize_labels(
                labels_2d, nz, gate_tp['z_min'], gate_tp['z_max'],
                fit_allowed=gate_tp['fit_allowed'],
                label_ids=gate_tp['label_ids'])
            effective_masks = _protect_inactive_xy_ownership(
                effective_masks, labels_2d, n_cells)
            tp_masks = np.ascontiguousarray(effective_masks, dtype=np.int32)
            tp_bboxes = np.ascontiguousarray(
                compute_bboxes(tp_masks, args.pad_xy, 0), dtype=np.int32)
            tp_label_counts = np.bincount(
                tp_masks.ravel(), minlength=n_cells + 2)[:n_cells + 1]
        else:
            tp_masks = masks_c
            tp_bboxes = bboxes_c
            tp_label_counts = None

        # Build per-cell arrays of {has_fit1, fit1 params} for this TP.
        has_fit1 = np.zeros(n_cells, dtype=np.int32)
        fit1_arr = np.zeros(n_cells * 8, dtype=np.float64)
        for c in range(n_cells):
            f = fits1.get((tp_name, c))
            if f is None:
                continue
            if gate_tp is not None:
                gate_axis = z_label_to_axis[c + 1]
                _validate_fit_gate_provenance(
                    f, gate_tp, gate_axis, tp_name, c)
                if not bool(gate_tp['evaluable'][gate_axis]):
                    continue
            has_fit1[c] = 1
            base = c * 8
            fit1_arr[base + 0] = f['amplitude']
            fit1_arr[base + 1] = f['z']
            fit1_arr[base + 2] = f['y']
            fit1_arr[base + 3] = f['x']
            fit1_arr[base + 4] = f['sigma_z']
            fit1_arr[base + 5] = f['sigma_y']
            fit1_arr[base + 6] = f['sigma_x']
            fit1_arr[base + 7] = f['background']

        img_c = np.ascontiguousarray(img, dtype=np.float64)

        tp_t0 = time.time()
        _BACKEND.batch_replication_qc(
            _as_double_ptr(img_c),
            tp_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(int(tp_masks.shape[0])),
            ctypes.c_int(int(tp_masks.shape[1])),
            ctypes.c_int(int(tp_masks.shape[2])),
            tp_bboxes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            has_fit1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            _as_double_ptr(fit1_arr),
            ctypes.c_int(n_cells),
            ctypes.c_double(cfg['sigma_xy_broad']),
            ctypes.c_double(cfg['sigma_z_broad']),
            ctypes.c_double(cfg['resid_score_thresh']),
            ctypes.c_double(cfg['min_lobe_sep_px']),
            ctypes.c_double(cfg['min_lobe_rel_amp']),
            ctypes.c_double(cfg['delta_bic_thresh']),
            ctypes.c_double(cfg['min_spot_sep_nm']),
            ctypes.c_double(cfg['min_amp_ratio']),
            ctypes.c_double(cfg['min_amp_recovery']),
            ctypes.c_double(cfg['min_peak_support_c2']),
            ctypes.c_double(cfg['max_valley_frac']),
            ctypes.c_double(cfg['min_resid_core_frac']),
            ctypes.c_double(cfg['sigma_z_max']),
            ctypes.c_double(cfg['center_shell_nm']),
            ctypes.c_double(Z_STEP_NM),
            ctypes.c_double(XY_PIXEL_NM),
            ctypes.c_int(20),    # min_n_voxels — matches original Python gate
            ctypes.c_int(200),   # max_iter_lm
            _as_double_ptr(out_floats),
            out_ints.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        # Unpack per-cell rows in index order. Skip cells with status 0/1.
        n_present = n_suspect = n_confirmed = 0
        n_no_fit1 = n_too_few_vox = 0
        n_gate_unevaluable = 0
        n_sigma_xy = n_sigma_z = n_resid = n_two_lobe = 0
        n_stage_b_attempted = n_stage_b_failed = 0
        n_overridden_two_lobe = 0   # confirmations reverted by --require_two_lobe_for_confirmation
        n_overridden_out_of_range = 0  # confirmations zeroed because TP outside call range
        n_ownership_unevaluable = 0  # loci made unevaluable by strict two-spot ownership
        n_overridden_xy_sep = 0     # confirmations zeroed by --min_spot_sep_xy_nm gate
        n_pinned_double_fit = 0     # diagnostic: 2-spot fits that pinned but pass thresholds
        n_merged_mask_suspect = 0   # stricter subset: pinned_double_fit + center-pinned + oversized mask
        dbic_suspect_sum = 0.0
        dbic_suspect_n = 0
        for c in range(n_cells):
            base_i = c * NCOLS_INT
            status_code = int(out_ints[base_i + 0])
            gate_axis = (z_label_to_axis[c + 1]
                         if gate_tp is not None else None)
            if (gate_axis is not None
                    and not bool(gate_tp['evaluable'][gate_axis])):
                quarantined = bool(gate_tp['quarantined'][gate_axis])
                gate_status = _gate_status_text(
                    gate_tp['status'][gate_axis])
                out_rows.append({
                    'schema_version': REPLICATION_QC_SCHEMA,
                    'calibration_version': args.calibration_version,
                    'tp': tp_name,
                    'WalkID': c,
                    'n_voxels': int(tp_label_counts[c + 1]),
                    'replication_suspect': 0,
                    'replication_confirmed': 0,
                    'auto_null_tested': 0,
                    'auto_null_reps': 0,
                    'pinned_double_fit_suspect': 0,
                    'merged_mask_suspect': 0,
                    'in_call_range': int(tp_in_call_range),
                    'n_spots_best': '',
                    'delta_BIC': '',
                    'spot_sep_nm': '',
                    'z_gate_status': gate_status,
                    'z_gate_fallback': _gate_status_text(
                        gate_tp['fallback'][gate_axis]),
                    'z_gate_min': _gate_bound_for_csv(
                        gate_tp['z_min'][gate_axis], nz),
                    'z_gate_max': _gate_bound_for_csv(
                        gate_tp['z_max'][gate_axis], nz),
                    'z_gate_quarantined': int(quarantined),
                    'z_gate_evaluable': 0,
                    'status': ('z_gate_quarantined:' if quarantined else
                               'z_gate_failed:') + gate_status,
                })
                n_gate_unevaluable += 1
                continue
            if status_code == 0:
                n_no_fit1 += 1
                continue
            if status_code == 1:
                n_too_few_vox += 1
                continue
            n_present += 1
            base_f = c * NCOLS_FLOAT
            suspect_flag   = int(out_ints[base_i + 2])
            confirmed_flag = int(out_ints[base_i + 3])
            two_lobe_flag  = int(out_ints[base_i + 8])

            # Optional stricter confirmation: require the Stage A two-lobe
            # residual flag. Overrides must happen BEFORE we bump the
            # status_totals / status-column so the CSV/summary stay consistent.
            if (args.require_two_lobe_for_confirmation
                    and confirmed_flag == 1 and two_lobe_flag == 0):
                confirmed_flag = 0
                status_code = 3   # tested_not_confirmed
                n_overridden_two_lobe += 1

            # Optional lateral-separation gate. NOTE: this thresholds the 95%
            # LOWER CONFIDENCE BOUND on the lateral separation (float 34,
            # spot_sep_xy_lcb_nm), not the raw fitted spot_sep_xy_nm -- the
            # same conservative convention as the 3D --min_spot_sep_nm gate.
            # The 3D separation folds in the axial component, which is
            # unreliable on these thin (~10-14 slice) stacks where the z fit is
            # frequently unidentifiable; gating on the lateral component
            # rejects splits that "separate" only in z.
            # Default 0 disables this (no behaviour change).
            if (cfg['min_spot_sep_xy_nm'] > 0
                    and confirmed_flag == 1):
                sep_xy_lcb = out_floats[base_f + 34]
                if not (np.isfinite(sep_xy_lcb)
                        and sep_xy_lcb >= cfg['min_spot_sep_xy_nm']):
                    confirmed_flag = 0
                    status_code = 3   # tested_not_confirmed
                    n_overridden_xy_sep += 1

            if (cfg.get('se_correction_factor', 1.0) > 1.0
                    and confirmed_flag == 1):
                axial_ok = int(out_ints[base_i + 19]) == 1
                _sep = out_floats[base_f + 3] if axial_ok else out_floats[base_f + 19]
                _se = out_floats[base_f + 31] if axial_ok else out_floats[base_f + 33]
                if not (np.isfinite(_sep) and np.isfinite(_se)
                        and _sep - 1.96 * _se * cfg['se_correction_factor']
                        >= cfg['min_spot_sep_nm']):
                    confirmed_flag = 0
                    status_code = 3

            # --- strict two-spot ownership -------------------------------
            # Both M2 component centres must lie in THIS cell's own registered
            # support, evaluated against the same effective mask the fit used and
            # with the same nearest-voxel rule as one-spot ownership. A component
            # in background or in another active label makes the locus
            # UNEVALUABLE: it is not a confirmed multispot, and it is NOT counted
            # as negative evidence either. The physical background shell remains
            # available for background ESTIMATION -- it never becomes legal
            # spot-centre territory.
            ownership_unevaluable = 0
            ownership_reason = ''
            if (getattr(args, 'center_ownership', 'allow') == 'strict'
                    and confirmed_flag == 1 and _own is not None):
                for _tag, _off in (('c1', 24), ('c2', 27)):
                    _cat = _own.classify_center(
                        tp_masks, c,
                        out_floats[base_f + _off],
                        out_floats[base_f + _off + 1],
                        out_floats[base_f + _off + 2])
                    if _cat not in _own.PASSING:
                        confirmed_flag = 0
                        # Its own status: NOT 4 (which the backend already uses
                        # for `confirmed`) and NOT 3 (`tested_not_confirmed`,
                        # which is positive evidence of a single spot).
                        status_code = STATUS_OWNERSHIP_UNEVALUABLE
                        ownership_unevaluable = 1
                        ownership_reason = f'm2_{_tag}_{_cat}'
                        n_ownership_unevaluable += 1
                        break

            # TP-range gate for the walk-level call: zero out confirmations
            # outside [call_start_0, call_end_0]. Stage A+B already ran (the
            # numeric metrics are still in the row for inspection); only the
            # final replication_confirmed flag changes.
            if confirmed_flag == 1 and not tp_in_call_range:
                confirmed_flag = 0
                status_code = 3   # tested_not_confirmed
                n_overridden_out_of_range += 1

            status_totals[status_code] += 1
            if suspect_flag: n_suspect += 1
            if confirmed_flag: n_confirmed += 1
            # Stage A reason breakdown (bitwise — a cell can trip multiple).
            if int(out_ints[base_i + 5]): n_sigma_xy += 1
            if int(out_ints[base_i + 6]): n_sigma_z += 1
            if int(out_ints[base_i + 7]): n_resid += 1
            if two_lobe_flag: n_two_lobe += 1
            # M2 is attempted for every row whose common-support M1 converges.
            if status_code in (3, 4, 5, 6, 7, STATUS_OWNERSHIP_UNEVALUABLE):
                n_stage_b_attempted += 1
                if status_code in (5, 6, 7):
                    n_stage_b_failed += 1
                dbic = out_floats[base_f + 2]
                if np.isfinite(dbic):
                    dbic_suspect_sum += dbic
                    dbic_suspect_n += 1

            def _float(idx):
                v = out_floats[base_f + idx]
                return '' if not np.isfinite(v) else float(v)

            # Derive amp_ratio from amp1/amp2 if Stage B ran (amp2 finite).
            amp1_v = out_floats[base_f + 4]
            amp2_v = out_floats[base_f + 5]
            if np.isfinite(amp1_v) and np.isfinite(amp2_v):
                amax = max(amp1_v, amp2_v)
                amp_ratio_v = float(min(amp1_v, amp2_v) / amax) if amax > 0 else 0.0
            else:
                amp_ratio_v = ''
            fit1_rc = int(out_ints[base_i + 12])
            fit2_rc = int(out_ints[base_i + 13])
            # `_v` suffix as elsewhere here: the module-level function of the
            # same name is the single definition, so it must not be shadowed.
            fit2_converged_v = fit2_converged(fit2_rc, status_code)
            fit2_pinned_v = int(out_ints[base_i + 9])
            fit2_center_pinned_v = int(out_ints[base_i + 10])
            fit2_sigma_pinned_v  = int(out_ints[base_i + 11])

            # Two diagnostic flags, broad → strict:
            #
            # pinned_double_fit_suspect (broad):
            #   Stage B met every numeric confirmation gate AND tripped the
            #   Stage A two-lobe heuristic AND is pinned at a bbox/sigma
            #   bound. This is the "fit looks like a double but ran into a
            #   bound" class. NOT a replication call.
            #
            # merged_mask_suspect (strict, segmentation-QC flag):
            #   pinned_double_fit_suspect == 1 AND the pin is on a center
            #   coordinate (not just on sigma) AND the cell mask is
            #   unusually large by the per-Location voxel-count baseline.
            #   This is the "the Cellpose mask probably covered two cells"
            #   class. NOT a replication call.
            n_voxels_v    = int(out_ints[base_i + 1])
            delta_bic_raw = out_floats[base_f + 2]
            sep_nm_raw    = (out_floats[base_f + 32]
                             if int(out_ints[base_i + 19])
                             else out_floats[base_f + 34])
            amp_recov_raw = out_floats[base_f + 14]
            amp_recov_gate_passes = (
                cfg['min_amp_recovery'] <= 0
                or (np.isfinite(amp_recov_raw)
                    and amp_recov_raw >= cfg['min_amp_recovery']))
            peak_support_raw = out_floats[base_f + 16]
            peak_support_gate_passes = (
                cfg['min_peak_support_c2'] <= 0
                or (np.isfinite(peak_support_raw)
                    and peak_support_raw >= cfg['min_peak_support_c2']))
            valley_raw = out_floats[base_f + 17]
            valley_gate_passes = (
                cfg['max_valley_frac'] <= 0
                or (np.isfinite(valley_raw)
                    and valley_raw <= cfg['max_valley_frac']))
            core_raw = out_floats[base_f + 18]
            core_gate_passes = (
                cfg['min_resid_core_frac'] <= 0
                or (np.isfinite(core_raw)
                    and core_raw >= cfg['min_resid_core_frac']))
            amp_ratio_num = (amp_ratio_v if isinstance(amp_ratio_v, float)
                             else float('nan'))
            pinned_double_fit_suspect = int(
                fit2_pinned_v == 1
                and two_lobe_flag == 1
                and np.isfinite(delta_bic_raw)
                and delta_bic_raw >= cfg['delta_bic_thresh']
                and np.isfinite(sep_nm_raw)
                and sep_nm_raw >= cfg['min_spot_sep_nm']
                and np.isfinite(amp_ratio_num)
                and amp_ratio_num >= cfg['min_amp_ratio']
                and amp_recov_gate_passes
                and peak_support_gate_passes
                and valley_gate_passes
                and core_gate_passes
                and int(out_ints[base_i + 20]) == 1
                and int(out_ints[base_i + 21]) == 1
                and np.isfinite(amp1_v) and amp1_v > 0
                and np.isfinite(amp2_v) and amp2_v > 0)
            merged_mask_suspect = int(
                pinned_double_fit_suspect == 1
                and fit2_center_pinned_v == 1
                and (size_by_cell[c]
                     if gate_axis is not None else n_voxels_v)
                >= mask_size_threshold)
            if pinned_double_fit_suspect:
                n_pinned_double_fit += 1
            if merged_mask_suspect:
                n_merged_mask_suspect += 1

            row = {
                'schema_version': REPLICATION_QC_SCHEMA,
                'calibration_version': args.calibration_version,
                'tp': tp_name, 'WalkID': c,
                'n_voxels': n_voxels_v,
                'replication_suspect':       suspect_flag,
                'replication_confirmed':     confirmed_flag,
                'auto_null_tested':          0,
                'auto_calibration_pass':     '',
                'auto_null_p':               '',
                'auto_null_q':               '',
                'auto_null_exceedances':     '',
                'auto_pooled_null_exceedances': '',
                'auto_pooled_fdr':           '',
                'auto_null_reps':            0,
                'pinned_double_fit_suspect': pinned_double_fit_suspect,
                'merged_mask_suspect':       merged_mask_suspect,
                'in_call_range':             int(tp_in_call_range),
                'ownership_unevaluable':     ownership_unevaluable,
                'ownership_failure_reason':  ownership_reason,
                'n_spots_best': 2 if confirmed_flag else 1,
                'BIC_1':          _float(0),
                'BIC_2':          _float(1),
                'delta_BIC':      _float(2),
                'spot_sep_nm':    _float(3),
                'spot_sep_xy_nm': _float(19),
                'spot_sep_se_nm': _float(31),
                'spot_sep_lcb_nm': _float(32),
                'spot_sep_xy_se_nm': _float(33),
                'spot_sep_xy_lcb_nm': _float(34),
                'amp1':           _float(4),
                'amp2':           _float(5),
                'amp_ratio':      amp_ratio_v,
                'amp_recovery':   _float(14),
                'peak_support_c1':_float(15),
                'peak_support_c2':_float(16),
                'valley_frac':    _float(17),
                'resid_core_frac':_float(18),
                'sigma_x':        _float(6),
                'sigma_y':        _float(7),
                'sigma_z':        _float(8),
                'sigma_z_se':     _float(30),
                'effective_sigma_z_max': _float(35),
                'm1_amplitude':   _float(20),
                'm1_sigma_z':     _float(21),
                'm1_sigma_y':     _float(22),
                'm1_sigma_x':     _float(23),
                'm1_z':           _float(36),
                'm1_y':           _float(37),
                'm1_x':           _float(38),
                'm1_background':  _float(39),
                'm2_c1_z':        _float(24),
                'm2_c1_y':        _float(25),
                'm2_c1_x':        _float(26),
                'm2_c2_z':        _float(27),
                'm2_c2_y':        _float(28),
                'm2_c2_x':        _float(29),
                'm2_background':  _float(40),
                'resid_score':    _float(9),
                'peak_rel_amp':   _float(10),
                'peak_offset_px': _float(11),
                'sse1':           _float(12),
                'sse2':           _float(13),
                'sigma_xy_broad':   int(out_ints[base_i + 5]),
                'sigma_z_broad':    int(out_ints[base_i + 6]),
                'resid_broad':      int(out_ints[base_i + 7]),
                'two_lobe_suspect': two_lobe_flag,
                'fit2_converged':       fit2_converged_v,
                'fit1_rc':              fit1_rc,
                'fit2_rc':              fit2_rc,
                'fit2_nfev':            int(out_ints[base_i + 14]),
                'fit2_n_starts':        int(out_ints[base_i + 15]),
                'fit2_pinned':          fit2_pinned_v,
                'fit2_center_pinned':   fit2_center_pinned_v,
                'fit2_sigma_pinned':    fit2_sigma_pinned_v,
                'fit2_center1_label':   int(out_ints[base_i + 16]),
                'fit2_center2_label':   int(out_ints[base_i + 17]),
                'fit2_sigma_z_censored': int(out_ints[base_i + 18]),
                'fit2_axial_identifiable': int(out_ints[base_i + 19]),
                'fit2_center1_supported': int(out_ints[base_i + 20]),
                'fit2_center2_supported': int(out_ints[base_i + 21]),
                'fit2_lower_width_pinned': int(out_ints[base_i + 22]),
                'fit2_xy_upper_pinned': int(out_ints[base_i + 23]),
                'status': status_label(status_code),
            }
            if gate_axis is not None:
                row.update({
                    'z_gate_status': _gate_status_text(
                        gate_tp['status'][gate_axis]),
                    'z_gate_fallback': _gate_status_text(
                        gate_tp['fallback'][gate_axis]),
                    'z_gate_min': _gate_bound_for_csv(
                        gate_tp['z_min'][gate_axis], nz),
                    'z_gate_max': _gate_bound_for_csv(
                        gate_tp['z_max'][gate_axis], nz),
                    'z_gate_quarantined': int(
                        gate_tp['quarantined'][gate_axis]),
                    'z_gate_evaluable': int(gate_tp['evaluable'][gate_axis]),
                })
            out_rows.append(row)

            # Keep a bounded gallery for QC plots.
            if args.max_gallery > 0 and suspect_flag:
                dbic = row['delta_BIC']
                rank = dbic if isinstance(dbic, float) and np.isfinite(dbic) else -1
                gallery.append((rank, tp_name, c, fits1.get((tp_name, c)),
                                None,  # params2 no longer needed in gallery
                                tp_bboxes[c].tolist(), bool(confirmed_flag)))
                if len(gallery) > args.max_gallery:
                    gallery.sort(key=lambda item: item[0], reverse=True)
                    del gallery[args.max_gallery:]

        # The physical/fit gates above define provisional calls. Pool their
        # candidate-matched null scores to estimate the false-discovery curve
        # directly, avoiding a separate extreme-tail p-value for every row.
        auto_rejected = 0
        if args.auto_null_bootstrap_reps > 0:
            tp_rows = out_rows[tp_row_start:]
            candidates = [row for row in tp_rows
                          if int(row['replication_confirmed']) == 1]
            result = (_auto_calibrate_candidates(
                img, tp_masks, tp_bboxes, candidates, cfg, args, tp_idx_0)
                if candidates else None)
            candidate_report = []
            for candidate_index, row in enumerate(candidates):
                q_value = result['q_values'][candidate_index]
                passed = bool(q_value <= args.auto_fdr_q)
                row['auto_null_tested'] = 1
                row['auto_calibration_pass'] = int(passed)
                row['auto_null_p'] = float(
                    result['marginal_p'][candidate_index])
                row['auto_null_q'] = float(q_value)
                row['auto_null_exceedances'] = int(
                    result['own_exceedances'][candidate_index])
                row['auto_pooled_null_exceedances'] = int(
                    result['pooled_exceedances'][candidate_index])
                row['auto_pooled_fdr'] = float(
                    result['raw_fdr'][candidate_index])
                row['auto_null_reps'] = int(result['replicates'])
                if not passed:
                    row['replication_confirmed'] = 0
                    row['n_spots_best'] = 1
                    row['status'] = 'tested_not_confirmed'
                    status_totals[4] -= 1
                    status_totals[3] += 1
                    auto_rejected += 1
                candidate_report.append({
                    'WalkID': int(row['WalkID']),
                    'observed_delta_BIC': float(row['delta_BIC']),
                    'marginal_p_value': float(
                        result['marginal_p'][candidate_index]),
                    'q_value': float(q_value),
                    'pooled_fdr_at_threshold': float(
                        result['raw_fdr'][candidate_index]),
                    'pooled_null_exceedances_at_threshold': int(
                        result['pooled_exceedances'][candidate_index]),
                    'passed': passed,
                    'candidate_null_exceedances': int(
                        result['own_exceedances'][candidate_index]),
                    'replicates': int(result['replicates']),
                    'detector_failures': int(
                        result['detector_failures'][candidate_index]),
                    'spatial_blocks': int(
                        result['n_blocks'][candidate_index]),
                })
            n_confirmed -= auto_rejected
            auto_reports.append({
                'tp': tp_name,
                'tp_index_0based': tp_idx_0,
                'in_call_range': bool(tp_in_call_range),
                'provisional_candidates': len(candidates),
                'requested_initial_replicates':
                    args.auto_null_bootstrap_reps,
                'maximum_replicates': args.auto_null_max_reps,
                'effective_replicates': (
                    result['replicates'] if result is not None else 0),
                'backend_batches': (
                    result['batches'] if result is not None else 0),
                'adaptive_stages': (
                    result['stages'] if result is not None else []),
                'retained': len(candidates) - auto_rejected,
                'rejected': auto_rejected,
                'candidates': candidate_report,
            })

        tp_dt = time.time() - tp_t0
        total_too_few_vox += n_too_few_vox
        total_no_fit1 += n_no_fit1
        total_gate_unevaluable += n_gate_unevaluable
        # ETA from mean per-TP time so far.
        elapsed_so_far = time.time() - t0
        mean_tp = elapsed_so_far / (tp_idx + 1)
        remaining = mean_tp * (n_use - (tp_idx + 1))
        dbic_mean = (dbic_suspect_sum / dbic_suspect_n) if dbic_suspect_n else float('nan')
        override_str = (f', two_lobe_overrides={n_overridden_two_lobe}'
                        if args.require_two_lobe_for_confirmation else '')
        out_of_range_str = (f', out_of_range_overrides={n_overridden_out_of_range}'
                            if not tp_in_call_range else '')
        xy_sep_str = (f', xy_sep_overrides={n_overridden_xy_sep}'
                      if cfg['min_spot_sep_xy_nm'] > 0 else '')
        auto_str = (f', auto_null_rejected={auto_rejected}'
                    if args.auto_null_bootstrap_reps > 0 else '')
        gate_str = (f', z_gate_unevaluable={n_gate_unevaluable}'
                    if z_support is not None else '')
        in_range_tag = '' if tp_in_call_range else '  [out-of-range]'
        print(f'[{tp_idx+1}/{n_use}] {tp_name}{in_range_tag}: '
              f'present={n_present}, suspect={n_suspect} '
              f'(xy={n_sigma_xy}, z*={n_sigma_z}, resid={n_resid}, lobe={n_two_lobe}), '
              f'stageB={n_stage_b_attempted} (fail={n_stage_b_failed}), '
              f'confirmed={n_confirmed}{override_str}{out_of_range_str}'
              f'{xy_sep_str}{auto_str}, '
              f'pinned_double_fit={n_pinned_double_fit}, '
              f'merged_mask={n_merged_mask_suspect}, '
              f'mean_dBIC={dbic_mean:.1f}, '
              f'skipped(no_fit1={n_no_fit1}, too_few_vox={n_too_few_vox}'
              f'{gate_str})  '
              f'({tp_dt:.1f}s, ETA {remaining/60:.1f}min)')

    elapsed = time.time() - t0
    print(f'Processed in {elapsed:.1f}s. Rows: {len(out_rows)}')
    # Global status breakdown across all TPs.
    print('Status totals across all TPs:')
    for code in sorted(status_totals):
        label = status_label(code)
        print(f'  {label:<30s} {status_totals[code]}')
    print(f'  {"skipped:no_fit1":<30s} {total_no_fit1}')
    print(f'  {"skipped:too_few_voxels":<30s} {total_too_few_vox}')
    if z_support is not None:
        print(f'  {"unevaluable:z_gate":<30s} {total_gate_unevaluable}')
    # Diagnostic flags (both separate from replication_confirmed):
    #   pinned_double_fit_suspect: broad — Stage B met thresholds but pinned.
    #   merged_mask_suspect:       strict — center pinning + oversized mask.
    n_pinned_rows = sum(int(r.get('pinned_double_fit_suspect', 0)) for r in out_rows)
    n_pinned_walks = len({int(r['WalkID']) for r in out_rows
                          if int(r.get('pinned_double_fit_suspect', 0)) == 1})
    n_merged_rows = sum(int(r.get('merged_mask_suspect', 0)) for r in out_rows)
    n_merged_walks = len({int(r['WalkID']) for r in out_rows
                          if int(r.get('merged_mask_suspect', 0)) == 1})
    print(f'Pinned double-fit suspects: {n_pinned_rows} (WalkID, tp) rows '
          f'across {n_pinned_walks} distinct WalkIDs '
          f'(broad diagnostic — does NOT set replication_confirmed=1).')
    print(f'Merged-mask suspects (strict, segmentation QC): '
          f'{n_merged_rows} (WalkID, tp) rows across {n_merged_walks} '
          f'distinct WalkIDs '
          f'(NOT a replication call — flags likely fused-cell masks).')

    # --- Write per-(WalkID, TP) CSV ---
    per_tp_csv = os.path.join(
        args.outdir, f'{args.output_prefix}_replication_qc.csv')
    fieldnames = [
        'schema_version', 'calibration_version',
        'tp', 'WalkID', 'n_voxels',
        'replication_suspect', 'replication_confirmed',
        'auto_null_tested', 'auto_calibration_pass',
        'auto_null_p', 'auto_null_q', 'auto_null_exceedances',
        'auto_pooled_null_exceedances', 'auto_pooled_fdr',
        'auto_null_reps',
        'pinned_double_fit_suspect', 'merged_mask_suspect',
        'in_call_range',
        'ownership_unevaluable',
        'ownership_failure_reason',
        'n_spots_best',
        'BIC_1', 'BIC_2', 'delta_BIC',
        'spot_sep_nm', 'spot_sep_xy_nm',
        'spot_sep_se_nm', 'spot_sep_lcb_nm',
        'spot_sep_xy_se_nm', 'spot_sep_xy_lcb_nm',
        'amp1', 'amp2', 'amp_ratio', 'amp_recovery',
        'peak_support_c1', 'peak_support_c2', 'valley_frac', 'resid_core_frac',
        'sigma_x', 'sigma_y', 'sigma_z', 'sigma_z_se',
        'effective_sigma_z_max',
        'm1_amplitude', 'm1_sigma_z', 'm1_sigma_y', 'm1_sigma_x',
        'm1_z', 'm1_y', 'm1_x', 'm1_background',
        'm2_c1_z', 'm2_c1_y', 'm2_c1_x',
        'm2_c2_z', 'm2_c2_y', 'm2_c2_x', 'm2_background',
        'resid_score', 'peak_rel_amp', 'peak_offset_px',
        'sse1', 'sse2',
        'sigma_xy_broad', 'sigma_z_broad', 'resid_broad',
        'two_lobe_suspect',
        'fit2_converged', 'fit1_rc', 'fit2_rc', 'fit2_nfev', 'fit2_n_starts',
        'fit2_pinned',
        'fit2_center_pinned', 'fit2_sigma_pinned',
        'fit2_center1_label', 'fit2_center2_label',
        'fit2_sigma_z_censored', 'fit2_axial_identifiable',
        'fit2_center1_supported', 'fit2_center2_supported',
        'fit2_lower_width_pinned',
        'fit2_xy_upper_pinned',
        'status',
    ]
    if z_support is not None:
        fieldnames[-1:-1] = [
            'z_gate_status', 'z_gate_fallback', 'z_gate_min', 'z_gate_max',
            'z_gate_quarantined', 'z_gate_evaluable',
        ]
    with open(per_tp_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print(f'Wrote {per_tp_csv}')

    if args.auto_null_bootstrap_reps > 0:
        auto_json = os.path.join(
            args.outdir, f'{args.output_prefix}_auto_calibration.json')
        auto_payload = {
            'schema': AUTO_CALIBRATION_SCHEMA,
            'schema_version': 1,
            'calibration_version': args.calibration_version,
            'calibration_type': 'model_based_null_calibration',
            'validation_claim': (
                'Controls candidate-family model-based FDR under the fitted '
                'one-spot spatial wild-bootstrap null; it is not biological '
                'ground-truth validation.'),
            'null_model': {
                'signal': 'refitted common-support M1',
                'resampling': 'Mammen spatial block wild bootstrap',
                'residual_centered_on_owned_support': True,
                'negative_intensity_policy': 'clip_to_zero',
                'block_z_slices': args.auto_null_block_z,
                'block_xy_pixels': args.auto_null_block_xy,
                'seed': args.auto_null_seed,
                'requested_initial_replicates_per_candidate':
                    args.auto_null_bootstrap_reps,
                'maximum_replicates_per_candidate': args.auto_null_max_reps,
                'adaptive_resolution': (
                    'extend only when an empty pooled null tail is blocked by '
                    'finite-simulation resolution'),
                'cross_candidate_batching': True,
            },
            'multiple_testing': {
                'method': 'pooled candidate-matched null FDR',
                'estimator': '(pooled_null_exceedances + 1) / '
                             '(replicates * observed_calls)',
                'scope': 'provisional candidates within each timepoint',
                'target_q': args.auto_fdr_q,
            },
            'model_policy': {
                **cfg,
                'segmentation_mode': args.segmentation_mode,
                'require_two_lobe_for_confirmation':
                    bool(args.require_two_lobe_for_confirmation),
                'pad_xy_pixels': args.pad_xy,
                'pad_z_slices': (0 if z_support is not None else args.pad_z),
                **({
                    'requested_pad_z_slices': args.pad_z,
                    'z_gate_failure_policy': args.z_gate_failure_policy,
                } if z_support is not None else {}),
                'min_n_voxels': 20,
                'max_fit_iterations': 200,
                'z_step_nm': Z_STEP_NM,
                'xy_pixel_nm': XY_PIXEL_NM,
            },
            'summary': {
                'timepoints': len(auto_reports),
                'provisional_candidates': sum(
                    report['provisional_candidates'] for report in auto_reports),
                'retained': sum(report['retained'] for report in auto_reports),
                'rejected': sum(report['rejected'] for report in auto_reports),
            },
            'timepoints': auto_reports,
        }
        with open(auto_json, 'w') as f:
            json.dump(auto_payload, f, indent=2, sort_keys=True)
            f.write('\n')
        print(f'Wrote {auto_json}')

    # --- Per-walk biological hierarchy summary ---
    # A confirmed two-spot is evidence that sister copies of one locus are
    # spatially resolved; it is not a direct measurement of DNA synthesis.
    # Missing/failed loci remain unevaluable rather than being converted into
    # negative one-spot evidence.
    summary_start_0 = args.replication_tp_start
    summary_end_0 = args.replication_tp_end
    summary_rows = [
        row for row in out_rows
        if summary_start_0 <= _tp_index_0based(row['tp']) <= summary_end_0
    ]
    if not summary_rows:
        raise ValueError(
            f'No replication-QC rows remain for walk-summary TP range '
            f'[{summary_start_0 + 1}, {summary_end_0 + 1}] 1-based.')
    print(f'Walk summary TP range: 1-based '
          f'[{summary_start_0 + 1}, {summary_end_0 + 1}] '
          f'({len(summary_rows)} / {len(out_rows)} rows)')

    rows_by_walk = defaultdict(list)
    all_rows_by_walk = defaultdict(list)
    for row in out_rows:
        wid = int(row['WalkID'])
        all_rows_by_walk[wid].append(row)
        tp_i = _tp_index_0based(row['tp'])
        if summary_start_0 <= tp_i <= summary_end_0:
            rows_by_walk[wid].append(row)

    evaluable_statuses = EVALUABLE_STATUSES
    walk_summaries = []
    state_counts = defaultdict(int)
    for wid in range(n_cells):
        rows = rows_by_walk.get(wid, [])
        # An ownership-rejected locus is unevaluable in BOTH directions: it
        # cannot confirm a two-spot and it is not evidence of a single spot.
        # Checked on the recorded column as well as the status string so the
        # two can never drift apart.
        def _own_rejected(row):
            return bool(int(float(row.get('ownership_unevaluable', 0) or 0)))

        confirmed_tps = {
            _tp_index_0based(row['tp']) for row in rows
            if int(row['replication_confirmed']) and not _own_rejected(row)
        }
        evaluated_tps = {
            _tp_index_0based(row['tp']) for row in rows
            if row['status'] in evaluable_statuses and not _own_rejected(row)
        }
        failed_tps = {
            _tp_index_0based(row['tp']) for row in rows
            if row['status'] not in evaluable_statuses or _own_rejected(row)
        }
        classification = classify_replication(
            confirmed_tps, evaluated_tps, failed_tps,
            arm_a, arm_b, args.ori_tp, args.ter_tp)
        classification['PinnedDoubleFitSuspect'] = int(any(
            int(row.get('pinned_double_fit_suspect', 0))
            for row in all_rows_by_walk.get(wid, [])))
        classification['MergedMaskSuspect'] = int(any(
            int(row.get('merged_mask_suspect', 0))
            for row in all_rows_by_walk.get(wid, [])))

        confirmed_names = sorted(confirmed_tps)
        finite_dbic = [
            float(row['delta_BIC']) for row in rows
            if row['delta_BIC'] != '' and np.isfinite(row['delta_BIC'])
        ]
        finite_sep = [
            float(row['spot_sep_nm']) for row in rows
            if row['spot_sep_nm'] != '' and np.isfinite(row['spot_sep_nm'])
        ]
        summary = {
            'WalkID': wid,
            # Deprecated compatibility alias: this is MultiSpot, not the
            # biological Replicating call.
            'spot_replicating': classification['MultiSpot'],
            'summary_tp_start_1based': summary_start_0 + 1,
            'summary_tp_end_1based': summary_end_0 + 1,
            'replication_tp_start': summary_start_0,
            'replication_tp_end': summary_end_0,
            'ori_tp': args.ori_tp,
            'ter_tp': args.ter_tp,
            **classification,
            'n_tps': len(rows),
            'n_suspect_tps': sum(
                int(row['replication_suspect']) for row in rows),
            'first_confirmed_tp': (
                f'TP{confirmed_names[0]:03d}' if confirmed_names else ''),
            'max_delta_BIC': max(finite_dbic) if finite_dbic else '',
            'max_spot_sep_nm': max(finite_sep) if finite_sep else '',
        }
        walk_summaries.append(summary)
        state_counts[classification['replication_state']] += 1

    per_walk_csv = os.path.join(
        args.outdir, f'{args.output_prefix}_walk_replication_summary.csv')
    summary_fields = [
        'WalkID', 'spot_replicating',
        'summary_tp_start_1based', 'summary_tp_end_1based',
        'replication_tp_start', 'replication_tp_end', 'ori_tp', 'ter_tp',
        *WALK_ANNOTATION_FIELDS,
        'n_tps', 'n_suspect_tps', 'first_confirmed_tp',
        'max_delta_BIC', 'max_spot_sep_nm',
    ]
    with open(per_walk_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(walk_summaries)
    print(f'Wrote {per_walk_csv}')
    print('Replication hierarchy states:')
    for state in sorted(state_counts):
        print(f'  {state:<30s} {state_counts[state]}')

    # --- QC figure (histograms + summary) ---
    if args.qc_png:
        _write_qc_figure(out_rows, args.outdir, args.output_prefix, cfg)

    return 0


def _write_qc_figure(rows, outdir, prefix, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    deltas = np.array([float(r['delta_BIC']) for r in rows
                       if r['delta_BIC'] != '' and np.isfinite(r['delta_BIC'])])
    seps = np.array([float(r['spot_sep_nm']) for r in rows
                     if r['spot_sep_nm'] != ''
                     and np.isfinite(r['spot_sep_nm'])])
    n_total = len(rows)
    n_suspect = sum(int(r['replication_suspect']) for r in rows)
    n_confirmed = sum(int(r['replication_confirmed']) for r in rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if deltas.size:
        axes[0].hist(deltas, bins=40, color='steelblue',
                     edgecolor='white', alpha=0.85)
    axes[0].axvline(cfg['delta_bic_thresh'], color='red', linestyle='--',
                    label=f'threshold = {cfg["delta_bic_thresh"]}')
    axes[0].set_xlabel('delta_BIC (BIC_1 - BIC_2, positive => 2-spot)')
    axes[0].set_ylabel('count')
    axes[0].set_title(f'delta_BIC over {deltas.size} Stage-B fits')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if seps.size:
        axes[1].hist(seps, bins=40, color='seagreen',
                     edgecolor='white', alpha=0.85)
    axes[1].axvline(cfg['min_spot_sep_nm'], color='red', linestyle='--',
                    label=f'threshold = {cfg["min_spot_sep_nm"]:.0f} nm')
    axes[1].set_xlabel('spot separation (nm)')
    axes[1].set_ylabel('count')
    axes[1].set_title('2-spot separation')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].axis('off')
    txt = (f'(WalkID, TP) rows: {n_total}\n'
           f'suspect (Stage A): {n_suspect} '
           f'({100*n_suspect/max(n_total,1):.1f}%)\n'
           f'confirmed (Stage A+B): {n_confirmed} '
           f'({100*n_confirmed/max(n_total,1):.1f}%)\n'
           f'\nthresholds:\n'
           f"  sigma_xy > {cfg['sigma_xy_broad']} px\n"
           f"  sigma_z  > {cfg['sigma_z_broad']} z-slices  (diagnostic only)\n"
           f"  resid_score > {cfg['resid_score_thresh']}\n"
           f"  lobe sep >= {cfg['min_lobe_sep_px']} px\n"
           f"  lobe rel amp >= {cfg['min_lobe_rel_amp']}\n"
           f"  delta_BIC >= {cfg['delta_bic_thresh']}\n"
           f"  spot sep >= {cfg['min_spot_sep_nm']:.0f} nm\n"
           f"  amp ratio >= {cfg['min_amp_ratio']}\n"
           + (f"  amp recovery >= {cfg['min_amp_recovery']}\n"
              if cfg['min_amp_recovery'] > 0 else
              "  amp recovery: DIAGNOSTIC ONLY (not gating)\n")
           + (f"  peak_support_c2 >= {cfg['min_peak_support_c2']}\n"
              if cfg['min_peak_support_c2'] > 0 else
              "  peak_support_c2: DIAGNOSTIC ONLY (not gating)\n")
           + (f"  valley_frac <= {cfg['max_valley_frac']}\n"
              if cfg['max_valley_frac'] > 0 else
              "  valley_frac: DIAGNOSTIC ONLY (not gating)\n")
           + (f"  resid_core_frac >= {cfg['min_resid_core_frac']}"
              if cfg['min_resid_core_frac'] > 0 else
              "  resid_core_frac: DIAGNOSTIC ONLY (not gating)"))
    axes[2].text(0.02, 0.97, txt, transform=axes[2].transAxes,
                 fontsize=10, va='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           alpha=0.85))
    axes[2].set_title('Summary + thresholds', fontweight='bold')

    fig.suptitle('Replication QC summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(outdir, f'{prefix}_replication_qc.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'Wrote {out_png}')


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _positive_int(s):
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError(f'must be >= 0, got {v}')
    return v


def _non_negative_int(s):
    """Parse a 0-based index; retained separately for truthful CLI semantics."""
    return _positive_int(s)


_CALIBRATION_KEYS = {
    'delta_bic_thresh', 'min_spot_sep_nm', 'min_amp_ratio',
    'min_amp_recovery', 'min_peak_support_c2', 'max_valley_frac',
    'min_resid_core_frac',
}


def _apply_calibration_config(args, parser):
    args.calibration_version = 'unvalidated-defaults'
    if not args.calibration_config:
        return args
    try:
        with open(args.calibration_config) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f'cannot load --calibration_config: {exc}')
    if payload.get('schema_version') != 1:
        parser.error('--calibration_config requires schema_version=1')
    if payload.get('validated') is not True:
        parser.error('--calibration_config is not marked validated=true')
    validation = payload.get('validation', {})
    try:
        fpr = float(validation['max_tp_false_positive_rate'])
    except (KeyError, TypeError, ValueError):
        parser.error('--calibration_config must record validation.max_tp_false_positive_rate')
    if not (math.isfinite(fpr) and 0.0 <= fpr <= 0.01):
        parser.error('--calibration_config exceeds the 1% TP false-positive target')
    version = str(payload.get('calibration_version', '')).strip()
    if not version:
        parser.error('--calibration_config requires calibration_version')
    thresholds = payload.get('thresholds')
    if not isinstance(thresholds, dict) or not thresholds:
        parser.error('--calibration_config requires a non-empty thresholds object')
    unknown = set(thresholds) - _CALIBRATION_KEYS
    if unknown:
        parser.error(f'unknown calibration threshold(s): {sorted(unknown)}')
    missing = _CALIBRATION_KEYS - set(thresholds)
    if missing:
        parser.error(f'missing calibration threshold(s): {sorted(missing)}')
    policy = payload.get('model_policy')
    if not isinstance(policy, dict):
        parser.error('--calibration_config requires model_policy')
    try:
        calibrated_sigma_z_max = float(policy['sigma_z_max'])
        calibrated_shell_nm = float(policy['center_shell_nm'])
        calibrated_z_step_nm = float(policy['z_step_nm'])
        calibrated_xy_pixel_nm = float(policy['xy_pixel_nm'])
        calibrated_min_sep_xy_nm = float(policy['min_spot_sep_xy_nm'])
    except (KeyError, TypeError, ValueError):
        parser.error('calibration model_policy requires numeric sigma_z_max, '
                     'center_shell_nm, z_step_nm, xy_pixel_nm, and '
                     'min_spot_sep_xy_nm')
    calibrated_require_two_lobe = policy.get('require_two_lobe_for_confirmation')
    if not isinstance(calibrated_require_two_lobe, bool):
        parser.error('calibration model_policy requires boolean '
                     'require_two_lobe_for_confirmation')
    numeric_policy = (
        calibrated_sigma_z_max, calibrated_shell_nm, calibrated_z_step_nm,
        calibrated_xy_pixel_nm, calibrated_min_sep_xy_nm,
    )
    if not all(math.isfinite(value) for value in numeric_policy):
        parser.error('calibration model_policy values must be finite')
    if (calibrated_sigma_z_max < 0 or calibrated_shell_nm < 0
            or calibrated_z_step_nm <= 0 or calibrated_xy_pixel_nm <= 0
            or calibrated_min_sep_xy_nm < 0):
        parser.error('calibration model_policy contains an invalid physical value')
    if abs(calibrated_sigma_z_max - args.sigma_z_max) > 1e-9:
        parser.error('calibration sigma_z_max does not match this run')
    if abs(calibrated_shell_nm - args.center_shell_nm) > 1e-9:
        parser.error('calibration center_shell_nm does not match this run')
    if abs(calibrated_z_step_nm - args.z_step_nm) > 1e-9:
        parser.error('calibration z_step_nm does not match this run')
    if abs(calibrated_xy_pixel_nm - args.xy_pixel_nm) > 1e-9:
        parser.error('calibration xy_pixel_nm does not match this run')
    if calibrated_require_two_lobe != bool(args.require_two_lobe_for_confirmation):
        parser.error('calibration two-lobe policy does not match this run')
    if abs(calibrated_min_sep_xy_nm - args.min_spot_sep_xy_nm) > 1e-9:
        parser.error('calibration min_spot_sep_xy_nm does not match this run')
    calibrated_thresholds = {}
    for key, value in thresholds.items():
        try:
            calibrated_thresholds[key] = float(value)
        except (TypeError, ValueError):
            parser.error(f'calibration threshold {key} must be numeric')
    if not all(math.isfinite(value) for value in calibrated_thresholds.values()):
        parser.error('calibration thresholds must be finite')
    if (calibrated_thresholds['min_spot_sep_nm'] < 0
            or not 0 <= calibrated_thresholds['min_amp_ratio'] <= 1
            or calibrated_thresholds['min_amp_recovery'] < 0
            or not 0 <= calibrated_thresholds['min_peak_support_c2'] <= 1
            or not 0 <= calibrated_thresholds['min_resid_core_frac'] <= 1):
        parser.error('calibration thresholds contain an invalid gate value')
    for key, value in calibrated_thresholds.items():
        setattr(args, key, value)
    args.calibration_version = version
    return args


def _one_based_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {v}')
    return v


def _parse_args():
    p = argparse.ArgumentParser(
        description='Replication QC: detect collapsed two-spot fits in the '
                    '640 channel using raw TP crops + a 2-spot model + BIC.')
    p.add_argument('--input_csv', required=True,
                   help='Existing localizations_640.csv (1-spot fits).')
    p.add_argument('--probe0_dir', required=True,
                   help='Directory containing the 640 TP TIFF stacks (or a Raw Images/ folder with --source dat).')
    p.add_argument('--source', choices=['tiff', 'dat'], default='tiff',
                   help='Image source: per-TP TIFFs (default) or raw Vutara .dat.')
    p.add_argument('--input_dir', default=None,
                   help='Override the image dir (a Raw Images/ folder when --source dat).')
    p.add_argument('--mask_path', required=True,
                   help='DAPI mask TIFF (same as the localizer used).')
    p.add_argument(
        '--segmentation-mode',
        choices=['stitched', '2p5d', '2p5d-zgated', '3d'],
        default='stitched',
        help='Segmentation support strategy. stitched, 2p5d, and 3d use the '
             'static mask volume; only 2p5d-zgated loads timepoint-specific '
             'axial support. (default: %(default)s)')
    p.add_argument(
        '--z-gate-path', default=None,
        help='2p5d-zgated NPZ sidecar. Defaults to z_support_2p5d.npz '
             'beside --mask_path.')
    p.add_argument(
        '--z-gate-failure-policy', choices=['quarantine', 'fail'],
        default='quarantine',
        help='Required failure policy recorded by the z-support sidecar. '
             '(default: %(default)s)')
    p.add_argument(
        '--chromatic-registration', '--chromatic_registration',
        dest='chromatic_registration', choices=['off', 'metadata'],
        default='off',
        help="Represent the DAPI (405) label mask on the 640 pixel grid using "
             "the acquisition's own widefield.json chromatic calibration, so "
             "the owned M1/M2 refit support and the centre-ownership test share "
             "the 640 frame. 'off' (default) preserves existing behaviour. "
             "'metadata' fails loudly if the calibration is absent or "
             "unsupported -- it never falls back to identity.")
    p.add_argument(
        '--chromatic-metadata', '--chromatic_metadata',
        dest='chromatic_metadata', default=None,
        help='Path to widefield.json. Default: alongside the image source dir.')
    p.add_argument(
        '--reference-wavelength-nm', '--reference_wavelength_nm',
        dest='reference_wavelength', default=None,
        help='Segmentation/reference channel wavelength (default 405).')
    p.add_argument(
        '--target-wavelength-nm', '--target_wavelength_nm',
        dest='target_wavelength', default=None,
        help='Probe target wavelength. widefield 640, confocal 561. Never '
             'inferred from the probe directory index.')
    p.add_argument(
        '--modality', choices=['widefield', 'confocal'], default='widefield',
        help='Selects the default reference/target wavelengths.')
    p.add_argument(
        '--center-ownership', '--center_ownership', dest='center_ownership',
        choices=['allow', 'strict'], default='allow',
        help="'strict' additionally requires BOTH two-spot component centres to "
             'lie in the same cell\'s own registered support. A component in '
             'background or another active label makes the locus UNEVALUABLE -- '
             'not a confirmed multispot and not negative evidence. Background '
             'shell pixels may still be used for background ESTIMATION; they are '
             'never legal spot-centre territory.')
    p.add_argument('--outdir', required=True,
                   help='Output directory for the QC CSV + summary.')
    p.add_argument('--output_prefix', default='localizations_640',
                   help='Prefix for the output files. Default: '
                        'localizations_640.')
    p.add_argument('--pad_xy', type=_positive_int, default=DEFAULT_PAD_XY)
    p.add_argument('--pad_z', type=_positive_int, default=DEFAULT_PAD_Z)
    p.add_argument('--sigma_z_max', type=float, default=0.0,
                   help='Replicate-specific upper sigma_z cap in slices. '
                        '0 applies no experimental cap and uses only the '
                        'per-cell crop depth as a numerical guard.')
    p.add_argument('--center_shell_nm', type=float, default=110.0,
                   help='Physical dilation around the owning Cellpose mask used '
                        'as common M1/M2 support. Other labelled cells remain '
                        'excluded. Default: %(default)s nm.')
    p.add_argument('--mask_dilation_xy_px', type=int, default=0,
                   help='Dilate the DAPI mask by N pixels in XY (in memory) '
                        'before bounding boxes and ownership. 0 = off.')
    p.add_argument('--calibration_config', default=None,
                   help='Optional legacy externally validated threshold JSON. '
                        'Automatic null calibration does not require this file.')
    p.add_argument('--auto_null_bootstrap_reps', type=_positive_int, default=0,
                   help='Initial spatial wild-bootstrap null replicates for '
                        'each provisional call. 0 disables automatic '
                        'calibration. Candidate nulls are pooled within each '
                        'TP, and more draws are used only when an empty null '
                        'tail is resolution-limited. '
                        '(default: %(default)s)')
    p.add_argument('--auto_null_max_reps', type=_one_based_int, default=128,
                   help='Maximum null replicates per candidate after bounded '
                        'adaptive extension. (default: %(default)s)')
    p.add_argument('--auto_fdr_q', type=float, default=0.01,
                   help='Pooled-null FDR target for provisional calls '
                        'within each TP. (default: %(default)s)')
    p.add_argument('--auto_null_seed', type=_positive_int, default=20260722,
                   help='Deterministic base seed for automatic null '
                        'calibration. (default: %(default)s)')
    p.add_argument('--auto_null_block_z', type=_one_based_int, default=2,
                   help='Spatial wild-bootstrap block depth in z slices. '
                        '(default: %(default)s)')
    p.add_argument('--auto_null_block_xy', type=_one_based_int, default=2,
                   help='Spatial wild-bootstrap block width in xy pixels. '
                        '(default: %(default)s)')
    p.add_argument('--auto_null_max_batch_voxels', type=_one_based_int,
                   default=5_000_000,
                   help='Memory guard for the number of synthetic crop voxels '
                        'sent to one C-backend batch. (default: %(default)s)')

    # Stage A
    p.add_argument('--sigma_xy_broad', type=float,
                   default=DEFAULT_SIGMA_XY_BROAD,
                   help='Stage A: flag if max(sigma_x, sigma_y) exceeds this '
                        'in pixel units. (default: %(default)s)')
    p.add_argument('--sigma_z_broad', type=float,
                   default=DEFAULT_SIGMA_Z_BROAD,
                   help='Stage A: sigma_z threshold in z-slice units. '
                        'NOTE: sigma_z_broad is REPORTED AS A DIAGNOSTIC '
                        'ONLY and no longer triggers Stage B on its own — '
                        'the physical z-PSF is broad enough that it would '
                        'flag nearly every cell. (default: %(default)s)')
    p.add_argument('--resid_score_thresh', type=float,
                   default=DEFAULT_RESID_SCORE_THRESH,
                   help='Stage A: flag if positive-residual sum / 1-spot '
                        'model sum exceeds this. Tightened 2026-04-23. '
                        '(default: %(default)s)')
    p.add_argument('--min_lobe_sep_px', type=float,
                   default=DEFAULT_MIN_LOBE_SEP_PX,
                   help='Stage A: minimum voxel separation between 1-spot '
                        'center and strongest positive residual peak. '
                        '(default: %(default)s)')
    p.add_argument('--min_lobe_rel_amp', type=float,
                   default=DEFAULT_MIN_LOBE_REL_AMP,
                   help='Stage A: minimum residual peak / 1-spot amplitude. '
                        'Tightened 2026-04-23. (default: %(default)s)')

    # Stage B + call
    p.add_argument('--delta_bic_thresh', type=float,
                   default=DEFAULT_DELTA_BIC_THRESH,
                   help='Stage B: minimum BIC_1 - BIC_2 to confirm '
                        'replication. Larger = stricter. '
                        'Default tightened 10 -> 15 on 2026-04-23. '
                        '(default: %(default)s)')
    p.add_argument('--min_spot_sep_nm', type=float,
                   default=DEFAULT_MIN_SPOT_SEP_NM,
                   help='Call: minimum 95%% lower confidence bound on spot '
                        'separation, in nm. Axially censored/unidentifiable '
                        'fits use the lateral confidence bound. '
                        'Default tightened 150 -> 250 on 2026-04-23. '
                        '(default: %(default)s)')
    p.add_argument('--min_amp_ratio', type=float,
                   default=DEFAULT_MIN_AMP_RATIO,
                   help='Call: minimum min(A1,A2)/max(A1,A2). Default '
                        'tightened 0.2 -> 0.3 on 2026-04-23. '
                        '(default: %(default)s)')
    p.add_argument('--min_spot_sep_xy_nm', type=float, default=0.0,
                   help='Optional confirm gate: require the 95%% lower '
                        'confidence bound on LATERAL spot separation (nm) to clear this, in '
                        'addition to the 3D --min_spot_sep_nm. The 3D '
                        'separation folds in the axial component, which is '
                        'unreliable on these thin (~10-14 slice) stacks where '
                        'the z fit is frequently unidentifiable. Gating on the '
                        'lateral component rejects 2-spot fits that separate '
                        'mostly in z. 0 disables (column still emitted). '
                        'Suggested value: ~250. (default: %(default)s)')
    p.add_argument('--min_amp_recovery', type=float,
                   default=DEFAULT_MIN_AMP_RECOVERY,
                   help='Optional confirm gate: require max(A1,A2)/amp_1spot '
                        '>= this. Real 2-spots split the original 1-spot '
                        'signal; a degenerate fit that converges on residual '
                        'lobes leaves both A1, A2 well below amp_1spot. '
                        '0 disables (metric still emitted to CSV for review). '
                        'Suggested value after inspection: ~0.5. '
                        '(default: %(default)s)')
    p.add_argument('--se_correction_factor', type=float, default=1.0,
                   help='Post-hoc SE scaling for the spot-separation LCB. '
                        'PSF voxel correlation makes the raw SE ~10x too small; '
                        'set to 10.0 to correct. The C backend runs with the '
                        'raw LCB; Python then rechecks with corrected SE. '
                        '1.0 disables. (default: %(default)s)')
    p.add_argument('--min_peak_support_c2', type=float,
                   default=DEFAULT_MIN_PEAK_SUPPORT_C2,
                   help='Optional confirm gate using the new raw-peak metric: '
                        'require the weaker fitted spot to sit on a real local '
                        'raw maximum, i.e. peak_support_c2 = (raw_at_c2 - bg) / '
                        '(raw_max_in_small_ball_at_c2 - bg) >= this. ~1 means '
                        'the center IS a peak; <1 means it sits on a slope. '
                        '0 disables (metric still emitted). Suggested value '
                        'after inspection: ~0.7. (default: %(default)s)')
    p.add_argument('--max_valley_frac', type=float,
                   default=DEFAULT_MAX_VALLEY_FRAC,
                   help='Optional confirm gate using the new valley metric: '
                        'sample raw intensity along the line between the two '
                        'fitted centers and require the minimum to be lower '
                        'than the centers, i.e. valley_frac = (raw_min_along_'
                        'line - bg) / (min(raw_c1, raw_c2) - bg) <= this. '
                        '<1 means a real saddle/valley exists between the '
                        'spots; >=1 means it is a shoulder/plateau. <=0 '
                        'disables (metric still emitted). Suggested value '
                        'after inspection: ~0.85-0.9. (default: %(default)s)')
    p.add_argument('--min_resid_core_frac', type=float,
                   default=DEFAULT_MIN_RESID_CORE_FRAC,
                   help='Optional confirm gate using the new residual '
                        'compactness metric: require the positive 1-spot '
                        'residual mass inside a small ball at the Stage A '
                        'residual peak to be at least this fraction of the '
                        'total positive residual. High = compact lobe; '
                        'low = diffuse shoulder smear. 0 disables (metric '
                        'still emitted). Suggested after inspection: '
                        '~0.25-0.35. (default: %(default)s)')
    # Default raised 1.5 -> 3.0 (2026-06).
    # The merged_mask_suspect size gate is `n_voxels >= factor * per-Location
    # median voxel count`. When the upstream clump-split step (split_clump_masks.py)
    # is run, it adds many small sub-cells, which DEFLATES the per-Location median
    # cell size. A 1.5x factor against that deflated median then sits around the
    # ~70-75th size percentile and over-fires on ordinary large cells (it flagged
    # ~50% of no-replicating walks), wrongly shrinking the no_rep_no_suspect set.
    # 3.0x median tracks ~the 90th percentile, so the gate flags only genuine
    # size outliers (residual merges the split missed) intersected with the
    # pinned-degenerate-fit condition. On split masks this recovered ~6,050 vs
    # ~3,615 clean walks. (For un-split/original masks 1.5x was also defensible;
    # 3.0x is the safe default now that clump-split is part of the pipeline.)
    p.add_argument('--merged_mask_size_factor', type=float, default=3.0,
                   help='Strict merged_mask_suspect rule: require the cell '
                        'mask 3D voxel count to be >= this multiple of the '
                        'per-Location median. Diagnostic only — never sets '
                        'replication_confirmed=1. Default raised 1.5->3.0 because '
                        'the clump-split step deflates the median (see code '
                        'comment). (default: %(default)s)')
    p.add_argument('--require_two_lobe_for_confirmation',
                   action='store_true',
                   help='Stricter call: a TP can only be '
                        'replication_confirmed=1 if the Stage A two-lobe '
                        'residual heuristic also fires (two_lobe_suspect=1). '
                        'OFF by default; useful when even one FP-confirmed '
                        'TP can drop a walk from topology analysis.')
    p.add_argument(
        '--replication_tp_start', type=_non_negative_int, default=None,
        help='First genomic TP used by the replication hierarchy, as a '
             '0-based TP filename index (inclusive). '
             f'Default: {REPLICATION_TP_START_DEFAULT} = '
             f'TP{REPLICATION_TP_START_DEFAULT:03d}.')
    p.add_argument(
        '--replication_tp_end', type=_non_negative_int, default=None,
        help='Last genomic TP used by the replication hierarchy, as a '
             '0-based TP filename index (inclusive). '
             f'Default: {REPLICATION_TP_END_DEFAULT} = '
             f'TP{REPLICATION_TP_END_DEFAULT:03d}.')
    p.add_argument(
        '--ori_tp', type=_non_negative_int, default=ORI_TP_DEFAULT,
        help='Ori genomic bin as a 0-based TP filename index. '
             f'Default: {ORI_TP_DEFAULT} = TP{ORI_TP_DEFAULT:03d}.')
    p.add_argument(
        '--ter_tp', type=_non_negative_int, default=TER_TP_DEFAULT,
        help='Ter genomic bin as a 0-based TP filename index. '
             f'Default: {TER_TP_DEFAULT} = TP{TER_TP_DEFAULT:03d}.')
    p.add_argument(
        '--summary_tp_start_1based', type=_one_based_int, default=None,
        help='Deprecated alias for --replication_tp_start, expressed as a '
             '1-based design timepoint.')
    p.add_argument(
        '--summary_tp_end_1based', type=_one_based_int, default=None,
        help='Deprecated alias for --replication_tp_end, expressed as a '
             '1-based design timepoint.')

    # Misc
    p.add_argument('--limit_tps', type=_positive_int, default=0,
                   help='If > 0, process only the first N TPs (for testing).')
    p.add_argument('--max_gallery', type=_positive_int, default=16,
                   help='Max number of suspect crops kept for a future '
                        'gallery figure (first-pass: histograms only).')
    p.add_argument('--qc_png', action='store_true',
                   help='Write a summary QC PNG next to the CSV.')

    # Pixel/z calibration (nm). Defaults = Vutara widefield; override for confocal.
    p.add_argument('--xy_pixel_nm', type=float, default=XY_PIXEL_NM,
                   help='nm per xy pixel (default %(default)s = Vutara widefield).')
    p.add_argument('--z_step_nm', type=float, default=Z_STEP_NM,
                   help='nm per z-slice (default %(default)s).')

    args = p.parse_args()
    legacy_start = (None if args.summary_tp_start_1based is None
                    else args.summary_tp_start_1based - 1)
    legacy_end = (None if args.summary_tp_end_1based is None
                  else args.summary_tp_end_1based - 1)
    if (args.replication_tp_start is not None and legacy_start is not None
            and args.replication_tp_start != legacy_start):
        p.error('--replication_tp_start conflicts with '
                '--summary_tp_start_1based')
    if (args.replication_tp_end is not None and legacy_end is not None
            and args.replication_tp_end != legacy_end):
        p.error('--replication_tp_end conflicts with '
                '--summary_tp_end_1based')
    args.replication_tp_start = (
        args.replication_tp_start
        if args.replication_tp_start is not None else
        legacy_start if legacy_start is not None else
        REPLICATION_TP_START_DEFAULT)
    args.replication_tp_end = (
        args.replication_tp_end
        if args.replication_tp_end is not None else
        legacy_end if legacy_end is not None else
        REPLICATION_TP_END_DEFAULT)
    if args.replication_tp_end < args.replication_tp_start:
        p.error('--replication_tp_end must be >= --replication_tp_start')
    for flag, value in (('--ori_tp', args.ori_tp), ('--ter_tp', args.ter_tp)):
        if not args.replication_tp_start <= value <= args.replication_tp_end:
            p.error(
                f'{flag} ({value}) must be within the inclusive replication '
                f'TP range [{args.replication_tp_start}, '
                f'{args.replication_tp_end}]')
    if args.ori_tp == args.ter_tp:
        p.error('--ori_tp and --ter_tp must be different')
    physical = (args.sigma_z_max, args.center_shell_nm,
                args.xy_pixel_nm, args.z_step_nm)
    if not all(math.isfinite(value) for value in physical):
        p.error('sigma/pixel/shell values must be finite')
    if args.sigma_z_max < 0:
        p.error('--sigma_z_max must be >= 0 (use 0 for uncapped)')
    if args.center_shell_nm < 0:
        p.error('--center_shell_nm must be >= 0')
    if args.xy_pixel_nm <= 0 or args.z_step_nm <= 0:
        p.error('--xy_pixel_nm and --z_step_nm must be > 0')
    if not math.isfinite(args.auto_fdr_q) or not 0 < args.auto_fdr_q <= 1:
        p.error('--auto_fdr_q must be finite and in (0, 1]')
    if (args.auto_null_bootstrap_reps > 0
            and args.auto_null_max_reps < args.auto_null_bootstrap_reps):
        p.error('--auto_null_max_reps must be >= --auto_null_bootstrap_reps')
    gates = (
        args.delta_bic_thresh, args.min_spot_sep_nm,
        args.min_spot_sep_xy_nm, args.min_amp_ratio,
        args.min_amp_recovery, args.min_peak_support_c2,
        args.max_valley_frac, args.min_resid_core_frac,
    )
    if not all(math.isfinite(value) for value in gates):
        p.error('replication-QC thresholds must be finite')
    if (args.min_spot_sep_nm < 0 or args.min_spot_sep_xy_nm < 0
            or not 0 <= args.min_amp_ratio <= 1
            or args.min_amp_recovery < 0
            or not 0 <= args.min_peak_support_c2 <= 1
            or not 0 <= args.min_resid_core_frac <= 1):
        p.error('replication-QC thresholds contain an invalid gate value')
    args = _apply_calibration_config(args, p)
    if args.auto_null_bootstrap_reps > 0:
        if args.calibration_config:
            args.calibration_version = (
                f'{args.calibration_version}+{AUTO_CALIBRATION_VERSION}')
        else:
            args.calibration_version = AUTO_CALIBRATION_VERSION
    return args


if __name__ == '__main__':
    args = _parse_args()
    XY_PIXEL_NM = args.xy_pixel_nm
    Z_STEP_NM = args.z_step_nm
    sys.exit(run(args))
