#!/usr/bin/env python3
"""Per-cell phase correlation XY drift estimation from 488 MIPs.

Replaces centroid-based XY drift estimation with rigid-body image
registration of the 488 rod within each cell's bounding box. Takes the
median across all cells to get field-wide XY drift per timepoint.
"""
import os

import numpy as np

from dat_io import iter_timepoints


def _hann_2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _phase_correlate_bbox(ref, mov, y0, y1, x0, x1):
    """Phase correlation of sub-images within a bounding box.

    Returns (dy, dx, peak_val) in pixel units.
    """
    sub_ref = ref[y0:y1, x0:x1].astype(np.float64)
    sub_mov = mov[y0:y1, x0:x1].astype(np.float64)

    if sub_ref.std() < 1e-6 or sub_mov.std() < 1e-6:
        return 0.0, 0.0, 0.0

    win = _hann_2d(sub_ref.shape)
    r = (sub_ref - sub_ref.mean()) * win
    m = (sub_mov - sub_mov.mean()) * win

    F = np.fft.rfft2(r)
    G = np.fft.rfft2(m)
    R = F * np.conj(G)
    denom = np.abs(R)
    denom[denom == 0] = 1
    R /= denom
    cc = np.fft.irfft2(R, s=r.shape)

    peak = np.unravel_index(np.argmax(cc), cc.shape)
    py, px = peak
    peak_val = float(cc[py, px])

    def _subpix(arr, idx, n):
        i0 = (idx - 1) % n
        i2 = (idx + 1) % n
        d = 2.0 * (2.0 * arr[idx] - arr[i0] - arr[i2])
        if abs(d) < 1e-12:
            return float(idx)
        return idx + (arr[i0] - arr[i2]) / d

    sy = _subpix(cc[:, px], py, cc.shape[0])
    sx = _subpix(cc[py, :], px, cc.shape[1])
    ny, nx = cc.shape
    if sy > ny / 2:
        sy -= ny
    if sx > nx / 2:
        sx -= nx
    return sy, sx, peak_val


def _cell_bboxes(mask, min_cell_px, pad):
    """Precompute padded bounding boxes for all qualifying cells."""
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]
    h, w = mask.shape
    bboxes = []
    for cid in cell_ids:
        ys, xs = np.where(mask == cid)
        if len(ys) < min_cell_px:
            continue
        y0 = max(0, ys.min() - pad)
        y1 = min(h, ys.max() + 1 + pad)
        x0 = max(0, xs.min() - pad)
        x1 = min(w, xs.max() + 1 + pad)
        if (y1 - y0) < 8 or (x1 - x0) < 8:
            continue
        bboxes.append((cid, y0, y1, x0, x1))
    return bboxes


def estimate_xy_drift(source_dir, mask_path, fmt='dat', probe_488=1,
                      min_cell_px=60, mad_k=3.0, pad=20):
    """Estimate per-TP XY drift via per-cell phase correlation of 488 MIPs.

    Parameters
    ----------
    source_dir : str
        Path to raw data (Raw Images/ for dat, probe_1/ for tiff).
    mask_path : str
        Path to masks_final.tif.
    fmt : str
        'dat' or 'tiff'.
    probe_488 : int
        Probe index for the 488 channel (default 1).
    min_cell_px : int
        Minimum cell size in pixels to include.
    mad_k : float
        MAD-based outlier rejection threshold for per-cell shifts.
    pad : int
        Padding around each cell's bounding box in pixels.

    Returns
    -------
    drift_xy : dict
        tp_name -> (dy_px, dx_px) drift relative to reference TP.
    stats : dict
        tp_name -> dict with 'n_kept', 'n_total', 'mad_sigma_y', 'mad_sigma_x'.
    ref_tp : str
        Name of the reference timepoint.
    """
    from PIL import Image
    mask = np.array(Image.open(mask_path))
    bboxes = _cell_bboxes(mask, min_cell_px, pad)
    n_cells = len(bboxes)
    print(f'[rod_drift] Cells qualifying (>= {min_cell_px} px): {n_cells}')

    tp_stacks = []
    for tp_name, stack in iter_timepoints(source_dir, probe_488, fmt):
        mip = stack.max(axis=0).astype(np.float64)
        tp_stacks.append((tp_name, mip))

    if not tp_stacks:
        raise RuntimeError("No timepoints loaded for 488 probe")

    ref_tp = tp_stacks[0][0]
    ref_mip = tp_stacks[0][1]
    print(f'[rod_drift] Reference TP: {ref_tp}, total TPs: {len(tp_stacks)}')

    drift_xy = {ref_tp: (0.0, 0.0)}
    stats = {ref_tp: {'n_kept': n_cells, 'n_total': n_cells,
                      'mad_sigma_y': 0.0, 'mad_sigma_x': 0.0}}

    for tp_name, mov_mip in tp_stacks:
        if tp_name == ref_tp:
            continue

        raw_shifts = []
        for cid, y0, y1, x0, x1 in bboxes:
            dy, dx, peak = _phase_correlate_bbox(ref_mip, mov_mip,
                                                  y0, y1, x0, x1)
            if peak > 0.01 and abs(dx) < 30 and abs(dy) < 30:
                raw_shifts.append((dy, dx))

        n_total = len(raw_shifts)

        if n_total >= 10:
            arr = np.array(raw_shifts)
            med = np.median(arr, axis=0)
            dists = np.sqrt(((arr - med) ** 2).sum(axis=1))
            mad_sigma = 1.4826 * np.median(dists)
            if mad_sigma > 0:
                keep = dists <= mad_k * mad_sigma
            else:
                keep = np.ones(len(dists), dtype=bool)
            kept = arr[keep]
        elif n_total > 0:
            kept = np.array(raw_shifts)
            mad_sigma = 0.0
        else:
            drift_xy[tp_name] = (0.0, 0.0)
            stats[tp_name] = {'n_kept': 0, 'n_total': 0,
                              'mad_sigma_y': 0.0, 'mad_sigma_x': 0.0}
            print(f'  {tp_name}: WARNING no valid cell shifts')
            continue

        med_dy = float(np.median(kept[:, 0]))
        med_dx = float(np.median(kept[:, 1]))
        mad_y = 1.4826 * float(np.median(np.abs(kept[:, 0] - med_dy)))
        mad_x = 1.4826 * float(np.median(np.abs(kept[:, 1] - med_dx)))

        drift_xy[tp_name] = (med_dy, med_dx)
        stats[tp_name] = {'n_kept': int(kept.shape[0]),
                          'n_total': n_total,
                          'mad_sigma_y': mad_y,
                          'mad_sigma_x': mad_x}

        print(f'  {tp_name}: dy={med_dy:+.3f} dx={med_dx:+.3f} px  '
              f'cells={kept.shape[0]}/{n_total}')

    return drift_xy, stats, ref_tp
