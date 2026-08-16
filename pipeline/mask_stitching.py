"""Cellpose-compatible helpers for stitching and cleaning 2D mask stacks.

The detector needs a fixed baseline stitching before it can identify broad
current-plane masks. These functions intentionally mirror Cellpose 4.1.1's
``utils.stitch3D`` label assignment so detection and finalization use identical
track identities.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix


def label_overlap(masks_true, masks_pred):
    """Return the pixel-overlap matrix, including background label zero."""
    if masks_true.shape != masks_pred.shape:
        raise ValueError(
            "mask planes must have identical shapes, got "
            f"{masks_true.shape} and {masks_pred.shape}"
        )
    return csr_matrix(
        (
            np.ones(masks_true.size, dtype=np.int64),
            (masks_true.ravel(), masks_pred.ravel()),
        ),
        shape=(int(masks_true.max()) + 1, int(masks_pred.max()) + 1),
    ).toarray()


def intersection_over_union(masks_true, masks_pred):
    """Return the all-label IoU matrix used by Cellpose stitching."""
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = overlap.sum(axis=0, keepdims=True)
    n_pixels_true = overlap.sum(axis=1, keepdims=True)
    denominator = n_pixels_pred + n_pixels_true - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = overlap / denominator
    iou[np.isnan(iou)] = 0.0
    return iou


def stitch_3d(masks, stitch_threshold=0.5, copy=True):
    """Stitch a ZYX stack exactly as Cellpose 4.1.1 ``utils.stitch3D``.

    Labels within each input plane are local. The returned labels are global
    tracks propagated across adjacent Z planes at the requested IoU threshold.
    """
    if masks.ndim != 3:
        raise ValueError(f"expected a ZYX mask stack, got shape {masks.shape}")
    if not 0 < stitch_threshold <= 1:
        raise ValueError("stitch_threshold must be in (0, 1]")

    stitched = masks.copy() if copy else masks
    if stitched.shape[0] <= 1:
        return stitched

    mmax = int(stitched[0].max())
    empty = 0
    for z in range(stitched.shape[0] - 1):
        iou = intersection_over_union(stitched[z + 1], stitched[z])[1:, 1:]
        if not iou.size and empty == 0:
            mmax = int(stitched[z + 1].max())
        elif not iou.size and empty != 0:
            current_count = int(stitched[z + 1].max())
            assignment = np.arange(
                mmax + 1,
                mmax + current_count + 1,
                dtype=stitched.dtype,
            )
            mmax += current_count
            assignment = np.concatenate(
                (np.zeros(1, dtype=stitched.dtype), assignment)
            )
            stitched[z + 1] = assignment[stitched[z + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            assignment = iou.argmax(axis=1) + 1
            unmatched = np.nonzero(iou.max(axis=1) == 0.0)[0]
            assignment[unmatched] = np.arange(
                mmax + 1,
                mmax + len(unmatched) + 1,
                dtype=stitched.dtype,
            )
            mmax += len(unmatched)
            assignment = np.concatenate(
                (np.zeros(1, dtype=assignment.dtype), assignment)
            )
            stitched[z + 1] = assignment[stitched[z + 1]]
            empty = 1
    return stitched


def renumber_labels(masks):
    """Renumber positive labels consecutively in ascending label order."""
    present = np.unique(masks)
    present = present[present > 0]
    if not present.size:
        return np.zeros_like(masks)
    lookup = np.zeros(int(masks.max()) + 1, dtype=masks.dtype)
    lookup[present] = np.arange(1, len(present) + 1, dtype=masks.dtype)
    return lookup[masks]


def filter_oversized_masks(masks, max_diameter=40.0):
    """Remove labels whose largest Z-plane area exceeds a diameter cutoff."""
    if masks.ndim != 3:
        raise ValueError(f"expected a ZYX mask stack, got shape {masks.shape}")
    max_label = int(masks.max())
    max_areas = np.zeros(max_label + 1, dtype=np.int64)
    for plane in masks:
        counts = np.bincount(plane.ravel(), minlength=max_label + 1)
        max_areas = np.maximum(max_areas, counts)
    area_threshold = math.pi * (max_diameter / 2.0) ** 2
    oversized = np.flatnonzero(
        (max_areas > area_threshold)
        & (np.arange(max_label + 1) > 0)
    )
    cleaned = masks.copy()
    if oversized.size:
        cleaned[np.isin(cleaned, oversized)] = 0
    return renumber_labels(cleaned), oversized


def broad_overlap_candidates(unstitched, stitched, min_iou=0.05):
    """Find current-plane masks overlapping two or more predecessor tracks.

    Detection uses a fixed, already-stitched baseline volume. This is the
    validated one-pass behavior: repairing one transition does not recursively
    create additional candidates in later planes during the same run.
    """
    if unstitched.shape != stitched.shape or unstitched.ndim != 3:
        raise ValueError(
            "unstitched and stitched masks must be same-shaped ZYX stacks"
        )
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0, 1]")

    candidates = []
    for z in range(1, unstitched.shape[0]):
        overlap = label_overlap(unstitched[z], stitched[z - 1])
        current_areas = overlap.sum(axis=1, keepdims=True)
        previous_areas = overlap.sum(axis=0, keepdims=True)
        denominator = current_areas + previous_areas - overlap
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = overlap / denominator
        iou[np.isnan(iou)] = 0.0

        for current_label in range(1, iou.shape[0]):
            previous_labels = np.flatnonzero(
                iou[current_label, 1:] >= min_iou
            ) + 1
            if len(previous_labels) < 2:
                continue
            candidates.append(
                {
                    "z": z,
                    "current_source_label": current_label,
                    "previous_labels": previous_labels.astype(int).tolist(),
                    "intersection_px": overlap[
                        current_label, previous_labels
                    ].astype(int).tolist(),
                    "ious": iou[
                        current_label, previous_labels
                    ].astype(float).tolist(),
                }
            )
    return candidates
