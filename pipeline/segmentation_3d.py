#!/usr/bin/env python3
"""Direct volumetric Cellpose segmentation for averaged DAPI stacks.

This mode asks Cellpose to infer one ``(z, y, x)`` instance volume with
``do_3D=True``.  It does not use the per-plane stitch/cross-Z-repair path and
does not project or extrude labels as the 2.5D modes do.
"""

import numpy as np


def resolve_anisotropy(xy_pixel_nm, z_step_nm, override=None):
    """Return Cellpose's z/xy sampling ratio with strict validation."""
    xy_pixel_nm = float(xy_pixel_nm)
    z_step_nm = float(z_step_nm)
    if not np.isfinite(xy_pixel_nm) or xy_pixel_nm <= 0:
        raise ValueError(
            f"xy_pixel_nm must be finite and positive, got {xy_pixel_nm}"
        )
    if not np.isfinite(z_step_nm) or z_step_nm <= 0:
        raise ValueError(
            f"z_step_nm must be finite and positive, got {z_step_nm}"
        )
    anisotropy = z_step_nm / xy_pixel_nm if override is None else float(override)
    if not np.isfinite(anisotropy) or anisotropy <= 0:
        raise ValueError(
            f"Cellpose 3D anisotropy must be finite and positive, got {anisotropy}"
        )
    return anisotropy


def filter_and_renumber_3d(labels, max_diameter=40.0):
    """Apply the established maximum-XY-diameter filter to a 3D label map.

    The stitched pipeline defines an oversized cell by its largest
    cross-sectional area in any z plane.  Keeping that definition here makes
    the post-Cellpose size policy comparable without flattening the 3D mask.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"expected a 3D label image, got shape {labels.shape}")
    if np.any(labels < 0):
        raise ValueError("label images cannot contain negative values")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"label image must be integer-valued, got {labels.dtype}")

    work = labels.astype(np.int64, copy=True)
    present_before = np.unique(work)
    present_before = present_before[present_before > 0]
    max_label = int(work.max()) if work.size else 0
    max_slice_areas = np.zeros(max_label + 1, dtype=np.int64)
    for plane in work:
        counts = np.bincount(plane.ravel(), minlength=max_label + 1)
        max_slice_areas = np.maximum(max_slice_areas, counts)

    max_area = np.pi * (float(max_diameter) / 2.0) ** 2
    oversized = np.flatnonzero(max_slice_areas > max_area)
    oversized = oversized[oversized > 0]
    if oversized.size:
        work[np.isin(work, oversized)] = 0

    present = np.unique(work)
    present = present[present > 0]
    if present.size > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"3D label count {present.size} exceeds uint16 capacity"
        )
    lookup = np.zeros(int(work.max()) + 1, dtype=np.uint16)
    lookup[present] = np.arange(1, present.size + 1, dtype=np.uint16)
    clean = lookup[work]
    stats = {
        "labels_before_oversized_filter": int(present_before.size),
        "oversized_labels_removed": int(oversized.size),
        "final_labels": int(present.size),
    }
    return clean, stats


def segment_3d(
    model,
    averaged_dapi,
    *,
    xy_pixel_nm,
    z_step_nm,
    anisotropy=None,
    diameter=20.0,
    min_size=15,
    max_diameter=40.0,
):
    """Return ``(raw_labels, final_labels, summary)`` for direct 3D inference."""
    averaged_dapi = np.asarray(averaged_dapi)
    if averaged_dapi.ndim != 3:
        raise ValueError(
            "3D segmentation requires a (z, y, x) averaged DAPI stack; "
            f"got shape {averaged_dapi.shape}"
        )
    resolved_anisotropy = resolve_anisotropy(
        xy_pixel_nm, z_step_nm, override=anisotropy
    )
    raw_labels, _, _ = model.eval(
        averaged_dapi,
        diameter=float(diameter),
        z_axis=0,
        do_3D=True,
        anisotropy=resolved_anisotropy,
        min_size=int(min_size),
    )
    raw_labels = np.asarray(raw_labels)
    if raw_labels.shape != averaged_dapi.shape:
        raise ValueError(
            f"Cellpose 3D returned shape {raw_labels.shape}; expected "
            f"{averaged_dapi.shape}"
        )
    final_labels, stats = filter_and_renumber_3d(
        raw_labels, max_diameter=max_diameter
    )
    summary = {
        "segmentation_mode": "3d",
        "inference": "CellposeModel.eval(do_3D=True)",
        "z_policy": "direct_volumetric_instances",
        "diameter_px": float(diameter),
        "min_size_voxels": int(min_size),
        "max_cross_section_diameter_px": float(max_diameter),
        "xy_pixel_nm": float(xy_pixel_nm),
        "z_step_nm": float(z_step_nm),
        "anisotropy_z_over_xy": float(resolved_anisotropy),
        "input_shape_zyx": [int(v) for v in averaged_dapi.shape],
        **stats,
    }
    return raw_labels, final_labels, summary
