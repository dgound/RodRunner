#!/usr/bin/env python3
"""Direct 2D DAPI-projection segmentation with full-z mask extrusion.

The downstream localizers require a label volume with the same ``(z, y, x)``
shape as every image stack.  In 2.5D mode, one Cellpose instance map is made
from the z-maximum projection and repeated unchanged through z.  This keeps XY
ownership fixed while allowing the existing 3D fitters to inspect every plane.
"""

import numpy as np


def z_max_projection(averaged_dapi):
    averaged_dapi = np.asarray(averaged_dapi)
    if averaged_dapi.ndim != 3:
        raise ValueError(
            "2.5D segmentation requires a 3D (z, y, x) averaged DAPI stack; "
            f"got shape {averaged_dapi.shape}"
        )
    return averaged_dapi.max(axis=0)


def filter_and_renumber_2d(labels, max_diameter=40.0):
    """Remove oversized 2D instances and return contiguous uint16 labels."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label image, got shape {labels.shape}")
    if np.any(labels < 0):
        raise ValueError("label images cannot contain negative values")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"label image must be integer-valued, got {labels.dtype}")

    work = labels.astype(np.int64, copy=True)
    counts = np.bincount(work.ravel())
    labels_before = np.flatnonzero(counts)
    labels_before = labels_before[labels_before > 0]
    max_area = np.pi * (float(max_diameter) / 2.0) ** 2
    oversized = np.flatnonzero(counts > max_area)
    oversized = oversized[oversized > 0]
    if oversized.size:
        work[np.isin(work, oversized)] = 0

    present = np.unique(work)
    present = present[present > 0]
    if present.size > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"2.5D label count {present.size} exceeds uint16 capacity"
        )
    lookup = np.zeros(int(work.max()) + 1, dtype=np.uint16)
    lookup[present] = np.arange(1, present.size + 1, dtype=np.uint16)
    clean = lookup[work]
    stats = {
        "labels_before_oversized_filter": int(labels_before.size),
        "oversized_labels_removed": int(oversized.size),
        "final_labels": int(present.size),
    }
    return clean, stats


def extrude_labels_through_z(labels_2d, z_planes):
    labels_2d = np.asarray(labels_2d)
    if labels_2d.ndim != 2:
        raise ValueError(f"expected a 2D label image, got shape {labels_2d.shape}")
    if int(z_planes) < 1:
        raise ValueError(f"z_planes must be positive, got {z_planes}")
    return np.broadcast_to(
        labels_2d[None, :, :],
        (int(z_planes),) + labels_2d.shape,
    ).copy()


def segment_2p5d(
    model,
    averaged_dapi,
    diameter=20.0,
    min_size=15,
    max_diameter=40.0,
):
    """Return ``(projection, labels_2d, labels_3d, summary)``."""
    projection = z_max_projection(averaged_dapi)
    raw_labels, _, _ = model.eval(
        projection,
        diameter=float(diameter),
        channels=[0, 0],
        do_3D=False,
        min_size=int(min_size),
    )
    labels_2d, stats = filter_and_renumber_2d(
        raw_labels,
        max_diameter=max_diameter,
    )
    labels_3d = extrude_labels_through_z(labels_2d, averaged_dapi.shape[0])
    summary = {
        "segmentation_mode": "2p5d",
        "projection": "z_max",
        "z_policy": "full_stack_extrusion",
        "diameter_px": float(diameter),
        "min_size_px": int(min_size),
        "max_diameter_px": float(max_diameter),
        "input_shape_zyx": [int(v) for v in averaged_dapi.shape],
        "projection_shape_yx": [int(v) for v in projection.shape],
        **stats,
    }
    return projection, labels_2d, labels_3d, summary
