#!/usr/bin/env python3
"""Repair per-plane Cellpose merges using predecessor masks across Z.

Cellpose first segments each Z plane independently. A current-plane mask is a
repair candidate when it overlaps at least two correctly separated predecessor
tracks. Those predecessor intersections become markers for a watershed on the
current DAPI plane. Only label boundaries change; the Cellpose foreground
footprint is preserved exactly.

The marker flood is implemented with NumPy/SciPy so this stage runs in the
primary Cellpose environment. It writes repaired *per-plane* masks. Stitching,
Cellpose's small-mask/hole cleanup, and the pipeline's oversized-mask filter are
performed afterward by ``finalize_cellpose_masks.py``.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    from .mask_stitching import broad_overlap_candidates, stitch_3d
except ImportError:  # direct script execution
    from mask_stitching import broad_overlap_candidates, stitch_3d


def _write_tiff(path, array):
    """Write compressed ZYX uint16 TIFF with old/new tifffile compatibility."""
    if array.max() > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"mask label {array.max()} exceeds uint16 output capacity"
        )
    array = array.astype(np.uint16, copy=False)
    if hasattr(tifffile.TiffWriter, "save"):
        tifffile.imwrite(path, array, compress=6, metadata={"axes": "ZYX"})
    else:
        tifffile.imwrite(
            path, array, compression="zlib", metadata={"axes": "ZYX"}
        )


def marker_watershed(image, markers, mask):
    """Marker-controlled 4-connected watershed for the cross-Z repair.

    Pixels are queued by elevation and then by insertion time, matching the
    scikit-image watershed behavior used in the validated diagnostic. Markers
    are full predecessor-overlap regions, so the inherited topology is fixed
    before ambiguous pixels are flooded.
    """
    if image.shape != markers.shape or image.shape != mask.shape:
        raise ValueError("image, markers, and mask must have identical shapes")
    height, width = image.shape
    output = markers.copy()
    queue = []
    age = 0
    for flat_index in np.flatnonzero(markers):
        heapq.heappush(
            queue,
            (
                float(image.flat[flat_index]),
                age,
                int(flat_index),
                int(markers.flat[flat_index]),
            ),
        )

    while queue:
        _, _, flat_index, label = heapq.heappop(queue)
        row, col = divmod(flat_index, width)
        neighbors = (
            flat_index - width if row > 0 else -1,
            flat_index - 1 if col > 0 else -1,
            flat_index + 1 if col + 1 < width else -1,
            flat_index + width if row + 1 < height else -1,
        )
        for neighbor in neighbors:
            if (
                neighbor < 0
                or not mask.flat[neighbor]
                or output.flat[neighbor] != 0
            ):
                continue
            output.flat[neighbor] = label
            age += 1
            heapq.heappush(
                queue,
                (
                    float(image.flat[neighbor]),
                    age,
                    int(neighbor),
                    label,
                ),
            )
    return output


def repair_cross_z(
    unstitched,
    image,
    *,
    stitch_threshold=0.5,
    candidate_iou=0.05,
    watershed_sigma=1.0,
):
    """Return repaired local labels, audit rows, and the baseline stitching."""
    if unstitched.ndim != 3:
        raise ValueError(
            f"unstitched masks must be ZYX, got shape {unstitched.shape}"
        )
    if image.shape != unstitched.shape:
        raise ValueError(
            f"image shape {image.shape} does not match masks "
            f"{unstitched.shape}"
        )

    baseline_stitched = stitch_3d(
        unstitched, stitch_threshold=stitch_threshold, copy=True
    )
    candidates = broad_overlap_candidates(
        unstitched, baseline_stitched, min_iou=candidate_iou
    )
    repaired = unstitched.copy()
    next_labels = [int(plane.max()) + 1 for plane in repaired]
    audit_rows = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        z = candidate["z"]
        source_label = candidate["current_source_label"]
        previous_labels = candidate["previous_labels"]
        footprint = unstitched[z] == source_label
        yy, xx = np.nonzero(footprint)
        if not yy.size:
            raise RuntimeError(
                f"candidate source label {source_label} is absent at z={z}"
            )

        margin = 4
        y0 = max(0, int(yy.min()) - margin)
        y1 = min(unstitched.shape[1], int(yy.max()) + margin + 1)
        x0 = max(0, int(xx.min()) - margin)
        x1 = min(unstitched.shape[2], int(xx.max()) + margin + 1)
        local_footprint = footprint[y0:y1, x0:x1]
        markers = np.zeros(local_footprint.shape, dtype=np.int32)

        marker_areas = []
        for marker_id, previous_label in enumerate(previous_labels, start=1):
            seed = (
                baseline_stitched[z - 1, y0:y1, x0:x1] == previous_label
            ) & local_footprint
            if not seed.any():
                raise RuntimeError(
                    f"candidate {candidate_index} at z={z} has no marker "
                    f"pixels for predecessor {previous_label}"
                )
            markers[seed] = marker_id
            marker_areas.append(int(seed.sum()))

        smooth = gaussian_filter(
            image[z, y0:y1, x0:x1],
            sigma=watershed_sigma,
            mode="nearest",
        )
        partition = marker_watershed(
            -smooth, markers=markers, mask=local_footprint
        )
        # The flood is 4-connected, so footprint pixels that touch the flooded
        # body only diagonally stay unlabelled. Left alone they are zeroed with
        # the footprint below and never rewritten, which shrinks the Cellpose
        # foreground and trips the support check. Attach each unlabelled pixel
        # to the nearest labelled partition instead.
        residue = local_footprint & (partition == 0)
        residue_px = int(residue.sum())
        if residue_px:
            _, (nearest_y, nearest_x) = distance_transform_edt(
                partition == 0, return_indices=True
            )
            partition[residue] = partition[
                nearest_y[residue], nearest_x[residue]
            ]
        expected_parts = len(previous_labels)
        present_parts = np.unique(partition[local_footprint])
        present_parts = present_parts[present_parts > 0]
        if len(present_parts) != expected_parts:
            raise RuntimeError(
                f"candidate {candidate_index} produced {len(present_parts)} "
                f"of {expected_parts} requested partitions"
            )

        repaired[z][footprint] = 0
        assigned_labels = []
        partition_areas = []
        for marker_id in range(1, expected_parts + 1):
            output_label = (
                source_label if marker_id == 1 else next_labels[z]
            )
            if marker_id > 1:
                next_labels[z] += 1
            target = partition == marker_id
            repaired[z, y0:y1, x0:x1][target] = output_label
            assigned_labels.append(int(output_label))
            partition_areas.append(int(target.sum()))

        audit_rows.append(
            {
                "candidate_index": candidate_index,
                "z": z,
                "current_source_label": source_label,
                "previous_labels": ";".join(map(str, previous_labels)),
                "previous_ious": ";".join(
                    f"{value:.9f}" for value in candidate["ious"]
                ),
                "marker_areas_px": ";".join(map(str, marker_areas)),
                "assigned_plane_labels": ";".join(map(str, assigned_labels)),
                "partition_areas_px": ";".join(map(str, partition_areas)),
                "unlabeled_residue_px": residue_px,
            }
        )

    if not np.array_equal(unstitched > 0, repaired > 0):
        raise RuntimeError("cross-Z repair changed Cellpose foreground support")
    return repaired, audit_rows, baseline_stitched


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Repair Cellpose plane merges from predecessor Z masks."
    )
    parser.add_argument("--unstitched", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--stitch_threshold", type=float, default=0.5)
    parser.add_argument("--candidate_iou", type=float, default=0.05)
    parser.add_argument("--watershed_sigma", type=float, default=1.0)
    args = parser.parse_args(argv)

    unstitched = tifffile.imread(args.unstitched)
    image = tifffile.imread(args.image).astype(np.float32)
    repaired, audit_rows, baseline_stitched = repair_cross_z(
        unstitched,
        image,
        stitch_threshold=args.stitch_threshold,
        candidate_iou=args.candidate_iou,
        watershed_sigma=args.watershed_sigma,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _write_tiff(args.out, repaired)

    output_dir = os.path.dirname(os.path.abspath(args.out))
    audit_csv = args.audit_csv or os.path.join(
        output_dir, "cross_z_repair_crosswalk.csv"
    )
    summary_json = args.summary_json or os.path.join(
        output_dir, "cross_z_repair_summary.json"
    )
    fields = [
        "candidate_index",
        "z",
        "current_source_label",
        "previous_labels",
        "previous_ious",
        "marker_areas_px",
        "assigned_plane_labels",
        "partition_areas_px",
        "unlabeled_residue_px",
    ]
    with open(audit_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    baseline_instances = int(
        sum(np.unique(plane).size - (1 if 0 in plane else 0)
            for plane in unstitched)
    )
    repaired_instances = int(
        sum(np.unique(plane).size - (1 if 0 in plane else 0)
            for plane in repaired)
    )
    summary = {
        "algorithm": "one-pass predecessor-guided cross-Z watershed",
        "candidate_count": len(audit_rows),
        "unstitched_instances_before": baseline_instances,
        "unstitched_instances_after": repaired_instances,
        "partitions_added": repaired_instances - baseline_instances,
        "foreground_support_identical": True,
        "candidates_with_unlabeled_residue": int(sum(
            1 for row in audit_rows if row["unlabeled_residue_px"])),
        "unlabeled_residue_px_total": int(sum(
            row["unlabeled_residue_px"] for row in audit_rows)),
        "baseline_stitched_label_count": int(baseline_stitched.max()),
        "parameters": {
            "stitch_threshold": args.stitch_threshold,
            "candidate_iou": args.candidate_iou,
            "watershed_sigma": args.watershed_sigma,
        },
        "outputs": {
            "repaired_unstitched": os.path.abspath(args.out),
            "audit_csv": os.path.abspath(audit_csv),
        },
    }
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "[cross-z] "
        f"{len(audit_rows)} broad masks repaired; "
        f"{baseline_instances} -> {repaired_instances} per-plane instances; "
        "foreground unchanged",
        flush=True,
    )
    print(f"[cross-z] repaired masks: {args.out}", flush=True)
    print(f"[cross-z] audit: {audit_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
