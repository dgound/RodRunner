#!/usr/bin/env python3
"""Stitch repaired per-plane masks and apply existing pipeline cleanup."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tifffile
from cellpose import utils

try:
    from .mask_stitching import filter_oversized_masks
except ImportError:  # direct script execution
    from mask_stitching import filter_oversized_masks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stitch repaired Cellpose masks and apply size filters."
    )
    parser.add_argument("--unstitched", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--stitch_threshold", type=float, default=0.5)
    parser.add_argument("--min_size", type=int, default=15)
    parser.add_argument("--max_diameter", type=float, default=40.0)
    parser.add_argument(
        "--min_z_extent", type=int, default=0,
        help="Drop labels spanning fewer than this many z-planes. 0 = off "
             "(rep1-3 behaviour). A ~1.5 um cell at 100 nm steps occupies ~15 "
             "planes, so a 1-2 plane label is not a cell. These matter beyond "
             "inflating counts: on Novo Loc-01, 5,091 of 7,951 labels spanned "
             "<=2 planes and 78% of them sat nested INSIDE a taller mask's XY "
             "footprint, where they trip the center-in-neighbour guard in "
             "gauss3d_fit.c and reject the genuine spot of the cell containing "
             "them. Symptom: tall masks fitting well per-TP (92%) yet almost "
             "never reaching full coverage (21%), because the rejection "
             "re-rolls independently at every TP. Set to 3 for Novo.",
    )
    args = parser.parse_args(argv)

    masks = tifffile.imread(args.unstitched)
    if masks.ndim != 3:
        raise ValueError(f"expected ZYX masks, got shape {masks.shape}")

    stitched = utils.stitch3D(
        masks.copy(), stitch_threshold=args.stitch_threshold
    )
    stitched = utils.fill_holes_and_remove_small_masks(
        stitched, min_size=args.min_size
    )
    n_stitched = int(stitched.max())
    n_thin = 0
    if args.min_z_extent > 0:
        # z-extent per label: count planes in which the label appears at all.
        present = np.zeros(int(stitched.max()) + 1, dtype=np.int32)
        for z in range(stitched.shape[0]):
            present[np.unique(stitched[z])] += 1
        present[0] = 0
        thin = np.nonzero(present[1:] < args.min_z_extent)[0] + 1
        n_thin = int(len(thin))
        if n_thin:
            drop = np.zeros(int(stitched.max()) + 1, dtype=bool)
            drop[thin] = True
            stitched[drop[stitched]] = 0
    cleaned, oversized = filter_oversized_masks(
        stitched, max_diameter=args.max_diameter
    )
    if cleaned.max() > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"final mask label {cleaned.max()} exceeds uint16 output capacity"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tifffile.imwrite(args.out, cleaned.astype(np.uint16))
    summary_json = args.summary_json or os.path.join(
        os.path.dirname(os.path.abspath(args.out)),
        "cross_z_finalize_summary.json",
    )
    summary = {
        "input": os.path.abspath(args.unstitched),
        "output": os.path.abspath(args.out),
        "stitched_labels_before_oversized_filter": n_stitched,
        "thin_labels_removed": n_thin,
        "oversized_labels_removed": int(len(oversized)),
        "final_labels": int(cleaned.max()),
        "parameters": {
            "stitch_threshold": args.stitch_threshold,
            "min_size": args.min_size,
            "max_diameter": args.max_diameter,
            "min_z_extent": args.min_z_extent,
        },
    }
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[cross-z] stitched labels={n_stitched}; "
        f"removed thin(<{args.min_z_extent}z)={n_thin}; "
        f"removed oversized={len(oversized)}; final={int(cleaned.max())}",
        flush=True,
    )
    print(f"[cross-z] final masks: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
