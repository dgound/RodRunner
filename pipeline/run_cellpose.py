#!/usr/bin/env python
"""Run cellpose on averaged DAPI from sequential FISH (oligoSTORM).

All TPs are the same FOV — different probes targeting 250kb bins of the
genome. DAPI is re-acquired each cycle, so we average a few
TPs for better SNR and segment once.
"""

from cellpose.models import CellposeModel
from cellpose import utils as cellpose_utils
import tifffile
import numpy as np
import os
import glob
import argparse
import json
import dat_io
from repair_cross_z_masks import main as repair_cross_z_main
from finalize_cellpose_masks import main as finalize_cellpose_main
from segmentation_3d import segment_3d
from segmentation_2p5d import segment_2p5d
from z_support_2p5d import compute_z_support, save_z_support

_ap = argparse.ArgumentParser(description='Segment averaged DAPI (probe_2).')
_ap.add_argument('--source', choices=['tiff', 'dat'], default='tiff',
                 help="Image source: per-TP TIFFs (default) or raw Vutara .dat.")
_ap.add_argument('--input_dir', default=None,
                 help="Override the DAPI dir (a Raw Images/ folder when --source dat).")
_ap.add_argument('--stitch_threshold', type=float, default=0.5,
                 help="IoU a 2D mask must share with one in the next plane to be "
                      "joined into the same 3D cell (do_3D=False, so cells are "
                      "built by stitching per-plane masks). Default 0.5 = rep1-3. "
                      "Too high for a given dataset and the axial chain keeps "
                      "breaking, so one cell becomes several short fragments that "
                      "each hold only part of its z-range: on the Novo tree at 0.5 "
                      "only 84%% of candidate links survive (rep3: 98%%), giving "
                      "masks a median 2 planes tall against rep3's 15. Set it so "
                      "link survival matches rep3's - 0.25 for Novo. Check with "
                      "the per-plane IoU in masks_per_plane_unstitched.tif.")
_ap.add_argument('--min_z_extent', type=int, default=0,
                 help="Drop final labels spanning fewer than this many z-planes "
                      "(0 = off = rep1-3). See finalize_cellpose_masks.py: thin "
                      "labels nested inside a real cell trip the "
                      "center-in-neighbour guard and reject that cell's genuine "
                      "spot. 3 for Novo.")
_ap.add_argument(
    '--segmentation-mode', '--segmentation_mode',
    dest='segmentation_mode',
    choices=['stitched', '2p5d', '2p5d-zgated', '3d'], default='stitched',
    help="Mask-generation strategy. 'stitched' preserves the existing per-plane "
         "Cellpose + cross-z repair path. '2p5d' segments the averaged-DAPI "
         "z-maximum projection once and extrudes that XY instance map through "
         "the full z stack for downstream 3D fitting. '2p5d-zgated' preserves "
         "the same XY ownership but derives per-cell, per-time-point axial "
         "support from DAPI intensity. '3d' runs one direct volumetric "
         "Cellpose inference on the averaged-DAPI stack.",
)
_ap.add_argument(
    '--xy-pixel-nm', '--xy_pixel_nm', dest='xy_pixel_nm', type=float,
    default=103.28,
    help='XY sampling used to derive Cellpose 3D anisotropy. Only used in '
         '3d mode. (default: %(default)s)',
)
_ap.add_argument(
    '--z-step-nm', '--z_step_nm', dest='z_step_nm', type=float,
    default=100.0,
    help='Z sampling used to derive Cellpose 3D anisotropy. Only used in '
         '3d mode. (default: %(default)s)',
)
_ap.add_argument(
    '--cellpose-3d-anisotropy', type=float, default=None,
    help='Optional direct override for Cellpose anisotropy (z spacing / xy '
         'spacing). By default it is derived from --z-step-nm / '
         '--xy-pixel-nm.',
)
_ap.add_argument(
    '--z-gate-failure-policy', choices=['quarantine', 'fail'],
    default='quarantine',
    help="Failure behavior for weak or ambiguous DAPI z gates in "
         "2p5d-zgated mode. (default: %(default)s)",
)
_ap.add_argument(
    '--diameter', type=int, default=20,
    help='Expected cell diameter in pixels for Cellpose. (default: %(default)s)',
)
_args, _ = _ap.parse_known_args()

# Output root. Env-overridable (PIPELINE_OUT) so independent pipeline runs can
# proceed concurrently in their own trees; unset reproduces the historical
# hardcoded path byte-for-byte. Set by run_pipeline_dat.sh /
# run_pipeline_confocal.sh, which no longer symlink $SALMO/output.
PIPELINE_OUT = os.environ.get('PIPELINE_OUT', '/mnt/altnas/work/dimos/Salmo_WF/output')

indir = os.path.join(PIPELINE_OUT, 'probe_2')
outdir = os.path.join(indir, 'cellpose_masks')
os.makedirs(outdir, exist_ok=True)

# DAPI = probe 2 (per-TP TIFFs, or de-interleaved from raw .dat)
src_dir = _args.input_dir or indir
n_have = dat_io.n_timepoints(src_dir, 2, _args.source)
print(f'Found {n_have} DAPI timepoints ({_args.source})')

# Average first 5 TPs' DAPI for better SNR
n_avg = min(5, n_have)
print(f'Averaging DAPI from first {n_avg} TPs...')
avg_img = None
for tp_idx, (tp_name, img) in enumerate(dat_io.iter_timepoints(src_dir, 2, _args.source)):
    if tp_idx >= n_avg:
        break
    img = img.astype(np.float32)
    avg_img = img if avg_img is None else avg_img + img
avg_img /= n_avg
print(f'  Averaged image shape: {avg_img.shape}, range: [{avg_img.min():.0f}, {avg_img.max():.0f}]')

# Save averaged DAPI
dapi_path = os.path.join(outdir, 'averaged_dapi.tif')
tifffile.imwrite(dapi_path, avg_img.astype(np.float32))

# Segment with the updated Cellpose-SAM model
model = CellposeModel(gpu=True, pretrained_model='cpsam_v2')
print(f'Running cellpose on averaged DAPI (mode={_args.segmentation_mode})...', flush=True)

if _args.segmentation_mode == '3d':
    raw_masks, masks_3d, summary = segment_3d(
        model,
        avg_img,
        xy_pixel_nm=_args.xy_pixel_nm,
        z_step_nm=_args.z_step_nm,
        anisotropy=_args.cellpose_3d_anisotropy,
        diameter=_args.diameter,
        min_size=15,
        max_diameter=40,
    )
    summary.update({
        'model': 'cpsam_v2',
        'timepoints_averaged': int(n_avg),
        'source': _args.source,
        'clump_split_policy': 'disabled',
        'cross_z_repair': False,
        'stitching': False,
    })
    raw_path = os.path.join(outdir, 'masks_cellpose_3d_raw.tif')
    final_path = os.path.join(outdir, 'masks_final.tif')
    summary_path = os.path.join(outdir, 'segmentation_3d_summary.json')
    tifffile.imwrite(
        raw_path,
        raw_masks.astype(np.uint32),
        compression='zlib',
        metadata={'axes': 'ZYX'},
    )
    tifffile.imwrite(
        final_path,
        masks_3d,
        compression='zlib',
        metadata={'axes': 'ZYX'},
    )
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(
        f"3D segmentation: {summary['labels_before_oversized_filter']} labels "
        f"before oversized filtering; removed "
        f"{summary['oversized_labels_removed']}; final={summary['final_labels']}"
    )
    print(
        '  Cellpose anisotropy (z/xy) = '
        f"{summary['anisotropy_z_over_xy']:.6g} "
        f"({summary['z_step_nm']:.6g}/{summary['xy_pixel_nm']:.6g} nm)"
    )
    print(f'Saved raw Cellpose 3D labels to {raw_path}')
    print(f'Saved final 3D masks to {final_path}')
    print(f'Saved 3D provenance to {summary_path}')
    print('Skipped per-plane stitching, cross-z repair, and finalization.')
    print('Done!')
    raise SystemExit(0)

if _args.segmentation_mode in ('2p5d', '2p5d-zgated'):
    projection, masks_2d, masks_3d, summary = segment_2p5d(
        model,
        avg_img,
        diameter=_args.diameter,
        min_size=15,
        max_diameter=40,
    )
    summary.update({
        'segmentation_mode': _args.segmentation_mode,
        'model': 'cpsam_v2',
        'timepoints_averaged': int(n_avg),
        'source': _args.source,
    })
    if _args.segmentation_mode == '2p5d-zgated':
        summary.update({
            'z_policy': 'per_timepoint_dapi_intensity_gate',
            'compatibility_mask_z_policy': 'full_stack_extrusion',
            'z_gate_failure_policy': _args.z_gate_failure_policy,
            'z_support_sidecar': 'z_support_2p5d.npz',
            'z_support_summary': 'z_support_2p5d_summary.json',
        })
    projection_path = os.path.join(outdir, 'averaged_dapi_zmax.tif')
    masks_2d_path = os.path.join(outdir, 'masks_2p5d_2d.tif')
    final_path = os.path.join(outdir, 'masks_final.tif')
    summary_path = os.path.join(outdir, 'segmentation_2p5d_summary.json')
    tifffile.imwrite(projection_path, projection.astype(np.float32), compression='zlib')
    tifffile.imwrite(
        masks_2d_path,
        masks_2d,
        compression='zlib',
        metadata={'axes': 'YX'},
    )
    tifffile.imwrite(
        final_path,
        masks_3d,
        compression='zlib',
        metadata={'axes': 'ZYX'},
    )

    z_support = None
    if _args.segmentation_mode == '2p5d-zgated':
        print(
            'Computing per-cell, per-time-point DAPI z support...',
            flush=True,
        )
        z_support = compute_z_support(
            masks_2d,
            dat_io.iter_timepoints(src_dir, 2, _args.source),
            failure_policy=_args.z_gate_failure_policy,
        )
        z_support_path = os.path.join(outdir, 'z_support_2p5d.npz')
        z_support_summary_path = os.path.join(
            outdir, 'z_support_2p5d_summary.json')
        save_z_support(
            z_support,
            z_support_path,
            z_support_summary_path,
            provenance={
                'segmentation_mode': _args.segmentation_mode,
                'model': 'cpsam_v2',
                'source': _args.source,
                'timepoints_averaged': int(n_avg),
                'averaged_dapi': os.path.basename(dapi_path),
                'ownership_mask': os.path.basename(masks_2d_path),
                'compatibility_mask': os.path.basename(final_path),
            },
        )
        statuses, status_counts = np.unique(
            z_support.status, return_counts=True)
        summary['z_gate_status_counts'] = {
            str(status): int(count)
            for status, count in zip(statuses, status_counts)
        }
        summary['z_gate_quarantined'] = int(
            np.count_nonzero(z_support.quarantined))
        summary['z_gate_evaluable'] = int(
            np.count_nonzero(z_support.evaluable))
        summary['z_gate_fit_allowed'] = int(
            np.count_nonzero(z_support.fit_allowed))

    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(
        f"2.5D segmentation: {summary['labels_before_oversized_filter']} labels "
        f"before oversized filtering; removed "
        f"{summary['oversized_labels_removed']}; final={summary['final_labels']}"
    )
    print(f'Saved direct 2D masks to {masks_2d_path}')
    if _args.segmentation_mode == '2p5d-zgated':
        print(f'Saved compatibility full-z ownership mask to {final_path}')
        print(f'Saved dynamic z support to {z_support_path}')
        print(f'Saved z-support provenance to {z_support_summary_path}')
    else:
        print(f'Saved full-z extruded masks to {final_path}')
    print(f'Saved 2.5D provenance to {summary_path}')
    print('Skipped per-plane stitching, cross-z repair, and finalization.')
    print('Done!')
    raise SystemExit(0)

# Cellpose does not expose the per-plane masks after stitching. Capture the
# exact input to its stitch3D call while leaving the normal stitched result
# untouched. The cross-Z stage below uses these local masks to repair cases
# where a current Z plane merged two correctly separated predecessor masks.
_unstitched_capture = []
_cellpose_stitch3d = cellpose_utils.stitch3D


def _capture_then_stitch(masks, stitch_threshold=0.25):
    _unstitched_capture.append(masks.copy())
    return _cellpose_stitch3d(masks, stitch_threshold=stitch_threshold)


cellpose_utils.stitch3D = _capture_then_stitch
print(f'  stitch_threshold={_args.stitch_threshold}')
try:
    masks, flows, styles = model.eval(
        avg_img,
        diameter=_args.diameter,
        channels=[0, 0],
        z_axis=0,
        do_3D=False,
        stitch_threshold=_args.stitch_threshold,
    )
finally:
    cellpose_utils.stitch3D = _cellpose_stitch3d

if len(_unstitched_capture) != 1:
    raise RuntimeError(
        "Expected one Cellpose stitch3D call, captured "
        f"{len(_unstitched_capture)}"
    )
masks_unstitched = _unstitched_capture[0]
if masks_unstitched.max() > np.iinfo(np.uint16).max:
    raise RuntimeError(
        f"Per-plane mask label {masks_unstitched.max()} exceeds uint16"
    )
unstitched_path = os.path.join(outdir, 'masks_per_plane_unstitched.tif')
tifffile.imwrite(
    unstitched_path,
    masks_unstitched.astype(np.uint16),
    compression='zlib',
    metadata={'axes': 'ZYX'},
)
print(f'  Saved per-plane masks for cross-Z repair: {unstitched_path}')

n_before = len(np.unique(masks)) - 1
print(f'  Baseline before cross-Z repair: {n_before} stitched labels')

# Filter out oversized masks (equivalent diameter > 40px)
# Vectorized: compute max per-slice area for each label
max_diameter = 40
max_area_thresh = np.pi * (max_diameter / 2) ** 2

# Process per z-slice, track max area per label
max_label = masks.max()
max_areas = np.zeros(max_label + 1, dtype=np.int64)
for z in range(masks.shape[0]):
    slice_counts = np.bincount(masks[z].ravel(), minlength=max_label + 1)
    max_areas = np.maximum(max_areas, slice_counts)

# Find labels to remove (skip background=0)
oversized = np.where((max_areas > max_area_thresh) & (np.arange(max_label + 1) > 0))[0]
mask_remove = np.isin(masks, oversized)
masks[mask_remove] = 0

# Relabel sequentially using fastremap (already installed with cellpose)
import fastremap
masks_clean = fastremap.renumber(masks)[0]
n_final = masks_clean.max()

print(f'  Baseline removed {len(oversized)} oversized masks (diameter > {max_diameter}px)')
print(f'  Baseline after size filtering: {n_final} labels')

# Preserve the unmodified Cellpose baseline for traceability, then replace
# masks_final.tif with the predecessor-guided repair.
pre_crossz_path = os.path.join(outdir, 'masks_final.pre_crossz.tif')
tifffile.imwrite(pre_crossz_path, masks_clean.astype(np.uint16))
print(f'Saved baseline before cross-Z repair to {pre_crossz_path}')

repaired_unstitched_path = os.path.join(
    outdir, 'masks_per_plane_crossz_repaired.tif'
)
# NB: masks_final.tif comes from finalize's OWN stitch of the cross-Z-repaired
# planes, not from Cellpose's stitched result (which only becomes the discarded
# masks_final.pre_crossz.tif baseline). So --stitch_threshold has to reach these
# two calls or it changes nothing downstream.
repair_cross_z_main([
    '--unstitched', unstitched_path,
    '--image', dapi_path,
    '--out', repaired_unstitched_path,
    '--stitch_threshold', str(_args.stitch_threshold),
    '--candidate_iou', '0.05',
    '--watershed_sigma', '1.0',
])
final_path = os.path.join(outdir, 'masks_final.tif')
finalize_cellpose_main([
    '--unstitched', repaired_unstitched_path,
    '--out', final_path,
    '--stitch_threshold', str(_args.stitch_threshold),
    '--min_size', '15',
    '--max_diameter', str(max_diameter),
    '--min_z_extent', str(_args.min_z_extent),
])
print(f'Saved cross-Z-repaired masks to {final_path}')
print('Done!')
