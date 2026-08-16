#!/usr/bin/env python
"""Localize 488 universal probe for drift correction.

Uses the same DAPI mask from probe_2 to localize 488 signal in probe_1.
"""

import numpy as np
import tifffile
import os
import glob
import csv
import ctypes
from scipy.ndimage import find_objects
from z_support_2p5d import load_z_support, materialize_labels
import center_ownership as _own
import chromatic_registration as _chrom

import argparse as _argparse
import dat_io
_ap = _argparse.ArgumentParser(description='Localize 488 (probe_1).')
_ap.add_argument('--source', choices=['tiff', 'dat'], default='tiff',
                 help="Image source: per-TP TIFFs (default) or raw Vutara .dat.")
_ap.add_argument('--input_dir', default=None,
                 help="Override the image dir (a Raw Images/ folder when --source dat).")
_ap.add_argument(
    '--segmentation-mode', '--segmentation_mode',
    dest='segmentation_mode',
    choices=['stitched', '2p5d', '2p5d-zgated', '3d'],
    default='stitched',
    help="Segmentation strategy. 'stitched', '2p5d', and '3d' use the static "
         "masks_final.tif volume; only '2p5d-zgated' activates dynamic z support.",
)
_ap.add_argument(
    '--z-gate-failure-policy', choices=['quarantine', 'fail'],
    default='quarantine',
    help="Required policy recorded in the 2p5d z-support sidecar.",
)
_ap.add_argument(
    '--z-gate-path', default=None,
    help="Override z_support_2p5d.npz (2p5d-zgated only).",
)
_ap.add_argument(
    '--chromatic-registration', '--chromatic_registration',
    dest='chromatic_registration', choices=['off', 'metadata'], default='off',
    help="Represent the DAPI (405) label mask on the 488 pixel grid using the "
         "acquisition's own widefield.json chromatic calibration. 'off' "
         "(default) preserves existing behaviour byte-for-byte. 'metadata' "
         "fails loudly if the calibration is absent or unsupported -- it never "
         "falls back to identity.",
)
_ap.add_argument(
    '--chromatic-metadata', '--chromatic_metadata',
    dest='chromatic_metadata', default=None,
    help="Path to widefield.json. Default: alongside --input_dir.",
)
_ap.add_argument(
    '--reference-wavelength-nm', '--reference_wavelength_nm',
    dest='reference_wavelength', default=None,
    help="Segmentation/reference channel wavelength (default: 405).")
_ap.add_argument(
    '--fiducial-wavelength-nm', '--fiducial_wavelength_nm',
    dest='fiducial_wavelength', default=None,
    help="Fiducial target wavelength (default: 488). Never inferred from the "
         "probe directory index.")
_ap.add_argument(
    '--modality', choices=['widefield', 'confocal'], default='widefield',
    help="Selects the default reference/fiducial wavelengths.")
_args, _ = _ap.parse_known_args()

# --- Build C library if missing or stale; prefer OpenMP, fall back to serial ---
_HERE = os.path.dirname(os.path.abspath(__file__))
c_src = os.path.join(_HERE, 'gauss3d_fit.c')
c_lib = os.path.join(_HERE, 'gauss3d_fit.so')
if not os.path.exists(c_lib) or os.path.getmtime(c_src) > os.path.getmtime(c_lib):
    base = f'gcc -O3 -shared -fPIC -o {c_lib} {c_src} -lm'
    print('Compiling C fitting library (trying OpenMP)...')
    if os.system(f'gcc -O3 -fopenmp -shared -fPIC -o {c_lib} {c_src} -lm') == 0:
        print('  Compiled with OpenMP.')
    else:
        print('  WARNING: OpenMP build failed; falling back to serial build.')
        if os.system(base) != 0:
            raise RuntimeError(
                f'Failed to compile {c_lib} from {c_src}. '
                f'Either prebuild it or ensure gcc is on PATH.')
        print('  Compiled without OpenMP (serial).')

lib = ctypes.CDLL(c_lib)

lib.has_openmp.restype = ctypes.c_int
lib.has_openmp.argtypes = []
lib.openmp_max_threads.restype = ctypes.c_int
lib.openmp_max_threads.argtypes = []
if lib.has_openmp():
    print(f'Fitter: OpenMP enabled, max threads = {lib.openmp_max_threads()}')
else:
    print('Fitter: serial (no OpenMP in this .so)')
lib.batch_fit.restype = None
lib.batch_fit.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
]

# --- Paths ---

# Output root. Env-overridable (PIPELINE_OUT) so independent pipeline runs can
# proceed concurrently in their own trees; unset reproduces the historical
# hardcoded path byte-for-byte. Set by run_pipeline_dat.sh /
# run_pipeline_confocal.sh, which no longer symlink $SALMO/output.
PIPELINE_OUT = os.environ.get('PIPELINE_OUT', '/mnt/altnas/work/dimos/Salmo_WF/output')
probe1_dir = os.path.join(PIPELINE_OUT, 'probe_1')
mask_path = os.path.join(PIPELINE_OUT, 'probe_2', 'cellpose_masks',
                         'masks_final.tif')
outdir = os.path.join(probe1_dir, 'localizations')
os.makedirs(outdir, exist_ok=True)

# --- Load mask ---
masks = tifffile.imread(mask_path).astype(np.int32)
nz, ny, nx = masks.shape
n_cells = masks.max()
print(f'Loaded mask: {masks.shape}, {n_cells} cells')

src_dir = _args.input_dir or probe1_dir

# --- Chromatic registration of the DAPI mask into the 488 grid --------------
# Same composition as the 640 path, only the target channel differs:
# p_405 = T_405 @ inverse(T_488) @ p_488. Default 'off' is byte-identical to the
# historical behaviour.
#
# NOTE: there is deliberately no --center-ownership strict here. Dropping 488
# fiducial fits would change which acquisitions contribute to the drift curve,
# and the relative 488(t) - 488(reference) formula must not be altered. Ownership
# is therefore RECORDED in the per-attempt audit table and never acted on, and
# localizations_488.csv keeps its historical column set so drift_correct.py sees
# exactly the same input schema.
_REF_NM = _args.reference_wavelength or _chrom.modality_default(_args.modality, 'reference')
_TGT_NM = _args.fiducial_wavelength or _chrom.modality_default(_args.modality, 'fiducial')

_chrom_calib = None
# Stage-specific filename: the 640 localizer and replication_qc write their own.
_chrom_prov_path = os.path.join(outdir, 'drift_488_chromatic_registration.json')
if _args.chromatic_registration == 'metadata':
    meta_path = _args.chromatic_metadata or os.path.join(src_dir, 'widefield.json')
    _chrom_calib = _chrom.load_calibration(meta_path)   # raises if unusable
    target = _TGT_NM
    print(f'Wavelengths: reference {_REF_NM} nm -> fiducial {_TGT_NM} nm')
    print(f'Chromatic registration: {meta_path}')
    print(f'  channel map (wavelength -> transform index): '
          f'{dict(_chrom_calib.transform_index)}')
    print(f'  M = T_{_REF_NM} @ inv(T_{target}):\n'
          f'{_chrom.lookup_matrix(_chrom_calib, target, _REF_NM)}')
    _chrom_provenance = _chrom.provenance(
        _chrom_calib, {'probe_1': target}, mode='metadata',
        stage='localize_488', reference_wavelength=_REF_NM,
        ownership_policy='record_only',
        cli_args={'modality': _args.modality,
                  'reference_wavelength_nm': _REF_NM,
                  'fiducial_wavelength_nm': _TGT_NM,
                  'chromatic_metadata': meta_path},
        notes='DAPI label mask represented on the probe_1 (488) pixel grid for '
              'bounding boxes and fitting support. Centre ownership is recorded '
              'only; no 488 fit is ever dropped, so the drift formula is '
              'untouched.')
    _labels_before = np.unique(masks)
    masks = _chrom.resample_labels_to_channel(
        masks, _chrom_calib, target, _REF_NM).astype(np.int32, copy=False)
    masks = np.ascontiguousarray(masks)
    if not set(np.unique(masks).tolist()).issubset(set(_labels_before.tolist())):
        raise RuntimeError('chromatic resample introduced new label values; '
                           'the lookup was not categorical')
else:
    _chrom_provenance = _chrom.disabled_provenance(
        f'Chromatic registration OFF: the {_REF_NM}-frame mask is used directly '
        f'as the {_TGT_NM} fitting support, so that chromatic offset is '
        'uncorrected. Ownership fields are still recorded.',
        stage='localize_488', ownership_policy='record_only',
        cli_args={'modality': _args.modality})
_chrom.write_provenance(_chrom_prov_path, _chrom_provenance)

tp_names = dat_io.timepoint_names(src_dir, 1, _args.source)
n_tp = len(tp_names)
if n_tp == 0 and _args.segmentation_mode == '2p5d-zgated':
    raise RuntimeError(f'No timepoints in {src_dir} ({_args.source})')

z_support = None
labels_2d = None
if _args.segmentation_mode == '2p5d-zgated':
    mask_dir = os.path.dirname(mask_path)
    labels_2d_path = os.path.join(mask_dir, 'masks_2p5d_2d.tif')
    z_gate_path = _args.z_gate_path or os.path.join(
        mask_dir, 'z_support_2p5d.npz')
    if not os.path.isfile(labels_2d_path):
        raise FileNotFoundError(
            f'2p5d-zgated requires the authoritative 2D mask: {labels_2d_path}')
    labels_2d = tifffile.imread(labels_2d_path).astype(np.int32)
    if labels_2d.shape != (ny, nx):
        raise ValueError(
            f'2D mask shape {labels_2d.shape} does not match 3D mask XY {(ny, nx)}')
    z_support = load_z_support(
        z_gate_path,
        labels_2d,
        expected_tp_names=tp_names,
        expected_nz=nz,
        expected_failure_policy=_args.z_gate_failure_policy,
    )
    expected_labels = np.arange(1, n_cells + 1, dtype=z_support.label_ids.dtype)
    if not np.array_equal(z_support.label_ids, expected_labels):
        raise ValueError(
            '2p5d z-support labels are not the contiguous IDs required by the fitter')
    print(
        f'Loaded dynamic z support: {len(z_support.tp_names)} timepoints, '
        f'failure_policy={z_support.failure_policy}')
    # The sidecar is validated against the UNREGISTERED 2D map (the frame it was
    # computed in); its per-label z intervals are frame-independent, so
    # materialization can use the registered map.
    if _chrom_calib is not None:
        labels_2d = _chrom.resample_labels_to_channel(
            labels_2d, _chrom_calib, _TGT_NM, _REF_NM
        ).astype(np.int32, copy=False)
        print(f'  registered the authoritative 2D ownership map into the {_TGT_NM} grid')

# --- Bounding boxes ---
print('Computing bounding boxes...')
slices = find_objects(masks)
pad_xy, pad_z = 3, 1
bboxes = np.zeros((n_cells, 6), dtype=np.int32)
for i, sl in enumerate(slices):
    if sl is None:
        continue
    bboxes[i] = [
        max(0, sl[0].start - pad_z), min(nz - 1, sl[0].stop - 1 + pad_z),
        max(0, sl[1].start - pad_xy), min(ny - 1, sl[1].stop - 1 + pad_xy),
        max(0, sl[2].start - pad_xy), min(nx - 1, sl[2].stop - 1 + pad_xy),
    ]

# --- Process all TPs (per-TP TIFFs, or de-interleaved from raw .dat) ---
print(f'Found {n_tp} timepoints ({_args.source})\n')

out_csv = os.path.join(outdir, 'localizations_488.csv')
fields = ['tp', 'WalkID', 'z', 'y', 'x', 'sigma_z', 'sigma_y', 'sigma_x',
          'amplitude', 'background', 'z_err', 'y_err', 'x_err']
if z_support is not None:
    fields.extend([
        'z_gate_status', 'z_gate_fallback', 'z_gate_min', 'z_gate_max',
        'z_gate_quarantined', 'z_gate_evaluable', 'z_gate_fit_allowed',
    ])

# Per-attempt status table. Previously written only in 2p5d-zgated mode; it is
# now the single auditable record for EVERY attempted (WalkID, tp) so the 488
# centre-ownership readout needs no extra reduced CSV. The z_gate_* columns are
# blank when the mode has no dynamic gate.
gate_audit_path = os.path.join(outdir, 'localizations_488_z_gate_status.csv')
gate_audit_file = open(gate_audit_path, 'w', newline='')
gate_audit_writer = csv.DictWriter(
    gate_audit_file,
    fieldnames=[
        'tp', 'WalkID', 'z_gate_status', 'z_gate_fallback',
        'z_gate_min', 'z_gate_max',
        'z_gate_quarantined', 'z_gate_evaluable', 'z_gate_fit_allowed',
        'fit_ret_code', 'localization_written',
        'center_ownership_initial', 'center_ownership_pass',
    ],
    restval='',
)
gate_audit_writer.writeheader()
ownership_counts = {name: 0 for name in _own.CATEGORIES}

total_fit = 0
total_fail = 0

with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    for tp_idx, (tp_name, img) in enumerate(
            dat_io.iter_timepoints(src_dir, 1, _args.source)):
        if tp_idx >= n_tp:
            break
        print(f'[{tp_idx+1}/{n_tp}] {tp_name}...', flush=True)

        img = img.astype(np.float64)
        if img.shape != masks.shape:
            raise ValueError(
                f'Shape mismatch at {tp_name}: img={img.shape} vs mask={masks.shape}. '
                f'Refusing to pass mismatched buffers to the C fitter.')

        fit_masks = masks
        fit_bboxes = bboxes
        gate = None
        if z_support is not None:
            gate = z_support.select_tp(tp_name)
            fit_masks = materialize_labels(
                labels_2d, nz, gate.z_min, gate.z_max,
                fit_allowed=gate.fit_allowed,
                label_ids=gate.label_ids,
            ).astype(np.int32, copy=False)
            fit_masks = np.ascontiguousarray(fit_masks)
            fit_bboxes = np.ascontiguousarray(
                np.zeros((n_cells, 6), dtype=np.int32))
            dynamic_slices = find_objects(fit_masks)
            for i, sl in enumerate(dynamic_slices):
                if sl is None:
                    continue
                fit_bboxes[i] = [
                    max(0, sl[0].start - pad_z),
                    min(nz - 1, sl[0].stop - 1 + pad_z),
                    max(0, sl[1].start - pad_xy),
                    min(ny - 1, sl[1].stop - 1 + pad_xy),
                    max(0, sl[2].start - pad_xy),
                    min(nx - 1, sl[2].stop - 1 + pad_xy),
                ]

        results = np.zeros((n_cells, 11), dtype=np.float64)
        success = np.zeros(n_cells, dtype=np.int32)

        lib.batch_fit(
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            fit_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            nz, ny, nx,
            fit_bboxes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_cells,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            success.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        n_ok = (success == 0).sum()
        n_bad = (success != 0).sum()
        total_fit += n_ok
        total_fail += n_bad

        # Centre ownership, against the SAME effective support used to fit.
        # Recorded only: no 488 fit is dropped on this basis.
        ownership = [''] * n_cells
        for c in range(n_cells):
            if success[c] != 0:
                continue
            r = results[c]
            ownership[c] = _own.classify_center(fit_masks, c, r[1], r[2], r[3])
            ownership_counts[ownership[c]] = (
                ownership_counts.get(ownership[c], 0) + 1)

        for c in range(n_cells):
            if success[c] != 0:
                continue
            r = results[c]
            row = {
                'tp': tp_name,
                'WalkID': c,
                'z': f'{r[1]:.4f}',
                'y': f'{r[2]:.4f}',
                'x': f'{r[3]:.4f}',
                'sigma_z': f'{r[4]:.4f}',
                'sigma_y': f'{r[5]:.4f}',
                'sigma_x': f'{r[6]:.4f}',
                'amplitude': f'{r[0]:.2f}',
                'background': f'{r[7]:.2f}',
                'z_err': f'{r[8]:.4f}',
                'y_err': f'{r[9]:.4f}',
                'x_err': f'{r[10]:.4f}',
            }
            if gate is not None:
                row.update({
                    'z_gate_status': str(gate.status[c]),
                    'z_gate_fallback': str(gate.fallback[c]),
                    'z_gate_min': int(gate.z_min[c]),
                    'z_gate_max': int(gate.z_max[c]),
                    'z_gate_quarantined': int(gate.quarantined[c]),
                    'z_gate_evaluable': int(gate.evaluable[c]),
                    'z_gate_fit_allowed': int(gate.fit_allowed[c]),
                })
            writer.writerow(row)

        for c in range(n_cells):
            audit = {
                'tp': tp_name,
                'WalkID': c,
                'fit_ret_code': int(success[c]),
                'localization_written': int(success[c] == 0),
                'center_ownership_initial': ownership[c],
                'center_ownership_pass': (
                    int(ownership[c] in _own.PASSING) if ownership[c] else ''),
            }
            if gate is not None:
                audit.update({
                    'z_gate_status': str(gate.status[c]),
                    'z_gate_fallback': str(gate.fallback[c]),
                    'z_gate_min': int(gate.z_min[c]),
                    'z_gate_max': int(gate.z_max[c]),
                    'z_gate_quarantined': int(gate.quarantined[c]),
                    'z_gate_evaluable': int(gate.evaluable[c]),
                    'z_gate_fit_allowed': int(gate.fit_allowed[c]),
                })
            gate_audit_writer.writerow(audit)

        print(f'  Fit: {n_ok}, Failed: {n_bad}')

print(f'\nTotal: {total_fit} localizations, {total_fail} failed')
_own_total = sum(ownership_counts.values())
if _own_total:
    print(f'Centre ownership over {_own_total} successful 488 fits '
          f'(registration={_args.chromatic_registration}; recorded only, '
          f'no fit dropped):')
    for name in _own.CATEGORIES:
        n = ownership_counts.get(name, 0)
        print(f'  {name:<20} {n:>9,}  {n / _own_total * 100:6.2f}%')
print(f'Saved to {out_csv}')
gate_audit_file.close()
print(f'Saved per-attempt status audit to {gate_audit_path}')
print(f'Saved chromatic provenance to {_chrom_prov_path}')
print('Done!')
