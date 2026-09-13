#!/usr/bin/env python
"""Localize probe_0 (640nm genomic probe) using DAPI mask and C fitting library."""

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

# Optional axial-PSF cap. --sigma_z_max V (voxels) clamps the fitted sigma_z to
# [0.2, V] instead of the loose [0.2, nz], removing the thin-stack sigma_z
# blow-up and the degeneracy-driven "confidently wrong" z it produces. Default
# 0.0 = OFF, identical to the original fit. Calibrate V from calibrate_sigma_z.py
# (p25 of well-conditioned in-situ fits; ~3.5 for this dataset's 640 channel).
import argparse as _argparse
import dat_io
_ap = _argparse.ArgumentParser(description='Localize probe_0 (640nm).')
_ap.add_argument('--sigma_z_max', type=float, default=0.0,
                 help='Upper cap on fitted sigma_z in voxels (<=0 disables; '
                      'default 0.0 preserves original behaviour).')
_ap.add_argument('--min_sbr', type=float, default=0.0,
                 help='Minimum signal-to-background ratio (amplitude/background) '
                      'to accept a fit. Spots below this are treated as failed. '
                      '0 disables (default).')
_ap.add_argument('--preprocess-radius', dest='preprocess_radius', type=int, default=20,
                 help='Morphological background-removal disk radius in px (default 20).')
_ap.add_argument('--preprocess-wavelet-levels', dest='preprocess_wavelet_levels', type=int, default=3,
                 help='Haar wavelet denoise levels, 0 disables (default 3).')
_ap.add_argument('--preprocess-diffusion-iters', dest='preprocess_diffusion_iters', type=int, default=5,
                 help='Anisotropic diffusion iterations, 0 disables (default 5).')
_ap.add_argument('--preprocess-nlm', dest='preprocess_nlm', type=float, nargs=3, default=None,
                 metavar=('SEARCH_R', 'PATCH_R', 'SIGMA'),
                 help='Add non-local-means denoising with these parameters (default off).')
_ap.add_argument('--no-preprocess', '--no_preprocess', dest='no_preprocess',
                 action='store_true',
                 help='Disable image preprocessing before fitting.')
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
    help="Represent the DAPI (405) label mask on the 640 pixel grid using the "
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
    '--center-ownership', '--center_ownership',
    dest='center_ownership',
    choices=['allow', 'strict', 'constrained_experimental'], default='allow',
    help="Ownership policy. 'allow' (default) records the ownership diagnostic "
         "and keeps every successful unconstrained fit -- byte-compatible with "
         "legacy output when registration is off. 'strict' is a FAIL-CLOSED "
         "FILTER: the ordinary unconstrained centre is classified against the "
         "registered effective mask and accepted only if it already lies in its "
         "own label; otherwise the locus is UNEVALUABLE. strict never moves, "
         "projects, clamps, snaps or re-fits a coordinate. "
         "'constrained_experimental' is research-only and NOT for analysis: it "
         "re-fits out-of-support centres against the mask boundary, which "
         "measures as boundary clamping.",
)
_ap.add_argument(
    '--reference-wavelength-nm', '--reference_wavelength_nm',
    dest='reference_wavelength', default=None,
    help="Segmentation/reference channel wavelength (default: 405).")
_ap.add_argument(
    '--target-wavelength-nm', '--target_wavelength_nm',
    dest='target_wavelength', default=None,
    help="Probe/localization target wavelength. Default depends on --modality: "
         "widefield 640, confocal 561. NEVER inferred from the probe index.")
_ap.add_argument(
    '--modality', choices=['widefield', 'confocal'], default='widefield',
    help="Selects the default reference/target wavelengths. confocal probe_0 is "
         "561 nm, not 640 -- pass --modality confocal for confocal data.")
_ap.add_argument(
    '--xy-pixel-nm', '--xy_pixel_nm', dest='xy_pixel_nm', type=float,
    default=0.0,
    help="XY pixel size in nm, for usability distance calculations. "
         "0 disables usability classification.")
_ap.add_argument(
    '--z-step-nm', '--z_step_nm', dest='z_step_nm', type=float,
    default=0.0,
    help="Z step size in nm, for usability distance calculations.")
_ap.add_argument(
    '--mask-dilation-xy-px', '--mask_dilation_xy_px',
    dest='mask_dilation_xy_px', type=int, default=0,
    help="Dilate the DAPI mask by N pixels in XY (in memory only) before "
         "fitting and ownership. Compensates for DAPI segmentation "
         "underestimating the nuclear boundary. 0 = off (default).")
_args, _ = _ap.parse_known_args()
lib.set_sigma_z_max.restype = None
lib.set_sigma_z_max.argtypes = [ctypes.c_double]
lib.set_sigma_z_max(ctypes.c_double(_args.sigma_z_max))
if _args.sigma_z_max > 0:
    print(f'Fitter: sigma_z capped at {_args.sigma_z_max} voxels')

# Runtime OpenMP diagnostic: tell the user whether the fitter is parallel and
# how many threads it will use.
lib.has_openmp.restype = ctypes.c_int
lib.has_openmp.argtypes = []
lib.openmp_max_threads.restype = ctypes.c_int
lib.openmp_max_threads.argtypes = []
if lib.has_openmp():
    print(f'Fitter: OpenMP enabled, max threads = {lib.openmp_max_threads()}')
else:
    print('Fitter: serial (no OpenMP in this .so)')
_preprocess_lib_path = os.path.join(_HERE, 'preprocess3d.so')
if _args.no_preprocess:
    _pp = None
    print('Preprocessing: disabled (--no-preprocess)')
elif os.path.exists(_preprocess_lib_path):
    _pp = ctypes.CDLL(_preprocess_lib_path)
    _pp.preprocess_denoise_morpho.restype = None
    _pp.preprocess_denoise_morpho.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _pp.preprocess_custom.restype = None
    _pp.preprocess_custom.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_double]
    _pp_custom = (_args.preprocess_radius != 20 or _args.preprocess_wavelet_levels != 3
                  or _args.preprocess_diffusion_iters != 5 or _args.preprocess_nlm is not None)
    if _pp_custom:
        print(f'Preprocessing: custom (radius={_args.preprocess_radius}, wavelet_levels='
              f'{_args.preprocess_wavelet_levels}, diffusion_iters={_args.preprocess_diffusion_iters}, '
              f'nlm={_args.preprocess_nlm})')
    else:
        print('Preprocessing: denoise+morpho enabled (preprocess3d.so)')
else:
    _pp = None
    print('Preprocessing: disabled (preprocess3d.so not found)')

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
# Constrained-refit entry point. Only ever called with strict_own=1, and only
# for the subset of walks whose unconstrained centre left its own support.
_own.declare_batch_fit_ex(lib)

# --- Paths ---

# Output root. Env-overridable (PIPELINE_OUT) so independent pipeline runs can
# proceed concurrently in their own trees; unset reproduces the historical
# hardcoded path byte-for-byte. Set by run_pipeline_dat.sh /
# run_pipeline_confocal.sh, which no longer symlink $SALMO/output.
PIPELINE_OUT = os.environ.get('PIPELINE_OUT', '/mnt/altnas/work/dimos/Salmo_WF/output')
probe0_dir = os.path.join(PIPELINE_OUT, 'probe_0')
mask_path = os.path.join(PIPELINE_OUT, 'probe_2', 'cellpose_masks',
                         'masks_final.tif')
outdir = os.path.join(probe0_dir, 'localizations')
os.makedirs(outdir, exist_ok=True)

# --- Load mask ---
masks = tifffile.imread(mask_path).astype(np.int32)
nz, ny, nx = masks.shape
n_cells = masks.max()
print(f'Loaded mask: {masks.shape}, {n_cells} cells')

src_dir = _args.input_dir or probe0_dir

# --- Chromatic registration of the DAPI mask into the 640 grid -------------
# The mask is segmented from 405 but used as the support for 640 fitting, so any
# 405<->640 chromatic offset misregisters it. Represent it on the 640 grid via
# p_405 = T_405 @ inverse(T_640) @ p_640. Default 'off' is byte-identical to the
# historical behaviour. Registration is XY-only and strictly categorical.
# Wavelengths are EXPLICIT. Never inferred from the probe directory index:
# confocal probe_0 is 561 nm, and inferring 640 from "probe_0" silently applied a
# 405->640 transform to confocal data.
_REF_NM = _args.reference_wavelength or _chrom.modality_default(_args.modality, 'reference')
_TGT_NM = _args.target_wavelength or _chrom.modality_default(_args.modality, 'probe')

_chrom_calib = None
_chrom_provenance = None
# Stage-specific provenance filename: replication_qc.py and the 488 localizer
# write their own, so no stage can overwrite another's record.
_chrom_prov_path = os.path.join(outdir, 'localizations_640_chromatic_registration.json')
if _args.chromatic_registration == 'metadata':
    meta_path = _args.chromatic_metadata or os.path.join(src_dir, 'widefield.json')
    _chrom_calib = _chrom.load_calibration(meta_path)   # raises if unusable
    target = _TGT_NM
    print(f'Wavelengths: reference {_REF_NM} nm -> target {_TGT_NM} nm '
          f'(modality={_args.modality})')
    print(f'Chromatic registration: {meta_path}')
    print(f'  channel map (wavelength -> transform index): '
          f'{dict(_chrom_calib.transform_index)}')
    print(f'  metadata pixel size: {_chrom_calib.pixel_size_nm} nm')
    print(f'  M = T_{_REF_NM} @ inv(T_{target}):\n'
          f'{_chrom.lookup_matrix(_chrom_calib, target, _REF_NM)}')
    _chrom_provenance = _chrom.provenance(
        _chrom_calib, {'probe_0': target}, mode='metadata',
        stage='localize_probe0',
        reference_wavelength=_REF_NM,
        ownership_policy=_args.center_ownership,
        cli_args={'modality': _args.modality,
                  'reference_wavelength_nm': _REF_NM,
                  'target_wavelength_nm': _TGT_NM,
                  'segmentation_mode': _args.segmentation_mode,
                  'sigma_z_max': _args.sigma_z_max,
                  'chromatic_metadata': meta_path},
        notes=f'Reference {_REF_NM} nm label mask represented on the {target} nm '
              'pixel grid for bounding boxes, fitting support and centre '
              'ownership. Ownership is decided in the raw target-channel frame.')
    _labels_before = np.unique(masks)
    masks = _chrom.resample_labels_to_channel(
        masks, _chrom_calib, target, _REF_NM).astype(np.int32, copy=False)
    masks = np.ascontiguousarray(masks)
    _labels_after = np.unique(masks)
    if not set(_labels_after.tolist()).issubset(set(_labels_before.tolist())):
        raise RuntimeError('chromatic resample introduced new label values; '
                           'the lookup was not categorical')
    _lost = sorted(set(_labels_before.tolist()) - set(_labels_after.tolist()))
    print(f'  registered mask: {len(_labels_after) - 1} of '
          f'{len(_labels_before) - 1} labels retain voxels'
          + (f'; {len(_lost)} shifted fully off the grid' if _lost else ''))
else:
    _chrom_provenance = _chrom.disabled_provenance(
        f'Chromatic registration OFF: the {_REF_NM}-frame mask is used directly '
        f'as the {_TGT_NM} fitting support, so any {_REF_NM}<->{_TGT_NM} '
        'chromatic offset is uncorrected. Ownership fields are still recorded.',
        stage='localize_probe0',
        ownership_policy=_args.center_ownership,
        cli_args={'modality': _args.modality,
                  'segmentation_mode': _args.segmentation_mode})
_chrom.write_provenance(_chrom_prov_path, _chrom_provenance)

# --- XY mask dilation (in memory only) ------------------------------------
if _args.mask_dilation_xy_px > 0:
    from scipy.ndimage import maximum_filter
    _dil = _args.mask_dilation_xy_px
    _k = 2 * _dil + 1
    _footprint = np.ones((1, _k, _k), dtype=bool)
    masks = maximum_filter(masks, footprint=_footprint).astype(np.int32, copy=False)
    masks = np.ascontiguousarray(masks)
    print(f'Mask dilation: {_dil} px XY (kernel {_k}x{_k}), in memory only')

tp_names = dat_io.timepoint_names(src_dir, 0, _args.source)
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
    # The z-support sidecar is validated against the UNREGISTERED 2D map, which
    # is the frame it was computed in; its intervals are per-label and therefore
    # frame-independent. Materialization then uses the registered map so the
    # effective support lives in the 640 grid.
    if _chrom_calib is not None:
        labels_2d = _chrom.resample_labels_to_channel(
            labels_2d, _chrom_calib, _TGT_NM, _REF_NM
        ).astype(np.int32, copy=False)
        print(f'  registered the authoritative 2D ownership map into the {_TGT_NM} grid')
    if _args.mask_dilation_xy_px > 0:
        from scipy.ndimage import maximum_filter as _mf2d
        _k2d = 2 * _args.mask_dilation_xy_px + 1
        labels_2d = _mf2d(labels_2d, size=_k2d).astype(np.int32, copy=False)

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

out_csv = os.path.join(outdir, 'localizations_640.csv')
fields = ['tp', 'WalkID', 'z', 'y', 'x', 'sigma_z', 'sigma_y', 'sigma_x',
          'amplitude', 'background', 'z_err', 'y_err', 'x_err']
# Centre-ownership lives in the per-attempt status table below, NOT here:
# localizations_640.csv keeps its historical column set so every downstream
# consumer and every archived comparison stays valid. In strict mode this file
# is the FILTERED result and the audit table records what was dropped and why.
OWNERSHIP_FIELDS = [
    'center_ownership_initial', 'center_refit_attempted',
    'center_refit_success', 'center_ownership_final', 'center_ownership_pass',
]
# The canonical ownership columns travel on the localization rows themselves, so
# a forced or rejected coordinate is identifiable from the comprehensive CSV
# without joining a sidecar. Appended AFTER the legacy columns so byte-position
# readers of the historical prefix are unaffected.
fields.extend(_own.OWNERSHIP_ROW_FIELDS)
fields.extend(_own.USABILITY_ROW_FIELDS)
if z_support is not None:
    fields.extend([
        'z_gate_status', 'z_gate_fallback', 'z_gate_min', 'z_gate_max',
        'z_gate_quarantined', 'z_gate_evaluable', 'z_gate_fit_allowed',
    ])

# Per-attempt status table. Previously written only in 2p5d-zgated mode; it is
# now the single auditable record for EVERY attempted (WalkID, tp), including
# failed fits and ownership rejections, so no reduced side CSVs are needed. The
# z_gate_* columns are blank when the mode has no dynamic gate.
#
# It is also the AUTHORITATIVE ATTEMPT UNIVERSE for the comprehensive downstream
# CSV, so it carries two further things:
#   * center_fit_{z,y,x}_px -- the unconstrained fitted centre, serialized
#     exactly as localizations_640.csv would write it, so a rejected coordinate
#     is still auditable numerically and the audit value is the very value the
#     ownership verdict was decided on;
#   * the canonical _own.OWNERSHIP_ROW_FIELDS, identical in name and convention
#     to the localization rows, so a rejected attempt can be reintroduced
#     downstream by copying columns rather than by re-deriving a verdict.
# Both are left blank for a failed fit: there is no centre and no verdict, which
# is different from a centre that was judged and rejected.
gate_audit_path = os.path.join(outdir, 'localizations_640_z_gate_status.csv')
gate_audit_file = open(gate_audit_path, 'w', newline='')
gate_audit_writer = csv.DictWriter(
    gate_audit_file,
    fieldnames=[
        'tp', 'WalkID', 'z_gate_status', 'z_gate_fallback',
        'z_gate_min', 'z_gate_max',
        'z_gate_quarantined', 'z_gate_evaluable', 'z_gate_fit_allowed',
        'fit_ret_code', 'localization_written',
        'center_fit_z_px', 'center_fit_y_px', 'center_fit_x_px',
    ] + OWNERSHIP_FIELDS + list(_own.OWNERSHIP_ROW_FIELDS)
    + list(_own.USABILITY_ROW_FIELDS) + ['unevaluable_reason'],
    restval='',
)
gate_audit_writer.writeheader()
ownership_counts = {name: 0 for name in _own.CATEGORIES}
ownership_final_counts = {name: 0 for name in _own.CATEGORIES}
n_ownership_rejected = 0
n_ownership_recovered = 0
n_refit_attempted = 0
rejected_reasons = {}
boundary_accepted = 0

total_fit = 0
total_fail = 0
n_sbr_rejected = 0

with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    for tp_idx, (tp_name, img) in enumerate(
            dat_io.iter_timepoints(src_dir, 0, _args.source)):
        if tp_idx >= n_tp:
            break
        print(f'[{tp_idx+1}/{n_tp}] {tp_name}...', flush=True)

        img = img.astype(np.float64)
        if _pp is not None:
            img = np.ascontiguousarray(img)
            if _pp_custom:
                nlm = _args.preprocess_nlm or (0, 0, 0.0)
                _pp.preprocess_custom(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    img.shape[0], img.shape[1], img.shape[2], _args.preprocess_radius,
                    _args.preprocess_wavelet_levels, _args.preprocess_diffusion_iters, 30.0,
                    int(nlm[0]), int(nlm[1]), float(nlm[2]))
            else:
                _pp.preprocess_denoise_morpho(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    img.shape[0], img.shape[1], img.shape[2], 20)
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

        # --- Centre ownership, against the SAME effective support used to fit ---
        initial = [''] * n_cells
        final = [''] * n_cells
        refit_attempted = [0] * n_cells
        refit_success = [0] * n_cells
        passed = [0] * n_cells
        unevaluable_reason = [''] * n_cells
        usability_rows = [_own.blank_usability_row_fields()] * n_cells
        _do_usability = (_args.xy_pixel_nm > 0 and _args.z_step_nm > 0)
        for c in range(n_cells):
            if success[c] != 0:
                unevaluable_reason[c] = f'fit_failed:{int(success[c])}'
                continue
            r = results[c]
            if _args.min_sbr > 0 and r[7] > 0 and r[0] / r[7] < _args.min_sbr:
                success[c] = -4
                unevaluable_reason[c] = f'sbr_below_threshold:{r[0]/r[7]:.3f}'
                n_sbr_rejected += 1
                continue
            cat = _own.classify_center(fit_masks, c, r[1], r[2], r[3])
            initial[c] = cat
            final[c] = cat
            ownership_counts[cat] = ownership_counts.get(cat, 0) + 1
            if _do_usability:
                usability_rows[c] = _own.classify_center_usable(
                    fit_masks, c, r[1], r[2], r[3],
                    _args.xy_pixel_nm, _args.z_step_nm,
                    registered=(_chrom_calib is not None))
                passed[c] = _own.is_usable_flag(
                    usability_rows[c].get("CenterOwnershipUsable", 0))
            else:
                passed[c] = int(cat in _own.PASSING)
            if not passed[c]:
                unevaluable_reason[c] = f'center_{cat}'

        # --- strict is a FILTER, not a corrector ------------------------------
        # Under POLICY_STRICT nothing happens here: the classification above IS
        # the decision. An out-of-support centre stays exactly where the
        # unconstrained fit put it and the locus is unevaluable. The coordinate is
        # never moved, projected, clamped, snapped, or re-fitted, so a rejected
        # spot can never be laundered into an apparently valid one.
        #
        # POLICY_CONSTRAINED_EXPERIMENTAL keeps the old constrained refit for
        # research only. On real data it behaves as boundary clamping through
        # optimization (see center_ownership.POLICY_CONSTRAINED_EXPERIMENTAL), so
        # it is never selected by strict and never used by production drivers.
        if _args.center_ownership == _own.POLICY_CONSTRAINED_EXPERIMENTAL:
            to_refit = [c for c in range(n_cells)
                        if success[c] == 0 and not passed[c]]
            if to_refit:
                _own.declare_batch_fit_ex(lib)
                refit_results, refit_codes = _own.constrained_refit(
                    lib, img, fit_masks, fit_bboxes, to_refit)
                for j, c in enumerate(to_refit):
                    refit_attempted[c] = 1
                    code = int(refit_codes[j])
                    if code != 0:
                        unevaluable_reason[c] = f'center_refit_failed:{code}'
                        continue
                    rr = refit_results[j]
                    cat = _own.classify_center(fit_masks, c, rr[1], rr[2], rr[3])
                    if cat in _own.PASSING:
                        results[c] = rr
                        refit_success[c] = 1
                        final[c] = cat
                        passed[c] = 1
                        unevaluable_reason[c] = ''
                        n_ownership_recovered += 1
                    else:
                        final[c] = cat
                        unevaluable_reason[c] = f'center_refit_outside:{cat}'

        for c in range(n_cells):
            if success[c] == 0:
                ownership_final_counts[final[c]] = (
                    ownership_final_counts.get(final[c], 0) + 1)

        n_refit_attempted += sum(refit_attempted)

        for c in range(n_cells):
            if success[c] != 0:
                continue
            if not _own.emits_coordinate_row(
                    _args.center_ownership, True, bool(passed[c])):
                n_ownership_rejected += 1
                rejected_reasons[unevaluable_reason[c] or 'unknown'] = (
                    rejected_reasons.get(unevaluable_reason[c] or 'unknown', 0) + 1)
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
            row.update(_own.row_fields(
                _args.center_ownership, initial[c], final=final[c],
                refit_attempted=refit_attempted[c],
                refit_accepted=refit_success[c],
                registration_applied=_chrom_calib is not None,
                reference_nm=_REF_NM if _chrom_calib is not None else None,
                target_nm=_TGT_NM if _chrom_calib is not None else None))
            row.update(usability_rows[c])
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

        # Every attempted (WalkID, tp) gets an auditable status, including
        # failed fits and ownership rejections.
        for c in range(n_cells):
            # The SAME predicate the emission loop above used, so the ledger can
            # never claim a coordinate row that was not written.
            written = int(_own.emits_coordinate_row(
                _args.center_ownership, success[c] == 0, bool(passed[c])))
            audit = {
                'tp': tp_name,
                'WalkID': c,
                'fit_ret_code': int(success[c]),
                'localization_written': written,
                'center_ownership_initial': initial[c],
                'center_refit_attempted': refit_attempted[c],
                'center_refit_success': refit_success[c],
                'center_ownership_final': final[c],
                'center_ownership_pass': passed[c],
                'unevaluable_reason': unevaluable_reason[c] if not written else '',
            }
            if success[c] == 0:
                # Same value and same serialization as the localization row, so
                # the audit centre of a REJECTED attempt is exactly the centre
                # the verdict was decided on -- not a re-derivation.
                r = results[c]
                audit.update({
                    'center_fit_z_px': f'{r[1]:.4f}',
                    'center_fit_y_px': f'{r[2]:.4f}',
                    'center_fit_x_px': f'{r[3]:.4f}',
                })
                audit.update(_own.row_fields(
                    _args.center_ownership, initial[c], final=final[c],
                    refit_attempted=refit_attempted[c],
                    refit_accepted=refit_success[c],
                    registration_applied=_chrom_calib is not None,
                    reference_nm=_REF_NM if _chrom_calib is not None else None,
                    target_nm=_TGT_NM if _chrom_calib is not None else None))
                audit.update(usability_rows[c])
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

print(f'\nTotal: {total_fit} localizations, {total_fail} failed'
      + (f', {n_sbr_rejected} SBR-rejected (amp/bg < {_args.min_sbr})' if n_sbr_rejected else ''))
_own_total = sum(ownership_counts.values())
if _own_total:
    print(f'Centre ownership over {_own_total} successful fits '
          f'(registration={_args.chromatic_registration}, '
          f'policy={_args.center_ownership}):')
    print(f'  {"category":<20} {"initial":>9}          {"final":>9}')
    for name in _own.CATEGORIES:
        n = ownership_counts.get(name, 0)
        nf = ownership_final_counts.get(name, 0)
        print(f'  {name:<20} {n:>9,}  {n / _own_total * 100:6.2f}%  '
              f'{nf:>9,}  {nf / _own_total * 100:6.2f}%')
    if _args.center_ownership != _own.POLICY_ALLOW:
        print(f'  policy:                                  {_args.center_ownership}')
        print(f'  constrained refits attempted:            {n_refit_attempted:,}')
        print(f'  recovered into own_mask by the refit:    {n_ownership_recovered:,}')
        print(f'  rejected as unevaluable:                 {n_ownership_rejected:,}')
        for reason in sorted(rejected_reasons):
            print(f'      {reason:<34} {rejected_reasons[reason]:,}')
    if _args.center_ownership == _own.POLICY_STRICT:
        # strict must never refit. Assert it rather than trusting the branch.
        assert n_refit_attempted == 0, (
            f'strict performed {n_refit_attempted} refits; it is a filter, '
            'not a corrector')
        assert n_ownership_recovered == 0, (
            f'strict recovered {n_ownership_recovered} centres; it must not '
            'move coordinates')
        print('  strict invariant held: 0 refits, 0 coordinates moved')
print(f'Saved to {out_csv}')
gate_audit_file.close()
print(f'Saved per-attempt status audit to {gate_audit_path}')
print(f'Saved chromatic provenance to {_chrom_prov_path}')
print('Done!')
