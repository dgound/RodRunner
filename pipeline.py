#!/usr/bin/env python3
"""RodRunner -- single entry point.

Runs the complete oligoSTORM widefield pipeline from raw input through
segmentation, localization, replication QC, drift correction, classification,
and dataset-level merge with trace salvage.

Usage:
    python pipeline.py --input /path/to/raw --output /path/to/results

All detection gates default to the tightened analysis (SE-corrected LCB,
amp_ratio >= 0.5). Algorithm internals are hardcoded.

Minimal example (.dat input, default geometry):
    python pipeline.py --input /data/experiment --output /data/results

Non-default geometry:
    python pipeline.py --input /data/novo --output /data/novo_results \\
        --xy-pixel-nm 101.38 --fov-size 2304

Confocal (no movement QC, different TP layout):
    python pipeline.py --input /data/confocal --output /data/confocal_out \\
        --xy-pixel-nm 81.25 --z-step-nm 180 --sigma-z-max 0 \\
        --replication-tp-start 0 --replication-tp-end 19 \\
        --movement-policy disabled --segmentation stitched

Re-run from existing masks:
    python pipeline.py --input /data/experiment --output /data/results \\
        --skip-segmentation --mask-dir /data/previous_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / 'pipeline'
DRIVERS_DIR = REPO_ROOT / 'drivers'


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    io = p.add_argument_group('input / output')
    io.add_argument('--input', required=True,
                    help='Root directory containing Location subdirectories')
    io.add_argument('--output', required=True,
                    help='Output directory for results')
    io.add_argument('--source', choices=('dat', 'tiff'), default='dat',
                    help='Input format (default: dat)')
    io.add_argument('--location-glob', default='Location-*',
                    help='Glob pattern for location discovery (default: Location-*)')
    io.add_argument('--locations', nargs='+', default=None,
                    help='Explicit location list (overrides --location-glob)')

    geom = p.add_argument_group('detector geometry')
    geom.add_argument('--xy-pixel-nm', type=float, default=None,
                      help='Lateral pixel size in nm (auto-detected from data.json)')
    geom.add_argument('--z-step-nm', type=float, default=None,
                      help='Axial step size in nm (auto-detected from data.json)')
    geom.add_argument('--fov-size', type=int, default=None,
                      help='FOV size in pixels, assumed square (auto-detected from data.json)')
    geom.add_argument('--sigma-z-max', type=float, default=0,
                      help='Axial sigma cap in voxels, 0 = uncapped (default: 0)')
    geom.add_argument('--min-sbr', type=float, default=0,
                      help='Minimum amplitude/background to accept a fit, 0 = off (default: 0)')
    geom.add_argument('--no-preprocess', action='store_true',
                      help='Disable image preprocessing (denoise + morpho background removal) '
                           'before fitting. Preprocessing is on by default.')

    seg = p.add_argument_group('segmentation')
    seg.add_argument('--segmentation',
                     choices=('2p5d-zgated', '2p5d', 'stitched', '3d'),
                     default='2p5d-zgated',
                     help='Segmentation mode (default: 2p5d-zgated)')
    seg.add_argument('--diameter', type=int, default=20,
                     help='Expected cell diameter in pixels for Cellpose (default: 20)')
    seg.add_argument('--skip-segmentation', action='store_true',
                     help='Use existing masks; requires --mask-dir or masks '
                          'already in --output')
    seg.add_argument('--mask-dir', default=None,
                     help='Tree with existing masks (with --skip-segmentation)')

    gates = p.add_argument_group('detection gates')
    gates.add_argument('--center-ownership', choices=('allow', 'strict'),
                       default='allow',
                       help='Centre ownership policy (default: allow)')
    gates.add_argument('--mask-dilation-xy-px', type=int, default=3,
                       help='XY mask dilation in pixels (default: 3)')
    gates.add_argument('--chromatic-registration',
                       choices=('off', 'metadata'), default='off',
                       help='Chromatic registration mode (default: off)')
    gates.add_argument('--min-spot-sep-nm', type=float, default=500.0,
                       help='3D spot separation gate in nm (default: 500)')
    gates.add_argument('--min-amp-ratio', type=float, default=0.5,
                       help='Amplitude ratio gate (default: 0.5)')
    gates.add_argument('--se-correction-factor', type=float, default=10.0,
                       help='SE correction factor for LCB gate (default: 10.0)')

    layout = p.add_argument_group('genomic layout')
    layout.add_argument('--ori-tp', type=int, default=17,
                        help='OriC timepoint index, 0-based (default: 17)')
    layout.add_argument('--ter-tp', type=int, default=7,
                        help='Ter timepoint index, 0-based (default: 7)')
    layout.add_argument('--replication-tp-start', type=int, default=2,
                        help='First genomic locus TP index, 0-based (default: 2)')
    layout.add_argument('--replication-tp-end', type=int, default=21,
                        help='Last genomic locus TP index, 0-based (default: 21)')

    drift = p.add_argument_group('drift correction')
    drift.add_argument('--drift-xy-method',
                       choices=('rod_registration', 'centroid'),
                       default='rod_registration',
                       help='XY drift estimation method (default: rod_registration)')

    mov = p.add_argument_group('movement QC')
    mov.add_argument('--movement-policy',
                     choices=('xy_repeatability', 'xyz_repeatability_legacy',
                              'disabled'),
                     default='xy_repeatability',
                     help='Movement QC policy (default: xy_repeatability)')
    mov.add_argument('--movement-max-nm', type=float, default=100.0,
                     help='Movement threshold in nm (default: 100.0)')
    mov.add_argument('--movement-pair', type=int, nargs=2, default=[3, 23],
                     metavar=('FROM', 'TO'),
                     help='Same-locus re-image pair, 1-based TPs (default: 3 23)')

    merge = p.add_argument_group('merge and salvage')
    merge.add_argument('--no-merge', action='store_true',
                       help='Skip dataset-level merge after all locations')
    merge.add_argument('--no-salvage', action='store_true',
                       help='Skip trace salvage during merge '
                            '(salvage runs by default)')
    merge.add_argument('--dataset-id', default=None,
                       help='Dataset identity for merge outputs '
                            '(default: output directory name)')

    return p.parse_args(argv)


_GEOM_DEFAULTS = dict(xy_pixel_nm=103.28, z_step_nm=100.0, fov_size=1866)


def _auto_detect_geometry(input_dir, locations):
    """Read geometry from the first location's data.json."""
    for loc in locations:
        data_json = Path(input_dir) / loc / 'Raw Images' / 'data.json'
        if not data_json.is_file():
            continue
        try:
            with open(data_json) as f:
                d = json.load(f)
            v = d['value']
            img = v['Image']
            rec = v['Recording']
            px = round((img['PixelSizeX'] + img['PixelSizeY']) / 2, 2)
            return dict(
                xy_pixel_nm=px,
                z_step_nm=abs(rec['ZStepSize']),
                fov_size=img['DimX'],
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f'WARNING: could not parse {data_json}: {exc}',
                  file=sys.stderr)
            return None
    return None


def discover_locations(input_dir, location_glob, source):
    input_path = Path(input_dir)
    locations = []
    for d in sorted(input_path.glob(location_glob)):
        if not d.is_dir():
            continue
        if source == 'dat' and (d / 'Raw Images').is_dir():
            locations.append(d.name)
        elif source == 'tiff' and (d / 'output').is_dir():
            locations.append(d.name)
    return locations


def run_step(label, cmd, env):
    print(f'### {label} ###', flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'{label} failed (exit {result.returncode})')


def _setup_work_tree(work, loc, args):
    """Create working directory and symlink inputs as needed."""
    for sub in ('probe_0/localizations', 'probe_1/localizations',
                'probe_2/cellpose_masks'):
        (work / sub).mkdir(parents=True, exist_ok=True)

    if args.source == 'tiff':
        input_path = Path(args.input)
        tif_src = input_path / loc / 'output'
        if not tif_src.is_dir():
            tif_src = input_path / loc
        for probe in ('probe_0', 'probe_1', 'probe_2'):
            src_probe = tif_src / probe
            if not src_probe.is_dir():
                continue
            for tif in sorted(src_probe.glob('TP*.tif')):
                dst = work / probe / tif.name
                if not dst.exists():
                    dst.symlink_to(tif)


def _cleanup_tiff_symlinks(work):
    """Remove input TIFF symlinks from the work tree."""
    for probe in ('probe_0', 'probe_1', 'probe_2'):
        probe_dir = work / probe
        if not probe_dir.is_dir():
            continue
        for tif in probe_dir.glob('TP*.tif'):
            if tif.is_symlink():
                tif.unlink()


def _link_masks(work, args, slug):
    """Symlink existing masks into work tree for --skip-segmentation."""
    mask = work / 'probe_2' / 'cellpose_masks' / 'masks_final.tif'
    if mask.exists():
        return
    if not args.mask_dir:
        raise FileNotFoundError(
            f'--skip-segmentation requires masks at {mask} or --mask-dir')
    mask_src = Path(args.mask_dir) / slug / 'probe_2' / 'cellpose_masks'
    if not mask_src.is_dir():
        raise FileNotFoundError(f'No mask directory at {mask_src}')
    for f in mask_src.iterdir():
        dst = work / 'probe_2' / 'cellpose_masks' / f.name
        if not dst.exists():
            dst.symlink_to(f)
    if not mask.exists():
        raise FileNotFoundError(f'No masks_final.tif in {mask_src}')


def process_location(loc, args, work, env):
    """Run pipeline steps 1-7 for one location."""
    py = sys.executable
    pd = str(PIPELINE_DIR)
    seg = args.segmentation

    mask = str(work / 'probe_2' / 'cellpose_masks' / 'masks_final.tif')
    loc640 = str(work / 'probe_0' / 'localizations' / 'localizations_640.csv')
    p0loc = str(work / 'probe_0' / 'localizations')
    status_csv = str(work / 'probe_0' / 'localizations' /
                     'localizations_640_z_gate_status.csv')
    repqc_csv = str(work / 'probe_0' / 'localizations' /
                    'localizations_640_replication_qc.csv')

    src_args = []
    if args.source == 'dat':
        raw = str(Path(args.input) / loc / 'Raw Images')
        src_args = ['--source', 'dat', '--input_dir', raw]
    else:
        raw = str(work / 'probe_1')
        src_args = ['--source', 'tiff']

    chrom_args = ['--chromatic-registration', args.chromatic_registration]

    # --- 1. Segmentation ---
    if not args.skip_segmentation:
        run_step('1. Segmentation', [
            py, os.path.join(pd, 'run_cellpose.py'),
            '--segmentation-mode', seg,
            '--z-gate-failure-policy', 'quarantine',
            '--xy_pixel_nm', str(args.xy_pixel_nm),
            '--z_step_nm', str(args.z_step_nm),
            '--stitch_threshold', '0.5',
            '--min_z_extent', '0',
            '--diameter', str(args.diameter),
        ] + src_args, env)
    else:
        _link_masks(work, args, loc.replace(' ', '_'))
        print('### 1. Segmentation SKIPPED ###')

    # --- 2. Clump split (stitched mode only) ---
    if seg in ('2p5d', '2p5d-zgated', '3d'):
        print(f'### 2. Clump-split disabled for {seg} ###')
    elif seg == 'stitched':
        dapi = str(work / 'probe_2' / 'cellpose_masks' / 'averaged_dapi.tif')
        if os.path.isfile(dapi):
            shutil.copy2(mask, mask.replace('.tif', '.presplit.tif'))
            run_step('2. Clump split', [
                py, os.path.join(pd, 'split_clump_masks.py'),
                '--mask', mask, '--dapi', dapi, '--out', mask,
            ], env)
        else:
            print('### 2. Clump-split skipped (no averaged_dapi.tif) ###')

    # --- 3. Localize 640 ---
    run_step('3. Localize 640', [
        py, os.path.join(pd, 'localize_probe0.py'),
        '--sigma_z_max', str(args.sigma_z_max),
        '--min_sbr', str(args.min_sbr),
    ] + (['--no-preprocess'] if args.no_preprocess else []) + [
        '--segmentation-mode', seg,
        '--z-gate-failure-policy', 'quarantine',
        '--center-ownership', args.center_ownership,
        '--xy-pixel-nm', str(args.xy_pixel_nm),
        '--z-step-nm', str(args.z_step_nm),
        '--mask-dilation-xy-px', str(args.mask_dilation_xy_px),
    ] + chrom_args + src_args, env)

    # --- 3b. Center ownership ---
    run_step('3b. Center ownership', [
        py, os.path.join(pd, 'center_ownership.py'),
        '--status_csv', status_csv,
        '--replication_tp_start', str(args.replication_tp_start),
        '--replication_tp_end', str(args.replication_tp_end),
    ], env)

    # --- 4. Localize 488 ---
    run_step('4. Localize 488', [
        py, os.path.join(pd, 'localize_488.py'),
        '--segmentation-mode', seg,
        '--z-gate-failure-policy', 'quarantine',
    ] + chrom_args + src_args, env)

    # --- 5. Replication QC ---
    run_step('5. Replication QC', [
        py, os.path.join(pd, 'replication_qc.py'),
        '--input_csv', loc640,
        '--probe0_dir', str(work / 'probe_0'),
        '--mask_path', mask,
        '--outdir', p0loc,
        '--output_prefix', 'localizations_640',
        '--segmentation-mode', seg,
        '--z-gate-failure-policy', 'quarantine',
        '--center-ownership', args.center_ownership,
        '--sigma_z_max', str(args.sigma_z_max),
        '--center_shell_nm', '0.0',
        '--auto_null_bootstrap_reps', '16',
        '--auto_null_max_reps', '128',
        '--auto_fdr_q', '0.01',
        '--require_two_lobe_for_confirmation',
        '--replication_tp_start', str(args.replication_tp_start),
        '--replication_tp_end', str(args.replication_tp_end),
        '--ori_tp', str(args.ori_tp),
        '--ter_tp', str(args.ter_tp),
        '--min_spot_sep_xy_nm', '0.0',
        '--min_amp_ratio', str(args.min_amp_ratio),
        '--min_amp_recovery', '0.0',
        '--se_correction_factor', str(args.se_correction_factor),
        '--delta_bic_thresh', '20.0',
        '--min_spot_sep_nm', str(args.min_spot_sep_nm),
        '--xy_pixel_nm', str(args.xy_pixel_nm),
        '--z_step_nm', str(args.z_step_nm),
        '--merged_mask_size_factor', '3.0',
        '--qc_png',
        '--mask_dilation_xy_px', str(args.mask_dilation_xy_px),
    ] + chrom_args + src_args, env)

    # --- 6. Drift correction ---
    analysis_start = args.replication_tp_start + 1
    analysis_end = args.replication_tp_end + 1

    drift_cmd = [
        py, os.path.join(pd, 'drift_correct.py'),
        '--tp_coverage', '1.0',
        '--movement_policy', args.movement_policy,
        '--analysis_tp_start_1based', str(analysis_start),
        '--analysis_tp_end_1based', str(analysis_end),
        '--xy_pixel_nm', str(args.xy_pixel_nm),
        '--z_step_nm', str(args.z_step_nm),
        '--fov_size', str(args.fov_size),
        '--drift_xy_method', args.drift_xy_method,
        '--raw_dir', raw,
        '--mask_path', mask,
        '--source_fmt', args.source,
    ]
    if args.movement_policy != 'disabled':
        axes = 'xyz' if args.movement_policy == 'xyz_repeatability_legacy' else 'xy'
        drift_cmd += [
            '--movement_max_nm', str(args.movement_max_nm),
            '--movement_axes', axes,
            '--movement_tp_from_1based', str(args.movement_pair[0]),
            '--movement_tp_to_1based', str(args.movement_pair[1]),
        ]
    run_step('6. Drift correction', drift_cmd, env)

    # --- 7. Classify replication ---
    run_step('7. Classify replication', [
        py, os.path.join(pd, 'classify_replication.py'),
        '--replication_qc_csv', repqc_csv,
        '--outdir', p0loc,
        '--output_prefix', 'localizations_640',
        '--require_attempt_ledger',
        '--attempt_ledger_csv', status_csv,
        '--ori_tp', str(args.ori_tp),
        '--ter_tp', str(args.ter_tp),
        '--replication_tp_start', str(args.replication_tp_start),
        '--replication_tp_end', str(args.replication_tp_end),
    ], env)


def _copy_results(work, dst):
    """Copy pipeline outputs from work tree to final output directory."""
    for sub in ('probe_0/localizations', 'probe_1/localizations',
                'probe_2/cellpose_masks'):
        src = work / sub
        tgt = dst / sub
        if not src.is_dir():
            continue
        tgt.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if not f.is_file():
                continue
            if f.is_symlink():
                shutil.copy2(str(f.resolve()), str(tgt / f.name))
            else:
                shutil.copy2(str(f), str(tgt / f.name))


def run_merge(args, output_dir, env):
    loc_prefix = args.location_glob.split('*')[0]
    dataset_id = args.dataset_id or Path(output_dir).name
    merged_dir = str(Path(output_dir) / 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    cmd = [
        sys.executable, str(DRIVERS_DIR / 'merge_all_walks.py'),
        '--base', output_dir,
        '--outdir', merged_dir,
        '--location_prefix', loc_prefix,
    ]

    n_locations = sum(
        1 for d in Path(output_dir).iterdir()
        if d.is_dir() and d.name.startswith(loc_prefix))
    run_salvage = not args.no_salvage and n_locations >= 2

    if run_salvage:
        chromosome_order = ','.join(
            str(i) for i in range(args.replication_tp_start,
                                  args.replication_tp_end + 1))
        expected = sorted(
            d.name for d in Path(output_dir).iterdir()
            if d.is_dir() and d.name.startswith(loc_prefix))
        loc_list = Path(merged_dir) / '_expected_locations.json'
        with open(loc_list, 'w') as f:
            json.dump(expected, f)

        cmd += [
            '--trace_salvage',
            '--trace_salvage_dataset', dataset_id,
            '--trace_salvage_outdir', merged_dir,
            '--trace_salvage_output_prefix', dataset_id,
            '--trace_salvage_chromosome_order', chromosome_order,
            '--trace_salvage_ori_tp', str(args.ori_tp),
            '--trace_salvage_ter_tp', str(args.ter_tp),
            '--trace_salvage_xy_pixel_nm', str(args.xy_pixel_nm),
            '--trace_salvage_z_step_nm', str(args.z_step_nm),
            '--trace_salvage_expected_locations', str(loc_list),
        ]
    elif not args.no_salvage and n_locations < 2:
        print('  [salvage] skipped: need >= 2 locations for '
              'leave-one-location-out calibration')

    label = 'Merge + salvage' if run_salvage else 'Merge'
    run_step(label, cmd, env)


def main(argv=None):
    args = parse_args(argv)

    if args.locations:
        locations = args.locations
    else:
        locations = discover_locations(
            args.input, args.location_glob, args.source)
    if not locations:
        print(f'No locations found under {args.input} '
              f'matching {args.location_glob}', file=sys.stderr)
        return 1

    if args.source == 'dat':
        detected = _auto_detect_geometry(args.input, locations)
        if detected:
            for key, val in detected.items():
                if getattr(args, key) is None:
                    setattr(args, key, val)
                    print(f'auto-detected {key}={val} from data.json')
        elif any(getattr(args, k) is None for k in _GEOM_DEFAULTS):
            print('WARNING: no data.json found, falling back to hardcoded '
                  'geometry defaults', file=sys.stderr)
    for key, val in _GEOM_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    output_dir = str(Path(args.output).resolve())
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(REPO_ROOT) + (':' + pp if pp else '')

    sep = '=' * 60
    print(sep)
    print(f'RodRunner | source={args.source} '
          f'| segmentation={args.segmentation}')
    print(f'input  = {args.input}')
    print(f'output = {output_dir}')
    print(f'geometry: {args.xy_pixel_nm} nm/px, z_step={args.z_step_nm} nm, '
          f'fov={args.fov_size} px, sigma_z_max={args.sigma_z_max}'
          f', min_sbr={args.min_sbr}'
          f', preprocess={"off" if args.no_preprocess else "on"}')
    print(f'gates: amp_ratio>={args.min_amp_ratio}, '
          f'sep>={args.min_spot_sep_nm} nm, '
          f'SE_corr={args.se_correction_factor}x, '
          f'ownership={args.center_ownership}, '
          f'dilation={args.mask_dilation_xy_px} px')
    print(f'loci: TP{args.replication_tp_start:03d}..TP{args.replication_tp_end:03d}'
          f'  ori=TP{args.ori_tp:03d}  ter=TP{args.ter_tp:03d}')
    print(f'movement: {args.movement_policy}'
          + (f' ({args.movement_max_nm} nm, pair {args.movement_pair})'
             if args.movement_policy != 'disabled' else ''))
    print(f'Locations ({len(locations)}): {" ".join(locations)}')
    print(sep)

    n_done = 0
    n_fail = 0
    t0 = time.time()

    for loc in locations:
        slug = loc.replace(' ', '_')
        dst = Path(output_dir) / slug
        work = dst / '_work'

        if work.exists():
            shutil.rmtree(str(work))

        print(sep)
        print(f'[{time.strftime("%H:%M:%S")}] START {loc}')
        ts = time.time()

        try:
            _setup_work_tree(work, loc, args)
            loc_env = dict(env, PIPELINE_OUT=str(work))
            process_location(loc, args, work, loc_env)
            _cleanup_tiff_symlinks(work)
            _copy_results(work, dst)
            elapsed = int(time.time() - ts)
            print(f'[{time.strftime("%H:%M:%S")}] DONE {loc} in {elapsed}s')
            n_done += 1
        except Exception as exc:
            print(f'[FAIL] {loc}: {exc}', file=sys.stderr)
            n_fail += 1
        finally:
            if work.exists():
                shutil.rmtree(str(work), ignore_errors=True)

    print(sep)
    total = int(time.time() - t0)
    print(f'Per-location processing: done={n_done} fail={n_fail} '
          f'elapsed={total}s')

    if n_done == 0:
        print('FATAL: no locations processed.', file=sys.stderr)
        return 1

    if n_fail > 0:
        print('Skipping merge due to per-location failures.', file=sys.stderr)
        return 1

    if not args.no_merge:
        print(sep)
        try:
            run_merge(args, output_dir, env)
        except Exception as exc:
            print(f'[FAIL] Merge: {exc}', file=sys.stderr)
            return 1

    print(sep)
    print(f'Pipeline complete. Output: {output_dir}')
    print(sep)
    return 0


if __name__ == '__main__':
    sys.exit(main())
