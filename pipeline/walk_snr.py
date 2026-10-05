"""Raw-image SNR and empirical repeat agreement for the final quality stage."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from . import dat_io
except ImportError:  # direct script execution by walk_quality.py
    import dat_io


SNR_COLUMNS = ('MedianSNR', 'SNRPercentile', 'SNRAgreementScore',
               'RepeatSeparation3D_nm', 'FirstSNR', 'RepeatSNR')
SNR_DEFINITION = (
    '(mean central 3x3x3 intensity - mean background) / background sample SD; '
    'background is the lateral annulus 5..8 pixels in the same three z planes; '
    'original pre-drift fit coordinates, original input images, probe 0')
_yy, _xx = np.mgrid[-8:9, -8:9]
_ANNULUS = (_xx ** 2 + _yy ** 2 >= 25) & (_xx ** 2 + _yy ** 2 <= 64)


def spot_snr(stack, xyz):
    """Return (SNR, status); retain negative signals, never truncate an edge crop."""
    if stack.ndim != 3:
        raise ValueError(f'SNR requires a ZYX stack, got {stack.shape}')
    if not np.isfinite(xyz).all():
        return float('nan'), 'nonfinite_center'
    x, y, z = np.rint(xyz).astype(int)
    nz, ny, nx = stack.shape
    if not (8 <= x < nx - 8 and 8 <= y < ny - 8 and 1 <= z < nz - 1):
        return float('nan'), 'edge_crop'
    cube = stack[z-1:z+2, y-8:y+9, x-8:x+9].astype(float)
    bg = cube[:, _ANNULUS]
    sd = float(bg.std(ddof=1))
    signal = float(cube[:, 7:10, 7:10].mean() - bg.mean())
    if not np.isfinite(sd) or not np.isfinite(signal):
        return float('nan'), 'nonfinite_intensity'
    if sd <= 0:
        return float('nan'), 'zero_background_sd'
    return signal / sd, 'ok'


def _stacks(input_root, location, source, needed):
    """Read only requested original probe-0 stacks, one stack at a time."""
    root = Path(input_root) / location
    if source == 'dat':
        raw = root / 'Raw Images'
        cfg = dat_io._read_dat_cfg(str(raw))
        frame, total = dat_io._dat_frames(str(raw), cfg)
        z_count, probes = cfg['Z'], cfg['P']
        for tp in sorted(needed):
            ti = int(tp[2:])
            ids = [ti * probes * z_count + z * probes if cfg['interleaved']
                   else ti * probes * z_count + z for z in range(z_count)]
            if ti < 0 or ti >= cfg['T'] or ids[-1] >= total:
                continue
            yield tp, np.stack([frame(i) for i in ids])
    else:
        import tifffile
        # Same input layout selection as pipeline.py:setup_work_dir.
        root = root / 'output' if (root / 'output').is_dir() else root
        paths = {}
        for path in sorted((root / 'probe_0').glob('TP*.tif')):
            tp = path.stem.replace('_CYC000', '')
            if tp in needed:
                if tp in paths:
                    raise ValueError(f'Ambiguous SNR input for {location}/{tp}')
                paths[tp] = path
        for tp, path in paths.items():
            yield tp, tifffile.imread(path)


def measure_spots(walks, loci, rep_a, rep_b, input_root, source):
    """Measure full-coverage walks before movement filtering, including repeats."""
    by_location = defaultdict(dict)
    for key, cell in walks.items():
        if all(tp in cell for tp in loci):
            by_location[key[0]][key] = cell
    measured = {}
    statuses = Counter()
    errors = {}
    needed = set(loci) | {rep_a, rep_b}
    for location, cells in sorted(by_location.items()):
        pending = {(key, tp) for key, cell in cells.items() for tp in needed if tp in cell}
        try:
            for tp, stack in _stacks(input_root, location, source, needed):
                for key, cell in cells.items():
                    if tp not in cell:
                        continue
                    spot = cell[tp]
                    value, status = spot_snr(stack, [spot['x_raw'], spot['y_raw'], spot['z_raw']])
                    measured[(key, tp)] = value
                    statuses[status] += 1
                    pending.discard((key, tp))
        except (OSError, KeyError, ValueError, IndexError) as exc:
            errors[location] = str(exc)
        for key_tp in pending:
            measured[key_tp] = float('nan')
        statuses['missing_image'] += len(pending)
        print(f'[walk_quality] SNR {location}: {len(cells)} full-coverage walks, '
              f'{len(pending)} unavailable image measurements', flush=True)
    return measured, dict(definition=SNR_DEFINITION, source=source,
                          input_root=str(input_root), spot_status_counts=dict(statuses),
                          image_errors=errors)


def percentile_scores(values, reference_keys):
    """Average ranks in the analysis set, shared between both CSVs.

    For reference values this equals (average rank - 1)/(n - 1).
    Other walks use the same empirical rank scale, clipped to [0, 1].
    """
    ref = np.sort([values[k] for k in reference_keys if k in values and np.isfinite(values[k])])
    result = {}
    for key, v in values.items():
        if not len(ref) or not np.isfinite(v):
            continue
        if len(ref) == 1:
            result[key] = 0.5 if v == ref[0] else float(v > ref[0])
        else:
            left, right = np.searchsorted(ref, v, side='left'), np.searchsorted(ref, v, side='right')
            result[key] = float(np.clip((left + right - 1) / (2 * (len(ref) - 1)), 0, 1))
    return result, len(ref)


def summarize(walks, loci, rep_a, rep_b, measured, analysis_keys, limit_nm):
    """Compute walk metrics and learn agreement in up to ten SNR quantile groups."""
    metrics = {}
    for key, cell in walks.items():
        if not all(tp in cell for tp in loci):
            continue
        snr = [measured.get((key, tp), float('nan')) for tp in loci]
        values = dict(MedianSNR=float(np.median(snr)) if np.isfinite(snr).all() else float('nan'),
                      FirstSNR=measured.get((key, rep_a), float('nan')),
                      RepeatSNR=measured.get((key, rep_b), float('nan')),
                      RepeatSeparation3D_nm=float('nan'))
        if rep_a in cell and rep_b in cell:
            delta = [cell[rep_b][ax] - cell[rep_a][ax] for ax in 'xyz']
            if np.isfinite(delta).all():
                values['RepeatSeparation3D_nm'] = float(np.linalg.norm(delta))
        metrics[key] = values
    medians = {k: v['MedianSNR'] for k, v in metrics.items()}
    ranks, n_reference = percentile_scores(medians, analysis_keys)
    for key, rank in ranks.items():
        metrics[key]['SNRPercentile'] = rank
    training = [v for v in metrics.values() if np.isfinite(v['MedianSNR'])
                and np.isfinite(v['RepeatSeparation3D_nm'])]
    info = dict(percentile_reference='delivered analysis set',
                percentile_reference_walks=len(analysis_keys), percentile_available_walks=n_reference,
                full_coverage_walks=len(metrics), training_available=len(training),
                training_unavailable=len(metrics) - len(training), groups=[],
                agreement_status='unavailable: no distance limit or no complete SNR/repeat pairs')
    if training and limit_nm is not None and np.isfinite(limit_nm):
        x = np.array([v['MedianSNR'] for v in training])
        d = np.array([v['RepeatSeparation3D_nm'] for v in training])
        # Identical SNR values must never be split between groups. Tied
        # quantiles collapse; constant SNR produces one group.
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        internal = []
        previous_count = 0
        for edge in edges[1:-1]:
            count = int((x < edge).sum())
            if previous_count < count < len(x):
                internal.append(edge)
                previous_count = count
        internal = np.asarray(internal)
        edges = np.r_[x.min(), internal, x.max()]
        assignments = np.searchsorted(internal, x, side='right')
        rates = {}
        for i in range(len(internal) + 1):
            take = assignments == i
            if not take.any():
                continue
            agreeing = int((d[take] <= limit_nm).sum())
            rates[i] = agreeing / int(take.sum())
            info['groups'].append(dict(group=i + 1, median_snr_low=float(edges[min(i, len(edges)-1)]),
                                       median_snr_high=float(edges[min(i+1, len(edges)-1)]),
                                       pairs=int(take.sum()), agreeing_pairs=agreeing,
                                       score=rates[i]))
        for values in metrics.values():
            if np.isfinite(values['MedianSNR']):
                i = int(np.searchsorted(internal, values['MedianSNR'], side='right'))
                if i in rates:
                    values['SNRAgreementScore'] = rates[i]
        info.update(agreement_status='ok', distance_limit_nm=float(limit_nm),
                    median_snr_edges=edges.tolist())
    return metrics, info
