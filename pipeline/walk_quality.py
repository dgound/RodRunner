#!/usr/bin/env python3
"""Walk quality: the chance that every locus of a walk is placed within a distance limit of its true position.

Runs after the merge, with or without trace salvage. Adds the column ``WalkChanceAllLociGood`` (0-1, one value per
walk, repeated on each of its rows) to ``all_locations_640_valid_walks.csv`` and
``all_locations_640_valid_walks_nosuspect_movement.csv`` and writes ``walk_quality.json`` next to them.

Method (all inputs are the per-Location ``localizations_640_drift_corrected.csv`` files; no raw images):

1. Each localization gets one label, first match wins:
   fit failed  - non-finite error, any axis error > 2 px, sigma_z at its lower bound or at the stack depth
   odd size    - mean lateral spot width < 1.2 px or > 5 px
   faint       - integrated fit intensity (amplitude x sigma_x x sigma_y x sigma_z) in the dataset's lowest 5 %
   clean       - otherwise, split into ten equal groups by the fit's own 3D uncertainty (axis errors rescaled per
                 dataset so the repeat-pair scatter is 1 per axis)
2. Distance limit = median over full-coverage walks of the 3D distance between neighbouring loci
   (1-2, ..., (n-1)-n and n-1, the chromosome is circular), with the measurement error removed:
   limit^2 = median^2 - 2 * (one localization's 3D error)^2, the error estimated from the repeat pair.
3. The chance that a localization of each label is further than the limit from its true position is fitted on the
   repeat pair (the same locus imaged twice, --repeat_pair_1based, all walks, before the movement filter):
   P(pair further apart than the limit) = 1 - (1 - q_label_first)(1 - q_label_repeat), maximum likelihood over q.
4. WalkChanceAllLociGood = product over the walk's loci of (1 - q_label).
5. Check: on a second repeat pair if the acquisition has one (the next locus re-imaged, accepted only if its two
   images agree like a repeat), otherwise on held-out half of the walks; the result goes into walk_quality.json.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import sys

import numpy as np
from scipy.optimize import minimize

csv.field_size_limit(10 ** 9)
COLUMN = 'WalkChanceAllLociGood'
MERGED_FILES = ('all_locations_640_valid_walks.csv', 'all_locations_640_valid_walks_nosuspect_movement.csv')
FLAG_LABELS = ('failed', 'width_off', 'low_signal')
N_CLEAN = 10


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _rsd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if len(v) else float('nan')


def _tp(name):
    return name.replace('_CYC000', '')


def load(base, prefix, px, zs):
    """(Location, WalkID) -> {tp: localization dict}"""
    walks = {}
    for locdir in sorted(glob.glob(os.path.join(base, prefix + '*'))):
        loc = os.path.basename(locdir)
        path = os.path.join(locdir, 'probe_0', 'localizations', 'localizations_640_drift_corrected.csv')
        if not os.path.isfile(path):
            continue
        nz = None
        for s in ('segmentation_2p5d_summary.json', 'segmentation_3d_summary.json'):
            p = os.path.join(locdir, 'probe_2', 'cellpose_masks', s)
            if os.path.isfile(p):
                try:
                    nz = int(json.load(open(p))['input_shape_zyx'][0])
                except Exception:
                    pass
                break
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if not r['x_nm'].strip():
                    continue
                ex, ey, ez = _f(r['x_err']) * px, _f(r['y_err']) * px, _f(r['z_err']) * zs
                sx, sy, sz = _f(r['sigma_x']), _f(r['sigma_y']), _f(r['sigma_z'])
                failed = (not all(np.isfinite([ex, ey, ez]))) or max(ex, ey, ez) > 2 * px or sz <= 0.21 \
                    or (nz is not None and sz >= nz - 0.5)
                walks.setdefault((loc, r['WalkID']), {})[_tp(r['tp'])] = dict(
                    x=_f(r['x_nm']), y=_f(r['y_nm']), z=_f(r['z_nm']), ex=ex, ey=ey, ez=ez,
                    failed=bool(failed), width_off=bool(0.5 * (sx + sy) < 1.2 or 0.5 * (sx + sy) > 5.0),
                    intensity=_f(r['amplitude']) * sx * sy * sz)
    return walks


def _d3(a, b):
    return math.sqrt((b['x'] - a['x']) ** 2 + (b['y'] - a['y']) ** 2 + (b['z'] - a['z']) ** 2)


class Model:
    def __init__(self, walks, loci, rep_a, rep_b, cells=None):
        self.loci = loci
        self.W = walks
        cells = list(walks) if cells is None else cells
        inten = np.array([r['intensity'] for c in walks.values() for r in c.values()])
        inten = inten[np.isfinite(inten)]
        self.low_signal_threshold = float(np.percentile(inten, 5)) if len(inten) else float('nan')
        for c in walks.values():
            for r in c.values():
                r['low_signal'] = bool(not np.isfinite(r['intensity']) or r['intensity'] < self.low_signal_threshold)
                r['flag'] = r['failed'] or r['width_off'] or r['low_signal']
        pairs = [(walks[k][rep_a], walks[k][rep_b]) for k in cells if rep_a in walks[k] and rep_b in walks[k]]
        clean = [(a, b) for a, b in pairs if not a['flag'] and not b['flag']]
        if len(clean) < 100:
            raise ValueError(f'only {len(clean)} clean repeat pairs ({rep_a} vs {rep_b}); need at least 100')
        # per-dataset rescaling of the fit's axis errors (sets the relative weight of x, y and z)
        rx = [(b['x'] - a['x']) / math.hypot(a['ex'], b['ex']) for a, b in clean]
        ry = [(b['y'] - a['y']) / math.hypot(a['ey'], b['ey']) for a, b in clean]
        rz = [(b['z'] - a['z']) / math.hypot(a['ez'], b['ez']) for a, b in clean]
        self.kxy = math.sqrt((_rsd(rx) ** 2 + _rsd(ry) ** 2) / 2)
        self.kz = _rsd(rz)
        # one localization's 3D error, from the repeat scatter
        self.error_3d_nm = math.sqrt(sum(_rsd([b[ax] - a[ax] for a, b in clean]) ** 2 for ax in 'xyz') / 2)
        # distance limit: median neighbour distance (circular), measurement error removed
        n = len(loci)
        gaps = []
        for k in cells:
            c = walks[k]
            if not all(t in c for t in loci):
                continue
            for i in range(n):
                a, b = c[loci[i]], c[loci[(i + 1) % n]]
                if not a['flag'] and not b['flag']:
                    gaps.append(_d3(a, b))
        self.neighbour_median_nm = float(np.median(gaps))
        self.limit_nm = math.sqrt(max(0.0, self.neighbour_median_nm ** 2 - 2 * self.error_3d_nm ** 2))
        # clean groups by the rescaled fit uncertainty
        unf = [max(self._sig(a), self._sig(b)) for a, b in pairs if self._cls(a) is None and self._cls(b) is None]
        e = np.quantile(unf, np.linspace(0, 1, N_CLEAN + 1))
        e[0], e[-1] = 0.0, np.inf
        self.edges = e
        # joint maximum-likelihood fit of the per-label chance of being further than the limit
        self.labels = list(FLAG_LABELS) + [f'clean_{i + 1}' for i in range(N_CLEAN)]
        idx = {l: i for i, l in enumerate(self.labels)}
        la = np.array([idx[self.label(a)] for a, b in pairs])
        lb = np.array([idx[self.label(b)] for a, b in pairs])
        y = np.array([_d3(a, b) > self.limit_nm for a, b in pairs], float)
        self.n_pairs, self.pairs_off = len(pairs), float(y.mean())

        def nll(v):
            q = 1 / (1 + np.exp(-v))
            p = np.clip(1 - (1 - q[la]) * (1 - q[lb]), 1e-9, 1 - 1e-9)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() + 0.01 * np.sum(v ** 2)
        res = minimize(nll, np.zeros(len(self.labels)), method='L-BFGS-B')
        q = 1 / (1 + np.exp(-res.x))
        self.q = {l: float(q[i]) for i, l in enumerate(self.labels)}
        self.label_counts = {l: int(((la == i) | (lb == i)).sum()) for i, l in enumerate(self.labels)}

    def _sig(self, r):
        v = (self.kxy * r['ex']) ** 2 + (self.kxy * r['ey']) ** 2 + (self.kz * r['ez']) ** 2
        return math.sqrt(v) if np.isfinite(v) else math.inf

    @staticmethod
    def _cls(r):
        return 'failed' if r['failed'] else 'width_off' if r['width_off'] else 'low_signal' if r['low_signal'] else None

    def label(self, r):
        c = self._cls(r)
        if c:
            return c
        return f'clean_{int(np.searchsorted(self.edges, self._sig(r), side="right"))}'

    def q_of(self, r):
        return self.q.get(self.label(r), self.q['failed'])

    def chance(self, c, exclude=()):
        return float(np.prod([1 - self.q_of(c[t]) for t in self.loci if t not in exclude]))


def _auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(np.asarray(score, float))
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def check(walks, loci, rep_a, rep_b, model):
    """Second repeat pair if it behaves like a repeat, else held-out half of the walks."""
    ia, ib = int(rep_a[2:]), int(rep_b[2:])
    cand_a, cand_b = f'TP{ia + 1:03d}', f'TP{ib + 1:03d}'
    full = [k for k, c in walks.items() if all(t in c for t in loci)]
    cand = [(walks[k][cand_a], walks[k][cand_b]) for k in walks if cand_a in walks[k] and cand_b in walks[k]]
    prim = [(walks[k][rep_a], walks[k][rep_b]) for k in walks if rep_a in walks[k] and rep_b in walks[k]]
    use_second = False
    if cand_a in loci and len(cand) >= 100:
        med_c = float(np.median([_d3(a, b) for a, b in cand if not a['flag'] and not b['flag']] or [np.inf]))
        med_p = float(np.median([_d3(a, b) for a, b in prim if not a['flag'] and not b['flag']]))
        use_second = med_c <= 1.5 * med_p and med_c < 0.75 * model.neighbour_median_nm
    if use_second:
        keys = [k for k in full if cand_b in walks[k]]
        pred = [1 - (1 - model.q_of(walks[k][cand_a])) * (1 - model.q_of(walks[k][cand_b])) for k in keys]
        obs = [_d3(walks[k][cand_a], walks[k][cand_b]) > model.limit_nm for k in keys]
        ch = [model.chance(walks[k], exclude={cand_a}) for k in keys]
        return dict(method=f'second repeat pair {cand_a} vs {cand_b}', n=len(keys),
                    predicted_pairs_off=float(np.mean(pred)), observed_pairs_off=float(np.mean(obs)),
                    auc_walk_chance=_auc(-np.array(ch), obs))
    random.seed(1)
    ks = [k for k in full if rep_b in walks[k]]
    random.shuffle(ks)
    h = len(ks) // 2
    reason = 'no second repeat pair in range' if cand_a not in loci or len(cand) < 100 else \
        f'{cand_a} vs {cand_b} does not behave like a repeat'
    try:
        m2 = Model(walks, loci, rep_a, rep_b, cells=ks[:h])
    except ValueError as exc:
        return dict(method=f'not possible ({reason}; held-out half too small: {exc})', n=0,
                    predicted_pairs_off=None, observed_pairs_off=None, auc_walk_chance=None)
    test = ks[h:]
    pred = [1 - (1 - m2.q_of(walks[k][rep_a])) * (1 - m2.q_of(walks[k][rep_b])) for k in test]
    obs = [_d3(walks[k][rep_a], walks[k][rep_b]) > m2.limit_nm for k in test]
    ch = [m2.chance(walks[k], exclude={rep_a}) for k in test]
    return dict(method=f'held-out half of the walks ({reason})', n=len(test),
                predicted_pairs_off=float(np.mean(pred)), observed_pairs_off=float(np.mean(obs)),
                auc_walk_chance=_auc(-np.array(ch), obs))


def add_column(path, scores):
    """Stream-rewrite a merged CSV with the score column appended (replaced if present); atomic rename."""
    tmp = path + '.walk_quality_tmp'
    n_rows = n_scored = 0
    with open(path, newline='') as src, open(tmp, 'w', newline='') as dst:
        rd = csv.DictReader(src)
        fields = [f for f in rd.fieldnames if f != COLUMN] + [COLUMN]
        wr = csv.DictWriter(dst, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        for row in rd:
            v = scores.get((row.get('Location', ''), str(row.get('orig_ID', ''))))
            row[COLUMN] = '' if v is None else f'{v:.4f}'
            n_rows += 1
            n_scored += v is not None
            wr.writerow(row)
    os.replace(tmp, path)
    return n_rows, n_scored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base', required=True, help='pipeline output directory holding the Location folders')
    ap.add_argument('--merged', default=None, help='merged directory (default: <base>/merged)')
    ap.add_argument('--location_prefix', required=True)
    ap.add_argument('--tp_start', type=int, required=True, help='first genomic locus, 0-based TP index')
    ap.add_argument('--tp_end', type=int, required=True, help='last genomic locus, 0-based TP index')
    ap.add_argument('--repeat_pair_1based', type=int, nargs=2, required=True, metavar=('FIRST', 'REPEAT'))
    ap.add_argument('--xy_pixel_nm', type=float, required=True)
    ap.add_argument('--z_step_nm', type=float, required=True)
    a = ap.parse_args(argv)
    merged = a.merged or os.path.join(a.base, 'merged')
    loci = [f'TP{i:03d}' for i in range(a.tp_start, a.tp_end + 1)]
    rep_a, rep_b = (f'TP{i - 1:03d}' for i in a.repeat_pair_1based)
    summary = dict(column=COLUMN, loci=[loci[0], loci[-1]], repeat_pair=[rep_a, rep_b],
                   xy_pixel_nm=a.xy_pixel_nm, z_step_nm=a.z_step_nm)
    walks = load(a.base, a.location_prefix, a.xy_pixel_nm, a.z_step_nm)
    print(f'[walk_quality] {len(walks)} walks loaded; repeat pair {rep_a} vs {rep_b}')
    try:
        model = Model(walks, loci, rep_a, rep_b)
    except ValueError as exc:
        print(f'[walk_quality] SKIPPED: {exc}; {COLUMN} left blank', file=sys.stderr)
        summary['status'] = f'skipped: {exc}'
        scores = {}
    else:
        full = [k for k, c in walks.items() if all(t in c for t in loci)]
        scores = {(k[0], str(k[1])): model.chance(walks[k]) for k in full}
        v = np.array(list(scores.values()))
        summary.update(
            status='ok', distance_limit_nm=model.limit_nm, neighbour_median_nm=model.neighbour_median_nm,
            localization_error_3d_nm=model.error_3d_nm, axis_error_scale_xy=model.kxy, axis_error_scale_z=model.kz,
            low_signal_intensity_threshold=model.low_signal_threshold, repeat_pairs=model.n_pairs,
            repeat_pairs_off=model.pairs_off, chance_localization_off=model.q, label_counts=model.label_counts,
            walks_scored=len(scores), score_median=float(np.median(v)),
            score_percentiles_10_25_50_75_90=[float(np.percentile(v, p)) for p in (10, 25, 50, 75, 90)],
            check=check(walks, loci, rep_a, rep_b, model))
        if model.limit_nm < model.error_3d_nm:
            warn = (f'localization error ({model.error_3d_nm:.0f} nm) is not smaller than the distance limit '
                    f'({model.limit_nm:.0f} nm): no walk can be trusted at locus resolution, scores are near 0')
            summary['warning'] = warn
            print(f'[walk_quality] WARNING: {warn}', file=sys.stderr)
        c = summary['check']
        print(f'[walk_quality] distance limit {model.limit_nm:.0f} nm (neighbour median '
              f'{model.neighbour_median_nm:.0f} nm, localization error {model.error_3d_nm:.0f} nm); '
              f'{len(scores)} walks scored, median {np.median(v):.2f}')
        if c['n']:
            print(f'[walk_quality] check ({c["method"]}): predicted {100 * c["predicted_pairs_off"]:.1f}% '
                  f'vs observed {100 * c["observed_pairs_off"]:.1f}% pairs off, AUC {c["auc_walk_chance"]:.3f}')
        else:
            print(f'[walk_quality] check {c["method"]}')
    for name in MERGED_FILES:
        p = os.path.join(merged, name)
        if os.path.isfile(p):
            n_rows, n_scored = add_column(p, scores)
            summary.setdefault('files', {})[name] = dict(rows=n_rows, rows_scored=n_scored)
            print(f'[walk_quality] {name}: {COLUMN} written to {n_scored}/{n_rows} rows')
    with open(os.path.join(merged, 'walk_quality.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
