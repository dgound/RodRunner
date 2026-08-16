#!/usr/bin/env python3
"""Rigorous link-detection workflow for theta / replicating objects.

A classical two-component linking number is ONLY valid for two DISJOINT closed
curves. The theta_trace.py arcs (A, B) share constructed fork-midpoint vertices,
so they are a valid THETA-GRAPH representation but NOT a valid classical link
representation. Therefore:

  * STRICT link mode builds the two daughter loops directly from the raw
    per-locus sister coordinates in theta_points_both_arm.csv (no fork vertices,
    no unreplicated arc), and is only eligible when EVERY analysis TP is fully
    duplicated (two valid replicated sisters). Ter (TP007) must be replicated
    (two sisters), not merely present as a single point.
  * PARTIAL-theta diagnostics (from theta_traced_arcs.csv) are descriptive
    geometry/crossing PROXIES only -- never a classical link, never written to
    theta_link_results.csv.

Inputs (pipeline_outputs_split/): theta_points_both_arm.csv,
theta_cells_both_arm.csv, theta_traced_arcs.csv.

Outputs:
  theta_link_candidates.csv             one row per theta cell (eligibility + reasons)
  theta_link_results.csv                one row per STRICT valid two-loop cell
  theta_partial_theta_diagnostics.csv   one row per partial theta cell (proxies)
"""
import argparse, csv, sys
import numpy as np, pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pipeline"))
import drift_correct as D

SPLIT = "/mnt/altnas/work/dimos/Salmo_WF/pipeline_outputs_split"
GB = D.GENOMIC_BIN_START
ANALYSIS_TPS = sorted(GB)                      # circular genomic order (tp_index)
N_ANALYSIS_BINS = len(ANALYSIS_TPS)            # 20
TER = D.TER_TP_DEFAULT

# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _unit_dirs(n, seed):
    rng = np.random.RandomState(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _basis(depth):
    depth = depth / np.linalg.norm(depth)
    a = np.array([1.0, 0, 0]) if abs(depth[0]) < 0.9 else np.array([0, 1.0, 0])
    u = a - depth * (a @ depth); u /= np.linalg.norm(u)
    return u, np.cross(depth, u)


def _inter_crossings(loopA, loopB, depth, signed=True):
    """Signed crossing sum between two 3D polylines projected along `depth`,
    using the standard linking sign: at each projected crossing pick the OVER
    strand (greater depth) and use eps = sign((t_over x t_under).depth).
    Returns (n_crossings, signed_sum); Lk = signed_sum / 2."""
    u, v = _basis(depth)
    PA = np.column_stack((loopA @ u, loopA @ v)); PB = np.column_stack((loopB @ u, loopB @ v))
    DA = loopA @ depth; DB = loopB @ depth
    n = 0; ssum = 0
    for i in range(len(loopA) - 1):
        a1 = PA[i]; ex, ey = PA[i+1] - a1; ta = loopA[i+1] - loopA[i]
        for j in range(len(loopB) - 1):
            b1 = PB[j]; gx, gy = PB[j+1] - b1
            den = -ex * gy + gx * ey
            if abs(den) < 1e-12:
                continue
            rx, ry = b1 - a1
            s = (-rx * gy + gx * ry) / den
            t = (ex * ry - ey * rx) / den
            if not (0.0 < s < 1.0 and 0.0 < t < 1.0):
                continue
            n += 1
            if signed:
                tb = loopB[j+1] - loopB[j]
                dA = DA[i] + s * (DA[i+1] - DA[i])
                dB = DB[j] + t * (DB[j+1] - DB[j])
                if dA >= dB:
                    ssum += 1 if (np.cross(ta, tb) @ depth) > 0 else -1
                else:
                    ssum += 1 if (np.cross(tb, ta) @ depth) > 0 else -1
    return n, ssum


def linking_number(loopA, loopB, dirs):
    """Lk for two CLOSED 3D loops over many projections.
    Returns (lk, median, values, stability_frac, min_dist_nm)."""
    vals = np.array([0.5 * _inter_crossings(loopA, loopB, d)[1] for d in dirs])
    rounded = np.round(vals).astype(int)
    if len(rounded):
        mode = np.bincount(rounded - rounded.min()).argmax() + rounded.min()
        stability = float(np.mean(rounded == mode))
    else:
        mode, stability = 0, 0.0
    return int(mode), float(np.median(vals)), vals.tolist(), stability, _min_seg_dist(loopA, loopB)


def _seg_seg_dist(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    r = p1 - q1
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c = d1 @ r
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    if a < 1e-12:
        t = np.clip(f / e, 0, 1); s = 0.0
    else:
        b = d1 @ d2; den = a * e - b * b
        s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
        t = (b * s + f) / e if e > 1e-12 else 0.0
        if t < 0: t = 0.0; s = np.clip(-c / a, 0, 1)
        elif t > 1: t = 1.0; s = np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (q1 + d2 * t)))


def _min_seg_dist(A, B, skip_shared=False):
    """Min 3D distance between segments of polylines A and B. skip_shared=True
    skips segment pairs that share an endpoint (NON-adjacent-segment distance)."""
    best = np.inf
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            if skip_shared and (np.allclose(A[i], B[j]) or np.allclose(A[i], B[j+1]) or
                                np.allclose(A[i+1], B[j]) or np.allclose(A[i+1], B[j+1])):
                continue
            best = min(best, _seg_seg_dist(A[i], A[i+1], B[j], B[j+1]))
    return float(best) if np.isfinite(best) else float("nan")


def _close(arr):
    arr = np.asarray(arr, float)
    return np.vstack([arr, arr[0]]) if len(arr) and not np.allclose(arr[0], arr[-1]) else arr


def _cyclic_len(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(np.vstack([P, P[0]]), axis=0), axis=1).sum())


# ----------------------------------------------------------------------------
# strict per-locus sister loops (from theta_points_both_arm.csv)
# ----------------------------------------------------------------------------
def load_replicated_points(points_csv):
    """Return {(Location,WalkID): {tp_index: {1:(x,y,z), 2:(x,y,z)}}} for finite
    replicated rows only. Coordinates as (x,y,z) for geometry."""
    df = pd.read_csv(points_csv)
    out = {}
    rep = df[df.region == "replicated"]
    for r in rep.itertuples():
        if int(r.copy) not in (1, 2):
            continue
        if not all(np.isfinite([r.z_nm, r.y_nm, r.x_nm])):
            continue
        out.setdefault((int(r.Location), int(r.WalkID)), {}).setdefault(int(r.tp_index), {})[int(r.copy)] = \
            (float(r.x_nm), float(r.y_nm), float(r.z_nm))
    return out


def two_copy_tps(cell_pts):
    """tp_index set that has BOTH valid sisters."""
    return {tp for tp, c in cell_pts.items() if 1 in c and 2 in c}


def build_full_sister_loops(cell_pts, circular_order):
    """Build two disjoint sister loops from per-TP sister pairs, ordered by
    circular genomic order. Sister identity assigned by nearest-neighbour
    continuity around the CYCLE; both initial assignments are tried and the one
    minimising total cyclic path length (incl. last->first closure) is kept.
    Returns (loopA, loopB) as (N,3) arrays (open; close with _close)."""
    tps = [t for t in circular_order if t in cell_pts and 1 in cell_pts[t] and 2 in cell_pts[t]]
    pairs = [(np.array(cell_pts[t][1]), np.array(cell_pts[t][2])) for t in tps]

    def thread(swap0):
        A, B = [], []
        for k, (p1, p2) in enumerate(pairs):
            if k == 0:
                a, b = (p2, p1) if swap0 else (p1, p2)
            else:
                d_same = np.linalg.norm(p1 - A[-1]) + np.linalg.norm(p2 - B[-1])
                d_swap = np.linalg.norm(p2 - A[-1]) + np.linalg.norm(p1 - B[-1])
                a, b = (p1, p2) if d_same <= d_swap else (p2, p1)
            A.append(a); B.append(b)
        return np.array(A), np.array(B)

    best = None
    for swap0 in (False, True):
        A, B = thread(swap0)
        L = _cyclic_len(A) + _cyclic_len(B)
        if best is None or L < best[0]:
            best = (L, A, B)
    return best[1], best[2]


# ----------------------------------------------------------------------------
# self-tests
# ----------------------------------------------------------------------------
def _hopf_loops(n=24, offset=1.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    c1 = np.column_stack((np.cos(t), np.sin(t), 0*t))
    c2 = np.column_stack((offset + np.cos(t), 0*t, np.sin(t)))
    return c1, c2


def _selftest(dirs):
    c1, c2 = _hopf_loops()
    far = c2 + np.array([10.0, 0, 0])
    return linking_number(_close(c1), _close(c2), dirs)[0], linking_number(_close(c1), _close(far), dirs)[0]


def _selftest_full(dirs):
    """Encode two Hopf-linked 20-pt loops as a per-TP sister-pair cell, run the
    FULL strict path (build_full_sister_loops -> linking_number)."""
    c1, c2 = _hopf_loops(n=N_ANALYSIS_BINS)
    far = c2 + np.array([10.0, 0, 0])

    def lk_from_pairs(loopL, loopR):
        cell = {tp: {1: tuple(loopL[k]), 2: tuple(loopR[k])} for k, tp in enumerate(ANALYSIS_TPS)}
        A, B = build_full_sister_loops(cell, ANALYSIS_TPS)
        return linking_number(_close(A), _close(B), dirs)[0]
    return lk_from_pairs(c1, c2), lk_from_pairs(c1, far)


# ----------------------------------------------------------------------------
def load_arcs(path):
    a = pd.read_csv(path)
    has_identity = "tp_index" in a.columns and "genomic_bp" in a.columns
    cells, dropped = {}, {}
    for (loc, w), g in a.groupby(["Location", "WalkID"]):
        arcs = {}; ndrop = 0
        for arc, ga in g.groupby("arc"):
            ga = ga.sort_values("idx")
            xyz = ga[["x_nm", "y_nm", "z_nm"]].to_numpy(float)
            fin = np.isfinite(xyz).all(axis=1)
            ndrop += int((~fin).sum())
            arcs[arc] = xyz[fin]
        cells[(int(loc), int(w))] = arcs
        dropped[(int(loc), int(w))] = ndrop
    return cells, dropped, has_identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near_complete_gap_bins", type=int, default=2)
    ap.add_argument("--min_run", type=int, default=None)
    ap.add_argument("--n_projections", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min_clearance_nm", type=float, default=50.0)
    a = ap.parse_args()
    dirs = _unit_dirs(a.n_projections, a.seed)

    h, u = _selftest(dirs)
    fh, fu = _selftest_full(dirs)
    print(f"[selftest] linking core: Hopf={h} unlink={u} (expect +/-1, 0)")
    print(f"[selftest] full strict path: Hopf={fh} unlink={fu} (expect +/-1, 0)")
    if abs(h) != 1 or u != 0 or abs(fh) != 1 or fu != 0:
        print("[selftest] WARNING: self-test failed!")

    cells_df = pd.read_csv(f"{SPLIT}/theta_cells_both_arm.csv")
    rep_pts = load_replicated_points(f"{SPLIT}/theta_points_both_arm.csv")
    arcs, dropped, has_identity = load_arcs(f"{SPLIT}/theta_traced_arcs.csv")
    if not has_identity:
        print("[warn] theta_traced_arcs.csv lacks tp_index/genomic_bp — locus identity limited.")

    cand_rows, result_rows, partial_rows = [], [], []
    n_strict = n_near = n_partial = 0
    n_ter_repl = n_missing_dup = 0
    n_strict_nonzero = n_closure_any = n_closure_stable_nz = 0

    for c in cells_df.itertuples():
        loc, w = int(c.Location), int(c.WalkID)
        ra, rb = int(c.run_armA), int(c.run_armB)
        min_run = min(ra, rb)
        cp = rep_pts.get((loc, w), {})
        twoc = two_copy_tps(cp)
        n_two = len(twoc)
        n_missing = N_ANALYSIS_BINS - n_two
        ter_repl = TER in twoc
        strict_eligible = (n_missing == 0)
        if n_missing > 0:
            n_missing_dup += 1
        if ter_repl:
            n_ter_repl += 1

        if strict_eligible:
            klass = "strict_eligible"; reason = ""
            n_strict += 1
        elif n_missing <= a.near_complete_gap_bins and (a.min_run is None or min_run >= a.min_run):
            klass = "near_complete"; reason = f"{n_missing} TPs lack two sisters (<= {a.near_complete_gap_bins})"
            n_near += 1; n_partial += 1
        else:
            klass = "partial"; reason = f"{n_missing} TPs lack two sisters; Ori-proximal bubble (min_run={min_run})"
            n_partial += 1

        cand_rows.append(dict(
            Location=loc, WalkID=w, run_armA=ra, run_armB=rb, min_run=min_run,
            n_replicated_tps_with_two_copies=n_two, n_missing_replicated_tps=n_missing,
            ter_replicated=int(ter_repl), strict_eligible=int(strict_eligible),
            candidate_class=klass, strict_ineligible_reason=("" if strict_eligible else reason)))

        # ---- STRICT: per-locus sister loops (no fork vertices, no unrep arc) ----
        if strict_eligible:
            A, B = build_full_sister_loops(cp, ANALYSIS_TPS)
            lk, lkmed, lkvals, stab, mind = linking_number(_close(A), _close(B), dirs)
            disjoint = mind >= a.min_clearance_nm
            ts = "ok" if (stab >= 0.99 and disjoint) else "low_clearance_or_not_disjoint"
            if lk != 0:
                n_strict_nonzero += 1
            result_rows.append(dict(
                Location=loc, WalkID=w, linking_number=lk, linking_number_median=round(lkmed, 3),
                linking_number_values=";".join(f"{v:.1f}" for v in lkvals[:20]) + ("..." if len(lkvals) > 20 else ""),
                projection_stability=round(stab, 3), min_intercomponent_distance_nm=round(mind, 1),
                trace_status=ts))
            continue

        # ---- PARTIAL diagnostics (proxies from theta_traced_arcs.csv) ----
        ac = arcs.get((loc, w))
        if ac is None or any(k not in ac for k in ("unrep", "A", "B")):
            continue
        U, A_open, B_open = ac["unrep"], ac["A"], ac["B"]
        ndrop = dropped.get((loc, w), 0)

        def arc_cross(P, Q):
            ns, ss = [], []
            for d in dirs:
                n, s = _inter_crossings(P, Q, d, signed=True)
                ns.append(n); ss.append(s)
            return int(np.median(ns)), float(np.median(np.abs(ss)))
        cAB, sAB = arc_cross(A_open, B_open)
        cAU, _ = arc_cross(A_open, U)
        cBU, _ = arc_cross(B_open, U)
        dAB = _min_seg_dist(A_open, B_open, skip_shared=True)
        dAU = _min_seg_dist(A_open, U, skip_shared=True)
        dBU = _min_seg_dist(B_open, U, skip_shared=True)
        braided = bool(sAB >= 1)

        # closure-MODEL sensitivity (NOT a classical link): close strands A,B into
        # loops 3 ways; report whether the conclusion is stable across models.
        closure = {}
        cen = np.vstack([A_open, B_open]).mean(axis=0)
        rad = np.linalg.norm(np.vstack([A_open, B_open]) - cen, axis=1).max() + 1.0
        eps = np.array([1e-3, 1e-3, 1e-3])
        for name, via in (("direct", None),
                          ("centroid", cen + (A_open[0] - cen) / (np.linalg.norm(A_open[0] - cen) + 1e-9) * 3 * rad),
                          ("farfield", cen + np.array([0, 0, 1e6]))):
            if via is None:
                loopA = _close(A_open); loopB = _close(B_open + eps)
            else:
                loopA = _close(np.vstack([A_open, via])); loopB = _close(np.vstack([B_open + eps, via + eps]))
            closure[name] = linking_number(loopA, loopB, dirs)[0]
        cvals = list(closure.values())
        closure_stable = len(set(cvals)) == 1
        if any(v != 0 for v in cvals):
            n_closure_any += 1
            if closure_stable:
                n_closure_stable_nz += 1

        partial_rows.append(dict(
            Location=loc, WalkID=w, run_armA=ra, run_armB=rb,
            n_missing_vertices_dropped=ndrop, proxy_status=("ok" if ndrop == 0 else "ok_dropped_missing"),
            n_cross_AB=cAB, n_cross_A_unrep=cAU, n_cross_B_unrep=cBU, braided_AB=int(braided),
            min_dist_AB_nm=round(dAB, 1), min_dist_A_unrep_nm=round(dAU, 1), min_dist_B_unrep_nm=round(dBU, 1),
            closure_link_direct=closure["direct"], closure_link_centroid=closure["centroid"],
            closure_link_farfield=closure["farfield"], closure_stable=int(closure_stable),
            trace_status="proxy_only"))

    def write(path, rows, cols):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
        print(f"Wrote {path}  ({len(rows)} rows)")
    write(f"{SPLIT}/theta_link_candidates.csv", cand_rows,
          ["Location", "WalkID", "run_armA", "run_armB", "min_run",
           "n_replicated_tps_with_two_copies", "n_missing_replicated_tps", "ter_replicated",
           "strict_eligible", "candidate_class", "strict_ineligible_reason"])
    write(f"{SPLIT}/theta_link_results.csv", result_rows,
          ["Location", "WalkID", "linking_number", "linking_number_median", "linking_number_values",
           "projection_stability", "min_intercomponent_distance_nm", "trace_status"])
    write(f"{SPLIT}/theta_partial_theta_diagnostics.csv", partial_rows,
          ["Location", "WalkID", "run_armA", "run_armB", "n_missing_vertices_dropped", "proxy_status",
           "n_cross_AB", "n_cross_A_unrep", "n_cross_B_unrep", "braided_AB",
           "min_dist_AB_nm", "min_dist_A_unrep_nm", "min_dist_B_unrep_nm",
           "closure_link_direct", "closure_link_centroid", "closure_link_farfield", "closure_stable", "trace_status"])

    print("\n=== validation summary ===")
    print(f"total theta cells                  : {len(cells_df)}")
    print(f"strict eligible cells (all TPs dup): {n_strict}")
    print(f"cells with TP007 replicated        : {n_ter_repl}")
    print(f"cells missing >=1 duplicated TP    : {n_missing_dup}")
    print(f"near-complete candidates           : {n_near}  (<= {a.near_complete_gap_bins} missing dup TPs"
          + (f", min_run >= {a.min_run}" if a.min_run is not None else "") + ")")
    print(f"partial-only cells                 : {n_partial}")
    print(f"strict cells with nonzero Lk       : {n_strict_nonzero}")
    print(f"partial: closure-STABLE nonzero proxy : {n_closure_stable_nz}  (hypothesis-generating, NOT a classical link)")
    print(f"partial: any closure nonzero          : {n_closure_any}  ({n_closure_any - n_closure_stable_nz} closure-DEPENDENT = artifact)")
    if n_strict == 0:
        print("\nNOTE: ZERO strict-eligible cells — no fully-duplicated two-loop object exists,")
        print("      so a classical linking number was NOT applicable to any cell. 'No links'")
        print("      means 'no fully-duplicated objects', not 'links tested and absent'.")
    cdf = pd.DataFrame(cand_rows).sort_values(["n_missing_replicated_tps", "min_run"], ascending=[True, False])
    print("\ntop candidates for manual review (cross-ref inspect_doublet.py):")
    print(cdf.head(10)[["Location", "WalkID", "run_armA", "run_armB", "min_run",
                        "n_replicated_tps_with_two_copies", "n_missing_replicated_tps",
                        "ter_replicated", "candidate_class"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
