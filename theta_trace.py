#!/usr/bin/env python3
"""Theta-curve trace for true-replicating ('both'-arm) cells.

Consumes the plumbing tables (theta_extract.py):
  pipeline_outputs_split/theta_points_both_arm.csv   (drift-corrected 3D coords + status)
  pipeline_outputs_split/theta_cells_both_arm.csv    (per-cell arm runs)

Builds, per cell, the replication-bubble theta-graph: two fork branch points
joined by three arcs -- unreplicated (fork1->Ter->fork2, single copy) and the
two sister strands A/B (fork1->Ori->fork2). Sister-strand assignment is
nearest-neighbour along the replicated arc.

Robustness contract (see CLAUDE_THETA_FIX_PROMPT.md):
  * Locus identity preserved: traced arcs carry tp / tp_index / genomic_bp /
    vertex_type, so idx is NOT a compressed complete genomic sequence.
  * Missing single-copy bins are emitted as NaN rows (vertex_type=missing), not
    silently dropped.
  * Fork vertices are computed ONLY from valid two-sister coordinates at the
    outermost split bins. A cell whose fork (or any interior replicated bin)
    lacks a valid sister pair is UNTRACEABLE -> excluded from the arcs CSV and
    recorded in theta_trace_diagnostics.csv.
  * No None is ever appended to an arc.

Outputs (under pipeline_outputs_split/):
  theta_traced_arcs.csv         ordered arc vertices (traceable cells only)
  theta_trace_diagnostics.csv   one row per both-arm cell (status + missing bins)
  + theta_trace_gallery.pdf     3D views (largest traceable bubbles)
"""
import csv, sys, math
import numpy as np, pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pipeline"))
import drift_correct as D

SPLIT = "/mnt/altnas/work/dimos/Salmo_WF/pipeline_outputs_split"
PTS = f"{SPLIT}/theta_points_both_arm.csv"
CELLS = f"{SPLIT}/theta_cells_both_arm.csv"
GB = D.GENOMIC_BIN_START
ARMA, ARMB = D.build_replichores(D.ORI_TP_DEFAULT, D.TER_TP_DEFAULT)
ORI, TER = D.ORI_TP_DEFAULT, D.TER_TP_DEFAULT
CIRC = ARMA + ARMB[1:-1][::-1]          # 20 analysis bins, Ori at index 0
N_GALLERY = 12
NAN = float("nan")


def _d(a, b):
    return math.dist(a, b)


def _finite(*vals):
    return all(v is not None and np.isfinite(v) for v in vals)


def trace_cell(pts_cell, run_a, run_b):
    """Return (arcs, gallery, diag).
    arcs: {arc_name: [vertex dict]} or None if untraceable.
    gallery: dict for plotting or None.
    diag: per-cell diagnostics dict (always returned)."""
    single, rep = {}, {}
    for r in pts_cell.itertuples():
        ti = int(r.tp_index)
        if r.region == "single" and _finite(r.z_nm, r.y_nm, r.x_nm):
            single[ti] = (float(r.z_nm), float(r.y_nm), float(r.x_nm))
        elif r.region == "replicated" and int(r.copy) in (1, 2) and _finite(r.z_nm, r.y_nm, r.x_nm):
            rep.setdefault(ti, {})[int(r.copy)] = (float(r.z_nm), float(r.y_nm), float(r.x_nm))

    rep_order = list(reversed(CIRC[0:run_a + 1])) + CIRC[-1:-run_b - 1:-1]
    unrep_order = CIRC[run_a + 1: len(CIRC) - run_b]
    rep_valid = {ti for ti in rep if 1 in rep[ti] and 2 in rep[ti]}

    invalid_rep = [ti for ti in rep_order if ti not in rep_valid]
    missing_single = [ti for ti in unrep_order if ti not in single]
    fork1_bin, fork2_bin = rep_order[0], rep_order[-1]
    fork_ok = fork1_bin in rep_valid and fork2_bin in rep_valid
    interior_bad = [ti for ti in invalid_rep if ti not in (fork1_bin, fork2_bin)]

    diag = dict(
        run_armA=run_a, run_armB=run_b,
        n_expected_bins=len(rep_order) + len(unrep_order),
        n_present_bins=(len(rep_order) - len(invalid_rep)) + (len(unrep_order) - len(missing_single)),
        n_missing_single=len(missing_single),
        n_missing_replicated=len(invalid_rep),
        missing_tp_indices=";".join(f"TP{ti:03d}" for ti in sorted(set(invalid_rep) | set(missing_single))),
        ter_present=int(TER in single),
        ori_present=int(ORI in rep_valid),
        min_sep_nm=NAN,
    )

    if not fork_ok:
        bad = [b for b in (fork1_bin, fork2_bin) if b not in rep_valid]
        diag["trace_status"] = "untraceable_missing_fork"
        diag["trace_message"] = "fork bin(s) lack valid sister pair: " + ",".join(f"TP{b:03d}" for b in bad)
        return None, None, diag
    if interior_bad:
        diag["trace_status"] = "untraceable_missing_replicated"
        diag["trace_message"] = "interior replicated bin(s) lack valid sister pair: " + \
            ",".join(f"TP{b:03d}" for b in interior_bad)
        return None, None, diag

    # ---- traceable: assign strands A/B along rep_order (all bins valid) ----
    A, B, seps, prevA, prevB = [], [], [], None, None
    for ti in rep_order:
        s1, s2 = rep[ti][1], rep[ti][2]
        seps.append(_d(s1, s2))
        if prevA is None:
            a, b = s1, s2
        else:
            a, b = (s1, s2) if _d(s1, prevA) + _d(s2, prevB) <= \
                               _d(s2, prevA) + _d(s1, prevB) else (s2, s1)
        A.append(a); B.append(b); prevA, prevB = a, b
    fork1 = tuple(np.mean([A[0], B[0]], axis=0))
    fork2 = tuple(np.mean([A[-1], B[-1]], axis=0))
    diag["min_sep_nm"] = min(seps)
    diag["trace_status"] = "ok_with_missing_single" if missing_single else "ok"
    diag["trace_message"] = ("missing single bins: " + ",".join(f"TP{ti:03d}" for ti in missing_single)) \
        if missing_single else "complete"

    def V(vtype, ti, coord):
        gbp = GB.get(ti, "") if ti is not None else ""
        tp = f"TP{ti:03d}" if ti is not None else ""
        z, y, x = coord if coord is not None else (NAN, NAN, NAN)
        return dict(vertex_type=vtype, tp=tp, tp_index=("" if ti is None else ti),
                    genomic_bp=gbp, z=z, y=y, x=x)

    unrep_v = [V("fork1", None, fork1)]
    for ti in unrep_order:
        unrep_v.append(V("locus", ti, single.get(ti)) if ti in single else V("missing", ti, None))
    unrep_v.append(V("fork2", None, fork2))

    A_v = [V("fork1", None, fork1)] + [V("locus", ti, A[k]) for k, ti in enumerate(rep_order)] + [V("fork2", None, fork2)]
    B_v = [V("fork1", None, fork1)] + [V("locus", ti, B[k]) for k, ti in enumerate(rep_order)] + [V("fork2", None, fork2)]

    arcs = {"unrep": unrep_v, "A": A_v, "B": B_v}
    gallery = dict(fork1=fork1, fork2=fork2,
                   ori=tuple(np.mean([rep[ORI][1], rep[ORI][2]], axis=0)) if ORI in rep_valid else None,
                   ter=single.get(TER), min_sep=min(seps),
                   arcs={k: np.array([(v["z"], v["y"], v["x"]) for v in vs], float) for k, vs in arcs.items()})
    return arcs, gallery, diag


def main():
    pts = pd.read_csv(PTS)
    cells = pd.read_csv(CELLS).sort_values(["run_armA", "run_armB"], ascending=False)

    arcs_path = f"{SPLIT}/theta_traced_arcs.csv"
    diag_path = f"{SPLIT}/theta_trace_diagnostics.csv"
    gallery_cells = []
    n_traced = 0
    with open(arcs_path, "w", newline="") as af, open(diag_path, "w", newline="") as df:
        aw = csv.writer(af); dw = csv.writer(df)
        aw.writerow(["Location", "WalkID", "arc", "idx", "tp", "tp_index",
                     "genomic_bp", "vertex_type", "z_nm", "y_nm", "x_nm"])
        dw.writerow(["Location", "WalkID", "run_armA", "run_armB", "n_expected_bins",
                     "n_present_bins", "n_missing_single", "n_missing_replicated",
                     "missing_tp_indices", "ter_present", "ori_present",
                     "trace_status", "trace_message", "min_sep_nm"])
        for c in cells.itertuples():
            pc = pts[(pts.Location == c.Location) & (pts.WalkID == c.WalkID)]
            arcs, gallery, diag = trace_cell(pc, int(c.run_armA), int(c.run_armB))
            ms = diag["min_sep_nm"]
            dw.writerow([c.Location, c.WalkID, diag["run_armA"], diag["run_armB"],
                         diag["n_expected_bins"], diag["n_present_bins"],
                         diag["n_missing_single"], diag["n_missing_replicated"],
                         diag["missing_tp_indices"], diag["ter_present"], diag["ori_present"],
                         diag["trace_status"], diag["trace_message"],
                         "" if not np.isfinite(ms) else f"{ms:.1f}"])
            if arcs is None:
                continue
            n_traced += 1
            for arc_name, vs in arcs.items():
                for i, v in enumerate(vs):
                    aw.writerow([c.Location, c.WalkID, arc_name, i, v["tp"], v["tp_index"],
                                 v["genomic_bp"], v["vertex_type"],
                                 "" if not np.isfinite(v["z"]) else f"{v['z']:.1f}",
                                 "" if not np.isfinite(v["y"]) else f"{v['y']:.1f}",
                                 "" if not np.isfinite(v["x"]) else f"{v['x']:.1f}"])
            gallery_cells.append((c, gallery))
    print(f"Wrote {arcs_path}  ({n_traced} traceable cells)")
    print(f"Wrote {diag_path}")

    # ---- validation summary ----
    diag_df = pd.read_csv(diag_path)
    n_cells = len(diag_df)
    n_ok = int(diag_df.trace_status.isin(["ok", "ok_with_missing_single"]).sum())
    n_missing_single = int((diag_df.trace_status == "ok_with_missing_single").sum())
    n_untraceable = int(diag_df.trace_status.str.startswith("untraceable").sum())
    n_missing_ter = int((diag_df.ter_present == 0).sum())
    n_refit_fail_bins = int(pd.read_csv(CELLS).n_refit_fail.sum())
    n_cells_refit_fail = int((pd.read_csv(CELLS).n_refit_fail > 0).sum())
    print("\n=== validation summary ===")
    print(f"both-arm cells               : {n_cells}")
    print(f"  traced (ok + missing-single): {n_ok}")
    print(f"  ok_with_missing_single      : {n_missing_single}")
    print(f"  untraceable                 : {n_untraceable} "
          f"({int((diag_df.trace_status=='untraceable_missing_fork').sum())} fork, "
          f"{int((diag_df.trace_status=='untraceable_missing_replicated').sum())} replicated)")
    print(f"  cells missing Ter           : {n_missing_ter}")
    print(f"invalid refits               : {n_refit_fail_bins} bins across {n_cells_refit_fail} cells")

    # ---- 3D gallery for the largest traceable bubbles ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    sel = gallery_cells[:N_GALLERY]
    out_pdf = "/mnt/altnas/work/dimos/cellpose/theta_trace_gallery.pdf"
    per_page = 4
    with PdfPages(out_pdf) as pdf:
        for start in range(0, len(sel), per_page):
            chunk = sel[start:start + per_page]
            fig = plt.figure(figsize=(13, 11))
            for j, (c, g) in enumerate(chunk):
                ax = fig.add_subplot(2, 2, j + 1, projection="3d")
                for arc, col, lab in (("unrep", "0.55", "unreplicated"),
                                      ("A", "#1f77b4", "copy A"), ("B", "#d62728", "copy B")):
                    P = g["arcs"][arc]              # NaN rows auto-break the line
                    ax.plot(P[:, 2], P[:, 1], P[:, 0], "-o", ms=3, lw=1.3, color=col, label=lab)
                for v in (g["fork1"], g["fork2"]):
                    ax.scatter(v[2], v[1], v[0], color="k", marker="*", s=90, depthshade=False)
                if g["ori"]:
                    ax.scatter(g["ori"][2], g["ori"][1], g["ori"][0], color="green", marker="^", s=60, depthshade=False, label="Ori")
                if g["ter"]:
                    ax.scatter(g["ter"][2], g["ter"][1], g["ter"][0], color="orange", marker="s", s=50, depthshade=False, label="Ter")
                ax.set_title(f"Loc{c.Location} W{c.WalkID}  runA={c.run_armA} runB={c.run_armB}\n"
                             f"min sister sep={g['min_sep']:.0f} nm", fontsize=9)
                ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
                ax.tick_params(labelsize=6)
                if j == 0:
                    ax.legend(fontsize=7, loc="upper left")
            fig.suptitle(f"Theta-curve trace · largest bubbles · page {start//per_page + 1}",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)
    print(f"Wrote {out_pdf}  ({len(sel)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
