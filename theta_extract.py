#!/usr/bin/env python3
"""Plumbing for theta-curve tracing of true-replicating ('both'-arm) cells.

For every cell whose Ori-anchored doubled run extends >=1 locus on BOTH
replichores (a genuine two-fork bubble), extract the drift-corrected 3D
coordinates of every genomic bin along the circular genome:

  * replicated-region bin (Ori-anchored contiguous two-spot run on each arm):
      TWO sister coordinates (re-fit 2-spot centres) -- but ONLY if the refit is
      VALID (status ok, converged, not pinned, finite separation that agrees
      with the production side-car spot_sep_nm within max(25 nm, 5%)). Invalid
      refits are recorded with a reason and counted in n_refit_fail; NO fake
      sister points are written.
  * every other analysis bin (the fork->Ter->fork single-copy arc):
      ONE coordinate, the existing 1-spot drift-corrected localization.

Outputs (under pipeline_outputs_split/):
  theta_points_both_arm.csv  long format, one row per (cell, bin, copy)
                             columns incl. `status` (ok / refit-fail reason)
  theta_cells_both_arm.csv   per-cell summary (arm runs, bin counts, n_refit_fail)

Coordinate transform (validated: valid re-fit sister 3D separation == side-car
spot_sep_nm): corrected_voxel = raw_voxel - drift_voxel; nm = voxel * (100 z /
108 xy). Sisters share the 1-spot drift vector (drift is a per-(TP,WalkID)
translation).
"""
import os, csv, sys
from types import SimpleNamespace
import numpy as np, pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pipeline"))
import drift_correct as D
import visualize_replication_qc_calls as V

SPLIT = "/mnt/altnas/work/dimos/Salmo_WF/pipeline_outputs_split"
SALMO = "/mnt/altnas/work/dimos/Salmo_WF"
LOCS = [1, 5, 6, 9, 10, 11, 12, 13, 14, 15]
Z_NM, XY_NM = 100.0, 103.28  # XY: Vutara config PixelSizeX/Y 102.97/103.59, mean; was 108
ARMA, ARMB = D.build_replichores(D.ORI_TP_DEFAULT, D.TER_TP_DEFAULT)
ORI = D.ORI_TP_DEFAULT
SEP_TOL_NM, SEP_TOL_FRAC = 25.0, 0.05   # refit vs side-car separation tolerance


def loc_paths(n):
    base = f"{SPLIT}/Location_{n}/probe_0/localizations"
    return SimpleNamespace(
        sidecar=f"{base}/localizations_640_replication_qc.csv",
        drift=f"{base}/localizations_640_drift_corrected.csv",
        loc640=f"{base}/localizations_640.csv",
        mask=f"{SPLIT}/Location_{n}/probe_2/cellpose_masks/masks_final.tif",
        probe0=f"{SALMO}/Location {n}/output/probe_0",
    )


def arm_run(conf, arm):
    """Loci doubled BEYOND Ori on this arm (contiguous from Ori)."""
    r = 0
    for tp in arm:                # arm[0] == ORI
        if tp in conf:
            r += 1
        else:
            break
    return r - 1


def replicated_region(conf):
    """tp_index set of the Ori-anchored contiguous doubled runs (both arms)."""
    region = {ORI}
    for arm in (ARMA, ARMB):
        for tp in arm[1:]:
            if tp in conf:
                region.add(tp)
            else:
                break
    return region


def validate_refit(ex, sidecar_sep):
    """Is this 2-spot refit usable? Returns (ok, reason)."""
    if ex is None or ex.get("fit2_params") is None:
        return False, "no_fit2"
    if ex.get("fit2_status") != "ok":
        return False, f"status:{ex.get('fit2_status')}"
    info = ex.get("fit2_info") or {}
    if "converged" in info and not info["converged"]:
        return False, "not_converged"
    if info.get("pinned", False):
        return False, "pinned"
    sep = ex.get("fit2_sep_nm")
    if sep is None or not np.isfinite(sep):
        return False, "nonfinite_sep"
    if np.isfinite(sidecar_sep):
        tol = max(SEP_TOL_NM, SEP_TOL_FRAC * sidecar_sep)
        if abs(sep - sidecar_sep) > tol:
            return False, f"sep_mismatch_{sep:.0f}vs{sidecar_sep:.0f}"
    return True, "ok"


def process_location(n, pts_writer, cells_writer):
    p = loc_paths(n)
    if not all(os.path.isfile(x) for x in (p.sidecar, p.drift, p.loc640, p.mask)):
        print(f"[skip] Location_{n}: missing inputs"); return 0
    sc = pd.read_csv(p.sidecar)
    sc["ti"] = sc["tp"].map(lambda s: int(s[2:5]))
    conf = {int(w): set(s["ti"]) for w, s in sc[sc["replication_confirmed"] == 1].groupby("WalkID")}
    both = [w for w, c in conf.items()
            if ORI in c and arm_run(c, ARMA) >= 1 and arm_run(c, ARMB) >= 1]
    if not both:
        return 0

    # per-(WalkID,ti) drift vector + 1-spot corrected nm, from drift CSV
    drift = {}
    with open(p.drift) as fh:
        for r in csv.DictReader(fh):
            drift[(int(r["WalkID"]), int(r["tp"][2:5]))] = (
                float(r["drift_z"]), float(r["drift_y"]), float(r["drift_x"]),
                float(r["z_nm"]), float(r["y_nm"]), float(r["x_nm"]))
    sc_idx = {(int(r.WalkID), int(r.ti)): r for r in sc.itertuples(index=False)}

    ren = V.CropRenderer(SimpleNamespace(
        mask_path=p.mask, localizations_csv=p.loc640, probe0_dir=p.probe0,
        pad_xy=3, pad_z=1, no_refit_2spot=False, z_step_nm=Z_NM, xy_pixel_nm=XY_NM))

    n_cells = 0
    for wid in sorted(both):
        region = replicated_region(conf[wid])
        present = sorted(ti for (w, ti) in drift if w == wid
                         and ti in D.GENOMIC_BIN_START)
        n_rep = n_sing = n_fail = 0
        for ti in present:
            gbp = D.GENOMIC_BIN_START[ti]
            dz, dy, dx, znm, ynm, xnm = drift[(wid, ti)]
            if ti in region:
                row = sc_idx.get((wid, ti))
                sidecar_sep = (float(row.spot_sep_nm)
                               if row is not None and np.isfinite(float(row.spot_sep_nm))
                               else float("nan"))
                ex = None
                if row is not None:
                    try:
                        ex = ren.build_example(pd.Series(row._asdict()))
                    except Exception as exc:
                        ex = None
                        _ex_err = f"build_example:{type(exc).__name__}"
                ok, reason = validate_refit(ex, sidecar_sep)
                if not ok:
                    # invalid refit: record reason, NO fake sisters (blank coords)
                    sep_str = f"{sidecar_sep:.1f}" if np.isfinite(sidecar_sep) else ""
                    pts_writer.writerow([n, wid, f"TP{ti:03d}", ti, gbp, "replicated", 0,
                                         "", "", "", sep_str, reason])
                    n_fail += 1
                    continue
                p2 = ex["fit2_params"]
                for copy, (zz, yy, xx) in enumerate(
                        [(p2[2], p2[3], p2[4]), (p2[5], p2[6], p2[7])], 1):
                    cz = (zz - dz) * Z_NM; cy = (yy - dy) * XY_NM; cx = (xx - dx) * XY_NM
                    pts_writer.writerow([n, wid, f"TP{ti:03d}", ti, gbp, "replicated", copy,
                                         f"{cz:.1f}", f"{cy:.1f}", f"{cx:.1f}",
                                         f"{sidecar_sep:.1f}", "ok"])
                n_rep += 1
            else:
                pts_writer.writerow([n, wid, f"TP{ti:03d}", ti, gbp, "single", 0,
                                     f"{znm:.1f}", f"{ynm:.1f}", f"{xnm:.1f}", "", "ok"])
                n_sing += 1
        cells_writer.writerow([n, wid, arm_run(conf[wid], ARMA), arm_run(conf[wid], ARMB),
                               len(region) - 1, n_rep, n_sing, n_fail, len(present)])
        n_cells += 1
    print(f"Location_{n}: {n_cells} both-arm cells extracted "
          f"({sum(1 for w in both)} candidates)")
    return n_cells


def main():
    pts_path = f"{SPLIT}/theta_points_both_arm.csv"
    cells_path = f"{SPLIT}/theta_cells_both_arm.csv"
    with open(pts_path, "w", newline="") as pf, open(cells_path, "w", newline="") as cf:
        pw = csv.writer(pf); cw = csv.writer(cf)
        pw.writerow(["Location", "WalkID", "tp", "tp_index", "genomic_bp",
                     "region", "copy", "z_nm", "y_nm", "x_nm", "spot_sep_nm", "status"])
        cw.writerow(["Location", "WalkID", "run_armA", "run_armB",
                     "n_doubled_beyond_ori", "n_replicated_bins", "n_single_bins",
                     "n_refit_fail", "n_present_bins"])
        tot = sum(process_location(n, pw, cw) for n in LOCS)
    print(f"\nTotal both-arm cells: {tot}")
    print(f"Wrote {pts_path}")
    print(f"Wrote {cells_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
