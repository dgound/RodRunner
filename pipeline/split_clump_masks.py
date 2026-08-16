#!/usr/bin/env python3
"""Split under-segmented clump masks via blob_log-seeded watershed.

For each Location's masks_final.tif, find labels whose averaged-DAPI footprint
contains >=2 nuclei (blob_log), split them with a watershed seeded on those
nuclei, propagate the 2D split through z, and renumber into a repaired
masks_final.tif written to pipeline_outputs_split/Location_N/ (production
pipeline_outputs/ untouched). Also writes a crosswalk CSV (orig_label ->
new_labels) for traceability.

Locked params (validated on Location 10): blob_log min_sigma=1.5, max_sigma=3.5,
threshold=0.05; watershed on -gaussian(norm, 1.0).
"""
import os, glob, csv, sys, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, tifffile
from scipy.ndimage import find_objects
from skimage.feature import blob_log
from skimage.segmentation import watershed
from skimage.filters import gaussian

PO   = "/mnt/altnas/work/dimos/Salmo_WF/pipeline_outputs"
OUT  = "/mnt/altnas/work/dimos/Salmo_WF/pipeline_outputs_split"
MINS, MAXS, NS, THR, OVL, PAD = 1.5, 3.5, 6, 0.05, 0.5, 14


def split_mask_array(m, dmax):
    """Split clump labels in 3D label volume `m` using DAPI image `dmax`
    (2D, max-projected). Returns (new_uint16_volume, crosswalk_rows, n_clumps)."""
    nz, ny, nx = m.shape
    sl = find_objects(m)
    new = np.zeros_like(m, dtype=np.uint16)
    counter = 0; n_clump = 0; crosswalk = []
    present = [L for L in range(1, int(m.max()) + 1) if sl[L - 1] is not None]
    for L in present:
        s = sl[L - 1]
        y0, y1 = s[1].start, s[1].stop; x0, x1 = s[2].start, s[2].stop
        yy0 = max(0, y0 - PAD); yy1 = min(ny, y1 + PAD); xx0 = max(0, x0 - PAD); xx1 = min(nx, x1 + PAD)
        crop = dmax[yy0:yy1, xx0:xx1]
        lo, hi = np.percentile(crop, [2, 99.5]); norm = np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1)
        foot = (m[:, yy0:yy1, xx0:xx1] == L).any(axis=0)
        bl = blob_log(norm, min_sigma=MINS, max_sigma=MAXS, num_sigma=NS, threshold=THR, overlap=OVL)
        seeds = [(int(round(b[0])), int(round(b[1]))) for b in bl
                 if 0 <= int(round(b[0])) < foot.shape[0] and 0 <= int(round(b[1])) < foot.shape[1]
                 and foot[int(round(b[0])), int(round(b[1]))]]
        if len(seeds) >= 2:
            mk = np.zeros(foot.shape, np.int32)
            for k, (r, c) in enumerate(seeds, 1): mk[r, c] = k
            ws = watershed(-gaussian(norm, 1.0, preserve_range=True), mk, mask=foot)
            base = counter; zslab = m[:, yy0:yy1, xx0:xx1] == L; part = ws[None, :, :]
            for k in range(1, int(ws.max()) + 1):
                counter += 1
                new[:, yy0:yy1, xx0:xx1][zslab & (part == k)] = counter
            crosswalk.append((L, int(ws.max()), base + 1, counter)); n_clump += 1
        else:
            counter += 1
            new[:, yy0:yy1, xx0:xx1][m[:, yy0:yy1, xx0:xx1] == L] = counter
            crosswalk.append((L, 1, counter, counter))
    return new, crosswalk, n_clump


def split_file(mask_path, dapi_path, out_path):
    m = tifffile.imread(mask_path).astype(np.int32)
    dmax = tifffile.imread(dapi_path).astype(np.float32)
    if dmax.ndim == 3: dmax = dmax.max(axis=0)
    before = int(np.unique(m).size - (1 if 0 in m else 0))
    new, crosswalk, n_clump = split_mask_array(m, dmax)
    tifffile.imwrite(out_path, new)
    cw = os.path.join(os.path.dirname(out_path) or '.', 'clump_split_crosswalk.csv')
    with open(cw, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["orig_label", "n_subcells", "new_label_start", "new_label_end"]); w.writerows(crosswalk)
    after = int(new.max())
    print(f"[split] {before} -> {after} cells (+{after-before}); {n_clump} clumps split -> {out_path}", flush=True)


def split_one_location(loc_dir):
    slug = os.path.basename(loc_dir)
    mpath = f"{loc_dir}/probe_2/cellpose_masks/masks_final.tif"
    dpath = f"{loc_dir}/probe_2/cellpose_masks/averaged_dapi.tif"
    if not (os.path.isfile(mpath) and os.path.isfile(dpath)):
        print(f"[skip] {slug}: missing mask/dapi"); return None
    m = tifffile.imread(mpath).astype(np.int32)
    nz, ny, nx = m.shape
    dmax = tifffile.imread(dpath).astype(np.float32).max(axis=0)
    sl = find_objects(m)
    new = np.zeros_like(m, dtype=np.uint16)
    counter = 0
    n_clump = 0; n_extra = 0
    crosswalk = []   # (orig_label, n_subcells, new_label_start, new_label_end)
    present = [L for L in range(1, int(m.max()) + 1) if sl[L - 1] is not None]
    for L in present:
        s = sl[L - 1]
        z0, z1 = s[0].start, s[0].stop; y0, y1 = s[1].start, s[1].stop; x0, x1 = s[2].start, s[2].stop
        yy0 = max(0, y0 - PAD); yy1 = min(ny, y1 + PAD); xx0 = max(0, x0 - PAD); xx1 = min(nx, x1 + PAD)
        crop = dmax[yy0:yy1, xx0:xx1]
        lo, hi = np.percentile(crop, [2, 99.5]); norm = np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1)
        foot = (m[:, yy0:yy1, xx0:xx1] == L).any(axis=0)
        bl = blob_log(norm, min_sigma=MINS, max_sigma=MAXS, num_sigma=NS, threshold=THR, overlap=OVL)
        seeds = [(int(round(b[0])), int(round(b[1]))) for b in bl
                 if 0 <= int(round(b[0])) < foot.shape[0] and 0 <= int(round(b[1])) < foot.shape[1]
                 and foot[int(round(b[0])), int(round(b[1]))]]
        if len(seeds) >= 2:
            mk = np.zeros(foot.shape, np.int32)
            for k, (r, c) in enumerate(seeds, 1): mk[r, c] = k
            ws = watershed(-gaussian(norm, 1.0, preserve_range=True), mk, mask=foot)  # (cy,cx) -> 1..k
            ksub = int(ws.max())
            base = counter
            # propagate the 2D partition through z for voxels of this label
            zslab = m[:, yy0:yy1, xx0:xx1] == L            # (z, cy, cx)
            part = ws[None, :, :]                          # broadcast partition over z
            for k in range(1, ksub + 1):
                counter += 1
                sel = zslab & (part == k)
                new[:, yy0:yy1, xx0:xx1][sel] = counter
            crosswalk.append((L, ksub, base + 1, counter))
            n_clump += 1; n_extra += (ksub - 1)
        else:
            counter += 1
            new[:, yy0:yy1, xx0:xx1][m[:, yy0:yy1, xx0:xx1] == L] = counter
            crosswalk.append((L, 1, counter, counter))
    # write
    od = f"{OUT}/{slug}/probe_2/cellpose_masks"; os.makedirs(od, exist_ok=True)
    tifffile.imwrite(f"{od}/masks_final.tif", new)
    with open(f"{od}/clump_split_crosswalk.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["orig_label", "n_subcells", "new_label_start", "new_label_end"])
        w.writerows(crosswalk)
    print(f"{slug}: {len(present)} -> {counter} cells  (+{counter-len(present)})  | "
          f"clumps split={n_clump}, extra cells={n_extra}", flush=True)
    return (slug, len(present), counter, n_clump)


def main():
    ap = argparse.ArgumentParser(description="Split clump masks (blob_log+watershed).")
    ap.add_argument('--mask'); ap.add_argument('--dapi'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.mask and a.dapi and a.out:        # single-file mode (used by the pipeline orchestrator)
        return split_file(a.mask, a.dapi, a.out)
    # else: batch mode over pipeline_outputs (original behaviour)
    locs = sorted(glob.glob(f"{PO}/Location_*"), key=lambda s: int(s.split('_')[-1]))
    tot_before = tot_after = tot_clump = 0
    print("=== clump-split repair (blob_log+watershed) ===")
    for ld in locs:
        r = split_one_location(ld)
        if r: tot_before += r[1]; tot_after += r[2]; tot_clump += r[3]
    print("-" * 60)
    print(f"TOTAL: {tot_before} -> {tot_after} cells (+{tot_after-tot_before}); "
          f"{tot_clump} clumps split. Repaired masks under {OUT}/")


if __name__ == "__main__":
    sys.exit(main())
