#!/usr/bin/env python3
"""Per-(probe, timepoint) z-stack loader for the Salmo_WF pipeline.

Every stage consumes one (probe, timepoint) z-stack. Historically those were
per-TP TIFFs under probe_N/; this lets a stage read the same units straight from
the raw Vutara .dat batches in a `Raw Images/` folder, selected by a --source
{tiff,dat} flag.

    for tp_name, img in iter_timepoints(source_dir, probe, fmt):
        ...  # img is a [z, y, x] uint16 stack, identical to tifffile.imread(TP.tif)
"""
import glob
import json
import os
import re

import numpy as np
import tifffile

_TP_RE = re.compile(r'^TP\d+(?:_CYC\d+)?\.tif$')


def _read_dat_cfg(raw_dir):
    v = json.load(open(os.path.join(raw_dir, "data.json")))["value"]
    img, rec = v["Image"], v["Recording"]
    return dict(dimX=int(img["RecordDimX"]), dimY=int(img["RecordDimY"]),
                P=int(rec["NumProbesImaging"]), Z=int(rec["NumZPos"]),
                T=int(rec["NumTimepoints"]))


def _dat_frames(raw_dir, cfg):
    """(frame_fn, total_complete_frames): concatenated COMPLETE frames of the
    *.dat batches; stops at the first in-progress (partial-frame) file, so it is
    safe to run against a folder the microscope is still writing."""
    fpx = cfg["dimX"] * cfg["dimY"]
    maps, offs = [], [0]
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.dat"))):
        elems = os.path.getsize(f) // 2
        n = elems // fpx
        if n == 0:
            break
        maps.append(np.memmap(f, dtype=np.uint16, mode="r")[:n * fpx].reshape(n, cfg["dimY"], cfg["dimX"]))
        offs.append(offs[-1] + n)
        if elems % fpx:                      # partial trailing frame -> in-progress
            break

    def frame(g):
        fi = 0
        while fi + 1 < len(offs) and offs[fi + 1] <= g:
            fi += 1
        return maps[fi][g - offs[fi]]

    return frame, offs[-1]


def n_timepoints(source_dir, probe, fmt="tiff"):
    """Cheap count of available timepoints for one channel (no image loading)."""
    if fmt == "tiff":
        return sum(1 for p in glob.glob(os.path.join(source_dir, "TP*.tif"))
                   if _TP_RE.match(os.path.basename(p)))
    cfg = _read_dat_cfg(source_dir)
    _, total = _dat_frames(source_dir, cfg)
    return min(cfg["T"], total // (cfg["P"] * cfg["Z"]))


def timepoint_names(source_dir, probe, fmt="tiff"):
    """Return canonical time-point names without loading image stacks.

    The returned order is exactly the order used by :func:`iter_timepoints`.
    This is used by sidecar-backed stages that must validate time-point identity
    before handing large image buffers to compiled fitters.
    """
    if fmt == "tiff":
        return [
            os.path.basename(path).replace(".tif", "")
            for path in sorted(glob.glob(os.path.join(source_dir, "TP*.tif")))
            if _TP_RE.match(os.path.basename(path))
        ]
    if fmt == "dat":
        return [f"TP{tp:03d}" for tp in range(n_timepoints(source_dir, probe, fmt))]
    raise ValueError(f"unknown fmt {fmt!r}")


def iter_timepoints(source_dir, probe, fmt="tiff", glob_pattern="TP*_CYC*.tif"):
    """Lazily yield (tp_name, z-stack[z,y,x] uint16) for one channel.

    fmt="tiff": source_dir is a probe_N/ folder of TP*.tif (probe is ignored --
                the folder already selects the channel).
    fmt="dat" : source_dir is a Raw Images/ folder (data.json + *.dat); frames are
                de-interleaved (z contiguous, probe interleaved, timepoint outer:
                frame = (tp*P + probe)*Z + z) and only `probe` is yielded, one
                z-stack per timepoint.
    """
    if fmt == "tiff":
        for f in sorted(p for p in glob.glob(os.path.join(source_dir, "TP*.tif"))
                        if _TP_RE.match(os.path.basename(p))):
            yield os.path.basename(f).replace(".tif", ""), tifffile.imread(f)
    elif fmt == "dat":
        cfg = _read_dat_cfg(source_dir)
        Z, P, T = cfg["Z"], cfg["P"], cfg["T"]
        frame, total = _dat_frames(source_dir, cfg)
        for tp in range(T):
            base = (tp * P + probe) * Z
            if base + Z > total:            # incomplete trailing block
                return
            yield f"TP{tp:03d}", np.stack([np.asarray(frame(base + z)) for z in range(Z)], 0)
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
