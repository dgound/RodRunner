# RodRunner

OligoSTORM widefield imaging pipeline for *Salmonella* chromosome tracing.

Segments cells, localizes fluorescent loci in 3D, detects replication via
two-spot Gaussian fitting, corrects stage drift, classifies replication
state, and optionally merges per-location results with trace salvage.

## Installation

Requires Python >= 3.9.

```bash
git clone https://github.com/dgound/Salmonella_WF.git
cd Salmonella_WF
pip install .
```

For GPU-accelerated Cellpose segmentation:

```bash
pip install ".[gpu]"
```

The localization step compiles a small C library (`pipeline/gauss3d_fit.c`)
automatically on first run. A working C compiler (gcc) is required.

## Quick start

```bash
python pipeline.py --input /path/to/raw --output /path/to/results
```

This runs the full pipeline with tightened-analysis defaults on `.dat` input.
All detection gates (SE-corrected LCB, amplitude ratio, spot separation) are
set to their publication values out of the box.

## Pipeline steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `run_cellpose.py` | Cell segmentation (2.5D z-gated by default) |
| 2 | `split_clump_masks.py` | Clump splitting (stitched mode only) |
| 3 | `localize_probe0.py` | 3D Gaussian localization of 640 nm probe |
| 3b | `center_ownership.py` | Assign localization ownership to nearest cell |
| 4 | `localize_488.py` | Localize 488 nm probe |
| 5 | `replication_qc.py` | Two-spot replication detection with BIC model selection |
| 6 | `drift_correct.py` | Stage drift correction + movement QC |
| 7 | `classify_replication.py` | Classify replication state (Ori/Ter assignment) |
| merge | `merge_all_walks.py` | Dataset-level merge with optional trace salvage |

All scripts live under `pipeline/` (steps 1-7) or `drivers/` (merge).

## Examples

Minimal `.dat` run with default geometry:

```bash
python pipeline.py --input /data/experiment --output /data/results
```

Non-default pixel size and FOV:

```bash
python pipeline.py --input /data/novo --output /data/novo_results \
    --xy-pixel-nm 101.38 --fov-size 2304
```

Confocal dataset (different geometry, no movement QC):

```bash
python pipeline.py --input /data/confocal --output /data/confocal_out \
    --xy-pixel-nm 81.25 --z-step-nm 180 --sigma-z-max 0 \
    --replication-tp-start 0 --replication-tp-end 19 \
    --movement-policy disabled --segmentation stitched
```

Re-run from existing masks:

```bash
python pipeline.py --input /data/experiment --output /data/results \
    --skip-segmentation --mask-dir /data/previous_run
```

With trace salvage:

```bash
python pipeline.py --input /data/experiment --output /data/results \
    --salvage --dataset-id my_experiment
```

## CLI reference

### Input / output

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Root directory containing Location subdirectories |
| `--output` | (required) | Output directory for results |
| `--source` | `dat` | Input format: `dat` (raw Vutara) or `tiff` (pre-extracted) |
| `--location-glob` | `Location-*` | Glob pattern for location discovery |
| `--locations` | | Explicit location list (overrides glob) |

### Detector geometry

| Flag | Default | Description |
|------|---------|-------------|
| `--xy-pixel-nm` | `103.28` | Lateral pixel size in nm |
| `--z-step-nm` | `100.0` | Axial step size in nm |
| `--fov-size` | `1866` | FOV size in pixels (assumed square) |
| `--sigma-z-max` | `3.5` | Axial sigma cap in voxels (0 = uncapped) |

### Segmentation

| Flag | Default | Description |
|------|---------|-------------|
| `--segmentation` | `2p5d-zgated` | Mode: `2p5d-zgated`, `2p5d`, `stitched`, `3d` |
| `--skip-segmentation` | off | Use existing masks |
| `--mask-dir` | | Path to existing mask tree (with `--skip-segmentation`) |

### Detection gates

| Flag | Default | Description |
|------|---------|-------------|
| `--center-ownership` | `allow` | Centre ownership policy: `allow` or `strict` |
| `--mask-dilation-xy-px` | `3` | XY mask dilation in pixels |
| `--chromatic-registration` | `off` | Chromatic registration: `off` or `metadata` |
| `--min-spot-sep-nm` | `500` | 3D spot separation gate in nm |
| `--min-amp-ratio` | `0.5` | Amplitude ratio gate |
| `--se-correction-factor` | `10.0` | SE correction factor for LCB gate |

### Genomic layout

| Flag | Default | Description |
|------|---------|-------------|
| `--ori-tp` | `17` | OriC timepoint index (0-based) |
| `--ter-tp` | `7` | Ter timepoint index (0-based) |
| `--replication-tp-start` | `2` | First genomic locus TP (0-based) |
| `--replication-tp-end` | `21` | Last genomic locus TP (0-based) |

### Movement QC

| Flag | Default | Description |
|------|---------|-------------|
| `--movement-policy` | `xy_repeatability` | Policy: `xy_repeatability`, `xyz_repeatability_legacy`, `disabled` |
| `--movement-max-nm` | `100.0` | Movement threshold in nm |
| `--movement-pair` | `3 23` | Same-locus re-image pair (1-based TPs) |

### Merge and salvage

| Flag | Default | Description |
|------|---------|-------------|
| `--no-merge` | off | Skip dataset-level merge |
| `--salvage` | off | Run trace salvage during merge |
| `--dataset-id` | output dir name | Dataset identity for merge outputs |

## Input layout

The pipeline expects Location subdirectories under `--input`. For `.dat`
sources, each location must contain a `Raw Images/` directory with Vutara
`.dat` files. For `.tiff` sources, each location must have an `output/`
directory with per-probe TIFF stacks.

```
input/
  Location-1/
    Raw Images/
      *.dat
  Location-2/
    Raw Images/
      *.dat
  ...
```

## Output layout

```
output/
  Location-1/
    probe_0/localizations/
      localizations_640.csv
      localizations_640_replication_qc.csv
      localizations_640_classified.csv
    probe_1/localizations/
      localizations_488.csv
    probe_2/cellpose_masks/
      masks_final.tif
  Location-2/
    ...
  merged/
    all_walks_merged.csv
```

## License

See [LICENSE](LICENSE).
