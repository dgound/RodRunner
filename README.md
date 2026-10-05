# RodRunner

OligoSTORM widefield imaging pipeline for *Salmonella* chromosome tracing.

Segments cells, localizes fluorescent loci in 3D, detects replication via
two-spot Gaussian fitting, corrects stage drift, classifies replication
state, merges per-location results with trace salvage, and scores the
localization quality of every walk.

## Installation

Requires Python >= 3.9.

```bash
git clone https://github.com/dgound/RodRunner.git
cd RodRunner
pip install .
```

For GPU-accelerated Cellpose segmentation:

```bash
pip install ".[gpu]"
```

The C helpers under `pipeline/` (`gauss3d_fit.c`, `preprocess3d.c`,
`replication_qc_fit.c`) are compiled automatically on first use. A working
C compiler (gcc) is required. The optional topology engine is built by hand
(see [Topology](#topology)).

## Quick start

```bash
python pipeline.py --input /path/to/raw --output /path/to/results
```

This runs the full pipeline on `.dat` input. Pixel size, z step and FOV size
are read from each dataset's `data.json`, image preprocessing is on, the
detection gates (SE-corrected LCB, amplitude ratio, spot separation) are at
their publication values, and after the per-location steps the merge runs
trace salvage and walk quality.

## Pipeline steps

Per location:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `run_cellpose.py` | Cell segmentation (2.5D z-gated by default) |
| 2 | `split_clump_masks.py` | Clump splitting (stitched mode only) |
| 3 | `localize_probe0.py` | Image preprocessing (wavelet denoise, anisotropic diffusion, background removal) and 3D Gaussian localization of the 640 nm probe |
| 3b | `center_ownership.py` | Assign localizations to cells |
| 4 | `localize_488.py` | Localize the 488 nm fiducial probe |
| 5 | `replication_qc.py` | Two-spot replication detection with BIC model selection |
| 6 | `drift_correct.py` | Stage drift correction + movement QC |
| 7 | `classify_replication.py` | Classify replication state (Ori/Ter assignment) |

After all locations:

| Step | Script | Description |
|------|--------|-------------|
| merge | `drivers/merge_all_walks.py` | Dataset-level merge; trace salvage runs by default (needs >= 2 locations) |
| walk quality | `pipeline/walk_quality.py` | Adds `WalkQuality` and SNR columns to both merged files |

Steps 1-7 and walk quality live under `pipeline/`, the merge under
`drivers/`. `--no-merge` skips both dataset-level steps.

## Examples

Minimal `.dat` run:

```bash
python pipeline.py --input /data/experiment --output /data/results
```

Override the detector geometry (normally read from `data.json`):

```bash
python pipeline.py --input /data/novo --output /data/novo_results \
    --xy-pixel-nm 101.38 --fov-size 2304
```

Centroid drift correction with chromatic registration from metadata:

```bash
python pipeline.py --input /data/experiment --output /data/results \
    --drift-xy-method centroid --chromatic-registration metadata
```

Confocal dataset (TIFF input, different geometry, no movement QC):

```bash
python pipeline.py --input /data/confocal --output /data/confocal_out \
    --source tiff --xy-pixel-nm 81.25 --z-step-nm 180 \
    --replication-tp-start 0 --replication-tp-end 19 \
    --movement-policy disabled --segmentation stitched
```

Re-run from existing masks:

```bash
python pipeline.py --input /data/experiment --output /data/results \
    --skip-segmentation --mask-dir /data/previous_run
```

Merge without trace salvage:

```bash
python pipeline.py --input /data/experiment --output /data/results --no-salvage
```

## CLI reference

### Input / output

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Root directory containing Location subdirectories |
| `--output` | (required) | Output directory for results |
| `--source` | `dat` | Input format: `dat` (raw Vutara) or `tiff` (pre-extracted stacks) |
| `--location-glob` | `Location-*` | Glob pattern for location discovery |
| `--locations` | | Explicit location list (overrides the glob) |

### Detector geometry and fitting

| Flag | Default | Description |
|------|---------|-------------|
| `--xy-pixel-nm` | from `data.json` | Lateral pixel size in nm |
| `--z-step-nm` | from `data.json` | Axial step size in nm |
| `--fov-size` | from `data.json` | FOV size in pixels (assumed square) |
| `--sigma-z-max` | `0` | Axial sigma cap in voxels (0 = uncapped) |
| `--min-sbr` | `0` | Minimum amplitude/background to accept a fit (0 = off) |

Geometry is read from `<Location>/Raw Images/data.json` for `.dat` input.
With `--source tiff`, or when no `data.json` is found, the fallback is
103.28 nm, 100 nm and 1866 px; pass the three flags explicitly in that case.

### Preprocessing

| Flag | Default | Description |
|------|---------|-------------|
| `--preprocess-radius` | `20` | Background-removal disk radius in px |
| `--preprocess-wavelet-levels` | `3` | Wavelet denoise levels (0 disables) |
| `--preprocess-diffusion-iters` | `5` | Anisotropic-diffusion iterations (0 disables) |
| `--preprocess-nlm SEARCH_R PATCH_R SIGMA` | off | Add non-local-means denoising |
| `--no-preprocess` | off | Disable preprocessing before fitting |

### Segmentation

| Flag | Default | Description |
|------|---------|-------------|
| `--segmentation` | `2p5d-zgated` | Mode: `2p5d-zgated`, `2p5d`, `stitched`, `3d` |
| `--diameter` | `20` | Expected cell diameter in pixels for Cellpose |
| `--skip-segmentation` | off | Use existing masks (in `--output` or `--mask-dir`) |
| `--mask-dir` | | Tree with existing masks, same layout as an output tree |

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

### Drift correction and movement QC

| Flag | Default | Description |
|------|---------|-------------|
| `--drift-xy-method` | `rod_registration` | XY drift estimation: `rod_registration` or `centroid` |
| `--movement-policy` | `xy_repeatability` | Policy: `xy_repeatability`, `xyz_repeatability_legacy`, `disabled` |
| `--movement-max-nm` | `100.0` | Movement threshold in nm |
| `--movement-pair` | `3 23` | Same-locus re-image pair (1-based TPs) |

### Merge and salvage

| Flag | Default | Description |
|------|---------|-------------|
| `--no-merge` | off | Skip the dataset-level merge and walk quality |
| `--no-salvage` | off | Merge without trace salvage (salvage runs by default; skipped automatically with fewer than two locations) |
| `--dataset-id` | output dir name | Prefix of the salvage output files |

## Input layout

For `.dat` sources each location holds a `Raw Images/` directory with the
Vutara `.dat` files and `data.json`:

```
input/
  Location-01/
    Raw Images/
      data.json
      *.dat
  Location-02/
    ...
```

For `tiff` sources each location holds per-probe stacks, either under an
`output/` directory or directly in the location (`probe_0` = 640 nm loci,
`probe_1` = 488 nm fiducial, `probe_2` = DAPI):

```
input/
  Location-01/
    output/
      probe_0/TP*.tif
      probe_1/TP*.tif
      probe_2/TP*.tif
```

## Output layout

```
output/
  Location-01/
    probe_0/localizations/
      localizations_640.csv
      localizations_640_center_ownership_walks.csv
      localizations_640_replication_qc.csv
      localizations_640_drift_corrected.csv
      localizations_640_drift_corrected_filtered.csv
      localizations_640_drift_corrected_full_coverage.csv
      localizations_640_replication_classification.csv
      localizations_640_walk_replication_summary.csv
    probe_1/localizations/
      localizations_488.csv
    probe_2/cellpose_masks/
      masks_final.tif
  Location-02/
    ...
  merged/
    all_locations_640_valid_walks.csv
    all_locations_640_valid_walks_nosuspect_movement.csv
    walk_quality.json
    <dataset-id>_trace_salvage_full_schema.csv
    <dataset-id>_trace_salvage_manifest.json
```

`all_locations_640_valid_walks.csv` holds every walk localized at all
genomic loci. `all_locations_640_valid_walks_nosuspect_movement.csv` is the
analysis set: the walks that pass the movement check and have no unresolved
two-spot locus after salvage. Both carry the same columns, including the
walk-quality columns:

- `WalkQuality` (0-1): mean over the genomic loci of each spot's chance of
  lying within the median neighbour-locus distance of its true position,
  learned from the same-locus re-image pair.
- `MedianSNR`, `SNRPercentile`, `SNRAgreementScore`, `FirstSNR`, `RepeatSNR`:
  signal-to-noise of the spots measured on the raw images.
- `RepeatSeparation3D_nm`: 3D distance between the first and repeat image of
  the re-imaged locus.

`walk_quality.json` records the calibration (distance limit, per-label
chances, SNR groups) for the run.

## Topology

`pipeline/detect_topology.py` detects theta curves and links on a merged
analysis set. It needs the C engine, built once by hand:

```bash
gcc -O3 -fopenmp -shared -fPIC -o pipeline/topology_engine.so pipeline/topology_engine.c -lm
```

## License

See [LICENSE](LICENSE).
