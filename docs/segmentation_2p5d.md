# DAPI segmentation modes

The canonical pipelines support four DAPI mask/support strategies:

- `stitched` (default): Cellpose runs independently on each z plane, followed by cross-z repair and stitching. The existing clump splitter runs unless `--skip-split` is supplied.
- `2p5d`: Cellpose runs once on the z-maximum projection of the time-averaged DAPI stack. The resulting 2D instance map is repeated through every z plane so downstream 3D fitters receive their required `(z, y, x)` mask. Cross-z repair, stitching, and clump splitting are skipped.
- `2p5d-zgated`: uses the same max-projection 2D Cellpose ownership map as `2p5d`, but derives a separate contiguous DAPI-intensity z interval for every cell and time point. Localization and replication QC use the intersection of the 2D label and that z interval. The full-z `masks_final.tif` remains only a compatibility/visualization product in this mode.
- `3d`: Cellpose runs once on the complete time-averaged DAPI volume with `do_3D=True`. Its direct volumetric labels become `masks_final.tif`; per-plane stitching, cross-z repair, projection/extrusion, z gating, and clump splitting are skipped.

`stitched`, legacy `2p5d`, and `2p5d-zgated` retain their existing behavior. Dynamic z support is never activated implicitly.

Use the mode through any canonical driver:

```bash
bash drivers/run_pipeline.sh --segmentation-mode 2p5d "Location 1"
bash drivers/run_pipeline_dat.sh --segmentation-mode 2p5d Location-01
bash drivers/run_pipeline_confocal.sh --segmentation-mode 2p5d Location_1

bash drivers/run_pipeline.sh --segmentation-mode 2p5d-zgated "Location 1"
bash drivers/run_pipeline_dat.sh --segmentation-mode 2p5d-zgated Location-01
bash drivers/run_pipeline_confocal.sh --segmentation-mode 2p5d-zgated Location_1

bash drivers/run_pipeline.sh --segmentation-mode 3d "Location 1"
bash drivers/run_pipeline_dat.sh --segmentation-mode 3d Location-01
bash drivers/run_pipeline_confocal.sh --segmentation-mode 3d Location_1
```

The default gate-failure policy is `quarantine`. Select it explicitly, or choose the strict failure policy, with:

```bash
bash drivers/run_pipeline.sh --segmentation-mode 2p5d-zgated \
  --z-gate-failure-policy quarantine "Location 1"

bash drivers/run_pipeline_dat.sh --segmentation-mode 2p5d-zgated \
  --z-gate-failure-policy fail Location-01
```

With `quarantine`, a weak or ambiguous gate uses the cell's consensus interval from valid time points, or the full stack if no valid interval exists. The localization is retained and marked, but replication QC treats it as unevaluable. With `fail`, the affected localization is not fitted and is recorded in the z-gate audit table. Neither policy allows an invalid gate to provide positive multispot evidence.

Slurm launchers forward the same option:

```bash
sbatch drivers/sbatch/sbatch_dat_all.sh --segmentation-mode 2p5d
sbatch drivers/sbatch/sbatch_confocal_all.sh --segmentation-mode 2p5d

sbatch drivers/sbatch/sbatch_dat_all.sh --segmentation-mode 2p5d-zgated \
  --z-gate-failure-policy quarantine

sbatch drivers/sbatch/sbatch_dat_all.sh --segmentation-mode 3d
sbatch drivers/sbatch/sbatch_confocal_all.sh --segmentation-mode 3d
```

The DAT driver builds masks in a job-specific scratch tree and copies results to `OUT_BASE`; set a new output root for controlled reruns. The TIFF and confocal drivers process their selected source tree in place before copying results, so use a copied/prepared source tree when the earlier segmentation must remain untouched.

## Legacy 2.5D output contract

In 2.5D mode, `probe_2/cellpose_masks/` contains:

- `averaged_dapi.tif`: time-averaged 3D DAPI stack;
- `averaged_dapi_zmax.tif`: 2D z-maximum projection used by Cellpose;
- `masks_2p5d_2d.tif`: direct non-overlapping XY instance map;
- `masks_final.tif`: the 2D labels repeated through the full z depth; and
- `segmentation_2p5d_summary.json`: model, projection, filtering, shape, label-count, and z-policy provenance.

The full-z mask intentionally changes fitted support, mask volumes, background shells, and the number of voxels entering multispot model selection. Localization and replication QC must therefore be rerun; stitched-mask downstream calls are not transferable.

## Direct 3D output contract

In `3d` mode, `probe_2/cellpose_masks/` contains:

- `averaged_dapi.tif`: the time-averaged `(z, y, x)` DAPI stack passed to Cellpose;
- `masks_cellpose_3d_raw.tif`: the direct Cellpose 3D labels before the established oversized-cross-section filter;
- `masks_final.tif`: filtered, contiguous `uint16` volumetric labels consumed downstream; and
- `segmentation_3d_summary.json`: model, shape, physical sampling, anisotropy, filtering, and label-count provenance.

Cellpose anisotropy is derived as `z_step_nm / xy_pixel_nm` from each canonical driver's acquisition calibration. The 3D branch passes `z_axis=0` and `do_3D=True`; `--stitch_threshold` and `--min_z_extent` do not apply. `--skip-segmentation --segmentation-mode 3d` requires a summary that explicitly records `segmentation_mode: 3d`.

The direct 3D mask is a static support volume. Both localizers and replication QC consume it in the same static-mask manner as stitched segmentation, while replication QC retains its 3D voxel-count merged-mask baseline. Because this support differs from every other mode, all downstream stages must be rerun before comparing results.

## Z-gated output contract

In `2p5d-zgated`, the same mask directory additionally contains:

- `z_support_2p5d.npz`: versioned machine-readable arrays keyed by exact time-point name and label ID;
- `z_support_2p5d_summary.json`: gate parameters, status counts, failure policy, and provenance; and
- a `segmentation_2p5d_summary.json` record identifying `2p5d-zgated` and the sidecars.

The two localization directories contain a companion audit table for every cell/time-point pair:

- `probe_0/localizations/localizations_640_z_gate_status.csv`; and
- `probe_1/localizations/localizations_488_z_gate_status.csv`.

Successful localization rows add:

```text
z_gate_status
z_gate_fallback
z_gate_min
z_gate_max
z_gate_quarantined
z_gate_evaluable
z_gate_fit_allowed
```

Replication-QC rows contain the same gate identity and evaluability fields. A quarantined or failed gate has `replication_confirmed=0` and a `z_gate_quarantined:*` or `z_gate_failed:*` status, so the downstream classifier treats that locus as unevaluable rather than as a clean negative.

When `--skip-segmentation` is used with `2p5d-zgated`, the driver requires the 2D ownership mask, NPZ sidecar, and JSON summary. Consumers also reject missing, duplicate, reordered, or mismatched time-point/label identities. They never fall back silently to the full-z compatibility mask.

## Interpretation

The z gate limits the axial support assigned to a cell; it does not remove time points under the default quarantine policy. It also does not establish that two fitted probe maxima are biological replication. Multispot model thresholds affected by voxel count, background support, or BIC must be validated on the z-gated outputs before publication-scale interpretation.

The `3d` mode is an explicit comparison arm, not a claim that volumetric Cellpose masks are more accurate. Its instance boundaries, cell counts, localization coverage, and downstream multispot behavior must be benchmarked against the other modes before selecting it for biological analysis.
