# 2.5D z-gated segmentation: implementation plan

## Objective

Add a new segmentation/localization mode that keeps the strong XY cell ownership of the current 2.5D Cellpose masks while preventing every cell mask from extending through the entire z stack during localization and replication QC.

The new mode will be named:

```text
--segmentation-mode 2p5d-zgated
```

For each time point and each 2D cell label, the pipeline will estimate a contiguous z interval from the DAPI intensity profile. Downstream fitting will use the intersection of:

1. the authoritative 2D Cellpose label in XY; and
2. that cell's time-point-specific z interval.

This is intended to reduce artifactual multispot calls caused by axially separated fluorescence being assigned to one full-depth extruded mask.

## Compatibility guarantees

- `stitched` remains the default and its segmentation, localization, and replication-QC behavior must remain unchanged.
- Existing `2p5d` remains available as the legacy full-z extrusion strategy and must remain unchanged.
- The new z-gating behavior is activated only by `2p5d-zgated`.
- Clump splitting remains enabled for `stitched` unless explicitly skipped.
- Clump splitting is disabled for both `2p5d` and `2p5d-zgated`, because Cellpose's 2D instance map is the authoritative XY ownership map in those modes.
- Old downstream results cannot be reused after selecting `2p5d-zgated`; localization and replication QC must be rerun.

## Scientific model

The 2D Cellpose segmentation answers: **which cell owns this XY pixel?**

The DAPI z gate answers: **over which z planes is that cell supported at this time point?**

The effective fitting mask for cell `L` at time point `T` is therefore:

```text
effective_mask(T, L, z, y, x) =
    [labels_2d(y, x) == L] AND [z_min(T, L) <= z <= z_max(T, L)]
```

The gate changes only axial support. It must never reassign an XY pixel from one cell to another.

## Z-gate algorithm

For every DAPI time point and every 2D cell label:

1. Compute the mean DAPI intensity in the label on each z plane.
2. Estimate per-plane background as the median intensity over unlabeled pixels.
3. Subtract that background from the label's z profile.
4. Smooth the profile along z with a Gaussian kernel of `sigma = 1` plane.
5. Estimate the profile baseline as its 10th percentile.
6. Normalize the baseline-subtracted profile by its peak value.
7. Mark planes active when normalized signal is at least `0.20`.
8. Fill internal gaps of one z plane.
9. Find contiguous active components and choose the component containing the strongest plane.
10. Expand the selected interval to a minimum width of three planes.
11. Pad the interval by one plane on each side, clipped to stack bounds.

The initial parameters will be explicit constants in one module and recorded in provenance so that they can be calibrated without changing the data contract.

### Gate quality flags

A gate is considered ambiguous when a separate active component:

- spans at least two planes; and
- contains at least 50% of the background-corrected signal of the primary component.

The implementation will also record:

- weak or flat profiles;
- missing/non-finite profiles;
- intervals clipped by the first or last z plane;
- fallback use; and
- exact reasons for quarantine or failure.

Per-plane Cellpose foreground will not be used to derive the gate. Inspection showed that it overlaps most labels across nearly the entire stack and therefore does not provide useful axial discrimination.

## Failure policy

Add:

```text
--z-gate-failure-policy quarantine|fail
```

The default will be `quarantine`.

### `quarantine`

- A weak or ambiguous time-point gate will use a per-cell consensus interval derived from that cell's valid time points.
- If the cell has no valid time-point gates, localization may fall back to the full z stack so that the trajectory is not silently discarded.
- Every such fit is marked as quarantined and records the fallback used.
- A quarantined observation is unevaluable for replication model selection and cannot independently produce a `MultiSpot` or `Replicating` call.

### `fail`

- A weak or ambiguous gate will not be localized.
- The output will contain an explicit failure/status record rather than silently omitting the observation.
- Replication QC will treat it as unevaluable.

This preserves time-point accounting while preventing uncertain axial support from becoming positive replication evidence.

## Data contract and provenance

### Existing mask products

The mode will continue to write:

- `averaged_dapi.tif`;
- `averaged_dapi_zmax.tif`;
- `masks_2p5d_2d.tif`; and
- `masks_final.tif`.

For compatibility, `masks_final.tif` may remain the 2D ownership map extruded through the full stack. In `2p5d-zgated`, it is a compatibility/visualization artifact, **not** the final fitting support.

### New z-support sidecar

Write a machine-readable sidecar:

```text
probe_2/cellpose_masks/z_support_2p5d.npz
```

It will be keyed by exact time-point names and label IDs and will contain, at minimum:

- ordered time-point names;
- ordered label IDs;
- `z_min` and `z_max` arrays;
- gate status codes;
- quarantine/fallback flags;
- peak plane and peak strength;
- selected-component signal;
- competing-component signal ratio;
- edge-clipping flags; and
- gate-algorithm/schema version.

Write a human-readable summary/provenance JSON beside it containing:

- segmentation mode and Cellpose model;
- mask and stack shapes;
- source time points;
- gate parameters;
- failure policy;
- counts by gate status and fallback type;
- software/schema version; and
- input/output identity checks where available.

### Validation on load

Consumers must reject the sidecar when any of the following occurs:

- duplicate time-point names or label IDs;
- missing or unexpected time points;
- missing or unexpected labels;
- array shapes inconsistent with the time-point/label axes;
- z bounds outside the source stack;
- mismatched XY mask shape or label identity; or
- unsupported schema version.

Matching must use exact time-point names and exact label IDs, not positional guessing.

## Implementation stages

### 1. Z-support library

Create a focused pipeline module responsible for:

- computing per-label DAPI z profiles;
- deriving and classifying gates;
- constructing per-cell consensus gates;
- saving/loading/validating the NPZ and JSON sidecars; and
- materializing a time-point-specific 3D label volume or equivalent label-specific bounds for downstream consumers.

Keep gate logic independent of TIFF, DAT, and confocal readers so all modalities use the same implementation.

### 2. Segmentation integration

Extend the canonical segmentation entry point to accept `2p5d-zgated`.

In this mode it will:

1. create the existing max-projection 2D Cellpose labels using `cpsam_v2`;
2. enumerate all downstream DAPI time points using their canonical names;
3. compute per-time-point, per-label z gates;
4. calculate consensus/fallback metadata;
5. write the mask products and z-support sidecars; and
6. report status counts and quality metrics.

The existing `stitched` and `2p5d` branches will not call the new gating path.

### 3. Localization integration

Update both localization paths (`probe0` and `488`) so that, only in `2p5d-zgated`:

- the exact time-point gate is loaded;
- fitting support is materialized from 2D ownership plus the selected z interval;
- z-aware bounding boxes are computed for that time point;
- each cell continues to be fitted independently; and
- gate provenance is added to each localization record.

Add at least these output columns:

```text
z_gate_status
z_gate_min
z_gate_max
z_gate_quarantined
```

Additional fallback/reason columns may be added if needed to preserve an auditable status trail.

In `stitched` and legacy `2p5d`, localizers must continue to use `masks_final.tif` exactly as they do now.

### 4. Replication-QC integration

In `2p5d-zgated`, replication QC must use the same effective support and gate status as localization.

- Multispot model fitting must not examine voxels outside the selected z gate.
- Quarantined or failed gates are unevaluable and cannot confirm `MultiSpot` or `Replicating`.
- The output must distinguish a clean negative result from an unevaluable result.
- The merged-mask size gate should use the authoritative 2D ownership area in this mode, because z-gated volume varies with gate width and time point.
- Thresholds affected by voxel count, background shells, BIC, or support size must be revalidated for the new mode.

The stitched branch must retain its existing 3D volume-based behavior.

### 5. Driver and Slurm wiring

Forward `2p5d-zgated` and `--z-gate-failure-policy` through:

- the canonical TIFF driver;
- the canonical DAT driver;
- the canonical confocal driver; and
- their Slurm wrappers.

Driver behavior must include:

- disabling clump splitting for `2p5d-zgated`;
- forwarding the selected failure policy to all relevant stages;
- recording the selected mode in logs/provenance; and
- requiring both the 2D mask and valid z-support sidecars when `--skip-segmentation` is used.

Missing z-support must be a hard error in `2p5d-zgated`, never an implicit fallback to legacy full-z behavior.

### 6. Documentation

Update user-facing pipeline documentation with:

- a comparison of `stitched`, `2p5d`, and `2p5d-zgated`;
- exact run examples for TIFF, DAT, and confocal data;
- output and status definitions;
- rerun requirements; and
- the scientific limitation that a DAPI-derived axial gate estimates cell support but does not prove that two probe spots are biological replication.

Update the sequential-FISH tracing skill after implementation so its operational guidance matches the tested CLI and output contract.

## Test plan

### Unit tests

Test the gate library with synthetic profiles covering:

- a single centered peak;
- a peak at each z boundary;
- a one-plane internal gap;
- a narrow peak requiring minimum-width expansion;
- two components below and above the ambiguity threshold;
- weak/flat and non-finite profiles;
- consensus fallback;
- no-valid-gate fallback;
- exact status-code serialization; and
- malformed/mismatched sidecars.

Test materialized masks to confirm that:

- XY ownership is unchanged;
- only the requested z interval is occupied;
- labels never overlap or change identity; and
- bounds and voxel support agree.

### Regression tests

- Existing `stitched` tests and representative outputs must remain unchanged.
- Existing legacy `2p5d` tests and full-z output must remain unchanged.
- The new mode must not be selected implicitly.
- Drivers must forward the mode and failure policy correctly.
- `--skip-segmentation` must reject absent or mismatched z-support in the new mode.

### Integration tests

Run a lightweight fake-data end-to-end test, then one representative field from each modality:

- rep1 TIFF;
- rep3 DAT;
- Novo DAT; and
- confocal-prepped TIFF.

For each field, verify:

- exact time-point alignment;
- gate-status counts;
- localization row accounting;
- mask/gate overlays in XY, XZ, and YZ;
- quarantined observations never become positive multispot evidence; and
- stitched and legacy 2.5D runs are unaffected.

After these pass, validate the six existing visual-comparison fields before starting a full campaign.

## Acceptance criteria

Implementation is complete when:

- all three segmentation modes are explicit and operational;
- the new sidecar is versioned, validated, and consumed by both localizers and replication QC;
- stitched and legacy 2.5D regression tests pass unchanged;
- every localization record in the new mode has auditable gate provenance;
- quarantine/failure cannot silently generate `MultiSpot` or `Replicating`;
- at least 95% of previously valid one-spot observations are retained before fallback on the validation set, with all losses reported by reason;
- visual checks confirm preservation of the accepted 2D cell boundaries;
- multispot-rate changes are reported by dataset and gate status rather than interpreted automatically as biological improvement; and
- documentation includes exact commands and rerun requirements.

## Rollout order

1. Implement and unit-test the z-support library.
2. Add the new segmentation mode and provenance outputs.
3. Integrate the support into both localizers.
4. Integrate the same support and quarantine semantics into replication QC.
5. Wire all drivers and Slurm wrappers.
6. Run regression and fake-data integration tests.
7. Validate one field per modality and inspect diagnostic panels.
8. Validate all six comparison fields and calibrate affected QC thresholds.
9. Freeze parameters and provenance schema before the full four-dataset rerun.

## Known risks and safeguards

- **Broad DAPI axial profiles:** retain explicit weak/ambiguous statuses and quarantine rather than forcing positive evidence.
- **Cell motion across time:** gates are computed independently at each time point, with consensus used only as a documented fallback.
- **Changed BIC or amplitude behavior:** recalibrate replication-QC thresholds on gated support before interpreting calls.
- **Hidden positional mismatches:** require exact time-point and label identity checks.
- **Silent legacy fallback:** prohibit it in `2p5d-zgated`; missing or invalid sidecars are fatal.
- **Loss of time points:** preserve rows and explicit statuses wherever possible; quarantine keeps localization available without treating uncertain support as replication evidence.
- **Publication reproducibility:** record every parameter, schema version, status count, and fallback in machine-readable provenance.
