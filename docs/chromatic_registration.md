# Chromatic channel registration and centre ownership

Two coupled options, both **default-off**, so every archived run reproduces
unchanged:

| flag | default | effect |
|---|---|---|
| `--chromatic-registration off\|metadata` | `off` | represent the 405 DAPI label mask on the probe's own pixel grid |
| `--center-ownership allow\|strict` | `allow` | require a one-spot centre to lie inside its own support |

They are separable and were measured separately (see *Measured effect*).

## Why

The DAPI mask is segmented from the 405 channel but used as the **fitting
support** for the 640 genomic probe and the 488 fiducial. Any 405↔640 chromatic
offset therefore misregisters the support: a genuine 640 centre can land in
background or inside a neighbour, and nothing recorded that it had.

On rep3 Location-01 the offset is real and large: **+209 nm in x, −71 nm in y**
at the field centre, i.e. about two pixels — comparable to a whole cell width in
x and far above the 79 nm same-locus XY repeatability.

## The transform convention, and how it was established

`ChromaticOffset.Transforms[i]` is a `PlaneTransformation` whose `Matrix**`
entries form an affine acting on **calibrated physical XY nanometres**, mapping
the common/reference plane **into** channel *i*'s raw plane. Translations are
physical, not pixel, values.

Transform index → wavelength binding comes from
`Configuration.Probes[i].value.WidefieldLaser`, **never** from array order. On
rep3 that yields `['640', '488', '405']`, with `T_640` exactly identity and
`RMSErrors[0] == 0` — both consistent with 640 being the reference plane. A unit
test reorders the probes and asserts the composition follows the labels, not the
indices.

To express the DAPI mask on target grid *c*, each target pixel is inverse-mapped
into the DAPI image:

```
p_405 = T_405 @ inverse(T_c) @ p_c
```

This was **discriminated by the data before being adopted**, not assumed.
Fraction of the 22,511 fitted 640 centres of rep3 Location-01 falling inside
their own DAPI label, under each candidate reading:

| candidate | own-mask |
|---|---|
| unregistered | 71.38 % |
| `p_405 = T_405 @ inv(T_640) @ p_640` | **81.89 %** ← adopted |
| inverse direction | 42.58 % |
| translations read as pixel values | 0.00 % |

The wrong direction and the pixel reading are both catastrophic, so the choice is
not merely plausible.

## What the helper will and will not do

`pipeline/chromatic_registration.py` does one composition and one categorical
resample. It is not an image-registration framework.

* Resampling is **strictly categorical**: nearest-voxel index lookup, so label
  IDs and the 0 background are preserved exactly. No interpolated label value can
  ever appear. A post-condition in each caller asserts the output label set is a
  subset of the input's.
* XY only. The axial axis is untouched, which is valid **only** because
  `ZOffsets` are zero; a non-zero axial offset raises rather than being ignored.
* `metadata` **fails loudly** — missing file, unreadable JSON, non-positive pixel
  size, probe/transform count mismatch, unusable `WidefieldLaser`, duplicate
  wavelength, non-finite or singular matrix, a non-empty `NonlinearTransform`,
  non-zero `ZOffsets`, or no 405 transform. It never falls back to identity,
  because a caller that asked for registration must not silently get an
  unregistered result.
* Every run writes `localizations_{640,488}_chromatic_registration.json`. With
  `off` that file carries an explicit warning that the channels are
  **unregistered**, so an unregistered run is self-documenting rather than silent.

### Pixel origin

Pixel centres are taken at integer indices, origin at index 0, and this is
recorded in the provenance as `pixel_centre_at_integer_index_0`. A half-pixel
convention would shift the result by ~51 nm. That variant has **not** been tested
against the data; the adopted convention is the one that produced the numbers
above.

## Centre ownership

`pipeline/center_ownership.py` classifies each fitted one-spot centre against
**the same effective support the fitter used** — registered into the probe's grid
and, in `2p5d-zgated` mode, already restricted to that timepoint's axial gate:

`own_mask` · `background` · `other_active_cell` · `out_of_bounds`

Only `own_mask` passes. Lookup is nearest-voxel and categorical.

Ownership is evaluated in the **raw target-channel image frame, before drift
correction**: the mask is registered into that frame and the fitted centre is
still in raw pixels there. Every localization row records this as
`OwnershipCoordinateFrame = raw_target_channel_pixels`.

### `allow` (default, backward compatible)

Records the classification and keeps every successful unconstrained fit. Nothing
is dropped, moved or re-fitted. With registration off this reproduces legacy
numerical output; the ownership columns are appended after the historical block.

### `strict` -- a FAIL-CLOSED FILTER

1. run the ordinary **unconstrained** one-spot fit;
2. classify **that original centre** against the registered effective mask;
3. accept only if it already lies in the cell's **own** label;
4. otherwise the locus is **unevaluable** -- background, another active label,
   off-grid, or non-finite.

**strict never moves, projects, clamps, snaps or re-fits a coordinate.**
`CenterRefitAttempted` and `CenterRefitAccepted` are always 0, and the localizer
asserts both at end of run rather than trusting the branch. A rejected locus is
excluded from `localizations_640.csv` and therefore cannot contribute to
complete-walk, replication, trace-salvage or clean-walk status. There is no silent
fallback to `allow`.

The decision is made on the coordinate **as serialized** (4 decimals, via
`center_ownership.quantize_for_output`), so a consumer recomputing ownership from
the written CSV gets the identical verdict. Deciding at full precision let 12 of
14,332 accepted centres on rep3 Location-01 re-round outside the mask on reload;
that is now fixed by construction rather than tolerated.

Strict two-spot ownership: in `replication_qc.py --center-ownership strict`, BOTH
M2 component centres must lie in the same cell's own registered support. A
component in background or another active label makes the locus **unevaluable** --
not a confirmed multispot and not negative evidence. The physical background shell
remains available for background *estimation*; it is never legal spot-centre
territory.

#### Unevaluable is its own status, not a negative call

The rejected locus gets status code `10` → `ownership_unevaluable`, its **own**
code. This matters concretely: the C backend already uses code `4` for
`confirmed`, so an earlier revision that reused `4` wrote `status=confirmed` on a
row with `replication_confirmed=0`, and every consumer that reads evaluability off
the status string then counted the locus as *measured and negative* — silently
converting "we cannot tell" into one-spot evidence. Code `3`
(`tested_not_confirmed`) would have been just as wrong for the same reason.

Walk-level replication is therefore **recomputed**, not copied:
`classify_replication._sets_for_walk` puts an ownership-rejected locus in `failed`
before anything else, so it cannot contribute a confirmation, cannot enter
`evaluated`, and cannot complete the pattern. `replication_qc.py`'s own walk
summary applies the same rule. Both read the explicit `ownership_unevaluable`
column as well as the status string, so the two cannot drift apart, and both
default to the previous behaviour when the column is absent.

Consequences, each covered by a test in
`tests/test_ownership_propagation.py`:

* an ownership-invalid **Ori** yields `indeterminate_ori`, never
  `non_replicating`;
* an ownership-invalid locus cannot set `MultiSpot` or `Replicating`, even if
  `replication_confirmed` still reads 1;
* an ownership-invalid locus is not evidence of a single spot, so
  `replication_pattern_complete` becomes 0;
* a fully owned walk keeps its expected classification unchanged, and a genuine
  two-spot still sets `MultiSpot`.

A one-spot ownership rejection needs no separate rule: strict drops the row from
`localizations_640.csv`, so `has_fit1` is 0, the two-spot test never runs, and the
locus is simply missing from the side-car.

### `constrained_experimental` -- research only, NOT for analysis

Retains the previous constrained refit that pushed an out-of-support centre back
inside the owned label. **This is boundary clamping through optimization** and must
not be used for results. Measured on rep3 Location-01: all 2,399 attempted refits
"succeeded", none were rejected, ~47% still rounded outside the mask after
serialization, and ~92% of those landing inside sat exactly on an XY mask
boundary. Downstream, at the archive's own `center_shell_nm 0`, it removed 53-86%
of complete traces -- overwhelmingly by making loci **unevaluable** (37-59% of
previously complete walks) rather than by detecting doublets. It destroys fits
rather than reinterpreting them. Never selected by `strict`, never used by a
production driver, and a test asserts no driver references it.

The unconstrained path is untouched: `batch_fit` delegates to
`batch_fit_ex(..., cell_ids=NULL, strict_own=0)`.

### Wavelengths are configurable, never inferred from a probe index

`probe_0` does **not** mean 640 nm. Confocal probe_0 is **561 nm**, and inferring
640 from the directory index silently applied a widefield 405->640 transform to
confocal data. Every stage takes explicit flags:

    --modality widefield|confocal
    --reference-wavelength-nm 405
    --target-wavelength-nm    640   (localize_probe0, replication_qc)
    --fiducial-wavelength-nm  488   (localize_488)

Defaults: widefield reference 405 / probe 640 / fiducial 488; confocal reference
405 / probe **561** / fiducial 488. Calibration entries bind by the actual
`WidefieldLaser` metadata, so requesting 640 against a confocal calibration fails
clearly instead of picking a neighbour. Confocal registration stays **disabled**
unless a matching 405->561 calibration measured on that microscope is supplied.

### Provenance is per stage and cannot be overwritten

`write_provenance` refuses to overwrite a record whose `stage` differs:

| stage | file |
|---|---|
| `localize_probe0` | `localizations_640_chromatic_registration.json` |
| `localize_488` | `drift_488_chromatic_registration.json` |
| `replication_qc` | `replication_qc_chromatic_registration.json` |

Each record carries stage, calibration path and SHA-256, reference and target
wavelengths, matrix and direction, pixel sizes, registration mode, ownership
policy and the relevant CLI arguments.

### Z-offset metadata must be complete

If `ZOffsets` is present it must cover every referenced plane. A short array is a
hard error -- silently substituting 0.0 would turn an unspecified axial offset
into an asserted zero. A wholly absent field is the documented "no axial
calibration" case. Non-zero offsets remain unsupported and fail.

### Rounding

`gauss3d_fit.c` resolves the nearest voxel with `lround` (half away from zero);
`np.rint` rounds half to **even**. They disagree at exactly `*.5`, which is not
hypothetical — rep3 Location-01 TP010 has a centre at x = 980.50 on a label
boundary where the two rules name different cells. `center_ownership.nearest_voxel`
implements the C rule so the constraint and the classifier can never disagree.

**Known limitation.** A constrained fit converges *against* its owned label's
boundary, so the localizer's 4-decimal output can land on the half-voxel tie and
re-round into the neighbouring voxel (`8.49999816` → `"8.5000"`). Ownership is
decided at full precision and **recorded**; consumers must read the recorded flag
rather than recompute it from the written coordinate. A test pins this behaviour.

## Where the fields live

`localizations_640.csv` keeps its **historical column set** so every downstream
consumer and archived comparison stays valid. In `strict` mode it is the
*filtered* result.

Ownership lives in the per-attempt status table
`localizations_640_z_gate_status.csv`, now written in **every** segmentation mode
(its `z_gate_*` columns are blank when there is no dynamic gate). One row per
attempted `(WalkID, tp)`, including failed fits and ownership rejections:

```
center_ownership_initial  center_refit_attempted  center_refit_success
center_ownership_final    center_ownership_pass   fit_ret_code
localization_written      unevaluable_reason
center_fit_z_px           center_fit_y_px         center_fit_x_px
```
plus the canonical `CenterOwnership*` / `CenterRefit*` /
`ChromaticRegistration*` columns, identical in name and convention to the
localization rows.

This table is therefore the **authoritative universe of localization attempts**,
and it is what lets the comprehensive downstream CSV be self-contained (see
`docs/trace_salvage.md`). Two properties matter:

* `center_fit_{z,y,x}_px` is the unconstrained fitted centre, serialized exactly
  as `localizations_640.csv` would write it — so the audit value of a rejected
  attempt *is* the value the verdict was decided on, not a re-derivation.
* Both the fit centre and the canonical ownership columns are **blank for a
  failed fit**: there is no centre and no verdict. `center_ownership.verdict_of`
  reads that as *no verdict* rather than as a rejection, so a fitter failure is
  counted as a missing locus and is never censored as an ownership rejection.
  The legacy lowercase `center_ownership_pass` keeps its historical `0`.

`localizations_488_z_gate_status.csv` mirrors it with
`center_ownership_initial` / `center_ownership_pass`.

### Walk level

Four fields travel on the merged and full-schema outputs
(`center_ownership.OWNERSHIP_WALK_FIELDS`):

| field | meaning |
|---|---|
| `CenterOwnershipExpectedCount` | size of the **configured** locus universe |
| `CenterOwnershipValidCount` | required loci present **and** valid |
| `CenterOwnershipRejectedCount` | required loci with an **explicit** rejection verdict |
| `CenterOwnershipComplete` | 1 only if every required locus is present and valid |

A locus that is entirely absent — never acquired, or fitted unsuccessfully so
that no verdict exists — is neither valid nor rejected. It is *missing*, and
`Expected − Valid − Rejected` recovers that count; no fifth column is added,
because the four fields already separate the three outcomes arithmetically.
Rejected and missing both make `CenterOwnershipComplete` zero.

**The denominator is the configured range, never the surviving rows.**
`merge_all_walks.py` reads it from the range the pipeline already persisted in
`localizations_640_replication_policy.json`
(`hierarchy.replication_tp_start/end`), or from `--ownership_locus_order`;
`trace_salvage.py` uses the config's `chromosome_order`. Deriving it from
surviving rows would let a filter that removed a locus shrink the denominator and
make an incomplete trace look complete. The `TP000`/`TP001` lead-in and the
`TP022`/`TP023` re-image controls therefore cannot affect the verdict.

**Rejections are counted from the audit table, not the coordinate CSV.** Under
`strict` the localizer *drops* a rejected row, so the coordinate CSV can only
represent a rejection as an absence. `localizations_640_z_gate_status.csv` is the
one place the rejection is still itemized, and it is the difference between
"rejected" and "never acquired". Both consumers prefer it and fall back to the
coordinate CSV's own ownership columns.

A dataset with **no** ownership schema at all yields blank fields, not `1`:
`walk_ownership_summary` raises `OwnershipSchemaMissing` rather than defaulting,
because filling absent ownership with "valid" would manufacture a clean result
and filling it with "invalid" would zero every walk in a pre-ownership dataset.

`python pipeline/center_ownership.py --status_csv <status> --replication_tp_start N
--replication_tp_end M` still writes the standalone
`localizations_640_center_ownership_walks.csv` audit
(`WalkID n_loci_expected n_loci_attempted n_loci_own_mask CenterOwnershipComplete`).

This is a **fifth** distinct meaning of "complete". It is not
`replication_pattern_complete`, not `valid_walks`, not `classified_walks`, and not
the merged `_nosuspect_movement` set. Never report it as any of those.

### Where ownership reaches, end to end

```
localizations_640.csv                        11 per-locus fields (appended suffix)
  -> localizations_640_drift_corrected*.csv  carried (all three products)
  -> localizations_640_replication_qc.csv    + ownership_unevaluable, ownership_failure_reason
  -> localizations_640_replication_classification.csv
                                             recomputed: unevaluable, not negative
  -> locations/<Loc>_trace_salvage_full_schema.csv
  -> all_locations_640_valid_walks.csv       + the 4 walk-level fields
  -> *_trace_salvage_full_schema.csv          unfiltered comprehensive
  -> *_trace_salvage_full_schema_complete.csv filtered to
                                              TraceSalvagedSingleSpotComplete == 1
```

Both trace-salvage products carry the **same full schema**; the second is the
first filtered by walk. Identifiers are unchanged: `WalkID` is continuous and
globally unique after combine, `OriginalWalkID` is the historical per-Location ID,
and joins use the local ID before global renumbering.

`tests/test_ownership_propagation.py` asserts each arrow above against real
module runs, including `drift_correct.py` as a subprocess.

## 488 and the drift formula

`localize_488.py` accepts `--chromatic-registration` and registers the mask into
the 488 grid by the same composition, and records ownership.

There is deliberately **no `--center-ownership strict` for 488**: dropping
fiducial fits would change which acquisitions contribute to the drift curve. The
relative `488(t) − 488(reference)` formula is unmodified, and
`localizations_488.csv` keeps its historical column set so `drift_correct.py`
sees an unchanged input schema.

**But registering the 488 support does change the 488 centres**, and therefore the
drift curve — not because the formula changed, but because the fits changed. The
measured size of that change is in *Measured effect*.

## Visualization frame consistency

`pipeline/display_frames.py` holds the **one** convention every overlay uses.
Three grids exist and all three look plausible on screen:

1. the **raw acquisition frame of a channel** — each channel has its own grid, so
   a DAPI-derived label map is not in the 640 grid until it is resampled there;
2. the **drift-corrected frame** — coordinates only. Drift correction is applied
   to fitted positions, never to the images;
3. **nanometre space** — a scaled view, not a separate grid; plotting nm on a
   pixel axis rescales by ~100×.

### The drift convention, read from the implementation

`drift_correct.py` computes

```
drift[tp] = current_488 - reference_488        # pixels, per acquisition
corrected = raw - drift[tp]                    # what the CSV stores
raw       = corrected + drift[tp]              # the inverse, for display
```

Drift is **per acquisition**: a 20-locus trace has 20 different displacements,
which is why the drift-corrected CSV carries `drift_z/drift_y/drift_x` on every
row. Never apply one walk-level displacement to a whole trace.
`display_frames.corrected_to_raw_pixels()` is the only implementation of that
inverse; `rows_to_raw_pixels()` prefers the `z_raw/y_raw/x_raw` columns that are
already raw and falls back to inverting with each row's own drift, raising when a
row offers neither.

### Declared, not inferred

A `Frame` states the channel wavelength, the kind (`image` / `labels` /
`coordinates`), the drift state, the acquisition, and the units. It is a
wavelength rather than a probe index because probe ordering has been reordered
between datasets. `require_displayable(image, coordinates=..., labels=...)`
raises on a channel mismatch, on drift-corrected coordinates over a raw image, on
nm values on a pixel axis, and on a drift-corrected frame that does not name its
TP. `labels_into_image_frame()` resamples categorically — nearest-neighbour only —
and verifies the output invents no label IDs.

### Production overlays

`diagnostics/trace_walks_to_fov.py`, `middle_grid_walks_fov.py`,
`random_grid_fov.py` and `overlay_chromatic_registration.py` all route through
`resolve_overlay()` and take `--modality` / `--chromatic-calibration`. The defect
this fixes is concrete: the mask is always segmented on DAPI, but
`trace_walks_to_fov --probe probe_0` displayed a probe image, and the 488 pages of
the two grid galleries drew the DAPI mask on the 488 grid — offset by the
calibration, with no warning. A cross-channel overlay without a calibration is now
**refused**, not rendered.

`report_cards_single_curve.py` and `qc_post_filter.py` need no frame handling:
they plot curves and distance matrices in nm space and never overlay an image.

`tests/test_display_frames.py` validates the sign, the inverse, per-acquisition
drift, registration direction, double registration, categorical labels, and that
the production scripts use this path.

## Confocal

**Refused.** The 15072026 Sal_AS tree is spinning-disk confocal at 81.25 nm xy /
180 nm z and its `data.json` reports `ConfocalReferencePixelSize = 0.0` — there is
no confocal chromatic solution in the metadata. The Vutara **widefield**
`ChromaticOffset` affine must not be transferred to it: different optical path,
different pixel size, unvalidated on this data.

`run_pipeline_confocal.sh` therefore exits with a clear message if
`--chromatic-registration metadata` is requested without
`CHROMATIC_METADATA=/path/to/a/confocal/calibration`, and the default `off` writes
the provenance warning rather than being silent.

### Confocal channel identities and wavelengths

`confocal_prep.py` de-muxes `C0=561` → `probe_0`, `C1=488` → `probe_1`,
`C2=405` → `probe_2`. The genomic probe is therefore **561 nm, not 640**, and
`run_pipeline_confocal.sh` passes `--modality confocal` plus explicit
`--reference-wavelength-nm 405`, `--target-wavelength-nm 561` and
`--fiducial-wavelength-nm 488` to every applicable stage, so no stage can fall
back to the widefield assumption. Overridable via `REFERENCE_WAVELENGTH_NM`,
`PROBE_WAVELENGTH_NM`, `FIDUCIAL_WAVELENGTH_NM`.

The 488 argument is not invented decoration: this workflow really does localize
the 488 fiducial and drive drift correction from it.

Geometry stays confocal throughout: 81.25 nm xy, 180 nm z, `fov_size 1152`. The
widefield 103.28 nm pixel and 1866 px field are never substituted.

If registration **is** requested with a calibration, the driver validates *up
front* that the file solves both `405→561` and `405→488`, before any Location
runs. A `405→640`-only file is **refused**, never substituted — a confocal run
must not borrow the widefield transform.

`tests/test_confocal_driver_commands.py` asserts all of the above against the
commands the driver actually constructs, using a stub interpreter that records
argv.

## Measured effect, current code (rep3 Location-01, 24 TPs, 698 cells)

Bounded smoke test on the archived 2.5D z-gated strict-ownership masks,
`center_shell_nm 0`, bootstrap 16/128, `q=0.01` — the archive parameters. Three
localization arms; the two registered arms carried through replication QC and
classification.

| | `off`/`allow` | `metadata`/`allow` | `metadata`/`strict` |
|---|---|---|---|
| fitted loci | 16,723 | 16,731 | 16,731 |
| own_mask | 9,015 (53.9 %) | 14,321 (**85.6 %**) | 14,321 (85.6 %) |
| background | 7,708 | 2,393 | 2,393 |
| other_active_cell | 0 | 17 | 17 |
| rows in `localizations_640.csv` | 16,723 | 16,731 | **14,321** (filtered) |
| constrained refits attempted | — | — | **0** |
| coordinates moved | — | — | **0** |
| rejected as unevaluable | — | — | **2,410** |
| outside own mask after CSV reload | — | — | **0** |
| accepted on an XY boundary | — | — | 3,713 (25.9 %) |

Walk level, expected universe `TP002..TP021` (20 loci), 698 walks:
`CenterOwnershipComplete` = **247** in both registered arms — identical, as it must
be, because `strict` filters rows and does not change ownership verdicts. 1,991 of
the 2,410 rejections fall inside the configured range; the remaining 419 are in the
`TP000`/`TP001` lead-in and the `TP022`/`TP023` controls and correctly do not count.

Replication, recomputed from accepted rows:

| | `metadata`/`allow` | `metadata`/`strict` | Δ |
|---|---|---|---|
| confirmed two-spot loci | 2,192 | 1,991 | −201 |
| loci `status=ownership_unevaluable` | 0 | **133** | +133 |
| `MultiSpot` walks | 444 | 416 | −28 |
| `Replicating` | 127 | 112 | −15 |
| `ActiveReplicating` | 98 | 87 | −11 |
| `replication_pattern_complete` | 500 | 171 | −329 |
| `non_replicating` | 249 | **233** | **−16** |
| `indeterminate_ori` | 9 | **92** | **+83** |

The last two rows are the load-bearing check. If unevaluable loci were being
converted into negative evidence, `non_replicating` would have **risen**; it fell,
while `indeterminate_ori` rose by 83. Directly:

> Of the walks whose **Ori** centre fails ownership, **79 are
> `indeterminate_ori` and 0 are `non_replicating`.**

`replication_pattern_complete` falling by 329 is the intended conservatism: a
rejected locus makes the pattern *incomplete* rather than silently negative.

One rejected locus traced end to end (WalkID 3, `TP002`, `center_background`):
absent from `localizations_640.csv` **and** from the replication side-car, so it
cannot contribute anywhere; retained in the per-attempt audit table with its
reason; `CenterRefitAttempted=0`; the walk survives with its other loci at
`CenterOwnershipComplete=0`, `ValidCount=2`, `ExpectedCount=20`,
`RejectedCount=16`, `replication_pattern_complete=0`.  Two of that walk's 20
required loci are fitter failures (`TP010`, `TP018`), which are *missing* rather
than rejected — hence 2 + 16 + 2 = 20.

In the comprehensive output that same locus appears exactly once, with
`LocalizationAttemptStatus=ownership_rejected`,
`LocalizationAttemptReason=center_background`, `CenterOwnershipValid=0`, its
unconstrained fitted centre in `UnconstrainedFit{Z,Y,X}Nm`, every canonical
coordinate blank, and zero rows in the filtered output for its walk.  See
`docs/trace_salvage.md` for that contract.

### The attempted-walk denominator

A walk whose **every** centre was rejected has no accepted localization and
therefore no replication side-car row at all. Classifying only side-car walks
dropped it from the totals entirely — neither counted nor reported, which reads as
if the cell had never been imaged. On rep3 Location-01 that was `WalkID 375` (mask
label 376): 24 attempts, all `center_background`, and 697 classifications for 698
attempted walks.

`classify_replication.py` therefore takes its walk universe from the **union** of
the attempt ledger and the side-car. Every attempted walk gets exactly one
classification row.

**Under strict ownership the ledger is mandatory, and an absent ledger is fatal.**
It is never permission to fall back — that is precisely how a strict run loses a
wholly rejected walk. The CLI is explicit about which case applies:

| invocation | behaviour |
|---|---|
| `--attempt_ledger_csv PATH` | that file or nothing: if it is missing or unusable the run fails, and no other path is ever tried |
| `--require_attempt_ledger` | required, discovered in `--outdir` when no path is given; nothing found, or more than one `*_z_gate_status.csv` candidate, is fatal |
| `--no_attempt_ledger` | deliberate legacy opt-out: side-car universe only, with a loud warning. Mutually exclusive with both options above |
| neither | strict provenance ⇒ a discoverable ledger is required and its absence is fatal; non-strict or unrecorded ⇒ historical behaviour, reported in the log |

Strictness is read from the provenance the stages already write —
`ownership_policy` in `*_chromatic_registration.json`. No new policy file is
introduced. If two provenance records disagree the run fails, listing both files
and both values, because nothing else can then decide whether the ledger is
mandatory.

Discovery looks **only** in `--outdir`: never a parent directory, never another
Location. Copying one Location's ledger next to another's side-car would classify
one cell population against a different denominator.

Every failure is one message naming whether the ledger was required, the exact
resolved path, why it failed, the ownership policy and where that came from, and
which option enables deliberate legacy behaviour. Distinct failures: required
ledger absent · explicit path absent · unreadable · empty despite attempts ·
required columns missing · duplicate `(WalkID, tp)` · malformed `WalkID` · blank
`tp` · ambiguous discovery · policy disagreement · conflicting flags.

On any of them the process exits **2** before the output path is opened: no walk
summary is written, no existing summary is overwritten, no policy JSON is
emitted, and no annotation runs — so the `|| exit 1` guard in every driver stops
the Location and a SLURM `afterok` dependency never starts merge or trace
salvage. The four strict-capable drivers (`run_from_masks.sh`,
`run_pipeline.sh`, `run_pipeline_dat.sh`, `run_pipeline_confocal.sh`) pass
`--require_attempt_ledger` **and** the explicit per-Location ledger path; none of
them may ever pass `--no_attempt_ledger`, which is asserted by test.

A walk with no evaluable locus is `indeterminate_ori` — the existing state, not a
new category — with `MultiSpot`, `Replicating`, `ActiveReplicating`,
`FullyReplicated`, `FullReplicationCandidate`, `replication_pattern_complete`,
`OriEvaluated` and `OriTwoSpot` all zero. It is **never** `non_replicating`:
absence of evidence is not evidence of a single spot. Nothing is manufactured —
no coordinates, no locus rows, only the walk-summary row that was missing.

Because `MultiSpot = 0` then means "no confirmed multispot evidence" rather than
"measured and clean", `ReplicationClassificationReason` states which it is:

| value | meaning |
|---|---|
| `''` | the walk had replication evidence; read the ordinary fields |
| `all_localizations_ownership_unevaluable` | every configured-range attempt was ownership-rejected |
| `all_localization_fits_failed` | every configured-range attempt failed to fit |
| `no_evaluable_localization` | mixed rejections and failures |
| `no_localization_attempt_in_configured_range` | attempts exist, but none inside `[tp_start, tp_end]` |

The reason field is not a status column and never encodes a biological call;
`replication_state` remains the only classification. It travels through
`WALK_ANNOTATION_FIELDS`, so it reaches the drift-corrected CSVs, the merged
deliverables and both comprehensive outputs. An archived summary CSV written
before it existed is still annotatable — it simply comes through blank.

Two notes on reading such a row. `n_missing_tps` counts loci missing from the
*replication measurement*; the reason field and the per-locus `CenterOwnership*`
columns say whether the locus was attempted and rejected — those are different
questions, and a walk can legitimately show `n_missing_tps=20` with
`all_localizations_ownership_unevaluable`. And a walk with side-car rows keeps its
existing annotation values unchanged: the ledger is consulted for the
attempted-but-unmeasured walks it adds, never to rewrite a walk that was already
classified.

## Superseded measurement (previous code vintage, constrained refit)

**Do not use these numbers.** They were produced by the `strict` policy *before*
it became reject-only, i.e. when it still ran the constrained refit — the arm
labelled `metadata`/`strict` below recovered every out-of-support centre by moving
it, which is the behaviour now quarantined in `constrained_experimental`. They are
kept only so the change is auditable.

Three arms, same code, same archived mask, `center_shell_nm 110` and bootstrap
calibration disabled identically in all three so the arms isolate these changes.

| | `off`/`allow` | `metadata`/`allow` | `metadata`/`strict` |
|---|---|---|---|
| 640 localizations | 22,511 | 22,511 | 22,511 |
| 640 failed fits | 4,105 | 4,105 | 4,105 |
| own_mask (initial) | 16,950 (75.30 %) | 20,675 (**91.84 %**) | 20,675 (91.84 %) |
| background (initial) | 5,561 (24.70 %) | 1,836 (8.16 %) | 1,836 (8.16 %) |
| own_mask (final) | 75.30 % | 91.84 % | **100.00 %** |
| constrained refits | — | — | 1,836 |
| recovered by refit | — | — | **1,836** |
| rejected unevaluable | — | — | **0** |
| `CenterOwnershipComplete` | 275 (24.80 %) | 634 (57.17 %) | **877 (79.08 %)** |
| full-coverage rows | 20,952 | 20,736 | 20,736 |
| 488 localizations | 22,547 | 22,534 | 22,534 |
| confirmed two-spot | 2,850 | 3,283 | 3,316 |
| MultiSpot walks | 600 | 672 | 674 |
| `Replicating` | 160 | 188 | 190 |
| `active_replicating` | 110 | 133 | 133 |
| `non_replicating` | 307 | 221 | 222 |
| `pattern_complete` | 462 | 433 | 429 |

### Registration changes the coordinates, not only the frame

Registration leaves the localization and failure **counts** identical
(22,511 / 4,105 in every arm). That is easy to misread as "only the support frame
moved". **It is not.** Moving the support by ~2 px changes each fit's voxel set,
its background estimate and its seed, so the fitted centres move:

| 640 centres vs `off`/`allow` | `metadata`/`allow` | `metadata`/`strict` |
|---|---|---|
| identical | 458 (2.0 %) | 448 (2.0 %) |
| \|ΔXY\| median | 25.3 nm | 25.9 nm |
| \|ΔXY\| p95 | 644.7 nm | 648.6 nm |
| \|ΔXY\| max | 3,787 nm | 3,787 nm |
| \|ΔZ\| p95 | 322.5 nm | 322.0 nm |
| **moved > 79 nm XY** (repeatability) | **7,246 (32.4 %)** | **7,366 (32.9 %)** |

Isolating the constrained refit alone (`strict` vs `metadata`/`allow`): 1,836 of
22,511 coordinates change (8.2 %), median \|ΔXY\| 112 nm among those.

So enabling registration is a **substantive change to the localization result**,
not a cosmetic reframing. A third of coordinates moving by more than the
measurement precision is consistent with the 15 % change in the replication
readout, and it is the reason the clean-walk set churns by ~34 %.

## Interaction with trace salvage

### Salvage cannot override ownership

Strict ownership is applied **before** any salvage decision, by censoring the
evidence at every ownership-invalid locus to unevaluable
(`_censor_unowned_evidence`, evidence status `ownership_unevaluable` -- the
canonical spelling from `replication_status`). One rule
produces every consequence:

* `call=False` — not a raw doublet, so it can never be salvaged; salvage cannot
  manufacture a coordinate where ownership rejected the locus;
* `confident_single=False` — it can never anchor a **neighbouring** locus's
  salvage decision, and never contributes a clean calibration link;
* `evaluable=False` — it is not read as evidence of a single spot;
* `components=None` — its M2 component centres are unavailable.

`TraceSalvagedSingleSpotComplete == 1` additionally **requires**
`CenterOwnershipComplete == 1` as an independent conjunct, so the invariant holds
by construction rather than by relying on the censor staying correct, and it also
covers a locus rejected in the audit table but absent from the side-car.

Salvage can still resolve an *accepted* multispot choice. It can never resurrect a
rejected locus.

An ownership-rejected walk keeps **all** its attempts in the unfiltered
comprehensive CSV — accepted loci with their coordinates, rejected loci as
audit-only rows — and contributes **zero** rows to the filtered one. Nothing is
fabricated: a locus never submitted to localization has no ledger row and so no
output row. The manifest records the ownership source per Location, the
censored-locus count, the rule text, and the per-category attempt counts.

Note that censoring counts only loci with an explicit rejection verdict. A locus
whose fit failed has no verdict at all (blank `CenterOwnershipValid` in the
ledger), so it is left alone rather than being relabelled
`ownership_unevaluable`. On
rep3 Location-01 that is the difference between 1,991 and 1,972 censored loci: 19
of the 21 fitter failures fall inside `TP002..TP021`.

### Effect sizes

**Not measured, and not measurable from a single Location.**
`trace_salvage.py` calibrates its clean-link cutoffs **leave-one-location-out**
with `min_calibration_links=100` and fails closed, so a one-Location run salvages
nothing in every arm and the comparison is uninformative by construction. All
numbers in this document are **pre-salvage**.

Salvage does not enter the clean-walk predicate — it adds
`TraceSalvagedSingleSpotComplete`, a separate field — so the clean-walk figures
above stand as the pre-salvage comparison. But salvage's **inputs all move**:

* its decision rule is **purely XY** (path score `d_xy(N-1,c) + d_xy(c,N+1)`, the
  80 % ratio gate, and the leave-one-location-out clean-link 95th percentile), and
  32 % of XY coordinates change by more than the repeatability;
* those percentiles are themselves fitted from the same coordinates, so the
  thresholds move as well as the scores;
* more confirmed doublets (2,850 → 3,316) means more candidate loci to salvage.

Salvage outcomes must therefore be **re-derived, not carried over**, on any
dataset run with registration enabled. Do not compare a registered salvage result
against an unregistered one as though only the representation changed.

### Effect on the clean-walk set

Clean = full localization coverage + movement pass + `PinnedDoubleFitSuspect=0` +
`MergedMaskSuspect=0`. With `--movement_axes xy`:

| | `off`/`allow` | `metadata`/`allow` | `metadata`/`strict` |
|---|---|---|---|
| full coverage | 873 | 864 | 864 |
| movement pass | 496 | 613 | 618 |
| not pinned | 449 | 413 | 410 |
| not merged-mask | 1,036 | 1,038 | 1,038 |
| **clean** | **142** | **153** | **157** |
| clean ∧ `CenterOwnershipComplete` | 56 | 126 | 157 |

The count rises, but the **membership churns**: only 93–94 of the baseline 142
remain clean (48–49 lost, 60–63 gained), so ~34 % of the baseline clean set is
lost. **The new clean set is not a superset of the old one** — a per-walk analysis
built on a previous clean set must be redone, not extended.

The +117 gain in *movement pass* is independent corroboration of the registration:
that filter compares `TP002` against its `TP022` re-image of the same locus, and
played no part in choosing the transform convention. Pinned suspects move the
wrong way (660 → 696 of 1,109), and full coverage loses 9 walks because 39 lost /
26 gained 488 fits change which walks are drift-correctable everywhere.

With the `xyz` movement default the clean set is only 42 walks (the 100 nm 3D
threshold sits below z's own 173 nm precision) and the comparison gives
42 → 49 → 52 with 21 of 42 lost — same direction, smaller and noisier base.

### Clean walks vs trace salvage: orthogonal axes

They are computed independently and must not be crossed.

The clean set (`*_valid_walks_nosuspect_movement.csv`) is written by
`merge_all_walks.py` **before** salvage runs, from the pre-salvage per-Location
drift-corrected outputs. Its coordinates are the raw drift-corrected values;
salvaged coordinates exist only in `*_trace_salvage_full_schema.csv`, where
`Trace{Z,Y,X}Nm` differs from `Original{Z,Y,X}Nm` at salvaged loci.

Measured on rep1 (10 Locations, 37,095 walks, 1,506 clean):

| `TraceSalvageStatus` within the clean set | walks | share |
|---|---|---|
| `incomplete_or_unevaluable` | 848 | 56.3 % |
| `ori_not_confident_single` | 313 | 20.8 % |
| `complete_clean` | 171 | 11.4 % |
| `unresolved_doublets` | 145 | 9.6 % |
| `complete_after_salvage` | 29 | 1.9 % |

147 clean walks (9.8 %) contain at least one salvaged locus, accounting for 170
of 16,614 salvaged loci (1.0 %).

Two warnings follow. **"Clean" does not mean "usable trace"**: only 200 of 1,506
clean walks have a complete trace (13.3 %), and 56 % are
`incomplete_or_unevaluable` -- clean is a statement about suspect flags and
measurement repeatability, not about whether every locus resolved. **And the
converse**: of 1,125 `complete_after_salvage` walks only 29 are clean (2.6 %),
because the movement filter and the 69 % pinned rate remove most of them.

Note also that `complete_clean` and `complete_after_salvage` are **mutually
exclusive** (`trace_salvage.py:1146`): `complete_clean` means a complete trace
that needed *no* salvage. It is not a subset of the salvaged-complete set, and it
is not the clean set defined above.

Finally, when joining these products: the merged clean file's `WalkID` is a
different numbering from trace-salvage v2's global `WalkID`. Join on
`(Location, orig_ID)` against `(Location, OriginalWalkID)`. Joining on `WalkID`
yields a plausible but meaningless overlap.

**488 and drift.** Registering the 488 support moves individual fiducial centres
(median |ΔXY| 5.1 nm, p95 122.5 nm, max 1,912 nm; only 9,171 of 22,508 shared
centres unchanged), but the median-based drift estimate absorbs it: **no per-TP
drift component changes by more than 3.58 nm**, against a 79 nm XY same-locus
repeatability. Median |drift_xy| 204.3 → 202.4 nm; median per-TP fiducial XY
residual 82.6 → 83.0 nm. So the drift curve is materially unchanged even though
its inputs are not.

### Caveats that must travel with these numbers

* **Registration changes the replication readout.** Confirmed two-spot calls rise
  15 % and `non_replicating` falls from 307 to 221. Given the blind-adjudication
  result that ~46 % of `Replicating` calls are judged not real, *more calls is not
  evidence of a better result*. Treat this as a change to be re-audited, not an
  improvement. `pattern_complete` also **falls** (462 → 433).
* **`strict` changes coordinates, it does not merely filter.** All 1,836
  out-of-support centres were recovered and none rejected, so `strict` rewrote
  8.2 % of coordinates rather than dropping them. Beyond registration it barely
  moves the replication readout (3,283 → 3,316 confirmed), because
  `replication_qc` refits M1/M2 on its own owned support.
* Those 1,836 recovered centres sit **against mask boundaries** by construction. A
  pile-up at the support edge is the same class of concern as
  `PinnedDoubleFitSuspect` and should be treated as a diagnostic, not as a
  precise localization.
* Registering the mask slightly changes per-cell mask areas, which shifts the
  `merged_mask_size_factor` median baseline. Default-off leaves legacy runs
  untouched.
* One Location, one replicate. Novo and the confocal tree have not been checked
  for non-zero `ZOffsets` or nonlinear terms; they will raise rather than silently
  mis-register if either is present.


## Canonical production contract (`production_contract_v1`)

Four distinct things must not be confused. This section is the boundary.

**1. Current production.** The four production drivers — `run_pipeline.sh` (rep1
TIFF), `run_pipeline_dat.sh` (rep3/Novo `.dat`), `run_pipeline_confocal.sh`,
`run_from_masks.sh` — now default to the contract and pass every value
EXPLICITLY rather than inheriting a module default:

| | value |
|---|---|
| model / segmentation | `cpsam_v2`, `2p5d-zgated`, z-gate failure `quarantine` (stitching and clump splitting are disabled by this mode) |
| one-spot ownership | `strict`, fail-closed — never moved, snapped, clamped, projected or re-fitted |
| M2 ownership | `--center-ownership strict` is now also passed to `replication_qc.py`, so both component centres must belong to the cell |
| replication support | `--center_shell_nm 0` |
| lateral gate | `--min_spot_sep_xy_nm 500` (95 % lateral LCB) — the calibrated direction |
| 3D gate | `--min_spot_sep_nm 500` retained as an additional **legacy diagnostic**, never as evidence |
| two-lobe | `--require_two_lobe_for_confirmation` |
| amplitudes | `--min_amp_ratio 0.3`, `--min_amp_recovery 0.5` |
| merged mask | `--merged_mask_size_factor 3.0` |
| attempt ledger | produced by every strict run; `--require_attempt_ledger` plus the explicit per-Location `--attempt_ledger_csv` |
| interval | rep1/rep3/Novo `TP002`–`TP021`, Ori `TP017`, Ter `TP007` |

Dataset specifics: rep1 `103.28`/`100`/`1866`/`sigma_z_max 3.5`/registration
**off** (no trustworthy metadata calibration exists — an explicit unregistered
limitation, not a preference); rep3 the same geometry with `sigma_z_max 0`
(unbounded) and registration `metadata`; confocal `81.25`/`180`/`1152`,
`sigma_z_max 0`, 405 reference / 561 probe / 488 fiducial, registration **off**
unless a confocal-specific calibration validates BOTH 405→561 and 405→488 — a
Vutara widefield calibration is never transferred.

Movement QC (`movement_qc_policy_v2`): rep1/rep3 are **XY-only at 100 nm**,
comparing `TP002` against its `TP022` re-image of the same locus. The 3D form put
a 100 nm threshold below z's own ~173 nm repeatability, culling walks for axial
imprecision rather than irreproducibility. Confocal movement QC is **disabled**
(`1e9`), because that layout has no same-locus re-image pair. A frozen registry
recording `xyz` keeps its provenance; the XY policy is versioned separately
rather than rewritten into it.

Each driver refuses to run if any contract value is overridden, listing the
offending settings. The 16/128 auto-null bootstrap in these drivers is the legacy
v2 detector and a diagnostic side-car only.

**2. Publication-v1 / v3 inference** is the paper-facing endpoint and is
unchanged: fractional SSE improvement as the calibrated statistic (never ordinary
BIC), raw lateral ≥ 500 nm and bootstrap 2.5th-percentile lateral ≥ 500 nm, z
separation alone never confirming a doublet, 63 → 32,767 draws, futility boundary
p = 0.05 with a 1e-6 total budget, 4,096 stability and 1,024 power draws, BY
q ≤ 0.01 primary with BH q ≤ 0.05 exploratory, seed 20260802, maximum bootstrap
failure fraction 0.05. The confirmatory biological endpoint is Ori **plus the
immediately adjacent locus** on either replichore; Ori-only and any-locus results
are secondary, and `MultiSpot` as a union across 20 loci is never the primary
endpoint.

**3. Explicit legacy reproduction** is `LEGACY_REPRODUCTION=1`, which restores
the historical permissive settings, prints a prominent warning, and produces
output that is *not* paper-facing and must not feed publication-v1. It is never
the default and no production wrapper sets it.

**4. Historical constrained-refit outputs are NOT acceptable upstream input.**
See below.

### Why `2.5d_gated_strict_ownership` is refused as publication upstream

That tree was produced by the earlier "strict" implementation, which constrained
and re-fitted out-of-support centres into masks. `preflight.py` now fails closed
per Location (`corrected_failclosed_upstream_v1`), requiring: ownership policy
`strict`; the current provenance schema; a per-attempt ledger carrying the
canonical `CenterOwnership*` and `center_fit_*_px` columns; **zero** constrained
refits; **zero** moved centres; every rejection carrying an explicit reason;
registration mode matching the dataset contract; `2p5d-zgated` mask plus
z-support provenance; and no contradiction between the localization- and
replication-stage provenance. A missing, old, contradictory or unverifiable
invariant is a hard failure naming the Location and the invariant. Historical
"strict" is never taken to mean current fail-closed strict.

Measured against the real archive, every Location fails: its ledger has only 11
columns and stops at `localization_written` — no ownership fields at all — and it
carries no chromatic-registration provenance, so nothing in it can show whether a
centre was re-fitted. Re-run the upstream stages with the current drivers and
point `upstream_root` at that output.

Tests passing does not constitute scientific validation. The synthetic
validation barrier in the campaign must pass before any publication result is
accepted, and it is currently blocked on the axial residual-correlation length
(the z ACF does not decay below ρ = 0.1 within the crop depth).
