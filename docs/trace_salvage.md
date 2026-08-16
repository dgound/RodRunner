# Trace-based multispot salvage

Trace salvage is an optional, post-drift-correction representation of a walk.
It does not change segmentation, localization, Gaussian fits, bootstrap
results, or the original multispot/replication calls.

The intended case is a walk whose Ori locus was evaluated as a single spot but
which contains an isolated two-component call at an internal locus `N`.  If
`N-1` and `N+1` are both evaluated single spots, the resolver asks whether one
of the two fitted components at `N` is substantially more consistent with the
lateral trace through the two neighbours.

## Frozen decision rule

For each candidate component `c`, the lateral path score is

```text
d_xy(N-1, c) + d_xy(c, N+1).
```

The lower-scoring component is accepted only when both conditions hold:

1. its path score is at most 80% of the other component's path score; and
2. each of its two lateral links is no longer than the clean-link 95th
   percentile calibrated for that location.

The percentile is fitted without using the target location: it is calculated
from full-coverage, clean walks in the other locations of the same dataset.
At least 100 calibration links are required.  Insufficient calibration fails
closed.  The selection calculation uses XY only.  After a component is
selected, its complete drift-corrected XYZ coordinate is retained in the
derived trace.

For v2, a confident single is specifically an evaluated
`tested_not_confirmed` locus; for v3 it is an evaluable locus without a primary
resolved lateral two-component call.  Missing, failed, or merely unqualified
rows are never used as trace neighbours or calibration links.  The coordinate
input must carry the drift vector: missing drift fails closed rather than
silently combining a raw component centre with corrected neighbour positions.

Ori and Ter are never salvaged.  A doublet adjacent to another doublet, a
missing or failed neighbour, a tie or weak winner, an overlong link, or a
missing component coordinate remains unresolved.  A location-only run cannot
fit its own policy; it must use a previously frozen dataset policy.

## Interpretation

An accepted component is a trace-continuity choice, not proof that the rejected
component was an imaging false positive.  In particular, this procedure can
collapse a real discordant biological doublet.  `TraceSalvagedSingleSpotComplete`
therefore means that a complete derived single-coordinate trace could be
constructed under this prespecified rule.  It does not mean experimentally
proven non-replication.

The resolver first writes one enriched CSV per location under `locations/`.
Replication-QC, walk-summary, and trace-decision columns are added to source
rows; no blank or synthetic filler row is ever created.

Python then combines the enriched location files into two flat, full-schema
dataset CSVs: the comprehensive unfiltered one and the subset retaining all rows
from walks with `TraceSalvagedSingleSpotComplete == 1`.  The filtered file is
derived from the comprehensive file at walk level; it does not select individual
timepoint rows.  The resolver also writes a row-level audit, an enriched walk
summary, a frozen policy, and a hash manifest.

## The comprehensive CSV is self-contained

`<prefix>_trace_salvage_full_schema.csv` is the **authoritative, lossless
output**: one row for every localization ATTEMPT, whatever became of it.  The
audit side-car remains for diagnostics, but no consumer needs it to identify
rejected loci, learn why they were rejected, count valid/rejected loci, compute
walk completeness, filter valid walks, or inspect the original unconstrained fit.

The attempt universe is the localizer's per-attempt ledger,
`localizations_640_z_gate_status.csv`, named per location as
`ownership_audit_csv`.  Accepted scientific results are joined **onto** that
universe; the comprehensive file is never assembled by starting from accepted
rows.  A location that declares no ledger keeps the historical behaviour, where
the coordinate rows are the row universe.

Two coordinate inputs are therefore distinguished, and they have different jobs:

| config field | file | job |
|---|---|---|
| `coordinates_csv` | `..._drift_corrected_full_coverage.csv` | the **decision** input: calibration, salvage, every derived trace coordinate |
| `all_coordinates_csv` | `..._drift_corrected.csv` (unfiltered) | canonical coordinates for accepted attempts whose walk lost full coverage |

`all_coordinates_csv` changes no decision.  It exists because a strict-rejected
locus *removes* its walk from the full-coverage set, so without it the 19 accepted
loci of a 19-plus-1 walk could not be represented at all.  The two files need not
share a header — the full-coverage file legitimately carries `movement_nm` /
`movement_def`, which are undefined off it — but the full-coverage row set must be
a subset, and a violation raises.

Every row carries:

| column | meaning |
|---|---|
| `LocalizationAttemptStatus` | `accepted`, `ownership_rejected`, `fit_failed`, or `no_drift_solution` |
| `LocalizationAttemptReason` | the localizer's exact reason string, blank when accepted |
| `UnconstrainedFitZNm` / `YNm` / `XNm` | **audit only**: the unconstrained fitted centre in nanometres, in the raw target-channel frame ownership was decided in |

The audit coordinates are populated for every attempt that produced a centre,
accepted or rejected, so the schema is uniform.  For an accepted row they are
`z_raw/y_raw/x_raw` scaled by the config geometry; for a rejected row they come
from the ledger's `center_fit_{z,y,x}_px`, which the localizer serializes exactly
as it would a localization row — so the audit value **is** the value the verdict
was decided on.  They are deliberately not `OriginalZNm` (drift-corrected) or
`z_raw` (pre-drift pixels), which already mean other things.

For an ownership-rejected or failed attempt, **every** canonical coordinate is
blank: `z_nm/y_nm/x_nm`, `z/y/x`, `z_raw/y_raw/x_raw`, `Original*Nm` and
`Trace*Nm`.  A rejected centre exists in the file for audit and nowhere else; it
cannot enter a trajectory, a replication call, or a salvage decision.  What such a
row does keep is its real `Location`, `OriginalWalkID`, `tp`, the canonical
`CenterOwnership*` verdict and reason, the walk-level replication annotations —
which are per-walk, not per-locus — and the four walk ownership fields.

To use the file:

```python
rows = pandas.read_csv("rep3_trace_salvage_full_schema.csv")
usable   = rows[rows.CenterOwnershipUsable == 1]           # distance-aware usable
exact    = rows[rows.CenterOwnershipValid == 1]             # exact voxel-level valid
rejected = rows[rows.LocalizationAttemptStatus == "ownership_rejected"]
complete = rows[rows.TraceSalvagedSingleSpotComplete == 1]  # == the filtered file
```

Under `production-v1`, `CenterOwnershipUsable == 1` is the appropriate filter:
it includes exact-own and near-background centres (within 1 pixel tolerance).
`CenterOwnershipValid == 1` is the strict voxel-level check and is always a
subset of usable.  The walk-level completeness uses
`CenterOwnershipUsableComplete` (all configured loci usable) and reports a
`TraceCompletenessTier` (exact20/usable20/partial19/partial18/incomplete).
Blank coordinates parse as NaN and do not crash a loader, but they are not zero
and must not be treated as one.

`WalkID` is the deterministic one-based identifier continuous across the whole
combined dataset; `OriginalWalkID` is the location-local identifier the location
files use.  Rejected attempts are reconstructed on the location-local key
`(Location, WalkID, tp)` *before* global renumbering, so location-local IDs
reused across locations cannot collide.

Because the row universe is now the ledger, the comprehensive file also contains
walks that never reached the full-coverage set.  That does not widen the filtered
product: completeness additionally requires a derived trace coordinate at every
configured locus, which only the decision input can supply.  On rep3 Location-01
the unfiltered file grew from 4,656 rows / 194 walks to 16,752 rows / 698 walks
while the filtered file stayed at exactly the same 816 rows and 34 walks, with
identical coordinates.

In the flat dataset-wide files, `WalkID` is a deterministic one-based
continuous identifier shared by all loci in a walk across all locations.
The historical per-location identifier is retained as `OriginalWalkID`.
Per-location files keep the historical identifier as `WalkID`; the rename and
global numbering happen only during the Python combine step.

There is one timepoint column, lowercase `tp`, canonicalized to `TP###`.
Legacy acquisition labels such as `TP002_CYC002` are normalized there; neither
an uppercase `TP` duplicate nor `OriginalTP` is emitted.  The original source
files remain unchanged.

Coordinate columns have explicit roles:

- `z_raw`, `y_raw`, `x_raw` are preserved upstream pre-drift pixel coordinates
  when the drift-corrected input provides them.
- `OriginalZNm`, `OriginalYNm`, `OriginalXNm` are the drift-corrected
  localization before trace salvage.
- `TraceZNm`, `TraceYNm`, `TraceXNm` are the derived trace choice.
- `z_nm`, `y_nm`, `x_nm` are the final downstream coordinates: the trace choice
  at a salvaged locus and the original drift-corrected coordinate otherwise.

For an output prefix `rep3`, the per-locus files are named
`rep3_trace_salvage_full_schema.csv` and
`rep3_trace_salvage_full_schema_complete.csv`.  A location such as
`Location-03` is written first as
`locations/Location-03_trace_salvage_full_schema.csv`.

## Interfaces

The standalone entry point is `pipeline/trace_salvage.py`.  Its JSON config
contains the dataset/evidence schema, chromosome order, Ori/Ter, physical
pixel sizes, fixed rule parameters, and explicit per-location evidence,
drift-corrected coordinate, and walk-summary paths.  No input glob is resolved
inside the statistical core.

```bash
python pipeline/trace_salvage.py run \
  --config trace_salvage_config.json \
  --outdir trace_salvage
```

`fit-policy` freezes a dataset policy separately, and `apply` uses such a
policy without recalibration.  This is the required route for applying an
already calibrated policy to a single location.

The v2 merged-walk finalizer exposes an explicit `--trace_salvage` switch.  The
publication-v1 v3 campaign exposes a separate opt-in registry setting and, when
enabled, schedules trace salvage after final reduction and before annotation.
Both paths leave their historical default outputs unchanged.

## Centre ownership

Each config location may name an optional `ownership_audit_csv`
(`localizations_640_z_gate_status.csv`).  It is preferred over the coordinate CSV
because under `--center-ownership strict` the localizer *drops* a rejected row, so
the coordinate CSV can only represent a rejection as an absence.
`merge_all_walks.py` fills it in automatically when the file exists, together with
`all_coordinates_csv`; it refuses to proceed if the ledger is present but the
unfiltered drift-corrected file is not, rather than dropping accepted loci.  With
neither that table nor per-locus ownership columns present, the dataset predates
the schema and behaviour is unchanged.

**Salvage cannot override ownership.**  Before any decision, the evidence at every
ownership-invalid locus is censored to unevaluable (status
`ownership_unevaluable`, the same spelling the upstream replication stage uses --
they used to differ, so a consumer matching one missed the other), so
that locus cannot be salvaged, cannot anchor a neighbour's salvage decision, is not
read as evidence of a single spot, and exposes no M2 component centres.
`TraceSalvagedSingleSpotComplete == 1` additionally requires
`CenterOwnershipComplete == 1` as an independent conjunct.  Salvage can still
resolve an *accepted* multispot choice; it can never resurrect a rejected locus.

Four walk-level fields join `DERIVED_WALK_FIELDS` and therefore appear in both
flat outputs and in each per-Location full-schema file:

| field | meaning |
|---|---|
| `CenterOwnershipExpectedCount` | size of the configured locus universe (`chromosome_order`), never the surviving rows |
| `CenterOwnershipValidCount` | distinct required loci with `CenterOwnershipValid == 1` |
| `CenterOwnershipRejectedCount` | required loci with an **explicit** rejection verdict |
| `CenterOwnershipComplete` | 1 only when every required locus is present and valid |

A required locus that is absent, or that has a ledger row but no verdict because
its fit failed, is neither valid nor rejected: it is *missing*, recoverable as
`Expected − Valid − Rejected`.  Rejected and missing both make
`CenterOwnershipComplete` zero.  Duplicate identical rows are counted once;
duplicates that contradict each other raise.  The fields describe actual ownership
validity, so they read the same under `allow` — where coordinates stay
operationally available — as under `strict`.  They are blank, not `1`, for a
dataset with no ownership schema.

The manifest gains a `center_ownership` block recording the source per Location,
whether enforcement was active, the locus universe, the censored-locus count, the
number of ownership-complete walks, and the rule text.

An ownership-rejected walk keeps **all** its attempts in the unfiltered
comprehensive CSV — the accepted loci with their coordinates and the rejected loci
as audit-only rows — and contributes **zero** rows to the filtered one.  Nothing
is fabricated: a locus that was never submitted to localization has no ledger row
and therefore no output row.

That extends to a walk whose *every* centre was rejected.  It has no accepted
localization and no replication side-car row, so it used to be absent from the
replication classification and from every merged deliverable.  The attempt ledger
is now also the walk denominator for `classify_replication.py`, which emits one row
per attempted walk and classifies a wholly unevaluable one as `indeterminate_ori`
with every positive flag zero — never `non_replicating` — recording why in
`ReplicationClassificationReason`.  Both comprehensive outputs therefore carry that
walk's audit rows with a populated walk-level classification rather than a blank
one, and it contributes zero rows to the filtered file.  On rep3 Location-01 that
walk is local `WalkID 375` (global 376): 24 audit-only rows,
`all_localizations_ownership_unevaluable`, `CenterOwnershipRejectedCount=20`.
