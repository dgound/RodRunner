# Theta-graph and link detection

`pipeline/detect_topology.py` is the canonical downstream topology stage. It
uses calibrated `replication_qc_v3` locus calls and the merged drift-corrected
full-coverage one-spot coordinate CSV. It never changes a spot call, fills a
missing locus, or moves a measured coordinate.

The detector is a single Python orchestrator over one C library,
`pipeline/topology_engine.so`, which holds every numeric step: theta-pattern
classification, global sister assignment, strand and loop validation, the
Yamada polynomial (skein resolution with OpenMP over states), and the classical
linking number. Build it once next to the script:

```bash
gcc -O3 -fopenmp -shared -fPIC -o pipeline/topology_engine.so pipeline/topology_engine.c -lm
```

Input CSVs are streamed in row chunks (`--chunk-rows`, default 100000): a file
with fewer rows loads as a single chunk, larger files never hold more than one
chunk of compact tuples. Only the columns needed are kept, and coordinates and
fitter evidence are retained only for walks that can reach a topology
calculation, so memory is set by the candidate count, not by the size of the
comprehensive evidence file.

## Configuration

The detector is not tied to a dataset registry. Any dataset can supply a small
JSON file with its locus order and physical sampling:

```json
{
  "family_id": "my-dataset-publication-v1",
  "chromosome_order": ["TP002", "TP003", "TP004", "TP005"],
  "ori_tp": "TP002",
  "ter_tp": "TP004",
  "geometry": {
    "xy_pixel_nm": 103.28,
    "z_step_nm": 100.0
  },
  "power_lcb_min": 0.8,
  "exploratory_bh_q": 0.05,
  "genomic_bp": {
    "TP002": 0,
    "TP003": 250000,
    "TP004": 500000,
    "TP005": 750000
  }
}
```

`genomic_bp` is optional. `family_id`, the chromosome fields, geometry,
`power_lcb_min`, and `exploratory_bh_q` are required. The existing
publication campaign JSON files already contain the required keys, but a new
dataset does not need to be added to that campaign or to a hard-coded manifest.

## Run

```bash
python pipeline/detect_topology.py \
  --locus-results /path/to/publication_v1_locus_results.csv \
  --coordinates-csv /path/to/publication_v1_trace_salvage_full_schema.csv \
  --config /path/to/dataset.json \
  --reduction-manifest /path/to/publication_v1_manifest.json \
  --outdir /path/to/topology
```

`--max-crossings` (default 16) bounds the skein resolution: a theta whose best
projection over all seeds needs more crossings is written with status
`theta_crossings_exceeded` and blank polynomial fields, but its reconstructed
strand points are still emitted. `--threads` sets the OpenMP thread count; under
SLURM it defaults to `SLURM_CPUS_PER_TASK`.

For exploratory work on an existing unfiltered trace-salvage full-schema CSV,
the publication-v1 reduction is optional:

```bash
python pipeline/detect_topology.py \
  --evidence-mode legacy-full-schema \
  --locus-results /path/to/trace_salvage_full_schema.csv \
  --coordinates-csv /path/to/trace_salvage_full_schema.csv \
  --config /path/to/dataset.json \
  --outdir /path/to/topology \
  --skip-yamada
```

This mode uses `RawTwoComponentCall`, `RawEvaluable`, and
`RawConfidentSingle` exactly as stored; it does not recalibrate or reinterpret
them. It also does not require strict-ownership provenance. The output manifest
records `publication_calibrated: false` and the corresponding limitations.
`--skip-yamada` still reconstructs and validates the complete three-strand XYZ
geometry, but leaves the polynomial fields blank so selected candidates can be
evaluated later. Link detection is unaffected.

Use the unfiltered comprehensive trace-salvage file, not its
`*_complete.csv` subset. The detector reads `OriginalXNm/OriginalYNm/OriginalZNm`,
never the salvaged `Trace*Nm` coordinates. A combined unfiltered drift-corrected
CSV containing Dataset and Location is also accepted. The detector requires only
the configured chromosome loci, so missing unrelated acquisitions do not
remove a walk before topology evaluation. When a combined coordinate CSV contains both a
global `WalkID` and `OriginalWalkID`, the detector joins the v3 evidence with
`(Dataset, Location, OriginalWalkID, TP)`.

The reduction manifest is mandatory. It binds the locus-results hash and the
exact config hash, and verifies the final family, chromosome order, Ori/Ter,
power threshold, and exploratory threshold before topology is evaluated.
For every coordinate actually used, the detector also requires its raw
one-spot centre to match the centre carried by the v3 result and recomputes
`(raw - drift) * pixel_size`; a same-key CSV from another localization run or
an inconsistent coordinate frame fails closed.

## What is called

A theta-pattern candidate must have a doubled Ori and a contiguous doubled run
extending beyond Ori on both chromosome arms. A double after a gap is
discordant and is reported but not traced. Every locus on the complementary
single-copy arc must have a one-spot coordinate and be a sensitivity-qualified
non-discovery: adequate power at the configured alternative and no exploratory
two-component signal. An ordinary non-significant locus is not assumed to be
single-copy.

The two fitted components at each doubled locus are assigned globally to two
sister paths by minimizing total adjacent 3D distance. The assignment is also
run in XY; disagreement or a numerical tie with the runner-up is reported as
unstable. The two fork coordinates are
explicit inferred boundary midpoints. They are not observations. The Yamada
polynomial is recomputed with a second, pair-midpoint fork model, and a change
is reported as fork-model instability.

A classical link candidate requires every configured locus to be a calibrated
primary double. Global circular sister assignment produces two polygonal 3D
loops. The link stage requires each loop to be simple and the two loops to be
disjoint, then computes the oriented linking number across deterministic
projections. Linking number zero is reported as `nominal_unlinked_polygon`,
not as a detected link.

The current v3 fitter does not provide two-component-center standard errors.
One-spot errors are retained when present, but sister/fork errors and
`PerturbationStable` remain blank. Zero uncertainty is never fabricated. For
that reason the positive status is explicitly `nominal_theta_graph` or
`nominal_linked_polygon`: it describes the fitted straight-segment geometry,
not topology proven stable under localization uncertainty or every admissible
interpolation between sparse loci.

## Outputs

Exactly five files are written:

- `theta_graph_summary.csv`: one row per theta-pattern candidate and its
  nominal-geometry or failure status, plus raw and normalized Yamada
  polynomials.
- `theta_strand_points.csv`: long-form XYZ-nm points for `unreplicated`,
  `sister_A`, and `sister_B`. All three paths share the same two explicit fork
  vertices.
- `link_summary.csv`: one row per fully doubled candidate, including linking
  number and minimum inter-loop distance.
- `link_loop_points.csv`: long-form XYZ-nm vertices for both sister loops.
  Closure is declared by `Closed=1`; the first vertex is not duplicated.
- `topology_manifest.json`: input hashes, coordinate transform, policies, row
  counts, and candidate counts.

To reconstruct one theta graph, filter `theta_strand_points.csv` on `GraphID`
and group by `StrandID` in `PointOrder`; the three strands share their first
and last rows exactly. To reconstruct a link, filter `link_loop_points.csv` on
`LinkID`, group by `LoopID` in `PointOrder`, and append vertex zero to close
each loop.

The CSV coordinate order is always `XNm, YNm, ZNm`. Two-component raw-pixel
centres are transformed with
`XYZ_nm = (raw_target_pixels - drift_pixels) * axis_scale_nm` before sister
assignment, matching the one-spot drift-corrected frame.
