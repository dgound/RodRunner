# Aug-4 trace-salvage walk flowchart

These counts are from the uploaded Aug-4 `trace_salvage_output_v2` summaries in
`dropbox:cellpose/trace_salvage/`. Counts are walks, not locus rows.

The status branches are mutually exclusive after the coordinate-coverage step;
they are not sequential filters that can be subtracted in arbitrary order.

## Replicate 1

```mermaid
flowchart TD
    A[18,840 decision-universe walks] --> B[17,705 walks represented after merge]
    B --> X[1,135 had no coordinate rows]
    B --> C[1,981 complete clean walks]
    B --> D[5,410 isolated-doublet salvage candidates]
    B --> E[4,249 Ori not confidently single]
    B --> F[6,065 incomplete or unevaluable]
    D --> G{Salvage checks: neighbor continuity,\nXY link calibration, isolated doublet,\nTP017/TP007 exclusion, winner margin <= 0.8}
    G --> H[450 complete after salvage]
    G --> I[4,960 unresolved doublet walks]
    C --> J[2,431 final complete walks]
    H --> J
```

## Replicate 3

```mermaid
flowchart TD
    A[25,094 decision-universe walks] --> B[22,900 walks represented after merge]
    B --> X[2,194 had no coordinate rows]
    B --> C[6,535 complete clean walks]
    B --> D[9,705 isolated-doublet salvage candidates]
    B --> E[3,958 Ori not confidently single]
    B --> F[2,702 incomplete or unevaluable]
    D --> G{Salvage checks: neighbor continuity,\nXY link calibration, isolated doublet,\nTP017/TP007 exclusion, winner margin <= 0.8}
    G --> H[3,421 complete after salvage]
    G --> I[6,284 unresolved doublet walks]
    C --> J[9,956 final complete walks]
    H --> J
```

## Final arithmetic

| Dataset | Clean | Complete after salvage | Final complete |
|---|---:|---:|---:|
| rep1 | 1,981 | 450 | **2,431** |
| rep3 | 6,535 | 3,421 | **9,956** |

The salvage policy required all of the following for a discordant locus:

- the doublet was isolated and Ori was confidently single;
- TP017 (Ori) and TP007 (Ter) were never salvaged;
- both neighboring XY links were within the location leave-one-out 95th-percentile calibration limit;
- the winning component's path was at most 0.8 times the runner-up path;
- the selected coordinate was the full drift-corrected XYZ coordinate of that component.

The manifests do not store cumulative walk counts after each individual salvage
criterion. They store the final mutually exclusive walk status, so the 450/3,421
and 4,960/6,284 values are the defensible pass/fail counts for the combined
salvage decision.
