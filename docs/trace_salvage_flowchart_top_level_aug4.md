# Aug-4 trace-salvage flowchart: top-level view

This is the reader-facing version of the Aug-4 results uploaded to
`dropbox:cellpose/trace_salvage/`. All values below are **walk counts**.

Important: movement, replication, and multispot status are parallel QC views.
They are not sequential filters whose counts should be multiplied together.
The final complete-walk count is determined by the clean/salvage decision.

## Replicate 1

```mermaid
flowchart TD
    A[18,840 initial decision-universe walks]
    A --> B[17,705 walks with coordinate rows]
    B --> C[QC and classification views]
    C --> C1[3,752 evaluated at all 20 planned timepoints]
    C --> C2[5,986 pass TP002-to-TP022 movement <=100 nm]
    C --> C3[1,996 classified Replicating]
    C --> C4[10,011 have at least one multispot locus]
    B --> D[Trace completion decision]
    D --> D1[1,981 complete clean walks]
    D --> D2[450 complete after multispot salvage]
    D1 --> E[2,431 final complete walks]
    D2 --> E
```

## Replicate 3

```mermaid
flowchart TD
    A[25,094 initial decision-universe walks]
    A --> B[22,900 walks with coordinate rows]
    B --> C[QC and classification views]
    C --> C1[18,275 evaluated at all 20 planned timepoints]
    C --> C2[7,972 pass TP002-to-TP022 movement <=100 nm]
    C --> C3[3,431 classified Replicating]
    C --> C4[13,907 have at least one multispot locus]
    B --> D[Trace completion decision]
    D --> D1[6,535 complete clean walks]
    D --> D2[3,421 complete after multispot salvage]
    D1 --> E[9,956 final complete walks]
    D2 --> E
```

## What each group means

**Initial decision-universe walks**

All walks present in the configured acquisition/locus universe before requiring
usable coordinate rows.

**Coordinate-bearing walks**

Walks for which the merged localization output contains at least one usable
coordinate row. Walks with no coordinate rows cannot enter trajectory analysis.

**Complete 20-timepoint walks**

Walks with evaluated localizations at all 20 planned chromosome loci (TP002–
TP021). This is a completeness/QC measurement; it is not identical to the
final clean-walk definition.

**Movement QC**

The displacement check compares the first and last planned loci, TP002 to
TP022, using the configured XYZ movement threshold of 100 nm. It flags or
filters excessive movement; it is reported separately from trace salvage.

**Replication classification**

The replication rules use the Ori/Ter hierarchy and the per-locus two-component
evidence to label walks as replicating, non-replicating, indeterminate, or
multispot-only. A replication label is a biological classification, not a
guarantee that every locus is a clean single spot.

**Multispot group**

Any walk with at least one multispot locus. These walks are not automatically
discarded: isolated, non-Ori/non-Ter doublets can enter the neighbor-guided
salvage test.

**Complete clean walks**

All planned loci are confidently single and have finite drift-corrected XYZ
coordinates.

**Complete after multispot salvage**

A walk that would otherwise contain an isolated discordant doublet, but passes
the neighbor-continuity, XY calibration, and winner-margin checks. The selected
component is retained as the trace coordinate.

**Final complete walks**

`complete clean + complete after salvage`: 2,431 for rep1 and 9,956 for rep3.
