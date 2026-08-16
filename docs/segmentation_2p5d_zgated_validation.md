# 2.5D z-gated implementation validation

Date: 2026-08-01

## Scope

This report validates the software contract for `--segmentation-mode 2p5d-zgated`. It does not claim that the gate parameters or downstream multispot thresholds are biologically calibrated for publication.

No source data or existing pipeline outputs were changed during the real-field checks.

## Automated validation

The full repository test suite passes:

```text
77 tests passed
```

Coverage added for the new mode includes:

- centered and edge-clipped gates;
- one-plane gap filling and minimum-width expansion;
- below/above-threshold competing z components;
- weak, flat, and non-finite profiles;
- consensus and full-z quarantine fallback;
- strict `fail` behavior;
- exact time-point and label matching;
- mask hash, depth, schema, type, shape, and failure-policy validation;
- pickle-free NPZ round trips and JSON provenance;
- vectorized mask materialization with noncontiguous label IDs;
- probe-0 and 488 localization using the gated mask;
- explicit audit output when localization is disallowed; and
- end-to-end replication QC proving that quarantined and failed gates remain unevaluable and cannot produce `MultiSpot` or `Replicating`.

The three canonical shell drivers pass syntax checks, and their help/argument paths expose the new mode and failure policy.

## Read-only real-field pilot

The gate generator was exercised against existing 2.5D ownership masks and real DAPI stacks from each modality. To keep this implementation check bounded, rep1, rep3, and Novo used the first five DAPI acquisitions; confocal completed all 20.

| Dataset/field | DAPI acquisitions checked | Labels | z planes | Evaluable gates | Raw status counts | Effective gate width p5/median/p95 |
|---|---:|---:|---:|---:|---|---:|
| rep1 `Location_1` | 5/23 | 2,530 | 11 | 99.98% | edge-clipped 11,516; good 1,131; ambiguous 3 | 8 / 9 / 10 |
| rep3 `Location-01` | 5/24 | 698 | 21 | 100.00% | edge-clipped 2,667; good 823 | 15 / 16 / 18 |
| Novo `Location-01` | 5/24 | 2,365 | 26 | 85.81% | edge-clipped 9,811; ambiguous 1,678; good 336 | 5 / 18 / 23 |
| confocal `Location_1` | 20/20 | 148 | 7 | 99.93% | edge-clipped 2,957; ambiguous 2; good 1 | 5 / 6 / 6 |

Widths are inclusive plane counts after the selected failure policy is applied. For the three five-acquisition pilots, consensus widths are provisional because the production consensus will use all acquisitions.

## Interpretation

- The implementation behaves consistently across TIFF, DAT, and confocal readers and preserves exact time-point identities.
- Most rep1, rep3, and confocal profiles touch a stack boundary after padding. This is recorded as `edge_clipped` but remains evaluable by design.
- The thin seven-plane confocal stack leaves limited room for axial restriction; the typical gate still spans six planes.
- Novo is materially different: 14.19% of the sampled cell/acquisition profiles contain a competing z component strong enough to be ambiguous. The default quarantine policy retains their one-spot localization using consensus support but excludes them from positive multispot evidence.
- The Novo result is a reason to inspect diagnostics and calibrate the ambiguity rule, not a reason to weaken it silently. Any parameter change must be versioned and rerun through the same false-positive and retention checks.

## Remaining publication acceptance work

Before a full four-dataset production rerun:

1. Complete all-acquisition validation on at least one field per modality.
2. Render XY/XZ/YZ gate overlays for good, edge-clipped, ambiguous, consensus-fallback, and full-z-fallback examples.
3. Measure one-spot localization retention relative to legacy 2.5D; target at least 95% before fallback, reporting every loss by reason.
4. Re-estimate multispot model behavior because z support changes voxel count, background support, and BIC.
5. Report outcome changes by dataset and gate status; do not interpret a lower multispot rate alone as improved biological specificity.
6. Freeze the gate parameters and schema version before launching the complete campaign.
