#!/usr/bin/env python3
"""Reconstruct theta-graph and two-loop geometry from calibrated v3 calls.

This is deliberately a downstream topology stage.  It does not reclassify
spots or interpolate missing loci.  Input two-component centres are the raw
target-channel pixel centres from replication QC; they are put into the same
drift-corrected XYZ-nm frame as the one-component trajectory before tracing.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

try:
    from pipeline.topology_core import (
        ChromosomeLayout, LocusPair, SisterAssignment, TopologyInputError,
        analyze_theta_pattern, assign_circular, assign_open,
    )
    from pipeline.topology_io import write_topology_outputs
    from pipeline.topology_yamada import (
        TopologyValidationError, evaluate_link, evaluate_theta, load_yamada,
        validate_theta_strands,
    )
except ModuleNotFoundError:  # direct ``python pipeline/detect_topology.py``
    from topology_core import (  # type: ignore
        ChromosomeLayout, LocusPair, SisterAssignment, TopologyInputError,
        analyze_theta_pattern, assign_circular, assign_open,
    )
    from topology_io import write_topology_outputs  # type: ignore
    from topology_yamada import (  # type: ignore
        TopologyValidationError, evaluate_link, evaluate_theta, load_yamada,
        validate_theta_strands,
    )


class TopologyDetectionError(ValueError):
    """Input data cannot support the requested topology analysis."""


@dataclass(frozen=True)
class DetectionConfig:
    layout: ChromosomeLayout
    xy_pixel_nm: float
    z_step_nm: float
    genomic_bp: Mapping[str, float]
    power_lcb_min: float
    exploratory_q: float
    family_id: str

    @property
    def geometry_id(self) -> str:
        payload = {"xy_pixel_nm": self.xy_pixel_nm, "z_step_nm": self.z_step_nm}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Coordinate:
    xyz_nm: tuple[float, float, float]
    raw_xyz_px: tuple[float, float, float]
    drift_zyx_px: tuple[float, float, float]
    xyz_se_nm: tuple[float, float, float] | None


WalkKey = tuple[str, str, int]
LocusKey = tuple[str, str, int, str]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ())
            if not fields:
                raise TopologyDetectionError(f"{path}: missing CSV header")
            if len(fields) != len(set(fields)):
                raise TopologyDetectionError(f"{path}: duplicate CSV columns")
            return fields, [dict(row) for row in reader]
    except OSError as exc:
        raise TopologyDetectionError(f"cannot read {path}: {exc}") from exc


def _csv_fields(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = list(csv.DictReader(handle).fieldnames or ())
    except OSError as exc:
        raise TopologyDetectionError(f"cannot read {path}: {exc}") from exc
    if not fields:
        raise TopologyDetectionError(f"{path}: missing CSV header")
    if len(fields) != len(set(fields)):
        raise TopologyDetectionError(f"{path}: duplicate CSV columns")
    return fields


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)
    except OSError as exc:
        raise TopologyDetectionError(f"cannot read {path}: {exc}") from exc


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TopologyDetectionError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise TopologyDetectionError(f"{label} must be finite, got {value!r}")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise TopologyDetectionError(f"{label} must be positive")
    return result


def _optional_finite(value: Any, label: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    return _finite(value, label)


def _walk_id(value: Any, label: str) -> int:
    text = str(value).strip()
    try:
        result = int(text)
    except ValueError as exc:
        raise TopologyDetectionError(f"{label} must be an integer, got {value!r}") from exc
    if text not in {str(result), f"+{result}"} or result < 0:
        raise TopologyDetectionError(f"{label} must be a non-negative integer")
    return result


def _bit(value: Any, label: str) -> int:
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    raise TopologyDetectionError(f"{label} must be 0 or 1, got {value!r}")


def _nonblank(value: Any, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise TopologyDetectionError(f"{label} must not be blank")
    return result


def load_config(path: str | Path) -> DetectionConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyDetectionError(f"cannot read configuration {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TopologyDetectionError("configuration must be a JSON object")
    try:
        order = tuple(payload["chromosome_order"])
        ori = str(payload["ori_tp"])
        ter = str(payload["ter_tp"])
        geometry = payload["geometry"]
        xy = _positive(geometry["xy_pixel_nm"], "geometry.xy_pixel_nm")
        z = _positive(geometry["z_step_nm"], "geometry.z_step_nm")
        power_lcb_min = _finite(payload["power_lcb_min"], "power_lcb_min")
        exploratory_q = _finite(payload["exploratory_bh_q"], "exploratory_bh_q")
    except (KeyError, TypeError) as exc:
        raise TopologyDetectionError(
            "configuration requires chromosome_order, ori_tp, ter_tp, and "
            "geometry.{xy_pixel_nm,z_step_nm}, power_lcb_min, and "
            "exploratory_bh_q"
        ) from exc
    try:
        layout = ChromosomeLayout(order, ori=ori, ter=ter)
    except TopologyInputError as exc:
        raise TopologyDetectionError(str(exc)) from exc
    bp_payload = payload.get("genomic_bp", {})
    if not isinstance(bp_payload, dict):
        raise TopologyDetectionError("genomic_bp must be a TP-to-number object")
    outside = set(bp_payload) - layout.universe
    if outside:
        raise TopologyDetectionError(
            f"genomic_bp contains loci outside chromosome_order: {sorted(outside)}")
    genomic_bp = {str(tp): _finite(value, f"genomic_bp.{tp}")
                  for tp, value in bp_payload.items()}
    if not 0.0 <= power_lcb_min <= 1.0:
        raise TopologyDetectionError("power_lcb_min must lie in [0, 1]")
    if not 0.0 < exploratory_q < 1.0:
        raise TopologyDetectionError("exploratory_bh_q must lie in (0, 1)")
    return DetectionConfig(
        layout, xy, z, genomic_bp, power_lcb_min, exploratory_q,
        str(payload.get("family_id", "")).strip())


def _locus_key(row: Mapping[str, Any], path: Path, number: int) -> LocusKey:
    return (
        _nonblank(row.get("Dataset", ""), f"{path}:{number} Dataset"),
        _nonblank(row.get("Location", ""), f"{path}:{number} Location"),
        _walk_id(row.get("WalkID", ""), f"{path}:{number} WalkID"),
        _nonblank(row.get("TP", ""), f"{path}:{number} TP"),
    )


def _exploratory_locus_key(
    row: Mapping[str, Any], path: Path, number: int,
) -> LocusKey:
    """Read a key from a merged legacy full-schema row.

    Comprehensive trace-salvage files renumber ``WalkID`` globally and retain
    the location-local identifier in ``OriginalWalkID``.  Replication evidence
    is location-local, so topology joins must use the latter when present.
    """
    original = str(row.get("OriginalWalkID", "")).strip()
    walk_field = "OriginalWalkID" if original else "WalkID"
    tp = _first(row, ("TP", "tp"), "TP")
    return (
        _nonblank(row.get("Dataset", ""), f"{path}:{number} Dataset"),
        _nonblank(row.get("Location", ""), f"{path}:{number} Location"),
        _walk_id(row.get(walk_field, ""), f"{path}:{number} {walk_field}"),
        _nonblank(tp, f"{path}:{number} TP"),
    )


def load_locus_results(
    path: str | Path, config: DetectionConfig,
) -> dict[WalkKey, dict[str, dict[str, str]]]:
    source = Path(path)
    fields, rows = _read_csv(source)
    required = {
        "Dataset", "Location", "WalkID", "TP", "v3_schema_version",
        "v3_locus_evaluable", "ResolvedLateralTwoComponentPrimary",
        "v3_reduction_phase", "v3_profile_hash", "v3_profile_id",
        "v3_geometry_id", "v3_power_lcb_effective", "v3_locus_holm_p",
        "z", "y", "x",
        "v3_m2_c1_z", "v3_m2_c1_y", "v3_m2_c1_x",
        "v3_m2_c2_z", "v3_m2_c2_y", "v3_m2_c2_x",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise TopologyDetectionError(
            f"{source}: not a calibrated v3 locus-results file; missing {missing}")
    if not rows:
        raise TopologyDetectionError(f"{source}: no locus rows")
    seen: set[LocusKey] = set()
    grouped: dict[WalkKey, dict[str, dict[str, str]]] = defaultdict(dict)
    witnessed_v3 = False
    profile_hashes: set[str] = set()
    profile_ids: set[str] = set()
    geometry_ids: set[str] = set()
    for number, row in enumerate(rows, 2):
        key = _locus_key(row, source, number)
        if key in seen:
            raise TopologyDetectionError(f"{source}:{number}: duplicate locus key {key}")
        seen.add(key)
        schema = str(row.get("v3_schema_version", "")).strip()
        if schema:
            if schema != "replication_qc_v3":
                raise TopologyDetectionError(
                    f"{source}:{number}: unexpected v3_schema_version {schema!r}")
            witnessed_v3 = True
        phase = str(row.get("v3_reduction_phase", "")).strip()
        if phase != "final":
            raise TopologyDetectionError(
                f"{source}:{number}: v3_reduction_phase must be 'final', got {phase!r}")
        evaluable = _bit(row.get("v3_locus_evaluable", ""),
                         f"{source}:{number} v3_locus_evaluable")
        doubled = _bit(row.get("ResolvedLateralTwoComponentPrimary", ""),
                       f"{source}:{number} ResolvedLateralTwoComponentPrimary")
        if doubled and not evaluable:
            raise TopologyDetectionError(
                f"{source}:{number}: a doubled locus cannot be unevaluable")
        if (evaluable or doubled) and schema != "replication_qc_v3":
            raise TopologyDetectionError(
                f"{source}:{number}: evaluable v3 locus has no v3 schema witness")
        for field, target in (("v3_profile_hash", profile_hashes),
                              ("v3_profile_id", profile_ids),
                              ("v3_geometry_id", geometry_ids)):
            value = str(row.get(field, "")).strip()
            if value:
                target.add(value)
            elif evaluable or doubled:
                raise TopologyDetectionError(
                    f"{source}:{number}: evaluable v3 locus lacks {field}")
        if key[3] in config.layout.universe:
            grouped[key[:3]][key[3]] = row
    if not witnessed_v3:
        raise TopologyDetectionError(
            f"{source}: no row identifies replication_qc_v3 evidence")
    for name, values in (("profile hashes", profile_hashes),
                         ("profile IDs", profile_ids),
                         ("geometry IDs", geometry_ids)):
        if len(values) != 1:
            raise TopologyDetectionError(
                f"{source}: mixed v3 {name}: {sorted(values)}")
    if geometry_ids != {config.geometry_id}:
        raise TopologyDetectionError(
            f"{source}: v3 geometry {next(iter(geometry_ids))!r} does not match "
            f"configuration geometry {config.geometry_id!r}")
    if not grouped:
        raise TopologyDetectionError(
            f"{source}: no rows overlap configured chromosome_order")
    return dict(grouped)


def load_exploratory_full_schema(
    path: str | Path, config: DetectionConfig,
) -> tuple[
    dict[WalkKey, dict[str, dict[str, str]]], int, Counter[str],
]:
    """Normalize legacy trace-salvage evidence for exploratory topology.

    This mode deliberately consumes the already-declared ``Raw*`` evidence
    columns.  It does not reinterpret fit statistics, invent power, or claim
    publication-v1 calibration.
    """
    source = Path(path)
    fields = _csv_fields(source)
    required = {
        "Dataset", "Location", "WalkID", "RawTwoComponentCall",
        "RawEvaluable", "RawConfidentSingle", "x_raw", "y_raw", "z_raw",
        "m2_c1_z", "m2_c1_y", "m2_c1_x",
        "m2_c2_z", "m2_c2_y", "m2_c2_x",
    }
    if not ({"TP", "tp"} & set(fields)):
        required.add("TP or tp")
    missing = sorted(required - set(fields))
    if missing:
        raise TopologyDetectionError(
            f"{source}: not a trace-salvage full-schema evidence file; "
            f"missing {missing}")
    tp_index = {tp: index for index, tp in enumerate(config.layout.order)}
    # seen, evaluable, doubled, confident-single bit masks per walk.  The
    # compact representation keeps million-row comprehensive inputs bounded.
    states: dict[WalkKey, list[int]] = {}
    row_count = 0
    for number, original in enumerate(_iter_csv_rows(source), 2):
        row_count += 1
        key = _exploratory_locus_key(original, source, number)
        if key[3] not in config.layout.universe:
            continue
        evaluable = _bit(
            original.get("RawEvaluable", ""),
            f"{source}:{number} RawEvaluable")
        doubled = _bit(
            original.get("RawTwoComponentCall", ""),
            f"{source}:{number} RawTwoComponentCall")
        confident_single = _bit(
            original.get("RawConfidentSingle", ""),
            f"{source}:{number} RawConfidentSingle")
        if doubled and not evaluable:
            raise TopologyDetectionError(
                f"{source}:{number}: RawTwoComponentCall cannot be set when "
                "RawEvaluable is zero")
        if confident_single and (doubled or not evaluable):
            raise TopologyDetectionError(
                f"{source}:{number}: RawConfidentSingle conflicts with the "
                "raw evaluability/two-component verdict")
        bit = 1 << tp_index[key[3]]
        state = states.setdefault(key[:3], [0, 0, 0, 0])
        if state[0] & bit:
            raise TopologyDetectionError(
                f"{source}:{number}: duplicate exploratory locus key {key}")
        state[0] |= bit
        if evaluable:
            state[1] |= bit
        if doubled:
            state[2] |= bit
        if confident_single:
            state[3] |= bit

    if row_count == 0:
        raise TopologyDetectionError(f"{source}: no locus rows")
    if not states:
        raise TopologyDetectionError(
            f"{source}: no rows overlap configured chromosome_order")

    pattern_counts: Counter[str] = Counter()
    candidate_walks: set[WalkKey] = set()
    for walk_key, state in states.items():
        doubled_loci = {
            tp for tp, index in tp_index.items() if state[2] & (1 << index)}
        confident_singles = {
            tp for tp, index in tp_index.items() if state[3] & (1 << index)}
        pattern = analyze_theta_pattern(
            config.layout, doubled_loci=doubled_loci,
            evaluable_loci=doubled_loci | confident_singles)
        pattern_counts[pattern.pattern_status] += 1
        if (pattern.is_candidate or
                pattern.pattern_status == "fully_doubled_not_theta"):
            candidate_walks.add(walk_key)

    # A second streaming pass retains detailed fitter coordinates only for the
    # walks that can reach either topology calculation.
    grouped: dict[WalkKey, dict[str, dict[str, str]]] = defaultdict(dict)
    for number, original in enumerate(_iter_csv_rows(source), 2):
        key = _exploratory_locus_key(original, source, number)
        if key[3] not in config.layout.universe or key[:3] not in candidate_walks:
            continue
        state = states[key[:3]]
        bit = 1 << tp_index[key[3]]
        evaluable = int(bool(state[1] & bit))
        doubled = int(bool(state[2] & bit))
        confident_single = int(bool(state[3] & bit))
        row = {
            "Dataset": key[0], "Location": key[1],
            "WalkID": str(key[2]), "TP": key[3],
            "v3_locus_evaluable": str(evaluable),
            "ResolvedLateralTwoComponentPrimary": str(doubled),
            "TopologyConfidentSingle": str(confident_single),
            # The comprehensive CSV's x/y/z are already drift-corrected;
            # replication-QC component centres and its source one-spot centre
            # are in the raw target-channel pixel frame.
            "x": str(original.get("x_raw", "")),
            "y": str(original.get("y_raw", "")),
            "z": str(original.get("z_raw", "")),
        }
        for component in (1, 2):
            for axis in ("z", "y", "x"):
                row[f"v3_m2_c{component}_{axis}"] = str(
                    original.get(f"m2_c{component}_{axis}", ""))
        grouped[key[:3]][key[3]] = row
    return dict(grouped), len(states), pattern_counts


def validate_reduction_manifest(
    path: str | Path, *, config_path: str | Path, locus_results: str | Path,
    config: DetectionConfig, loci_by_walk: Mapping[WalkKey, Any],
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyDetectionError(
            f"cannot read final reduction manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TopologyDetectionError("final reduction manifest must be a JSON object")
    if payload.get("schema_version") != "replication_publication_v1_results_v1":
        raise TopologyDetectionError(
            f"{source}: unsupported final reduction manifest schema")
    if payload.get("reduction_phase") != "final":
        raise TopologyDetectionError(f"{source}: reduction_phase is not final")
    datasets = {key[0] for key in loci_by_walk}
    if len(datasets) != 1 or payload.get("dataset") not in datasets:
        raise TopologyDetectionError(
            f"{source}: manifest dataset does not match locus results")
    if not config.family_id or payload.get("family_id") != config.family_id:
        raise TopologyDetectionError(
            f"{source}: family_id does not match the configuration")
    if (tuple(payload.get("chromosome_order", ())) != config.layout.order or
            payload.get("ori_tp") != config.layout.ori or
            payload.get("ter_tp") != config.layout.ter):
        raise TopologyDetectionError(
            f"{source}: chromosome order/Ori/Ter differ from configuration")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise TopologyDetectionError(f"{source}: thresholds are missing")
    exploratory = thresholds.get("exploratory")
    if (not isinstance(exploratory, dict) or exploratory.get("method") != "BH" or
            _finite(exploratory.get("q"), "manifest exploratory q") !=
            config.exploratory_q or
            _finite(thresholds.get("power_lcb_min"),
                    "manifest power_lcb_min") != config.power_lcb_min):
        raise TopologyDetectionError(
            f"{source}: single-copy thresholds differ from final reduction")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise TopologyDetectionError(f"{source}: input/output hashes are missing")
    if inputs.get("config_sha256") != _sha256(Path(config_path)):
        raise TopologyDetectionError(
            f"{source}: configuration hash differs from final reduction")
    if outputs.get("locus_results_sha256") != _sha256(Path(locus_results)):
        raise TopologyDetectionError(
            f"{source}: locus-results hash differs from final reduction")
    return payload


def _coordinate_walk_id(row: Mapping[str, Any], path: Path, number: int) -> int:
    original = str(row.get("OriginalWalkID", "")).strip()
    field = "OriginalWalkID" if original else "WalkID"
    return _walk_id(row.get(field, ""), f"{path}:{number} {field}")


def _first(row: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    for name in names:
        value = row.get(name, "")
        if str(value).strip():
            return value
    raise TopologyDetectionError(f"missing {label}; tried {list(names)}")


def load_coordinates(
    path: str | Path, config: DetectionConfig, *, require_ownership: bool = True,
    allowed_walks: set[WalkKey] | None = None,
) -> dict[LocusKey, Coordinate]:
    source = Path(path)
    fields = _csv_fields(source)
    required_identity = {"Location", "WalkID"}
    if not required_identity <= set(fields) or not ({"TP", "tp"} & set(fields)):
        raise TopologyDetectionError(
            f"{source}: coordinates need Location, WalkID, and TP or tp")
    raw_fields = {"x_raw", "y_raw", "z_raw"}
    if not raw_fields <= set(fields):
        raise TopologyDetectionError(
            f"{source}: coordinates need x_raw, y_raw, and z_raw to bind them "
            "to the v3 localization source")
    original_names = ("OriginalXNm", "OriginalYNm", "OriginalZNm")
    coordinate_names = (original_names if set(original_names) <= set(fields)
                        else ("x_nm", "y_nm", "z_nm"))
    if not set(coordinate_names) <= set(fields):
        raise TopologyDetectionError(
            f"{source}: coordinates need OriginalXNm/OriginalYNm/OriginalZNm "
            "or x_nm/y_nm/z_nm")
    result: dict[LocusKey, Coordinate] = {}
    drift_by_acquisition: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for number, row in enumerate(_iter_csv_rows(source), 2):
        # Audit-only rejected attempts deliberately have no usable coordinates.
        xyz_values = [row.get(name, "") for name in coordinate_names]
        populated = [bool(str(value).strip()) for value in xyz_values]
        if any(populated) and not all(populated):
            raise TopologyDetectionError(
                f"{source}:{number}: partially populated XYZ-nm coordinates")
        if not any(populated):
            continue
        dataset = _nonblank(row.get("Dataset", ""), f"{source}:{number} Dataset")
        location = _nonblank(row.get("Location", ""), f"{source}:{number} Location")
        walk = _coordinate_walk_id(row, source, number)
        tp = _nonblank(_first(row, ("TP", "tp"), "TP"),
                       f"{source}:{number} TP")
        if tp not in config.layout.universe:
            continue
        key = (dataset, location, walk, tp)
        if allowed_walks is not None and key[:3] not in allowed_walks:
            continue
        if config.family_id and "v3_family_id" in fields:
            family = _nonblank(
                row.get("v3_family_id", ""), f"{source}:{number} v3_family_id")
            if family != config.family_id:
                raise TopologyDetectionError(
                    f"{source}:{number}: coordinate family {family!r} does not "
                    f"match configuration family {config.family_id!r}")
        if key in result:
            raise TopologyDetectionError(f"{source}:{number}: duplicate coordinate key {key}")
        xyz = tuple(_finite(row[name], f"{source}:{number} {name}")
                    for name in coordinate_names)
        raw_xyz = tuple(_finite(row[name], f"{source}:{number} {name}")
                        for name in ("x_raw", "y_raw", "z_raw"))
        drift = tuple(_finite(_first(row, (name, alias), f"{name}"),
                              f"{source}:{number} {name}")
                      for name, alias in (("drift_z", "DriftZ"),
                                          ("drift_y", "DriftY"),
                                          ("drift_x", "DriftX")))
        acquisition = (dataset, location, tp)
        previous_drift = drift_by_acquisition.setdefault(acquisition, drift)
        if previous_drift != drift:
            raise TopologyDetectionError(
                f"{source}:{number}: drift differs within acquisition {acquisition}")
        if require_ownership and "CenterOwnershipValid" not in fields:
            raise TopologyDetectionError(
                f"{source}: coordinates lack CenterOwnershipValid; strict-owned "
                "one-spot coordinates are required")
        if (require_ownership and
                _bit(row.get("CenterOwnershipValid", ""),
                     f"{source}:{number} CenterOwnershipValid") != 1):
            raise TopologyDetectionError(
                f"{source}:{number}: usable coordinate is not strict-owned")
        error_values = [row.get(name, "") for name in ("x_err", "y_err", "z_err")]
        error_populated = [bool(str(value).strip()) for value in error_values]
        if any(error_populated) and not all(error_populated):
            raise TopologyDetectionError(
                f"{source}:{number}: partially populated x_err/y_err/z_err")
        xyz_se = None
        if all(error_populated):
            raw_errors = tuple(_finite(value, f"{source}:{number} localization error")
                               for value in error_values)
            if all(value >= 0.0 for value in raw_errors):
                xyz_se = (raw_errors[0] * config.xy_pixel_nm,
                          raw_errors[1] * config.xy_pixel_nm,
                          raw_errors[2] * config.z_step_nm)
        result[key] = Coordinate(
            xyz_nm=xyz, raw_xyz_px=raw_xyz, drift_zyx_px=drift,
            xyz_se_nm=xyz_se)  # type: ignore[arg-type]
    if not result and (allowed_walks is None or allowed_walks):
        raise TopologyDetectionError(f"{source}: no usable coordinate rows")
    return result


def _component_pair(
    row: Mapping[str, Any], coordinate: Coordinate, config: DetectionConfig,
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    dz, dy, dx = coordinate.drift_zyx_px
    components = []
    for component in (1, 2):
        z = _finite(row.get(f"v3_m2_c{component}_z", ""),
                    f"{label} component {component} z")
        y = _finite(row.get(f"v3_m2_c{component}_y", ""),
                    f"{label} component {component} y")
        x = _finite(row.get(f"v3_m2_c{component}_x", ""),
                    f"{label} component {component} x")
        components.append(((x - dx) * config.xy_pixel_nm,
                           (y - dy) * config.xy_pixel_nm,
                           (z - dz) * config.z_step_nm))
    if components[0] == components[1]:
        raise TopologyDetectionError(f"{label}: coincident two-component centres")
    return components[0], components[1]


def _verify_coordinate_source(
    row: Mapping[str, Any], coordinate: Coordinate, config: DetectionConfig,
    label: str,
) -> None:
    source_xyz = tuple(_finite(row.get(axis, ""), f"{label} source {axis}")
                       for axis in ("x", "y", "z"))
    if any(abs(first - second) > 5e-4
           for first, second in zip(source_xyz, coordinate.raw_xyz_px)):
        raise TopologyDetectionError(
            f"{label}: coordinate CSV raw centre differs from the v3 source")
    dz, dy, dx = coordinate.drift_zyx_px
    raw_x, raw_y, raw_z = coordinate.raw_xyz_px
    expected = ((raw_x - dx) * config.xy_pixel_nm,
                (raw_y - dy) * config.xy_pixel_nm,
                (raw_z - dz) * config.z_step_nm)
    # drift_correct serializes XYZ nm to 0.1 nm.
    if any(abs(observed - calculated) > 0.07
           for observed, calculated in zip(coordinate.xyz_nm, expected)):
        raise TopologyDetectionError(
            f"{label}: drift-corrected XYZ is inconsistent with raw centre, "
            "drift, and configured pixel size")


def _component_id(
    point: tuple[float, float, float],
    original: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> int:
    if point == original[0]:
        return 1
    if point == original[1]:
        return 2
    raise TopologyDetectionError("assigned coordinate is not an input component")


def _mean(first: Sequence[float], second: Sequence[float]) -> tuple[float, float, float]:
    return tuple((float(a) + float(b)) / 2.0 for a, b in zip(first, second))  # type: ignore[return-value]


def _theta_strands(
    assignment: SisterAssignment,
    singles: Sequence[tuple[float, float, float]],
    *, boundary_midpoint: bool,
) -> tuple[list[tuple[float, float, float]], ...]:
    first_pair_mid = _mean(assignment.sister_a[0], assignment.sister_b[0])
    last_pair_mid = _mean(assignment.sister_a[-1], assignment.sister_b[-1])
    if boundary_midpoint:
        fork_1 = _mean(first_pair_mid, singles[0])
        fork_2 = _mean(last_pair_mid, singles[-1])
    else:
        fork_1, fork_2 = first_pair_mid, last_pair_mid
    return (
        [fork_1, *singles, fork_2],
        [fork_1, *assignment.sister_a, fork_2],
        [fork_1, *assignment.sister_b, fork_2],
    )


def _blank_uncertainty() -> dict[str, str]:
    return {"XSEnm": "", "YSEnm": "", "ZSEnm": ""}


def _theta_points(
    key: WalkKey, graph_id: str, config: DetectionConfig, pattern,
    assignment: SisterAssignment,
    singles: Sequence[tuple[float, float, float]],
    single_errors: Sequence[tuple[float, float, float] | None],
    component_pairs: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> list[dict[str, Any]]:
    strands = _theta_strands(assignment, singles, boundary_midpoint=True)
    strand_loci = {
        "unreplicated": tuple(pattern.unreplicated_path),
        "sister_A": tuple(pattern.replicated_path),
        "sister_B": tuple(pattern.replicated_path),
    }
    result: list[dict[str, Any]] = []
    for strand_id, points in zip(("unreplicated", "sister_A", "sister_B"), strands):
        loci = strand_loci[strand_id]
        for index, point in enumerate(points):
            endpoint = index in (0, len(points) - 1)
            locus = "" if endpoint else loci[index - 1]
            component: str | int = ""
            source = "inferred_boundary_midpoint" if endpoint else "one_spot_drift_corrected"
            if not endpoint and strand_id != "unreplicated":
                component = _component_id(point, component_pairs[locus])
                source = "two_component_drift_corrected"
            uncertainty = _blank_uncertainty()
            if not endpoint and strand_id == "unreplicated":
                standard_errors = single_errors[index - 1]
                if standard_errors is not None:
                    uncertainty = {
                        "XSEnm": standard_errors[0], "YSEnm": standard_errors[1],
                        "ZSEnm": standard_errors[2]}
            result.append({
                "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                "GraphID": graph_id, "StrandID": strand_id,
                "PointOrder": index,
                "PointRole": ("fork_1" if index == 0 else
                              "fork_2" if index == len(points) - 1 else "locus"),
                "TP": locus,
                "TPIndex": "" if endpoint else config.layout.order.index(locus),
                "GenomicBP": "" if endpoint else config.genomic_bp.get(locus, ""),
                "CoordinateSource": source, "ComponentID": component,
                "XNm": point[0], "YNm": point[1], "ZNm": point[2],
                **uncertainty,
                "Observed": int(not endpoint), "InferredFork": int(endpoint),
            })
    return result


def _link_points(
    key: WalkKey, link_id: str, config: DetectionConfig,
    assignment: SisterAssignment,
    component_pairs: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for loop_id, points in (("sister_A", assignment.sister_a),
                            ("sister_B", assignment.sister_b)):
        for index, (locus, point) in enumerate(zip(assignment.loci, points)):
            result.append({
                "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                "LinkID": link_id, "LoopID": loop_id, "PointOrder": index,
                "PointRole": "locus", "TP": locus,
                "TPIndex": config.layout.order.index(locus),
                "GenomicBP": config.genomic_bp.get(locus, ""),
                "CoordinateSource": "two_component_drift_corrected",
                "ComponentID": _component_id(point, component_pairs[locus]),
                "XNm": point[0], "YNm": point[1], "ZNm": point[2],
                **_blank_uncertainty(), "Closed": 1,
            })
    return result


def _failure_summary(
    key: WalkKey, identifier: str, *, theta: bool, count: int,
    status: str, reason: str,
) -> dict[str, Any]:
    if theta:
        return {
            "Dataset": key[0], "Location": key[1], "WalkID": key[2],
            "GraphID": identifier, "ThetaStatus": status,
            "FailureReason": reason, "NumDoubledLoci": count,
            "AssignmentCost": "", "AssignmentMargin": "",
            "YamadaPolynomial": "", "YamadaNormalized": "",
            "ProjectionStable": "", "PerturbationStable": "",
            "ForkModelStable": "",
        }
    return {
        "Dataset": key[0], "Location": key[1], "WalkID": key[2],
        "LinkID": identifier, "LinkStatus": status, "FailureReason": reason,
        "NumLoci": count, "AssignmentCost": "", "AssignmentMargin": "",
        "LinkingNumber": "", "MinInterLoopDistanceNm": "",
        "AssignmentStable": "", "PerturbationStable": "",
        "DisjointStable": "",
    }


def detect(
    *, locus_results: str | Path, coordinates_csv: str | Path,
    config_path: str | Path, reduction_manifest: str | Path | None,
    output_dir: str | Path,
    yamada_path: str | Path, yamada_seeds: Sequence[int] = (0, 1, 2, 3),
    yamada_tries: int = 60, link_projections: int = 96,
    evidence_mode: str = "calibrated-v3",
    evaluate_yamada_polynomial: bool = True,
) -> dict[str, Path]:
    config = load_config(config_path)
    if evidence_mode == "calibrated-v3":
        if reduction_manifest is None:
            raise TopologyDetectionError(
                "--reduction-manifest is required in calibrated-v3 mode")
        loci_by_walk = load_locus_results(locus_results, config)
        validate_reduction_manifest(
            reduction_manifest, config_path=config_path,
            locus_results=locus_results, config=config,
            loci_by_walk=loci_by_walk)
        require_ownership = True
        walks_total = len(loci_by_walk)
        precomputed_pattern_counts: Counter[str] | None = None
    elif evidence_mode == "legacy-full-schema":
        (loci_by_walk, walks_total,
         precomputed_pattern_counts) = load_exploratory_full_schema(
             locus_results, config)
        require_ownership = False
    else:
        raise TopologyDetectionError(
            f"unsupported evidence_mode {evidence_mode!r}")
    coordinates = load_coordinates(
        coordinates_csv, config, require_ownership=require_ownership,
        allowed_walks=(set(loci_by_walk)
                       if evidence_mode == "legacy-full-schema" else None))
    # Fail before writing anything if the requested external implementation is
    # absent or does not provide the expected API.
    if evaluate_yamada_polynomial:
        load_yamada(yamada_path)

    theta_summaries: list[dict[str, Any]] = []
    theta_points: list[dict[str, Any]] = []
    link_summaries: list[dict[str, Any]] = []
    link_points: list[dict[str, Any]] = []
    pattern_counts: Counter[str] = (
        Counter(precomputed_pattern_counts)
        if precomputed_pattern_counts is not None else Counter())

    for key in sorted(loci_by_walk):
        by_tp = loci_by_walk[key]
        doubled = {tp for tp, row in by_tp.items()
                   if _bit(row["ResolvedLateralTwoComponentPrimary"],
                           "ResolvedLateralTwoComponentPrimary")}
        confident_single = set()
        for tp, row in by_tp.items():
            if tp in doubled or not _bit(
                    row["v3_locus_evaluable"], "v3_locus_evaluable"):
                continue
            if evidence_mode == "legacy-full-schema":
                is_confident_single = _bit(
                    row.get("TopologyConfidentSingle", ""),
                    f"{key + (tp,)} RawConfidentSingle") == 1
            else:
                power = _optional_finite(
                    row.get("v3_power_lcb_effective", ""),
                    f"{key + (tp,)} v3_power_lcb_effective")
                holm_p = _finite(row.get("v3_locus_holm_p", ""),
                                 f"{key + (tp,)} v3_locus_holm_p")
                is_confident_single = (
                    power is not None and power >= config.power_lcb_min
                    and holm_p > config.exploratory_q)
            if is_confident_single:
                confident_single.add(tp)
        pattern = analyze_theta_pattern(
            config.layout, doubled_loci=doubled,
            evaluable_loci=doubled | confident_single)
        if precomputed_pattern_counts is None:
            pattern_counts[pattern.pattern_status] += 1
        is_theta = pattern.is_candidate
        is_link = pattern.pattern_status == "fully_doubled_not_theta"
        if not (is_theta or is_link):
            continue
        identifier = ("theta" if is_theta else "link") + ":" + ":".join(
            (key[0], key[1], str(key[2])))

        if is_theta and not pattern.is_traceable:
            theta_summaries.append(_failure_summary(
                key, identifier, theta=True, count=len(doubled),
                status="pattern_not_traceable",
                reason=("discordant or unevaluable loci prevent a coherent "
                        "three-strand reconstruction")))
            continue

        try:
            needed_double = (pattern.replicated_path if is_theta
                             else config.layout.order)
            component_pairs = {}
            pairs = []
            for tp in needed_double:
                coordinate = coordinates.get((*key, tp))
                if coordinate is None:
                    raise TopologyDetectionError(f"missing coordinate for {key + (tp,)}")
                _verify_coordinate_source(
                    by_tp[tp], coordinate, config, f"{key + (tp,)}")
                components = _component_pair(
                    by_tp[tp], coordinate, config, f"{key + (tp,)}")
                component_pairs[tp] = components
                pairs.append(LocusPair(tp, *components))
            assignment = (assign_open(pairs) if is_theta else assign_circular(pairs))
            xy_assignment = (assign_open(pairs, dims=(0, 1)) if is_theta
                             else assign_circular(pairs, dims=(0, 1)))
            margin = assignment.cost_margin
            xy_margin = xy_assignment.cost_margin
            margin_clear = (
                margin is not None and
                margin > 1e-9 * max(1.0, assignment.total_cost) and
                xy_margin is not None and
                xy_margin > 1e-9 * max(1.0, xy_assignment.total_cost))
            assignment_stable = (
                assignment.state_path == xy_assignment.state_path and margin_clear)
        except (TopologyDetectionError, TopologyInputError) as exc:
            summary = _failure_summary(
                key, identifier, theta=is_theta, count=len(doubled),
                status="geometry_incomplete", reason=str(exc))
            (theta_summaries if is_theta else link_summaries).append(summary)
            continue

        if is_theta:
            try:
                singles = []
                single_errors = []
                for tp in pattern.unreplicated_path:
                    coordinate = coordinates.get((*key, tp))
                    if coordinate is None:
                        raise TopologyDetectionError(
                            f"missing single-copy coordinate for {key + (tp,)}")
                    _verify_coordinate_source(
                        by_tp[tp], coordinate, config, f"{key + (tp,)}")
                    singles.append(coordinate.xyz_nm)
                    single_errors.append(coordinate.xyz_se_nm)
                primary = _theta_strands(
                    assignment, singles, boundary_midpoint=True)
                alternate = _theta_strands(
                    assignment, singles, boundary_midpoint=False)
                reasons = []
                if assignment.state_path != xy_assignment.state_path:
                    reasons.append("XY and XYZ sister assignments disagree")
                if not margin_clear:
                    reasons.append("sister assignment has no numerical cost margin")
                if evaluate_yamada_polynomial:
                    primary_eval = evaluate_theta(
                        primary, yamada_path, seeds=yamada_seeds,
                        tries=yamada_tries)
                    alternate_eval = evaluate_theta(
                        alternate, yamada_path, seeds=yamada_seeds,
                        tries=yamada_tries)
                    fork_stable: int | str = int(
                        alternate_eval.stable and
                        primary_eval.normalized_terms ==
                        alternate_eval.normalized_terms)
                    if not primary_eval.stable:
                        reasons.append(
                            "Yamada polynomial changes across projections")
                    if not fork_stable:
                        reasons.append(
                            "Yamada polynomial changes with fork model")
                    vertices_ok = all(
                        result.vertices == 2
                        for result in (*primary_eval.seed_results,
                                       *alternate_eval.seed_results))
                    if not vertices_ok:
                        reasons.append(
                            "Yamada reconstruction has extra graph vertices")
                    status = ("nominal_theta_graph" if not reasons
                              else "theta_unstable")
                    raw_polynomial = primary_eval.seed_results[0].raw_polynomial
                    normalized_polynomial = primary_eval.normalized_polynomial
                    projection_stable: int | str = int(primary_eval.stable)
                else:
                    validate_theta_strands(primary)
                    validate_theta_strands(alternate)
                    status = ("nominal_theta_geometry" if not reasons
                              else "theta_geometry_unstable")
                    raw_polynomial = ""
                    normalized_polynomial = ""
                    projection_stable = ""
                    fork_stable = ""
                theta_summaries.append({
                    "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                    "GraphID": identifier, "ThetaStatus": status,
                    "FailureReason": "; ".join(reasons),
                    "NumDoubledLoci": len(doubled),
                    "AssignmentCost": assignment.total_cost,
                    "AssignmentMargin": assignment.cost_margin,
                    "YamadaPolynomial": raw_polynomial,
                    "YamadaNormalized": normalized_polynomial,
                    "ProjectionStable": projection_stable,
                    "PerturbationStable": "", "ForkModelStable": fork_stable,
                })
                theta_points.extend(_theta_points(
                    key, identifier, config, pattern, assignment, singles,
                    single_errors, component_pairs))
            except (TopologyDetectionError, TopologyValidationError, RuntimeError) as exc:
                theta_summaries.append(_failure_summary(
                    key, identifier, theta=True, count=len(doubled),
                    status="geometry_invalid", reason=str(exc)))
        else:
            loop_a, loop_b = assignment.sister_a, assignment.sister_b
            try:
                evaluation = evaluate_link(
                    loop_a, loop_b, n_projections=link_projections)
                reasons = []
                if assignment.state_path != xy_assignment.state_path:
                    reasons.append("XY and XYZ sister assignments disagree")
                if not margin_clear:
                    reasons.append("sister assignment has no numerical cost margin")
                if not evaluation.stable:
                    reasons.append("linking number changes across projections")
                if reasons:
                    status = "link_unstable"
                elif evaluation.absolute_linking_number:
                    status = "nominal_linked_polygon"
                else:
                    status = "nominal_unlinked_polygon"
                link_summaries.append({
                    "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                    "LinkID": identifier, "LinkStatus": status,
                    "FailureReason": "; ".join(reasons),
                    "NumLoci": len(config.layout.order),
                    "AssignmentCost": assignment.total_cost,
                    "AssignmentMargin": assignment.cost_margin,
                    "LinkingNumber": evaluation.linking_number,
                    "MinInterLoopDistanceNm": evaluation.min_segment_distance_nm,
                    "AssignmentStable": int(assignment_stable),
                    "PerturbationStable": "", "DisjointStable": "",
                })
                link_points.extend(_link_points(
                    key, identifier, config, assignment, component_pairs))
            except TopologyValidationError as exc:
                link_summaries.append(_failure_summary(
                    key, identifier, theta=False, count=len(config.layout.order),
                    status="geometry_invalid", reason=str(exc)))

    locus_path = Path(locus_results)
    coordinate_path = Path(coordinates_csv)
    config_source = Path(config_path)
    reduction_source = (Path(reduction_manifest)
                        if reduction_manifest is not None else None)
    yamada_file = Path(yamada_path)
    if yamada_file.is_dir():
        yamada_file = yamada_file / "yamada.py"
    provenance = {
        "detector": "pipeline.detect_topology",
        "software_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("detect_topology.py", "topology_core.py",
                         "topology_io.py", "topology_yamada.py")
        },
        "input_sha256": {
            "locus_results": _sha256(locus_path),
            "coordinates": _sha256(coordinate_path),
            "config": _sha256(config_source),
        },
        "input_paths": {
            "locus_results": str(locus_path.resolve()),
            "coordinates": str(coordinate_path.resolve()),
            "config": str(config_source.resolve()),
        },
        "evidence_mode": evidence_mode,
        "publication_calibrated": evidence_mode == "calibrated-v3",
        "ownership_required": require_ownership,
        "evidence_limitations": (
            [] if evidence_mode == "calibrated-v3" else [
                "two-component and confident-single calls come from the "
                "legacy trace-salvage Raw* columns",
                "no publication-v1 reduction manifest or power calibration "
                "was required",
                "strict center-ownership provenance was not required",
            ]),
        "coordinate_transform": "XYZ_nm=(raw_target_px-drift_px)*axis_scale_nm",
        "fork_models": ["boundary_midpoint", "replicated_pair_midpoint"],
        "uncertainty": "not_available_from_v3_component_centres",
        "walks_total": walks_total,
        "walks_reaching_topology_geometry": len(loci_by_walk),
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "theta_candidates": len(theta_summaries),
        "link_candidates": len(link_summaries),
        "yamada_seeds": list(yamada_seeds),
        "yamada_tries": yamada_tries,
        "yamada_evaluated": evaluate_yamada_polynomial,
        "link_projections": link_projections,
        "link_projection_seed": 0,
        "link_min_valid_projections": 16,
        "link_min_projection_stability": 0.95,
        "geometry_intersection_tolerance_nm": 1e-6,
    }
    if reduction_source is not None:
        provenance["input_sha256"]["reduction_manifest"] = _sha256(
            reduction_source)
        provenance["input_paths"]["reduction_manifest"] = str(
            reduction_source.resolve())
    if evaluate_yamada_polynomial:
        provenance["input_sha256"]["yamada"] = _sha256(yamada_file)
        provenance["input_paths"]["yamada"] = str(yamada_file.resolve())
    return write_topology_outputs(
        output_dir, theta_summaries=theta_summaries,
        theta_points=theta_points, link_summaries=link_summaries,
        link_points=link_points, provenance=provenance)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TopologyDetectionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _seed_list(text: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locus-results", required=True,
                        help=("calibrated v3 locus results, or a trace-salvage "
                              "full-schema CSV in legacy-full-schema mode"))
    parser.add_argument("--coordinates-csv", required=True,
                        help="combined comprehensive or unfiltered drift-corrected coordinates")
    parser.add_argument("--config", required=True,
                        help="dataset config with chromosome order and geometry")
    parser.add_argument(
        "--evidence-mode",
        choices=("calibrated-v3", "legacy-full-schema"),
        default="calibrated-v3",
        help=("calibrated-v3 enforces the publication reduction contract; "
              "legacy-full-schema uses explicit Raw* calls for exploratory "
              "analysis"),
    )
    parser.add_argument(
        "--reduction-manifest",
        help="final v3 publication_v1_manifest.json (required for calibrated-v3)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--yamada-path", default="/mnt/altnas/work/dimos/Yamada")
    parser.add_argument(
        "--skip-yamada", action="store_true",
        help=("reconstruct and validate theta geometry without evaluating the "
              "Yamada polynomial; link detection is unchanged"))
    parser.add_argument("--yamada-seeds", type=_seed_list, default=(0, 1, 2, 3))
    parser.add_argument("--yamada-tries", type=int, default=60)
    parser.add_argument("--link-projections", type=int, default=96)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.yamada_tries < 1 or args.link_projections < 16:
        print("error: --yamada-tries must be positive and --link-projections >= 16",
              file=sys.stderr)
        return 2
    try:
        paths = detect(
            locus_results=args.locus_results,
            coordinates_csv=args.coordinates_csv,
            config_path=args.config, output_dir=args.outdir,
            reduction_manifest=args.reduction_manifest,
            yamada_path=args.yamada_path, yamada_seeds=args.yamada_seeds,
            yamada_tries=args.yamada_tries,
            link_projections=args.link_projections,
            evidence_mode=args.evidence_mode,
            evaluate_yamada_polynomial=not args.skip_yamada)
    except (TopologyDetectionError, TopologyInputError,
            TopologyValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
