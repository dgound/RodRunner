#!/usr/bin/env python3
"""Audited trace-based resolution of isolated two-component loci.

The resolver is deliberately downstream of drift correction and never changes
the image-model call.  When Ori is confidently single, an isolated non-endpoint
two-component locus may contribute one component to a *derived* trace if that
component connects its two single-locus neighbours substantially better than
the alternative.  Component choice uses XY only; the chosen component's full
drift-corrected XYZ coordinate is written to a separate output.

Calibration is dataset-level and leave-one-location-out.  For each target
location, the maximum accepted neighbour-link distance is the configured
quantile of links in full-coverage, confidently-single walks from every other
location.  Fewer than ``min_calibration_links`` links fails closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


try:
    from . import center_ownership as _own
    from . import replication_status as _rstatus
except ImportError:      # direct execution
    import center_ownership as _own
    import replication_status as _rstatus

INPUT_SCHEMA_VERSION = "trace_salvage_input_v1"
POLICY_SCHEMA_VERSION = "trace_salvage_policy_v1"
OUTPUT_SCHEMA_VERSION = "trace_salvage_output_v2"
# rep1's legacy v2 sidecars label the same genomic locus as, for example,
# ``TP000_CYC000``.  The suffix is acquisition bookkeeping, not part of the
# chromosome position; retain a strict known-form parser rather than accepting
# arbitrary trailing text.
_TP_RE = re.compile(r"^TP(\d+)(?:_CYC\d+)?$", re.IGNORECASE)


class TraceSalvageError(ValueError):
    """Raised when an input or policy violates the resolver contract."""


@dataclass(frozen=True)
class LocationInputs:
    location: str
    evidence_csv: str
    coordinates_csv: str
    walk_summary_csv: str | None = None
    #: Per-attempt centre-ownership audit table from the localizer. Optional and
    #: explicit: under `--center-ownership strict` the localizer DROPS a rejected
    #: locus from the coordinate CSV, so that CSV records the rejection only as
    #: an absence. This table is the one place a rejection is still itemized, and
    #: it is the difference between "rejected" and "never acquired".
    ownership_audit_csv: str | None = None
    #: The UNFILTERED drift-corrected coordinate CSV, when ``coordinates_csv`` is
    #: its full-coverage subset. ``coordinates_csv`` alone decides salvage, so the
    #: calibration and every decision stay exactly as before; this file only
    #: supplies the canonical coordinates of accepted attempts whose walk lost
    #: full coverage -- which is precisely what happens to a walk with one
    #: strict-rejected locus, and without it the walk's 19 accepted siblings
    #: could not be represented in the comprehensive output at all.
    all_coordinates_csv: str | None = None


@dataclass(frozen=True)
class TraceSalvageConfig:
    dataset: str
    evidence_schema: str
    chromosome_order: tuple[str, ...]
    ori_tp: str
    ter_tp: str
    xy_pixel_nm: float
    z_step_nm: float
    locations: tuple[LocationInputs, ...]
    min_calibration_links: int = 100
    clean_link_quantile: float = 0.95
    winner_ratio: float = 0.80
    salvage_policy: str = _own.SALVAGE_CALIBRATED_LOO


@dataclass(frozen=True)
class _Evidence:
    call: bool
    evaluable: bool
    confident_single: bool
    status: str
    components_px: tuple[tuple[float, float, float],
                         tuple[float, float, float]] | None


@dataclass(frozen=True)
class _Coordinate:
    z_nm: float
    y_nm: float
    x_nm: float
    drift_z_px: float
    drift_y_px: float
    drift_x_px: float


def _canonical_tp(value: Any) -> str:
    if isinstance(value, int):
        if value < 0:
            raise TraceSalvageError("TP indices must be non-negative")
        return f"TP{value:03d}"
    text = str(value).strip()
    match = _TP_RE.fullmatch(text)
    if not match:
        if text.isdigit():
            return f"TP{int(text):03d}"
        raise TraceSalvageError(f"invalid TP label {value!r}")
    return f"TP{int(match.group(1)):03d}"


def _finite(value: Any, field: str, *, blank_ok: bool = False) -> float | None:
    if value is None or str(value).strip() == "":
        if blank_ok:
            return None
        raise TraceSalvageError(f"blank {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceSalvageError(f"non-numeric {field}={value!r}") from exc
    if not math.isfinite(result):
        if blank_ok:
            return None
        raise TraceSalvageError(f"non-finite {field}={value!r}")
    return result


def _binary(value: Any, field: str) -> bool:
    try:
        number = int(float(value))
    except (TypeError, ValueError) as exc:
        raise TraceSalvageError(f"invalid binary {field}={value!r}") from exc
    if number not in (0, 1):
        raise TraceSalvageError(f"invalid binary {field}={value!r}")
    return bool(number)


def _walk_id(value: Any) -> int:
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError) as exc:
        raise TraceSalvageError(f"invalid WalkID={value!r}") from exc
    if not math.isfinite(number) or number != result:
        raise TraceSalvageError(f"invalid WalkID={value!r}")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _resolve_path(base: Path, value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TraceSalvageError(f"blank {field}")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def load_config(path: os.PathLike[str] | str) -> TraceSalvageConfig:
    """Load and validate a ``trace_salvage_input_v1`` JSON document."""
    config_path = Path(path).resolve()
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    # Both pipeline integrations historically called the evidence generation
    # ``schema`` and used flat geometry fields.  Accept that frozen on-disk
    # contract as well as the more explicit names used by this module.
    schema_version = raw.get("schema_version")
    if schema_version not in (None, INPUT_SCHEMA_VERSION):
        raise TraceSalvageError(
            f"config schema_version must be {INPUT_SCHEMA_VERSION!r}")
    dataset = str(raw.get("dataset", "")).strip()
    if not dataset:
        raise TraceSalvageError("config dataset is required")
    evidence_schema = str(
        raw.get("evidence_schema", raw.get("schema", ""))).strip().lower()
    if evidence_schema not in {"v2", "v3"}:
        raise TraceSalvageError("evidence_schema must be 'v2' or 'v3'")
    order = tuple(_canonical_tp(tp) for tp in raw.get("chromosome_order", ()))
    if len(order) < 3 or len(order) != len(set(order)):
        raise TraceSalvageError(
            "chromosome_order must contain at least three unique TPs")
    ori_tp = _canonical_tp(raw.get("ori_tp", ""))
    ter_tp = _canonical_tp(raw.get("ter_tp", ""))
    if ori_tp == ter_tp or ori_tp not in order or ter_tp not in order:
        raise TraceSalvageError(
            "ori_tp and ter_tp must be distinct chromosome_order members")
    geometry = raw.get("geometry") or {}
    xy_pixel_nm = _finite(
        geometry.get("xy_pixel_nm", raw.get("xy_pixel_nm")), "xy_pixel_nm")
    z_step_nm = _finite(
        geometry.get("z_step_nm", raw.get("z_step_nm")), "z_step_nm")
    assert xy_pixel_nm is not None and z_step_nm is not None
    if xy_pixel_nm <= 0 or z_step_nm <= 0:
        raise TraceSalvageError("geometry values must be positive")
    min_links = int(raw.get("min_calibration_links", 100))
    quantile = float(raw.get("clean_link_quantile", 0.95))
    winner_ratio = float(raw.get("winner_ratio", 0.80))
    salvage_pol = str(raw.get("salvage_policy",
                              _own.SALVAGE_CALIBRATED_LOO)).strip()
    if salvage_pol not in (_own.SALVAGE_SIMPLE_NEIGHBOR,
                           _own.SALVAGE_CALIBRATED_LOO):
        raise TraceSalvageError(
            f"salvage_policy must be {_own.SALVAGE_SIMPLE_NEIGHBOR!r} or "
            f"{_own.SALVAGE_CALIBRATED_LOO!r}, got {salvage_pol!r}")
    if min_links < 1:
        raise TraceSalvageError("min_calibration_links must be positive")
    if not 0 < quantile < 1:
        raise TraceSalvageError("clean_link_quantile must be in (0,1)")
    if not 0 < winner_ratio < 1:
        raise TraceSalvageError("winner_ratio must be in (0,1)")
    base = config_path.parent
    locations: list[LocationInputs] = []
    seen: set[str] = set()
    for item in raw.get("locations", ()):
        location = str(item.get("location", "")).strip()
        if not location or location in seen:
            raise TraceSalvageError(
                "each config location must be non-empty and unique")
        seen.add(location)
        summary = item.get("walk_summary_csv")
        ownership = item.get("ownership_audit_csv")
        all_coordinates = item.get("all_coordinates_csv")
        locations.append(LocationInputs(
            location=location,
            evidence_csv=_resolve_path(base, item.get("evidence_csv"),
                                       "evidence_csv"),
            coordinates_csv=_resolve_path(base, item.get("coordinates_csv"),
                                          "coordinates_csv"),
            walk_summary_csv=(
                _resolve_path(base, summary, "walk_summary_csv")
                if summary not in (None, "") else None),
            ownership_audit_csv=(
                _resolve_path(base, ownership, "ownership_audit_csv")
                if ownership not in (None, "") else None),
            all_coordinates_csv=(
                _resolve_path(base, all_coordinates, "all_coordinates_csv")
                if all_coordinates not in (None, "") else None),
        ))
    if not locations:
        raise TraceSalvageError("config locations must not be empty")
    return TraceSalvageConfig(
        dataset=dataset, evidence_schema=evidence_schema,
        chromosome_order=order, ori_tp=ori_tp, ter_tp=ter_tp,
        xy_pixel_nm=xy_pixel_nm, z_step_nm=z_step_nm,
        locations=tuple(locations), min_calibration_links=min_links,
        clean_link_quantile=quantile, winner_ratio=winner_ratio,
        salvage_policy=salvage_pol)


def _config_contract(config: TraceSalvageConfig) -> dict[str, Any]:
    return {
        "dataset": config.dataset,
        "evidence_schema": config.evidence_schema,
        "chromosome_order": list(config.chromosome_order),
        "ori_tp": config.ori_tp,
        "ter_tp": config.ter_tp,
        "geometry": {
            "xy_pixel_nm": config.xy_pixel_nm,
            "z_step_nm": config.z_step_nm,
        },
        "parameters": {
            "min_calibration_links": config.min_calibration_links,
            "clean_link_quantile": config.clean_link_quantile,
            "winner_ratio": config.winner_ratio,
            "salvage_policy": config.salvage_policy,
        },
    }


def _read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TraceSalvageError(f"{path} has no CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise TraceSalvageError(f"{path} has duplicate CSV columns")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _row_key(row: Mapping[str, Any], location: str, path: str,
             number: int) -> tuple[str, int, str]:
    embedded = str(row.get("Location", "")).strip()
    if embedded and embedded != location:
        raise TraceSalvageError(
            f"{path}:{number} Location={embedded!r}, expected {location!r}")
    tp_value = row.get("TP", row.get("tp", ""))
    try:
        return location, _walk_id(row.get("WalkID")), _canonical_tp(tp_value)
    except TraceSalvageError as exc:
        raise TraceSalvageError(f"{path}:{number}: {exc}") from exc


def _components(row: Mapping[str, Any], prefix: str) -> tuple[
        tuple[float, float, float], tuple[float, float, float]] | None:
    values: list[float] = []
    for component in (1, 2):
        for axis in ("z", "y", "x"):
            value = _finite(row.get(f"{prefix}_c{component}_{axis}"),
                            f"{prefix}_c{component}_{axis}", blank_ok=True)
            if value is None:
                return None
            values.append(value)
    return ((values[0], values[1], values[2]),
            (values[3], values[4], values[5]))


#: Columns that only ONE schema's evidence file can legitimately carry. Used to
#: refuse a declared schema that the file contradicts.
_SCHEMA_WITNESS = {
    "v2": ("replication_confirmed", "status"),
    "v3": ("ResolvedLateralTwoComponentPrimary", "v3_locus_evaluable",
           "v3_aggregation_status"),
}


def _verify_evidence_schema(header: Sequence[str], schema: str, path) -> None:
    """Refuse an evidence file whose columns contradict the declared schema.

    Necessary because v3 preparation used to copy v2 columns in UNPREFIXED, so a
    v3 `*_locus_results.csv` also carried `status`, `replication_confirmed` and
    `m2_c1_*` -- the exact v2 witness set. Declaring `evidence_schema: v2` on a
    v3 file therefore succeeded and silently produced v2 calls from publication
    evidence. Prefixing fixed the cause; this check makes the failure loud even
    for files written before the fix.
    """
    present = set(header)
    other = "v3" if schema == "v2" else "v2"
    foreign = [name for name in _SCHEMA_WITNESS[other] if name in present]
    if foreign:
        raise TraceSalvageError(
            f"{path}: evidence_schema is {schema!r} but the file carries {other} "
            f"columns {foreign}. Refusing to read {other} evidence as {schema}: "
            "that would emit calls from the wrong detector. If this is a "
            "pre-fix v3 file with unprefixed legacy columns, re-run prepare.")
    missing = [name for name in _SCHEMA_WITNESS[schema] if name not in present]
    if missing:
        raise TraceSalvageError(
            f"{path}: evidence_schema is {schema!r} but required {schema} "
            f"columns are absent: {missing}")


def _normalize_evidence(row: Mapping[str, Any], schema: str) -> _Evidence:
    if schema == "v2":
        call = _binary(row.get("replication_confirmed"),
                       "replication_confirmed")
        status = str(row.get("status", "")).strip()
        # This deliberately matches the strict smoke-test definition.  A
        # generic not_suspect row is not silently promoted to a confident
        # single-locus observation.
        single = (not call and status == "tested_not_confirmed")
        return _Evidence(call, call or single, single, status,
                         _components(row, "m2"))
    call = _binary(row.get("ResolvedLateralTwoComponentPrimary"),
                   "ResolvedLateralTwoComponentPrimary")
    evaluable = _binary(row.get("v3_locus_evaluable"),
                        "v3_locus_evaluable")
    status = str(row.get("v3_aggregation_status", "")).strip()
    single = evaluable and not call
    return _Evidence(call, evaluable, single, status,
                     _components(row, "v3_m2"))


def _first_finite(row: Mapping[str, Any], names: Sequence[str],
                  label: str, *, default: float | None = None) -> float:
    for name in names:
        value = _finite(row.get(name), name, blank_ok=True)
        if value is not None:
            return value
    if default is not None:
        return default
    raise TraceSalvageError(f"missing {label}; tried {list(names)}")


def _normalize_coordinate(row: Mapping[str, Any]) -> _Coordinate:
    z_nm = _first_finite(row, ("z_nm", "ZNm"), "z_nm")
    y_nm = _first_finite(row, ("y_nm", "YNm"), "y_nm")
    x_nm = _first_finite(row, ("x_nm", "XNm"), "x_nm")
    # Component centres are fitted in the raw image frame.  Treating a missing
    # drift vector as zero would silently mix raw M2 centres with corrected
    # neighbour localizations, so this post-drift resolver fails closed.
    drift_z = _first_finite(row, ("drift_z", "DriftZ"), "drift_z")
    drift_y = _first_finite(row, ("drift_y", "DriftY"), "drift_y")
    drift_x = _first_finite(row, ("drift_x", "DriftX"), "drift_x")
    return _Coordinate(z_nm, y_nm, x_nm, drift_z, drift_y, drift_x)


def _ownership_rows(item: LocationInputs) -> tuple[list[dict[str, str]], str]:
    """Per-locus ownership rows for one location, and where they came from.

    The audit table is preferred because it retains STRICT-rejected loci, which
    the coordinate CSV can only represent as an absence.  Falling back to the
    coordinate CSV keeps a dataset that has ownership columns but no audit table
    working; a dataset with neither predates the schema and yields ``("", [])``
    so nothing is fabricated.
    """
    if item.ownership_audit_csv:
        if not Path(item.ownership_audit_csv).is_file():
            raise TraceSalvageError(
                f"missing ownership audit CSV: {item.ownership_audit_csv}")
        fields, rows = _read_csv(item.ownership_audit_csv)
        if not any(name in fields
                   for name in _own.VALID_COLUMN_CANDIDATES):
            raise TraceSalvageError(
                f"{item.ownership_audit_csv} has no centre-ownership column "
                f"(expected one of {list(_own.VALID_COLUMN_CANDIDATES)})")
        return rows, "ownership_audit_csv"
    fields, rows = _read_csv(item.coordinates_csv)
    if any(name in fields for name in _own.VALID_COLUMN_CANDIDATES):
        return rows, "coordinates_csv"
    return [], ""


#: Outcome of one localization attempt, recorded on every comprehensive row.
ATTEMPT_ACCEPTED = "accepted"
ATTEMPT_OWNERSHIP_REJECTED = "ownership_rejected"
ATTEMPT_FIT_FAILED = "fit_failed"
ATTEMPT_NO_DRIFT_SOLUTION = "no_drift_solution"

#: Audit-only physical coordinates of the unconstrained fitted centre, in the RAW
#: target-channel frame -- the frame ownership is decided in, before drift
#: correction. Named for their units, and deliberately NOT reusing z_raw/OriginalZNm
#: (pre-drift pixels / drift-corrected nm), which already mean other things.
#: Populated for every attempt that produced a centre, accepted or rejected, so
#: the audit schema is uniform; a rejected centre appears ONLY here.
AUDIT_COORDINATE_FIELDS = (
    "UnconstrainedFitZNm", "UnconstrainedFitYNm", "UnconstrainedFitXNm",
)

#: Why a row exists and whether its canonical coordinates are usable.
ATTEMPT_FIELDS = ("LocalizationAttemptStatus", "LocalizationAttemptReason")

#: Ledger columns. The first two are required to define the attempt universe; the
#: centre columns are newer, and a ledger without them degrades to blank audit
#: coordinates rather than being rejected, which would make an archived tree
#: unmergeable.
_LEDGER_RET_CODE = "fit_ret_code"
_LEDGER_WRITTEN = "localization_written"
_LEDGER_REASON = "unevaluable_reason"
_LEDGER_CENTRE = ("center_fit_z_px", "center_fit_y_px", "center_fit_x_px")
#: Legacy lowercase ownership columns -> the canonical names carried on rows.
_LEDGER_LEGACY_OWNERSHIP = {
    "center_ownership_initial": "CenterOwnershipInitial",
    "center_ownership_final": "CenterOwnershipFinal",
    "center_ownership_pass": "CenterOwnershipValid",
    "center_refit_attempted": "CenterRefitAttempted",
    "center_refit_success": "CenterRefitAccepted",
}


@dataclass(frozen=True)
class _Attempt:
    """One localization attempt as recorded in the authoritative ledger."""
    walk_id: int
    tp: str
    fit_succeeded: bool
    written: bool
    reason: str
    centre_px: tuple[float, float, float] | None
    ownership: dict[str, str]


def _ledger_ownership(row: Mapping[str, str]) -> dict[str, str]:
    """Canonical per-locus ownership + usability columns for one ledger row."""
    values = {field: "" for field in _own.OWNERSHIP_ROW_FIELDS}
    for field in _own.USABILITY_ROW_FIELDS:
        values[field] = ""
    present = False
    for field in _own.OWNERSHIP_ROW_FIELDS:
        if field in row:
            present = True
            values[field] = str(row.get(field, ""))
    for field in _own.USABILITY_ROW_FIELDS:
        if field in row:
            present = True
            values[field] = str(row.get(field, ""))
    if present:
        return values
    for source, target in _LEDGER_LEGACY_OWNERSHIP.items():
        if source in row:
            values[target] = str(row.get(source, ""))
    return values


def _read_attempt_ledger(item: LocationInputs) -> tuple[
        dict[tuple[int, str], _Attempt], bool] | None:
    """The authoritative attempt universe for one location, or None.

    Returns ``({(walk, tp): attempt}, has_centres)``.  ``None`` when the location
    declares no ledger at all, in which case the comprehensive output keeps its
    historical coordinate-row universe.
    """
    if not item.ownership_audit_csv:
        return None
    fields, rows = _read_csv(item.ownership_audit_csv)
    for required in (_LEDGER_RET_CODE, _LEDGER_WRITTEN):
        if required not in fields:
            raise TraceSalvageError(
                f"{item.ownership_audit_csv} has no {required!r} column, so it "
                "cannot define the localization-attempt universe")
    has_centres = all(name in fields for name in _LEDGER_CENTRE)
    attempts: dict[tuple[int, str], _Attempt] = {}
    for number, row in enumerate(rows, start=2):
        _, walk_id, tp = _row_key(
            row, item.location, item.ownership_audit_csv, number)
        key = (walk_id, tp)
        if key in attempts:
            raise TraceSalvageError(
                f"{item.ownership_audit_csv}:{number}: duplicate attempt key "
                f"{(item.location, walk_id, tp)!r}")
        # `or 0` made a blank code mean "converged". It does not: it means the
        # column was never written, so no coordinate may be trusted from it.
        try:
            outcome = _own.attempt_outcome(row)
        except _own.LedgerRowError as exc:
            raise TraceSalvageError(
                f"{item.ownership_audit_csv}:{number}: {exc}") from None
        ret_code = 0 if outcome != _own.ATTEMPT_FIT_FAILED else 1
        centre = None
        if has_centres:
            values = [_finite(row.get(name), name, blank_ok=True)
                      for name in _LEDGER_CENTRE]
            if all(value is not None for value in values):
                centre = (values[0], values[1], values[2])
        attempts[key] = _Attempt(
            walk_id=walk_id, tp=tp,
            fit_succeeded=(ret_code == 0),
            written=_own.is_valid_flag(row.get(_LEDGER_WRITTEN)),
            reason=str(row.get(_LEDGER_REASON, "") or "").strip(),
            centre_px=centre,
            ownership=_ledger_ownership(row),
        )
    return attempts, has_centres


def _raw_pixel_centre(row: Mapping[str, Any],
                      attempt: "_Attempt | None"
                      ) -> tuple[float, float, float] | None:
    """Pre-drift pixel centre of an accepted row, from the row or the ledger.

    ``z_raw/y_raw/x_raw`` are drift correction's copy of the localizer's fitted
    centre, so they are the same number the ownership verdict was decided on. The
    ledger is the fallback for a coordinate CSV that predates those columns.
    """
    values = [_finite(row.get(name), name, blank_ok=True)
              for name in ("z_raw", "y_raw", "x_raw")]
    if all(value is not None for value in values):
        return (values[0], values[1], values[2])
    return attempt.centre_px if attempt is not None else None


def _audit_coordinates(centre_px: tuple[float, float, float] | None,
                       config: TraceSalvageConfig) -> dict[str, str]:
    """Audit-only nm coordinates from a raw-frame pixel centre."""
    if centre_px is None:
        return {field: "" for field in AUDIT_COORDINATE_FIELDS}
    z, y, x = centre_px
    return {
        "UnconstrainedFitZNm": _fmt(z * config.z_step_nm),
        "UnconstrainedFitYNm": _fmt(y * config.xy_pixel_nm),
        "UnconstrainedFitXNm": _fmt(x * config.xy_pixel_nm),
    }


def _load_inputs(config: TraceSalvageConfig) -> tuple[
        dict[tuple[str, int, str], _Evidence],
        dict[tuple[str, int, str], _Coordinate],
        dict[tuple[str, int], dict[str, str]], list[str], dict[str, Any],
        dict[tuple[str, int], dict[str, int]],
        dict[tuple[str, int, str], bool], dict[str, str]]:
    evidence: dict[tuple[str, int, str], _Evidence] = {}
    coordinates: dict[tuple[str, int, str], _Coordinate] = {}
    summaries: dict[tuple[str, int], dict[str, str]] = {}
    summary_fields: list[str] = []
    hashes: dict[str, Any] = {}
    #: (location, walk) -> the four OWNERSHIP_WALK_FIELDS. Absent for a location
    #: with no ownership schema at all, so enforcement stays off there.
    ownership_walks: dict[tuple[str, int], dict[str, int]] = {}
    #: (location, walk, TP) -> ownership validity, for the configured loci only.
    ownership_loci: dict[tuple[str, int, str], bool] = {}
    ownership_sources: dict[str, str] = {}
    locus_order = [_own.tp_index(tp) for tp in config.chromosome_order]
    order_tps = set(config.chromosome_order)
    for item in config.locations:
        for kind, path in (("evidence", item.evidence_csv),
                           ("coordinates", item.coordinates_csv)):
            if not Path(path).is_file():
                raise TraceSalvageError(f"missing {kind} CSV: {path}")
        _evidence_header, rows = _read_csv(item.evidence_csv)
        _verify_evidence_schema(_evidence_header, config.evidence_schema,
                                item.evidence_csv)
        for number, row in enumerate(rows, start=2):
            key = _row_key(row, item.location, item.evidence_csv, number)
            if key in evidence:
                raise TraceSalvageError(f"duplicate evidence key {key!r}")
            evidence[key] = _normalize_evidence(row, config.evidence_schema)
        _, rows = _read_csv(item.coordinates_csv)
        for number, row in enumerate(rows, start=2):
            key = _row_key(row, item.location, item.coordinates_csv, number)
            if key in coordinates:
                raise TraceSalvageError(f"duplicate coordinate key {key!r}")
            try:
                coordinates[key] = _normalize_coordinate(row)
            except TraceSalvageError as exc:
                raise TraceSalvageError(
                    f"{item.coordinates_csv}:{number}: {exc}") from exc
        if item.walk_summary_csv:
            fields, rows = _read_csv(item.walk_summary_csv)
            for field in fields:
                if field not in summary_fields:
                    summary_fields.append(field)
            for number, row in enumerate(rows, start=2):
                embedded = str(row.get("Location", "")).strip()
                if embedded and embedded != item.location:
                    raise TraceSalvageError(
                        f"{item.walk_summary_csv}:{number} has unexpected Location")
                key = (item.location, _walk_id(row.get("WalkID")))
                if key in summaries:
                    raise TraceSalvageError(f"duplicate walk summary key {key!r}")
                summaries[key] = dict(row)
        own_rows, own_source = _ownership_rows(item)
        ownership_sources[item.location] = own_source
        if own_rows:
            try:
                per_walk = _own.walk_ownership_summary(own_rows, locus_order)
            except (ValueError, _own.OwnershipSchemaMissing) as exc:
                raise TraceSalvageError(
                    f"centre-ownership aggregation failed for "
                    f"{item.location}: {exc}") from exc
            # Also compute usability walk fields when the columns exist.
            try:
                per_walk_usability = _own.walk_usability_summary(
                    own_rows, locus_order)
            except (ValueError, _own.OwnershipSchemaMissing):
                per_walk_usability = {}
            for walk, values in per_walk.items():
                merged = dict(values)
                merged.update(per_walk_usability.get(walk, {}))
                ownership_walks[(item.location, walk)] = merged
            # Prefer CenterOwnershipUsable for the censor decision when it
            # exists; fall back to CenterOwnershipValid for legacy data.
            has_usable = any(_own.USABLE_COLUMN in row for row in own_rows[:1])
            if has_usable:
                verdict_fn = _own.usability_verdict_of
            else:
                column = _own.detect_valid_column(own_rows)
                verdict_fn = lambda row: _own.verdict_of(row, column)
            for row in own_rows:
                try:
                    tp = _canonical_tp(row.get("tp", row.get("TP", "")))
                except TraceSalvageError:
                    continue
                if tp not in order_tps:
                    continue
                key = (item.location, _walk_id(row.get("WalkID")), tp)
                verdict = verdict_fn(row)
                if verdict is None:
                    continue
                valid = verdict
                if ownership_loci.get(key, valid) != valid:
                    raise TraceSalvageError(
                        f"contradictory centre-ownership verdicts for {key!r}")
                ownership_loci[key] = valid
        hashes[item.location] = {
            "evidence_csv": str(Path(item.evidence_csv).resolve()),
            "coordinates_csv": str(Path(item.coordinates_csv).resolve()),
            "walk_summary_csv": (
                str(Path(item.walk_summary_csv).resolve())
                if item.walk_summary_csv else None),
            "ownership_audit_csv": (
                str(Path(item.ownership_audit_csv).resolve())
                if item.ownership_audit_csv else None),
            "ownership_source": own_source or "absent_predates_schema",
        }
    return (evidence, coordinates, summaries, summary_fields, hashes,
            ownership_walks, ownership_loci, ownership_sources)


#: Evidence status recorded for a locus censored by centre ownership.
#:
#: This used to be the distinct spelling "ownership_invalid" while the upstream
#: replication stage wrote "ownership_unevaluable" for the same situation. Both
#: landed in the same `RawEvidenceStatus` column, so a consumer matching one
#: spelling silently missed every row carrying the other -- and "silently missed
#: an ownership censor" means treating a censored locus as ordinary evidence.
#: There is now ONE spelling, owned by replication_status.
OWNERSHIP_CENSORED_STATUS = _rstatus.OWNERSHIP_UNEVALUABLE_STATUS


def _censor_unowned_evidence(
        evidence: dict[tuple[str, int, str], _Evidence],
        ownership_loci: Mapping[tuple[str, int, str], bool],
) -> dict[tuple[str, int, str], _Evidence]:
    """Replace the evidence at every ownership-invalid locus with unevaluable.

    Under the usability-aware path, ``ownership_loci`` carries the USABILITY
    verdict (``CenterOwnershipUsable``); under legacy it carries the exact
    verdict (``CenterOwnershipValid``). The censor itself is unchanged: a
    non-usable locus is censored exactly like a non-valid one.

    Loci absent from ``ownership_loci`` are untouched, so a dataset with no
    ownership schema behaves exactly as before.  Only loci that ALREADY have an
    evidence row are censored.
    """
    if not ownership_loci:
        return evidence
    censored = dict(evidence)
    for key, valid in ownership_loci.items():
        if valid or key not in censored:
            continue
        censored[key] = _Evidence(
            False, False, False, OWNERSHIP_CENSORED_STATUS, None)
    return censored


def _xy_distance(a: _Coordinate | tuple[float, float, float],
                 b: _Coordinate | tuple[float, float, float]) -> float:
    ay, ax = ((a.y_nm, a.x_nm) if isinstance(a, _Coordinate)
              else (a[1], a[2]))
    by, bx = ((b.y_nm, b.x_nm) if isinstance(b, _Coordinate)
              else (b[1], b[2]))
    return math.hypot(ay - by, ax - bx)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    """Hyndman-Fan type 7 quantile (NumPy's default), without NumPy."""
    if not values:
        raise TraceSalvageError("cannot take quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _walk_keys(evidence: Mapping[tuple[str, int, str], _Evidence],
               coordinates: Mapping[tuple[str, int, str], _Coordinate],
               summaries: Mapping[tuple[str, int], Mapping[str, Any]],
               extra: Sequence[tuple[str, int]] = ()) -> list[
                   tuple[str, int]]:
    keys = {(key[0], key[1]) for key in evidence}
    keys.update((key[0], key[1]) for key in coordinates)
    keys.update(summaries)
    keys.update(extra)
    return sorted(keys, key=lambda key: (key[0], key[1]))


def fit_policy(config: TraceSalvageConfig) -> dict[str, Any]:
    """Fit leave-one-location-out clean-link cutoffs for a dataset."""
    # Calibration deliberately ignores the ownership tables: it is fitted from
    # clean walks, and a walk with an ownership-invalid locus is not clean, so it
    # is already excluded by the confident-single requirement below.
    (evidence, coordinates, summaries, _, input_hashes,
     _own_walks, ownership_loci, _own_sources) = _load_inputs(config)
    evidence = _censor_unowned_evidence(evidence, ownership_loci)
    links_by_location: dict[str, list[tuple[str, float]]] = defaultdict(list)
    clean_walks = Counter()
    order = config.chromosome_order
    for location, walk_id in _walk_keys(evidence, coordinates, summaries):
        loci = [(evidence.get((location, walk_id, tp)),
                 coordinates.get((location, walk_id, tp))) for tp in order]
        if not all(ev is not None and ev.confident_single and coord is not None
                   for ev, coord in loci):
            continue
        clean_walks[location] += 1
        for index in range(len(order) - 1):
            left = loci[index][1]
            right = loci[index + 1][1]
            assert left is not None and right is not None
            link_id = f"{location}|{walk_id}|{order[index]}|{order[index + 1]}"
            links_by_location[location].append((link_id, _xy_distance(left, right)))

    calibrations: dict[str, Any] = {}
    for item in config.locations:
        pooled = [entry for source, entries in links_by_location.items()
                  if source != item.location for entry in entries]
        values = [value for _, value in pooled]
        enough = len(values) >= config.min_calibration_links
        threshold = (_linear_quantile(values, config.clean_link_quantile)
                     if enough else None)
        calibrations[item.location] = {
            "status": "calibrated" if enough else "insufficient_calibration_links",
            "clean_link_quantile_nm": threshold,
            "max_link_xy_nm": threshold,
            "n_calibration_links": len(values),
            "min_calibration_links": config.min_calibration_links,
            "source_locations": sorted(
                source for source, entries in links_by_location.items()
                if source != item.location and entries),
            "clean_walks_by_source_location": {
                source: clean_walks[source]
                for source in sorted(links_by_location) if source != item.location
            },
        }
    contract = _config_contract(config)
    policy: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        **contract,
        "single_locus_definition": {
            "v2": "replication_confirmed == 0 and status == tested_not_confirmed",
            "v3": "v3_locus_evaluable == 1 and ResolvedLateralTwoComponentPrimary == 0",
        }[config.evidence_schema],
        "clean_walk_definition": (
            "every planned locus is confidently single and has a finite "
            "drift-corrected XYZ localization"),
        "calibration": {
            "method": "leave_one_location_out",
            "links": "XY Euclidean links between consecutive chromosome_order loci",
            "quantile_method": "Hyndman-Fan type 7",
            "locations": calibrations,
        },
        "selection": {
            "distance_axes": "XY only",
            "path_score": "d_xy(N-1, component) + d_xy(component, N+1)",
            "winner_test": "best_path <= winner_ratio * runner_up_path",
            "link_test": "both winning links <= location LOO quantile",
            "selected_output": "full drift-corrected XYZ of winning component",
            "never_salvage": [config.ori_tp, config.ter_tp],
            "requires_ori_confident_single": True,
            "requires_isolated_doublet": True,
        },
        "input_paths": input_hashes,
        "interpretation": (
            "A salvaged coordinate is a derived trace representation, not proof "
            "that the raw doublet was false or that the cell was non-replicating."),
    }
    return policy


def _validate_policy(config: TraceSalvageConfig,
                     policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise TraceSalvageError("unsupported trace-salvage policy schema")
    contract = _config_contract(config)
    for key, expected in contract.items():
        actual = policy.get(key)
        if _canonical_json(actual) != _canonical_json(expected):
            raise TraceSalvageError(
                f"policy {key!r} disagrees with config: "
                f"{actual!r} vs {expected!r}")


def _component_nm(component_px: tuple[float, float, float],
                  coordinate: _Coordinate,
                  config: TraceSalvageConfig) -> tuple[float, float, float]:
    z, y, x = component_px
    return ((z - coordinate.drift_z_px) * config.z_step_nm,
            (y - coordinate.drift_y_px) * config.xy_pixel_nm,
            (x - coordinate.drift_x_px) * config.xy_pixel_nm)


LOCUS_FIELDS = (
    "Dataset", "Location", "WalkID", "TP", "RawTwoComponentCall",
    "RawEvaluable", "RawConfidentSingle", "RawEvidenceStatus",
    "TraceZNm", "TraceYNm", "TraceXNm", "TraceCoordinateSource",
    "TraceSelectedComponent", "TraceSalvaged", "TraceLocusState",
    "TraceSalvageReason",
)

AUDIT_FIELDS = (
    "Dataset", "Location", "WalkID", "TP", "PrevTP", "NextTP",
    "RawTwoComponentCall", "OriConfidentSingle", "PrevConfidentSingle",
    "NextConfidentSingle", "StructurallyEligible", "CalibrationStatus",
    "CalibrationLinkQuantileNm", "Component1PrevLinkXYNm",
    "Component1NextLinkXYNm", "Component1PathXYNm",
    "Component2PrevLinkXYNm", "Component2NextLinkXYNm",
    "Component2PathXYNm", "BestPathXYNm", "RunnerUpPathXYNm",
    "BestToRunnerRatio", "WinnerRatioThreshold", "WinningPrevLinkXYNm",
    "WinningNextLinkXYNm", "SelectedComponent", "TraceSalvaged",
    "TraceSalvageReason",
)

DERIVED_WALK_FIELDS = (
    "TraceSalvageStatus", "TraceSalvagedMultiSpot",
    "TraceSalvagedSingleSpotComplete", "TraceRawDoubletLociCount",
    "TraceSalvagedLociCount", "TraceUnresolvedDoubletLociCount",
    "TraceSalvageEligibleLociCount", "TraceSalvagedTPs",
    "TraceUnresolvedDoubletTPs",
    *_own.OWNERSHIP_WALK_FIELDS,
    *_own.USABILITY_WALK_FIELDS,
)

#: Identity of a walk in the combined comprehensive CSV. `OriginalWalkID` is
#: carried through when present but is NOT part of the key: it is a 1:1 function
#: of the identity below, so requiring it would reject an otherwise valid file.
WALK_KEY_FIELDS = ("Dataset", "Location", "WalkID")
WALK_CARRIED_ID_FIELDS = ("OriginalWalkID",)


def walk_level_rows(comprehensive_csv) -> list[dict[str, str]]:
    """One row per walk, DERIVED from the comprehensive CSV. One reader.

    The reduced walk-summary CSV is no longer a default output: every field it
    held is already carried on every row of the comprehensive file, so shipping
    it as a second file created two representations of the same facts that could
    disagree. Consumers that genuinely need walk-level information -- the
    campaign's annotate stage is the only one -- call this instead.

    Walk-invariant by construction: a field whose value differs between two rows
    of the same walk is an error rather than a silently-taken first value, since
    that would hide exactly the kind of corruption this reader exists to avoid.
    """
    seen: dict[tuple[str, ...], dict[str, str]] = {}
    #: Columns observed to vary within some walk. Those are locus-level, so they
    #: have no walk-level value and are dropped from the projection.
    per_locus: set[str] = set()
    with open(comprehensive_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [name for name in WALK_KEY_FIELDS if name not in header]
        if missing:
            raise TraceSalvageError(
                f"{comprehensive_csv} lacks walk identity columns {missing}")
        # Declared walk-level fields MUST be walk-invariant; a violation is
        # corruption, not a column to reclassify.
        declared = [name for name in header
                    if name in DERIVED_WALK_FIELDS
                    or name in WALK_KEY_FIELDS
                    or name in WALK_CARRIED_ID_FIELDS]
        for number, row in enumerate(reader, start=2):
            key = tuple(str(row.get(name, "")).strip()
                        for name in WALK_KEY_FIELDS)
            current = {name: str(row.get(name, "")) for name in header}
            existing = seen.get(key)
            if existing is None:
                seen[key] = current
                continue
            for name, value in current.items():
                if existing[name] == value:
                    continue
                if name in declared:
                    raise TraceSalvageError(
                        f"{comprehensive_csv}:{number}: walk-level field "
                        f"{name!r} differs within walk {key}: "
                        f"{existing[name]!r} vs {value!r}")
                # Anything else that varies is per-locus by observation.
                per_locus.add(name)
    # Carry every column that is walk-invariant, not only the ones this module
    # derives. The upstream summary fields ride on every comprehensive row -- the
    # campaign's annotate stage requires CalibratedReplicatingComplexID among
    # them -- and projecting to DERIVED_WALK_FIELDS alone silently dropped them,
    # which made annotate fail closed as soon as trace salvage was enabled.
    keep = [name for name in header if name not in per_locus]
    return [{name: row[name] for name in keep} for row in seen.values()]


def _fmt(value: float | None) -> str:
    return "" if value is None else format(value, ".17g")


def _atomic_csv(path: Path, fields: Sequence[str],
                rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields),
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_SOURCE_ID_FIELDS = {
    "Dataset", "Location", "WalkID", "OriginalWalkID", "TP", "tp",
    "OriginalTP",
}
_FINAL_ID_FIELDS = ("Dataset", "Location", "WalkID", "OriginalWalkID", "tp")
_ORIGINAL_COORDINATE_FIELDS = ("OriginalZNm", "OriginalYNm", "OriginalXNm")
_STANDARD_COORDINATE_FIELDS = ("z_nm", "y_nm", "x_nm")


def _csv_header(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            fields = next(reader)
        except StopIteration as exc:
            raise TraceSalvageError(f"{path} has no CSV header") from exc
    if not fields or len(fields) != len(set(fields)):
        raise TraceSalvageError(f"{path} has an invalid or duplicate CSV header")
    return fields


def _unique_source_name(field: str, occupied: set[str], suffix: str) -> str:
    candidate = field
    if candidate not in occupied:
        return candidate
    candidate = f"{field}{suffix}"
    counter = 2
    while candidate in occupied:
        candidate = f"{field}{suffix}_{counter}"
        counter += 1
    return candidate


def _source_field_map(fields: Sequence[str], occupied: set[str],
                      suffix: str) -> list[tuple[str, str]]:
    mapping: list[tuple[str, str]] = []
    for field in fields:
        if field in _SOURCE_ID_FIELDS:
            continue
        output = _unique_source_name(field, occupied, suffix)
        occupied.add(output)
        mapping.append((field, output))
    return mapping


def _location_output_schema(item: LocationInputs) -> dict[str, Any]:
    """Build a deterministic, collision-safe schema for one source location."""
    coordinate_fields = _csv_header(item.coordinates_csv)
    if item.all_coordinates_csv:
        # The two files are not required to have identical headers: the
        # full-coverage file legitimately carries `movement_nm`/`movement_def`,
        # which are undefined for a walk that is not full coverage. Take the
        # union, full-coverage order first, and let the writer leave a column
        # blank for a row whose source file does not have it.
        coordinate_fields = coordinate_fields + [
            field for field in _csv_header(item.all_coordinates_csv)
            if field not in coordinate_fields]
    evidence_fields = _csv_header(item.evidence_csv)
    summary_fields = (
        _csv_header(item.walk_summary_csv) if item.walk_summary_csv else [])

    # Derived names are reserved so a previously enriched input cannot silently
    # overwrite the new decision.  Upstream fields with those names are retained
    # under an explicit source suffix.
    derived_fields = [
        *(_ORIGINAL_COORDINATE_FIELDS),
        *(field for field in LOCUS_FIELDS if field not in _SOURCE_ID_FIELDS),
        *DERIVED_WALK_FIELDS,
        *ATTEMPT_FIELDS,
        *AUDIT_COORDINATE_FIELDS,
    ]
    occupied = {"Dataset", "Location", "WalkID", "tp", *derived_fields}
    coordinate_map = _source_field_map(
        coordinate_fields, occupied, "_coordinate")
    for field in _STANDARD_COORDINATE_FIELDS:
        if field not in occupied:
            occupied.add(field)
            coordinate_map.append((field, field))
    # A ledger carries the canonical verdict for attempts the coordinate CSV
    # cannot represent, so those columns must exist even when the coordinate CSV
    # predates them. Columns the coordinate CSV already provides are untouched,
    # so the ordinary schema is unchanged.
    ownership_only: list[str] = []
    if item.ownership_audit_csv:
        for field in (*_own.OWNERSHIP_ROW_FIELDS, *_own.USABILITY_ROW_FIELDS):
            if field not in occupied:
                occupied.add(field)
                ownership_only.append(field)
    evidence_map = _source_field_map(evidence_fields, occupied, "_qc")
    summary_map = _source_field_map(
        summary_fields, occupied, "_walk_summary")

    fields = ["Dataset", "Location", "WalkID", "tp"]
    fields.extend(ATTEMPT_FIELDS)
    fields.extend(output for _, output in coordinate_map)
    fields.extend(ownership_only)
    fields.extend(field for field in _ORIGINAL_COORDINATE_FIELDS
                  if field not in fields)
    fields.extend(AUDIT_COORDINATE_FIELDS)
    fields.extend(output for _, output in evidence_map)
    fields.extend(output for _, output in summary_map)
    fields.extend(field for field in LOCUS_FIELDS
                  if field not in _SOURCE_ID_FIELDS and field not in fields)
    fields.extend(field for field in DERIVED_WALK_FIELDS if field not in fields)
    if len(fields) != len(set(fields)):
        raise AssertionError("trace-salvage output schema contains duplicates")
    # Coordinate columns that classify_replication ANNOTATED onto the coordinate
    # rows from the walk summary. They are per-walk, not per-locus, so a
    # reintroduced attempt can and must carry them: leaving `replication_state`
    # blank on a rejected row would hide walk-level context that is not
    # coordinate-bearing and is available for that very walk.
    walk_level_map = [(field, output) for field, output in coordinate_map
                      if field in set(summary_fields)]
    return {
        "fields": fields,
        "coordinate_map": coordinate_map,
        "evidence_map": evidence_map,
        "summary_map": summary_map,
        "walk_level_map": walk_level_map,
    }


def _copy_mapped(row: dict[str, Any], source: Mapping[str, Any],
                 mapping: Sequence[tuple[str, str]]) -> None:
    for input_field, output_field in mapping:
        if input_field in source:
            row[output_field] = source[input_field]


def _outside_salvage_locus(config: TraceSalvageConfig, location: str,
                           walk_id: int, tp: str, evidence: _Evidence | None,
                           coordinate: _Coordinate) -> dict[str, Any]:
    """Describe an acquired locus outside the configured salvage interval."""
    return {
        "Dataset": config.dataset,
        "Location": location,
        "WalkID": walk_id,
        "TP": tp,
        "RawTwoComponentCall": int(bool(evidence and evidence.call)),
        "RawEvaluable": int(bool(evidence and evidence.evaluable)),
        "RawConfidentSingle": int(bool(evidence and evidence.confident_single)),
        "RawEvidenceStatus": evidence.status if evidence else "missing",
        "TraceZNm": _fmt(coordinate.z_nm),
        "TraceYNm": _fmt(coordinate.y_nm),
        "TraceXNm": _fmt(coordinate.x_nm),
        "TraceCoordinateSource": "original_drift_corrected_outside_salvage_range",
        "TraceSelectedComponent": -1,
        "TraceSalvaged": 0,
        "TraceLocusState": "outside_salvage_range",
        "TraceSalvageReason": "outside_salvage_range",
    }


#: ``TraceSalvageReason`` for a reintroduced attempt at a timepoint the resolver
#: never evaluated -- an acquired control outside the configured interval. Why the
#: row has no coordinate is stated separately, in ``LocalizationAttemptStatus``,
#: so ``TraceSalvage*`` keeps meaning exactly what it means in the audit CSV.
UNEVALUATED_LOCUS_REASON = "no_usable_coordinate"


def _unavailable_locus(config: TraceSalvageConfig, location: str, walk_id: int,
                       tp: str, evidence: _Evidence | None) -> dict[str, Any]:
    """Describe a reintroduced attempt outside the configured locus interval.

    Every ``Trace*`` field is blank: a rejected or failed attempt must not
    receive a trace, salvaged, or drift-corrected coordinate, and blanks are the
    only representation that cannot be mistaken for one.
    """
    return {
        "Dataset": config.dataset,
        "Location": location,
        "WalkID": walk_id,
        "TP": tp,
        "RawTwoComponentCall": 0,
        "RawEvaluable": 0,
        "RawConfidentSingle": 0,
        "RawEvidenceStatus": evidence.status if evidence else "missing",
        "TraceZNm": "", "TraceYNm": "", "TraceXNm": "",
        "TraceCoordinateSource": "",
        "TraceSelectedComponent": -1,
        "TraceSalvaged": 0,
        "TraceLocusState": "unevaluable_or_unqualified",
        "TraceSalvageReason": UNEVALUATED_LOCUS_REASON,
    }


def _safe_location_filename(location: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", location).strip("._")
    if not safe:
        raise TraceSalvageError(f"location {location!r} has no safe filename")
    return safe


def _write_location_full_schema(
        config: TraceSalvageConfig, item: LocationInputs,
        schema: Mapping[str, Any],
        evidence: Mapping[tuple[str, int, str], _Evidence],
        locus_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
        walk_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
        output_path: Path) -> dict[str, Any]:
    """Write one comprehensive row for every localization ATTEMPT.

    The attempt universe is the localizer's per-attempt ledger when the location
    declares one; accepted scientific results are joined onto it.  Without a
    ledger the universe stays exactly what it historically was -- the coordinate
    rows -- so a dataset that predates the schema is unaffected.

    An attempt with no usable canonical coordinate (ownership-rejected, failed
    fit, or no drift solution) is still written, with every canonical coordinate
    blank and its unconstrained fitted centre in the audit-only
    ``UnconstrainedFit*Nm`` columns.
    """
    raw_evidence: dict[tuple[int, str], dict[str, str]] = {}
    _, evidence_rows = _read_csv(item.evidence_csv)
    for number, source in enumerate(evidence_rows, start=2):
        location, walk_id, tp = _row_key(
            source, item.location, item.evidence_csv, number)
        key = (walk_id, tp)
        if key in raw_evidence:
            raise TraceSalvageError(
                f"duplicate evidence key {(location, walk_id, tp)!r}")
        raw_evidence[key] = source

    raw_summaries: dict[int, dict[str, str]] = {}
    if item.walk_summary_csv:
        _, summary_rows = _read_csv(item.walk_summary_csv)
        for number, source in enumerate(summary_rows, start=2):
            embedded = str(source.get("Location", "")).strip()
            if embedded and embedded != item.location:
                raise TraceSalvageError(
                    f"{item.walk_summary_csv}:{number} has unexpected Location")
            walk_id = _walk_id(source.get("WalkID"))
            if walk_id in raw_summaries:
                raise TraceSalvageError(
                    f"duplicate walk summary key {(item.location, walk_id)!r}")
            raw_summaries[walk_id] = source

    ledger = _read_attempt_ledger(item)
    attempts, ledger_has_centres = (
        ledger if ledger is not None else ({}, False))
    coordinate_source = item.all_coordinates_csv or item.coordinates_csv
    if item.all_coordinates_csv:
        # The full-coverage file must be a row subset of the unfiltered one, or
        # switching the universe would silently drop accepted coordinate rows.
        _, subset_rows = _read_csv(item.coordinates_csv)
        subset_keys = {
            _row_key(row, item.location, item.coordinates_csv, number)[1:]
            for number, row in enumerate(subset_rows, start=2)}
    else:
        subset_keys = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent)
    seen_loci: set[tuple[int, str]] = set()
    source_walks: set[int] = set()
    source_tps: set[str] = set()
    row_count = 0
    attempt_counts: Counter = Counter()
    try:
        with open(coordinate_source, newline="", encoding="utf-8-sig") as source_handle, \
                os.fdopen(fd, "w", newline="", encoding="utf-8") as output_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise TraceSalvageError(
                    f"{coordinate_source} has no CSV header")
            writer = csv.DictWriter(
                output_handle, fieldnames=list(schema["fields"]),
                extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for number, source in enumerate(reader, start=2):
                location, walk_id, tp = _row_key(
                    source, item.location, coordinate_source, number)
                local_key = (walk_id, tp)
                if local_key in seen_loci:
                    raise TraceSalvageError(
                        f"duplicate coordinate key {(location, walk_id, tp)!r}")
                seen_loci.add(local_key)
                source_walks.add(walk_id)
                source_tps.add(tp)
                try:
                    coordinate = _normalize_coordinate(source)
                except TraceSalvageError as exc:
                    raise TraceSalvageError(
                        f"{coordinate_source}:{number}: {exc}") from exc
                attempt = attempts.get(local_key)
                if ledger is not None and attempt is None:
                    raise TraceSalvageError(
                        f"{coordinate_source}:{number}: "
                        f"{(location, walk_id, tp)!r} has a coordinate row but no "
                        f"entry in {item.ownership_audit_csv}; the attempt ledger "
                        "and the coordinates are from different runs")
                if attempt is not None and not attempt.written:
                    raise TraceSalvageError(
                        f"{coordinate_source}:{number}: the attempt ledger says "
                        f"{(location, walk_id, tp)!r} produced no localization, "
                        "but a coordinate row exists; the two inputs are from "
                        "different runs")

                key = (location, walk_id, tp)
                derived = locus_by_key.get(key)
                if derived is None:
                    derived = _outside_salvage_locus(
                        config, location, walk_id, tp, evidence.get(key),
                        coordinate)
                walk = walk_by_key.get((location, walk_id))
                if walk is None:
                    raise TraceSalvageError(
                        f"coordinate row has no trace walk result: "
                        f"{(location, walk_id, tp)!r}")

                row: dict[str, Any] = {
                    "Dataset": config.dataset,
                    "Location": location,
                    "WalkID": walk_id,
                    "tp": tp,
                }
                _copy_mapped(row, source, schema["coordinate_map"])
                _copy_mapped(
                    row, raw_evidence.get(local_key, {}),
                    schema["evidence_map"])
                _copy_mapped(
                    row, raw_summaries.get(walk_id, {}),
                    schema["summary_map"])
                for field in LOCUS_FIELDS:
                    if field not in _SOURCE_ID_FIELDS:
                        row[field] = derived.get(field, "")
                for field in DERIVED_WALK_FIELDS:
                    row[field] = walk.get(field, "")

                row["OriginalZNm"] = _fmt(coordinate.z_nm)
                row["OriginalYNm"] = _fmt(coordinate.y_nm)
                row["OriginalXNm"] = _fmt(coordinate.x_nm)
                trace_values = {
                    "z_nm": row.get("TraceZNm", ""),
                    "y_nm": row.get("TraceYNm", ""),
                    "x_nm": row.get("TraceXNm", ""),
                }
                originals = {
                    "z_nm": row["OriginalZNm"],
                    "y_nm": row["OriginalYNm"],
                    "x_nm": row["OriginalXNm"],
                }
                for field in _STANDARD_COORDINATE_FIELDS:
                    value = str(trace_values[field]).strip()
                    row[field] = value if value else originals[field]
                row[ATTEMPT_FIELDS[0]] = ATTEMPT_ACCEPTED
                row[ATTEMPT_FIELDS[1]] = ""
                # The same unconstrained centre and the same transform as a
                # rejected row, so the audit schema is uniform across attempts.
                row.update(_audit_coordinates(
                    _raw_pixel_centre(source, attempt), config))
                writer.writerow(row)
                row_count += 1
                attempt_counts[ATTEMPT_ACCEPTED] += 1

            if subset_keys is not None and not subset_keys <= seen_loci:
                raise TraceSalvageError(
                    f"{item.coordinates_csv} has "
                    f"{len(subset_keys - seen_loci)} rows absent from "
                    f"{item.all_coordinates_csv}; the unfiltered file must be a "
                    "superset of the full-coverage file")

            for local_key in sorted(set(attempts) - seen_loci):
                attempt = attempts[local_key]
                walk_id, tp = local_key
                location = item.location
                key = (location, walk_id, tp)
                if not attempt.fit_succeeded:
                    status = ATTEMPT_FIT_FAILED
                elif not attempt.written:
                    status = ATTEMPT_OWNERSHIP_REJECTED
                elif item.all_coordinates_csv:
                    status = ATTEMPT_NO_DRIFT_SOLUTION
                else:
                    raise TraceSalvageError(
                        f"{key!r} has an accepted localization that is absent "
                        f"from {item.coordinates_csv}; supply "
                        "all_coordinates_csv (the unfiltered drift-corrected "
                        "file) so its coordinate is not lost")
                walk = walk_by_key.get((location, walk_id))
                if walk is None:
                    raise TraceSalvageError(
                        f"attempt row has no trace walk result: {key!r}")
                row = {
                    "Dataset": config.dataset,
                    "Location": location,
                    "WalkID": walk_id,
                    "tp": tp,
                    ATTEMPT_FIELDS[0]: status,
                    ATTEMPT_FIELDS[1]: attempt.reason,
                }
                # No blanket coordinate_map copy: every canonical coordinate
                # column stays blank. Only the per-walk annotations and the
                # ownership verdict travel, under the same column names an
                # accepted row uses.
                _copy_mapped(
                    row, raw_summaries.get(walk_id, {}),
                    schema["walk_level_map"])
                for field, value in attempt.ownership.items():
                    if field in schema["fields"]:
                        row[field] = value
                _copy_mapped(
                    row, raw_evidence.get(local_key, {}),
                    schema["evidence_map"])
                _copy_mapped(
                    row, raw_summaries.get(walk_id, {}),
                    schema["summary_map"])
                # The resolver's own verdict for a configured locus, so the
                # comprehensive row and the audit CSV never disagree about
                # TraceSalvageReason. It is blank-coordinate by construction: the
                # resolver only produces a trace coordinate from the decision
                # input, and this attempt is absent from it. Asserted, not assumed.
                derived = locus_by_key.get(key) or _unavailable_locus(
                    config, location, walk_id, tp, evidence.get(key))
                if any(str(derived.get(field, "")).strip()
                       for field in ("TraceZNm", "TraceYNm", "TraceXNm")):
                    raise TraceSalvageError(
                        f"{key!r} has no localization coordinate yet carries a "
                        "derived trace coordinate")
                for field in LOCUS_FIELDS:
                    if field not in _SOURCE_ID_FIELDS:
                        row[field] = derived.get(field, "")
                for field in DERIVED_WALK_FIELDS:
                    row[field] = walk.get(field, "")
                for field in (*_ORIGINAL_COORDINATE_FIELDS,
                              *_STANDARD_COORDINATE_FIELDS):
                    row[field] = ""
                row.update(_audit_coordinates(attempt.centre_px, config))
                writer.writerow(row)
                row_count += 1
                seen_loci.add(local_key)
                source_walks.add(walk_id)
                source_tps.add(tp)
                attempt_counts[status] += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "location": item.location,
        "path": output_path,
        "fields": list(schema["fields"]),
        "rows": row_count,
        "walk_ids": source_walks,
        "tps": source_tps,
        "attempt_universe": (
            "attempt_ledger" if ledger is not None else "coordinate_rows"),
        "attempt_counts": dict(sorted(attempt_counts.items())),
        "ledger_attempts": len(attempts),
        "ledger_has_fit_centres": bool(ledger_has_centres),
    }


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", value))


def _combine_location_outputs(
        locations: Sequence[Mapping[str, Any]], full_path: Path,
        complete_path: Path,
        complete_walks: set[tuple[str, int]]) -> tuple[
            dict[tuple[str, int], int], int, int, list[str]]:
    """Combine enriched location files and assign dataset-wide walk IDs."""
    source_walks = {
        (str(item["location"]), int(walk_id))
        for item in locations for walk_id in item["walk_ids"]
    }
    walk_order = sorted(
        source_walks, key=lambda key: (_natural_key(key[0]), key[1]))
    global_ids = {key: index for index, key in enumerate(walk_order, start=1)}

    fields = list(_FINAL_ID_FIELDS)
    for item in locations:
        for field in item["fields"]:
            if field not in _SOURCE_ID_FIELDS and field not in fields:
                fields.append(field)

    for path in (full_path, complete_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    full_fd, full_temporary = tempfile.mkstemp(
        prefix=f".{full_path.name}.", dir=full_path.parent)
    complete_fd, complete_temporary = tempfile.mkstemp(
        prefix=f".{complete_path.name}.", dir=complete_path.parent)
    full_rows = complete_rows = 0
    observed_global_ids: set[int] = set()
    try:
        with os.fdopen(full_fd, "w", newline="", encoding="utf-8") as full_handle, \
                os.fdopen(complete_fd, "w", newline="", encoding="utf-8") as complete_handle:
            full_writer = csv.DictWriter(
                full_handle, fieldnames=fields, extrasaction="ignore",
                lineterminator="\n")
            complete_writer = csv.DictWriter(
                complete_handle, fieldnames=fields, extrasaction="ignore",
                lineterminator="\n")
            full_writer.writeheader()
            complete_writer.writeheader()
            for item in locations:
                with Path(item["path"]).open(
                        newline="", encoding="utf-8-sig") as source_handle:
                    reader = csv.DictReader(source_handle)
                    for number, source in enumerate(reader, start=2):
                        location = str(source.get("Location", "")).strip()
                        if location != item["location"]:
                            raise TraceSalvageError(
                                f"{item['path']}:{number} has unexpected Location")
                        original = _walk_id(source.get("WalkID"))
                        tp = _canonical_tp(source.get("tp", ""))
                        key = (location, original)
                        global_id = global_ids[key]
                        row = dict(source)
                        for field in _SOURCE_ID_FIELDS:
                            row.pop(field, None)
                        row.update({
                            "Dataset": source.get("Dataset", ""),
                            "Location": location,
                            "WalkID": global_id,
                            "OriginalWalkID": original,
                            "tp": tp,
                        })
                        full_writer.writerow(row)
                        full_rows += 1
                        observed_global_ids.add(global_id)
                        expected_complete = key in complete_walks
                        flag = str(source.get(
                            "TraceSalvagedSingleSpotComplete", "")).strip()
                        if flag not in {"0", "1"} or (flag == "1") != expected_complete:
                            raise TraceSalvageError(
                                f"{item['path']}:{number} has inconsistent "
                                "TraceSalvagedSingleSpotComplete")
                        if expected_complete:
                            usable_val = str(source.get(
                                "CenterOwnershipUsableComplete", "")).strip()
                            exact_val = str(source.get(
                                "CenterOwnershipComplete", "")).strip()
                            if usable_val and usable_val == "0":
                                raise TraceSalvageError(
                                    f"{item['path']}:{number} is salvage-complete "
                                    "with CenterOwnershipUsableComplete=0; the "
                                    "filtered output must never contain an "
                                    "ownership-incomplete walk")
                            if not usable_val and exact_val == "0":
                                raise TraceSalvageError(
                                    f"{item['path']}:{number} is salvage-complete "
                                    "with CenterOwnershipComplete=0 (legacy); the "
                                    "filtered output must never contain an "
                                    "ownership-incomplete walk")
                        if expected_complete:
                            complete_writer.writerow(row)
                            complete_rows += 1
            full_handle.flush()
            os.fsync(full_handle.fileno())
            complete_handle.flush()
            os.fsync(complete_handle.fileno())
        expected_ids = set(range(1, len(global_ids) + 1))
        if observed_global_ids != expected_ids:
            raise TraceSalvageError(
                "combined WalkID values are not continuous and complete")
        os.replace(full_temporary, full_path)
        os.replace(complete_temporary, complete_path)
    finally:
        for temporary in (full_temporary, complete_temporary):
            if os.path.exists(temporary):
                os.unlink(temporary)
    return global_ids, full_rows, complete_rows, fields


def apply_policy(config: TraceSalvageConfig, policy: Mapping[str, Any],
                 outdir: os.PathLike[str] | str,
                 output_prefix: str = "trace_salvage",
                 diagnostics: bool = False) -> dict[str, Any]:
    """Apply a frozen calibration policy and atomically write audited outputs.

    Four public outputs by default: the comprehensive unfiltered CSV, the
    comprehensive `TraceSalvagedSingleSpotComplete == 1` CSV, the policy JSON and
    the manifest. The comprehensive file retains every ownership, audit,
    replication, original-coordinate and trace-coordinate field, so nothing is
    lost by not writing the others -- the per-locus audit table and the walk
    summary are both derivable from it, and were separate files that could drift.

    ``diagnostics=True`` additionally writes the per-locus audit CSV, the reduced
    walk summary and the per-Location comprehensive files. It is ONE switch
    rather than one per file, so "diagnostics" is a single, checkable state, and
    it changes only WHICH files appear -- never a coordinate, a global/
    OriginalWalkID mapping, walk membership, or a salvage decision.
    """
    _validate_policy(config, policy)
    (evidence, coordinates, summaries, summary_fields, input_hashes,
     ownership_walks, ownership_loci, ownership_sources) = _load_inputs(config)
    evidence = _censor_unowned_evidence(evidence, ownership_loci)
    calibrations = policy["calibration"]["locations"]
    order = config.chromosome_order
    index_by_tp = {tp: index for index, tp in enumerate(order)}
    locus_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    walk_rows: list[dict[str, Any]] = []
    counts = Counter()

    # Every walk in the ownership ledger needs a decision row, because the
    # comprehensive output now contains its attempts: a walk whose every locus
    # was rejected has no evidence, no coordinates and no summary, and would
    # otherwise have rows with no walk-level result at all. It cannot change any
    # salvage decision -- with no evidence it is never a confident single -- and
    # the calibration in fit_policy is deliberately left on the original universe.
    for location, walk_id in _walk_keys(evidence, coordinates, summaries,
                                        sorted(ownership_walks)):
        one_evidence = {tp: evidence.get((location, walk_id, tp)) for tp in order}
        one_coords = {tp: coordinates.get((location, walk_id, tp)) for tp in order}
        ori = one_evidence[config.ori_tp]
        ori_single = bool(ori and ori.confident_single)
        location_cal = calibrations.get(location, {})
        cal_status = str(location_cal.get("status", "missing_location_calibration"))
        threshold_raw = location_cal.get("clean_link_quantile_nm")
        threshold = (float(threshold_raw) if threshold_raw is not None else None)
        per_walk: list[dict[str, Any]] = []

        _simple = (config.salvage_policy == _own.SALVAGE_SIMPLE_NEIGHBOR)

        for tp in order:
            ev = one_evidence[tp]
            coord = one_coords[tp]
            call = bool(ev and ev.call)
            prev_tp = next_tp = ""
            structurally_eligible = False
            reason = "not_a_raw_doublet"
            selected = -1
            accepted = False
            metrics: dict[str, float | None] = {
                "c1_prev": None, "c1_next": None, "c1_path": None,
                "c2_prev": None, "c2_next": None, "c2_path": None,
                "best": None, "runner": None, "ratio": None,
                "win_prev": None, "win_next": None,
            }
            trace_xyz: tuple[float, float, float] | None = None
            trace_source = ""

            if call:
                index = index_by_tp[tp]
                if tp in {config.ori_tp, config.ter_tp}:
                    reason = "endpoint_never_salvaged"
                elif not ori_single:
                    reason = "ori_not_confident_single"
                elif index == 0 or index == len(order) - 1:
                    reason = "missing_configured_neighbor"
                else:
                    prev_tp, next_tp = order[index - 1], order[index + 1]
                    prev_ev, next_ev = one_evidence[prev_tp], one_evidence[next_tp]
                    prev_call = bool(prev_ev and prev_ev.call)
                    next_call = bool(next_ev and next_ev.call)
                    if not (prev_ev and prev_ev.confident_single
                            and next_ev and next_ev.confident_single):
                        reason = "not_isolated_between_confident_singles"
                    elif prev_call or next_call:
                        reason = "adjacent_is_raw_doublet"
                    else:
                        structurally_eligible = True
                        prev_coord, next_coord = one_coords[prev_tp], one_coords[next_tp]
                        if coord is None or prev_coord is None or next_coord is None:
                            reason = "missing_required_coordinates"
                        elif ev is None or ev.components_px is None:
                            reason = "missing_component_xyz"
                        elif not _simple and (cal_status != "calibrated"
                                              or threshold is None):
                            reason = "calibration_unavailable"
                        else:
                            c1 = _component_nm(ev.components_px[0], coord, config)
                            c2 = _component_nm(ev.components_px[1], coord, config)
                            metrics["c1_prev"] = _xy_distance(prev_coord, c1)
                            metrics["c1_next"] = _xy_distance(c1, next_coord)
                            metrics["c2_prev"] = _xy_distance(prev_coord, c2)
                            metrics["c2_next"] = _xy_distance(c2, next_coord)
                            metrics["c1_path"] = metrics["c1_prev"] + metrics["c1_next"]
                            metrics["c2_path"] = metrics["c2_prev"] + metrics["c2_next"]
                            if not (math.isfinite(metrics["c1_path"])
                                    and math.isfinite(metrics["c2_path"])):
                                reason = "nonfinite_path_metric"
                            elif _simple and metrics["c1_path"] == metrics["c2_path"]:
                                reason = "exact_tie"
                            elif metrics["c1_path"] == metrics["c2_path"]:
                                reason = "winner_not_clear"
                            else:
                                if metrics["c1_path"] < metrics["c2_path"]:
                                    selected, winner = 1, c1
                                    metrics["best"], metrics["runner"] = (
                                        metrics["c1_path"], metrics["c2_path"])
                                    metrics["win_prev"], metrics["win_next"] = (
                                        metrics["c1_prev"], metrics["c1_next"])
                                else:
                                    selected, winner = 2, c2
                                    metrics["best"], metrics["runner"] = (
                                        metrics["c2_path"], metrics["c1_path"])
                                    metrics["win_prev"], metrics["win_next"] = (
                                        metrics["c2_prev"], metrics["c2_next"])
                                runner = metrics["runner"]
                                metrics["ratio"] = (
                                    metrics["best"] / runner
                                    if runner and runner > 0
                                    else math.inf)
                                if _simple:
                                    accepted = True
                                    reason = "salvaged"
                                    trace_xyz = winner
                                    trace_source = (
                                        f"m2_component_{selected}_drift_corrected")
                                else:
                                    if metrics["best"] > config.winner_ratio * runner:
                                        reason = "winner_not_clear"
                                    elif (metrics["win_prev"] > threshold
                                          or metrics["win_next"] > threshold):
                                        reason = "winning_link_exceeds_calibration"
                                    else:
                                        accepted = True
                                        reason = "salvaged"
                                        trace_xyz = winner
                                        trace_source = (
                                            f"m2_component_{selected}"
                                            "_drift_corrected")
            elif ev and ev.confident_single and coord is not None:
                trace_xyz = (coord.z_nm, coord.y_nm, coord.x_nm)
                trace_source = "raw_confident_single_drift_corrected"
                reason = "raw_confident_single"
            elif coord is not None:
                trace_xyz = (coord.z_nm, coord.y_nm, coord.x_nm)
                trace_source = "raw_unqualified_drift_corrected"
                reason = "raw_not_confident_single"
            else:
                reason = "missing_or_unevaluable"

            state = (
                "doublet_salvaged" if accepted else
                "doublet_unresolved" if call else
                "confident_single" if ev and ev.confident_single else
                "unevaluable_or_unqualified")
            locus = {
                "Dataset": config.dataset, "Location": location,
                "WalkID": walk_id, "TP": tp,
                "RawTwoComponentCall": int(call),
                "RawEvaluable": int(bool(ev and ev.evaluable)),
                "RawConfidentSingle": int(bool(ev and ev.confident_single)),
                "RawEvidenceStatus": ev.status if ev else "missing",
                "TraceZNm": _fmt(trace_xyz[0] if trace_xyz else None),
                "TraceYNm": _fmt(trace_xyz[1] if trace_xyz else None),
                "TraceXNm": _fmt(trace_xyz[2] if trace_xyz else None),
                "TraceCoordinateSource": trace_source,
                "TraceSelectedComponent": selected if accepted else -1,
                "TraceSalvaged": int(accepted),
                "TraceLocusState": state,
                "TraceSalvageReason": reason,
            }
            audit = {
                "Dataset": config.dataset, "Location": location,
                "WalkID": walk_id, "TP": tp, "PrevTP": prev_tp,
                "NextTP": next_tp, "RawTwoComponentCall": int(call),
                "OriConfidentSingle": int(ori_single),
                "PrevConfidentSingle": int(bool(
                    prev_tp and one_evidence[prev_tp]
                    and one_evidence[prev_tp].confident_single)),
                "NextConfidentSingle": int(bool(
                    next_tp and one_evidence[next_tp]
                    and one_evidence[next_tp].confident_single)),
                "StructurallyEligible": int(structurally_eligible),
                "CalibrationStatus": cal_status,
                "CalibrationLinkQuantileNm": _fmt(threshold),
                "Component1PrevLinkXYNm": _fmt(metrics["c1_prev"]),
                "Component1NextLinkXYNm": _fmt(metrics["c1_next"]),
                "Component1PathXYNm": _fmt(metrics["c1_path"]),
                "Component2PrevLinkXYNm": _fmt(metrics["c2_prev"]),
                "Component2NextLinkXYNm": _fmt(metrics["c2_next"]),
                "Component2PathXYNm": _fmt(metrics["c2_path"]),
                "BestPathXYNm": _fmt(metrics["best"]),
                "RunnerUpPathXYNm": _fmt(metrics["runner"]),
                "BestToRunnerRatio": _fmt(metrics["ratio"]),
                "WinnerRatioThreshold": _fmt(config.winner_ratio),
                "WinningPrevLinkXYNm": _fmt(metrics["win_prev"]),
                "WinningNextLinkXYNm": _fmt(metrics["win_next"]),
                "SelectedComponent": selected if accepted else -1,
                "TraceSalvaged": int(accepted),
                "TraceSalvageReason": reason,
            }
            locus_rows.append(locus)
            audit_rows.append(audit)
            per_walk.append(locus | {"StructurallyEligible": int(structurally_eligible)})

        # --- Second pass: pair-salvage adjacent doublets ---
        # When two consecutive doublets N, N+1 are both unresolved and the
        # outer anchors N-1, N+2 are confident singles, try all four
        # component pairings and pick the one that minimises the total XY
        # path  anchor_prev -> N -> N+1 -> anchor_next.
        if ori_single:
            walk_row_offset = len(locus_rows) - len(order)
            for i in range(len(order) - 1):
                pw_n, pw_n1 = per_walk[i], per_walk[i + 1]
                if not (pw_n["RawTwoComponentCall"] and not pw_n["TraceSalvaged"]
                        and pw_n1["RawTwoComponentCall"] and not pw_n1["TraceSalvaged"]):
                    continue
                tp_n, tp_n1 = order[i], order[i + 1]
                if tp_n in {config.ori_tp, config.ter_tp}:
                    continue
                if tp_n1 in {config.ori_tp, config.ter_tp}:
                    continue
                if i == 0 or i + 1 == len(order) - 1:
                    continue
                tp_prev, tp_next = order[i - 1], order[i + 2]
                prev_ev = one_evidence[tp_prev]
                next_ev = one_evidence[tp_next]
                if not (prev_ev and prev_ev.confident_single
                        and next_ev and next_ev.confident_single):
                    continue
                prev_coord = one_coords[tp_prev]
                next_coord = one_coords[tp_next]
                coord_n = one_coords[tp_n]
                coord_n1 = one_coords[tp_n1]
                ev_n = one_evidence[tp_n]
                ev_n1 = one_evidence[tp_n1]
                if any(v is None for v in (prev_coord, next_coord,
                                           coord_n, coord_n1)):
                    continue
                if ev_n is None or ev_n.components_px is None:
                    continue
                if ev_n1 is None or ev_n1.components_px is None:
                    continue
                n_c1 = _component_nm(ev_n.components_px[0], coord_n, config)
                n_c2 = _component_nm(ev_n.components_px[1], coord_n, config)
                n1_c1 = _component_nm(ev_n1.components_px[0], coord_n1, config)
                n1_c2 = _component_nm(ev_n1.components_px[1], coord_n1, config)
                best_path = math.inf
                best_pair = None
                for sel_n, comp_n in ((1, n_c1), (2, n_c2)):
                    for sel_n1, comp_n1 in ((1, n1_c1), (2, n1_c2)):
                        path = (_xy_distance(prev_coord, comp_n)
                                + _xy_distance(comp_n, comp_n1)
                                + _xy_distance(comp_n1, next_coord))
                        if path < best_path:
                            best_path = path
                            best_pair = (sel_n, comp_n, sel_n1, comp_n1)
                if best_pair is None or not math.isfinite(best_path):
                    continue
                sel_n, winner_n, sel_n1, winner_n1 = best_pair
                for j, tp, sel, winner in (
                    (i, tp_n, sel_n, winner_n),
                    (i + 1, tp_n1, sel_n1, winner_n1),
                ):
                    source = f"m2_component_{sel}_drift_corrected"
                    update = {
                        "TraceZNm": _fmt(winner[0]),
                        "TraceYNm": _fmt(winner[1]),
                        "TraceXNm": _fmt(winner[2]),
                        "TraceCoordinateSource": source,
                        "TraceSelectedComponent": sel,
                        "TraceSalvaged": 1,
                        "TraceLocusState": "doublet_salvaged",
                        "TraceSalvageReason": "pair_salvaged",
                    }
                    per_walk[j].update(update)
                    locus_rows[walk_row_offset + j].update(update)
                    audit_rows[walk_row_offset + j].update({
                        "SelectedComponent": sel,
                        "TraceSalvaged": 1,
                        "TraceSalvageReason": "pair_salvaged",
                        "BestPathXYNm": _fmt(best_path),
                    })

        raw_doublets = [row for row in per_walk if row["RawTwoComponentCall"]]
        salvaged = [row for row in raw_doublets if row["TraceSalvaged"]]
        unresolved = [row for row in raw_doublets if not row["TraceSalvaged"]]
        eligible_count = sum(row["StructurallyEligible"] for row in per_walk)
        # Walk-level centre ownership over the configured locus universe. Absent
        # only when the dataset has no ownership schema at all, in which case the
        # gate below is not applicable and the fields stay blank.
        own_walk = ownership_walks.get((location, walk_id))
        # Prefer usability-based completeness when the columns exist;
        # fall back to exact ownership for legacy data.
        usable_complete_field = "CenterOwnershipUsableComplete"
        if own_walk is not None and usable_complete_field in own_walk:
            ownership_usable_complete = bool(own_walk[usable_complete_field])
        elif own_walk is not None:
            ownership_usable_complete = bool(
                own_walk.get("CenterOwnershipComplete", 0))
        else:
            ownership_usable_complete = None

        complete = (
            ori_single
            and all(row["TraceLocusState"] in {"confident_single", "doublet_salvaged"}
                    and str(row["TraceZNm"]).strip()
                    for row in per_walk)
            and one_evidence[config.ter_tp] is not None
            and one_evidence[config.ter_tp].confident_single)
        if ownership_usable_complete is False:
            complete = False
        if complete and salvaged:
            status = "complete_after_salvage"
        elif complete:
            status = "complete_clean"
        elif not ori_single:
            status = "ori_not_confident_single"
        elif unresolved:
            status = "unresolved_doublets"
        else:
            status = "incomplete_or_unevaluable"
        original = dict(summaries.get((location, walk_id), {}))
        original["Dataset"] = config.dataset
        original["Location"] = location
        original["WalkID"] = walk_id
        original.update({
            "TraceSalvageStatus": status,
            # Post-correction MultiSpot: one means at least one raw doublet is
            # still unresolved.  This deliberately becomes zero when all raw
            # doublets were resolved into the derived single-coordinate trace.
            "TraceSalvagedMultiSpot": int(bool(unresolved)),
            "TraceSalvagedSingleSpotComplete": int(complete),
            "TraceRawDoubletLociCount": len(raw_doublets),
            "TraceSalvagedLociCount": len(salvaged),
            "TraceUnresolvedDoubletLociCount": len(unresolved),
            "TraceSalvageEligibleLociCount": eligible_count,
            "TraceSalvagedTPs": ";".join(row["TP"] for row in salvaged),
            "TraceUnresolvedDoubletTPs": ";".join(row["TP"] for row in unresolved),
        })
        blank_own = {field: "" for field in _own.OWNERSHIP_WALK_FIELDS}
        blank_usable = {field: "" for field in _own.USABILITY_WALK_FIELDS}
        if own_walk is None:
            original.update(blank_own)
            original.update(blank_usable)
        else:
            for field in _own.OWNERSHIP_WALK_FIELDS:
                original[field] = own_walk.get(field, "")
            for field in _own.USABILITY_WALK_FIELDS:
                original[field] = own_walk.get(field, "")
        walk_rows.append(original)
        counts[status] += 1
        counts["raw_doublet_loci"] += len(raw_doublets)
        counts["salvaged_loci"] += len(salvaged)
        counts["unresolved_doublet_loci"] += len(unresolved)

    # The drift-corrected coordinate CSV is the authoritative row universe.
    # The configured chromosome order controls salvage decisions only; it must
    # never truncate acquired rows such as TP022/TP023 or invent blank filler.
    locus_by_key = {
        (row["Location"], int(row["WalkID"]), row["TP"]): row
        for row in locus_rows
    }
    walk_by_key = {
        (row["Location"], int(row["WalkID"])): row
        for row in walk_rows
    }
    root = Path(outdir)
    outputs = {
        "full_schema": root / f"{output_prefix}_trace_salvage_full_schema.csv",
        "full_schema_complete": (
            root / f"{output_prefix}_trace_salvage_full_schema_complete.csv"),
        "policy": root / f"{output_prefix}_trace_salvage_policy.json",
        "manifest": root / f"{output_prefix}_trace_salvage_manifest.json",
    }
    if diagnostics:
        outputs["audit"] = (
            root / f"{output_prefix}_trace_salvage_audit.csv")
        outputs["walk_summary"] = (
            root / f"{output_prefix}_trace_salvage_walk_summary.csv")

    location_dir = root / "locations"
    location_outputs: list[dict[str, Any]] = []
    used_location_paths: set[Path] = set()
    for item in config.locations:
        location_path = location_dir / (
            f"{_safe_location_filename(item.location)}_trace_salvage_full_schema.csv")
        if location_path in used_location_paths:
            raise TraceSalvageError(
                f"location filenames collide at {location_path.name!r}")
        used_location_paths.add(location_path)
        schema = _location_output_schema(item)
        location_outputs.append(_write_location_full_schema(
            config, item, schema, evidence, locus_by_key, walk_by_key,
            location_path))

    complete_walks = {
        key for key, row in walk_by_key.items()
        if int(row["TraceSalvagedSingleSpotComplete"]) == 1
    }
    global_ids, full_row_count, complete_row_count, _ = (
        _combine_location_outputs(
            location_outputs, outputs["full_schema"],
            outputs["full_schema_complete"], complete_walks))

    # The enriched walk summary follows the same coordinate-source universe as
    # the flat file.  Evidence-only walks remain represented in the audit and
    # manifest counts, but cannot receive a global coordinate-table ID without
    # introducing gaps.
    output_walk_rows: list[dict[str, Any]] = []
    for row in walk_rows:
        original = int(row["WalkID"])
        key = (str(row["Location"]), original)
        if key not in global_ids:
            counts["walks_without_coordinate_rows"] += 1
            continue
        row["OriginalWalkID"] = original
        row["WalkID"] = global_ids[key]
        output_walk_rows.append(row)
    output_counts = Counter()
    for row in output_walk_rows:
        output_counts[str(row["TraceSalvageStatus"])] += 1
        output_counts["raw_doublet_loci"] += int(
            row["TraceRawDoubletLociCount"])
        output_counts["salvaged_loci"] += int(row["TraceSalvagedLociCount"])
        output_counts["unresolved_doublet_loci"] += int(
            row["TraceUnresolvedDoubletLociCount"])
    walk_fields = ["Dataset", "Location", "WalkID", "OriginalWalkID"]
    walk_fields.extend(field for field in summary_fields
                       if field not in _SOURCE_ID_FIELDS
                       and field not in walk_fields)
    walk_fields.extend(field for field in DERIVED_WALK_FIELDS
                       if field not in walk_fields)

    if diagnostics:
        _atomic_csv(outputs["audit"], AUDIT_FIELDS, audit_rows)
        _atomic_csv(outputs["walk_summary"], walk_fields, output_walk_rows)
    _atomic_json(outputs["policy"], dict(policy))
    manifest: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "dataset": config.dataset,
        "evidence_schema": config.evidence_schema,
        "input_paths": input_hashes,
        "counts": {
            "walks": len(output_walk_rows),
            "decision_walks": len(walk_rows),
            "locus_rows": len(locus_rows),
            "walk_summary_rows": len(output_walk_rows),
            "full_schema_rows": full_row_count,
            "full_schema_complete_rows": complete_row_count,
            "global_walks": len(global_ids),
            "walks_without_coordinate_rows": int(
                counts["walks_without_coordinate_rows"]),
            **dict(sorted(output_counts.items())),
        },
        "decision_counts": {
            "walks": len(walk_rows),
            **dict(sorted(counts.items())),
        },
        "outputs": {},
        "location_outputs": {},
        "interpretation": policy.get("interpretation", ""),
        "center_ownership": {
            "sources": dict(sorted(ownership_sources.items())),
            "enforced": bool(ownership_walks),
            "locus_universe": list(config.chromosome_order),
            "censored_loci": sum(1 for valid in ownership_loci.values()
                                 if not valid),
            "ownership_complete_walks": sum(
                1 for row in output_walk_rows
                if str(row.get("CenterOwnershipComplete", "")).strip() == "1"),
            "usability_complete_walks": sum(
                1 for row in output_walk_rows
                if str(row.get("CenterOwnershipUsableComplete",
                               "")).strip() == "1"),
            "salvage_policy": config.salvage_policy,
            "rule": (
                "a non-usable locus is censored to unevaluable; "
                "TraceSalvagedSingleSpotComplete requires "
                "CenterOwnershipUsableComplete (or CenterOwnershipComplete "
                "for legacy data)"),
        },
    }
    for name, path in outputs.items():
        if name == "manifest":
            continue
        manifest["outputs"][name] = {"path": path.name}
    attempt_totals: Counter = Counter()
    for item in location_outputs:
        path = Path(item["path"])
        attempt_totals.update(item["attempt_counts"])
        manifest["location_outputs"][str(item["location"])] = {
            "path": str(path.relative_to(root)),
            "rows": int(item["rows"]),
            "walks": len(item["walk_ids"]),
            "timepoints": sorted(item["tps"]),
            "attempt_universe": item["attempt_universe"],
            "attempt_counts": item["attempt_counts"],
            "ledger_attempts": int(item["ledger_attempts"]),
            "ledger_has_fit_centres": bool(item["ledger_has_fit_centres"]),
            # The content hash is recorded whether or not the file is kept, so
            # dropping the intermediate loses provenance of nothing.
            "retained": bool(diagnostics),
        }
    if not diagnostics:
        # The per-Location comprehensive files are an INTERMEDIATE: the combine
        # step above has already read them, and every row is present in the two
        # comprehensive outputs. Their hashes and counts stay in the manifest.
        for item in location_outputs:
            Path(item["path"]).unlink(missing_ok=True)
        try:
            location_dir.rmdir()
        except OSError:
            pass          # a diagnostic file from an earlier run may remain
    manifest["attempt_universe"] = {
        "source": sorted({str(item["attempt_universe"])
                          for item in location_outputs}),
        "counts": dict(sorted(attempt_totals.items())),
        "audit_coordinate_fields": list(AUDIT_COORDINATE_FIELDS),
        "rule": (
            "one comprehensive row per localization attempt in the localizer's "
            "per-attempt ledger; an attempt with no usable canonical coordinate "
            "keeps every canonical coordinate blank and appears only in the "
            "audit-only UnconstrainedFit*Nm columns"),
    }
    _atomic_json(outputs["manifest"], manifest)
    manifest["outputs"]["manifest"] = {"path": outputs["manifest"].name}
    return manifest


def run_dataset(config: TraceSalvageConfig,
                outdir: os.PathLike[str] | str,
                output_prefix: str = "trace_salvage",
                diagnostics: bool = False) -> dict[str, Any]:
    """Fit the dataset LOO policy, then apply it without changing inputs."""
    policy = fit_policy(config)
    return apply_policy(config, policy, outdir, output_prefix,
                        diagnostics=diagnostics)


def _load_policy(path: os.PathLike[str] | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TraceSalvageError("policy JSON must contain an object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser(
        "fit-policy", help="fit leave-one-location-out link calibration")
    fit.add_argument("--config", required=True)
    fit.add_argument("--output-policy", required=True)
    apply = subparsers.add_parser(
        "apply", help="apply a frozen policy to one or more locations")
    apply.add_argument("--config", required=True)
    apply.add_argument("--policy", required=True)
    apply.add_argument("--outdir", required=True)
    apply.add_argument("--output-prefix", default="trace_salvage")
    apply.add_argument(
        "--diagnostics", action="store_true",
        help="Additionally write the per-locus audit CSV, the reduced walk "
             "summary and the per-Location comprehensive files. Off by "
             "default: every field they hold is already on every row of the "
             "comprehensive CSV, and shipping them as separate files created "
             "two representations of the same facts that could disagree. This "
             "changes only WHICH files appear, never a coordinate, a WalkID "
             "mapping, walk membership or a salvage decision.")
    run = subparsers.add_parser("run", help="fit and apply in one dataset run")
    run.add_argument("--config", required=True)
    run.add_argument("--outdir", required=True)
    run.add_argument("--output-prefix", default="trace_salvage")
    run.add_argument(
        "--diagnostics", action="store_true",
        help="Additionally write the per-locus audit CSV, the reduced walk "
             "summary and the per-Location comprehensive files. Off by "
             "default: every field they hold is already on every row of the "
             "comprehensive CSV, and shipping them as separate files created "
             "two representations of the same facts that could disagree. This "
             "changes only WHICH files appear, never a coordinate, a WalkID "
             "mapping, walk membership or a salvage decision.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "fit-policy":
        policy = fit_policy(config)
        _atomic_json(Path(args.output_policy), policy)
        print(f"[trace_salvage] wrote policy {args.output_policy}")
        return 0
    if args.command == "apply":
        manifest = apply_policy(
            config, _load_policy(args.policy), args.outdir, args.output_prefix,
            diagnostics=args.diagnostics)
    elif args.command == "run":
        manifest = run_dataset(config, args.outdir, args.output_prefix,
                               diagnostics=args.diagnostics)
    else:
        raise AssertionError(f"unhandled command {args.command!r}")
    counts = manifest["counts"]
    print(f"[trace_salvage] dataset={config.dataset} walks={counts['walks']} "
          f"salvaged_loci={counts.get('salvaged_loci', 0)} "
          f"unresolved_doublet_loci={counts.get('unresolved_doublet_loci', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
