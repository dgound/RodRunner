#!/usr/bin/env python3
"""Small, strict interchange format for reconstructed chromosome topology.

Coordinates in this module are always ordered ``X, Y, Z`` and measured in nm.
Theta CSVs contain three open paths with exactly shared endpoint coordinates.
Link CSVs contain unique polygon vertices; their loader appends vertex zero to
make the closure explicit in memory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "chromosome-topology-v1"
COORDINATE_FRAME = "XYZ_nm"

THETA_SUMMARY_FIELDS = (
    "Dataset", "Location", "WalkID", "GraphID", "ThetaStatus",
    "FailureReason", "NumDoubledLoci", "AssignmentCost",
    "AssignmentMargin", "YamadaPolynomial", "YamadaNormalized",
    "ProjectionStable", "PerturbationStable", "ForkModelStable",
)
THETA_POINT_FIELDS = (
    "Dataset", "Location", "WalkID", "GraphID", "StrandID",
    "PointOrder", "PointRole", "TP", "TPIndex", "GenomicBP",
    "CoordinateSource", "ComponentID", "XNm", "YNm", "ZNm",
    "XSEnm", "YSEnm", "ZSEnm", "Observed", "InferredFork",
)
LINK_SUMMARY_FIELDS = (
    "Dataset", "Location", "WalkID", "LinkID", "LinkStatus",
    "FailureReason", "NumLoci", "AssignmentCost", "AssignmentMargin",
    "LinkingNumber", "MinInterLoopDistanceNm", "AssignmentStable",
    "PerturbationStable", "DisjointStable",
)
LINK_POINT_FIELDS = (
    "Dataset", "Location", "WalkID", "LinkID", "LoopID",
    "PointOrder", "PointRole", "TP", "TPIndex", "GenomicBP",
    "CoordinateSource", "ComponentID", "XNm", "YNm", "ZNm",
    "XSEnm", "YSEnm", "ZSEnm", "Closed",
)

OUTPUT_NAMES = {
    "theta_summary": "theta_graph_summary.csv",
    "theta_points": "theta_strand_points.csv",
    "link_summary": "link_summary.csv",
    "link_points": "link_loop_points.csv",
    "manifest": "topology_manifest.json",
}

_THETA_STRANDS = ("unreplicated", "sister_A", "sister_B")
_LINK_LOOPS = ("sister_A", "sister_B")
_COORDS = ("XNm", "YNm", "ZNm")
_UNCERTAINTIES = ("XSEnm", "YSEnm", "ZSEnm")


class TopologyIOError(ValueError):
    """The topology table is malformed or internally inconsistent."""


def _text(value: Any, field: str) -> str:
    result = "" if value is None else str(value)
    if not result:
        raise TopologyIOError(f"{field} must not be blank")
    return result


def _integer(value: Any, field: str, *, optional: bool = False) -> int | None:
    if value is None or value == "":
        if optional:
            return None
        raise TopologyIOError(f"{field} must not be blank")
    try:
        number = int(str(value))
    except (TypeError, ValueError) as error:
        raise TopologyIOError(f"{field} must be an integer, got {value!r}") from error
    if str(value).strip() not in {str(number), f"+{number}"}:
        raise TopologyIOError(f"{field} must be an integer, got {value!r}")
    return number


def _finite(value: Any, field: str, *, optional: bool = False) -> float | None:
    if value is None or value == "":
        if optional:
            return None
        raise TopologyIOError(f"{field} must not be blank")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TopologyIOError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise TopologyIOError(f"{field} must be finite, got {value!r}")
    return number


def _bit(value: Any, field: str, *, optional: bool = False) -> int | None:
    if value is None or value == "":
        if optional:
            return None
        raise TopologyIOError(f"{field} must not be blank")
    if value in (0, "0", False):
        return 0
    if value in (1, "1", True):
        return 1
    raise TopologyIOError(f"{field} must be 0 or 1, got {value!r}")


def _exact_rows(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[str], table: str,
) -> list[dict[str, Any]]:
    expected = set(fields)
    result = []
    for number, source in enumerate(rows, 1):
        actual = set(source)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise TopologyIOError(
                f"{table} row {number} has wrong schema; missing={missing}, extra={extra}")
        result.append({field: source[field] for field in fields})
    return result


def _validate_identity(row: Mapping[str, Any], id_field: str) -> None:
    _text(row["Dataset"], "Dataset")
    _text(row["Location"], "Location")
    walk = _integer(row["WalkID"], "WalkID")
    if walk is not None and walk < 0:
        raise TopologyIOError("WalkID must be non-negative")
    _text(row[id_field], id_field)


def _validate_summary(
    rows: list[dict[str, Any]], *, id_field: str, status_field: str,
    count_field: str, optional_numbers: Sequence[str], optional_bits: Sequence[str],
) -> None:
    seen: set[str] = set()
    for row in rows:
        _validate_identity(row, id_field)
        identifier = str(row[id_field])
        if identifier in seen:
            raise TopologyIOError(f"duplicate {id_field} {identifier!r}")
        seen.add(identifier)
        _text(row[status_field], status_field)
        count = _integer(row[count_field], count_field)
        if count is not None and count < 0:
            raise TopologyIOError(f"{count_field} must be non-negative")
        for field in optional_numbers:
            _finite(row[field], field, optional=True)
        for field in optional_bits:
            _bit(row[field], field, optional=True)


def _point_xyz(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([_finite(row[field], field) for field in _COORDS], dtype=float)


def _validate_common_point(row: Mapping[str, Any], id_field: str) -> None:
    _validate_identity(row, id_field)
    order = _integer(row["PointOrder"], "PointOrder")
    if order is not None and order < 0:
        raise TopologyIOError("PointOrder must be non-negative")
    _point_xyz(row)
    for field in _UNCERTAINTIES:
        # The current two-component fitter does not report coordinate standard
        # errors.  Keep those values explicitly missing rather than inventing
        # zero uncertainty; future fitters may populate them.
        uncertainty = _finite(row[field], field, optional=True)
        if uncertainty is not None and uncertainty < 0:
            raise TopologyIOError(f"{field} must be non-negative")
    _finite(row["GenomicBP"], "GenomicBP", optional=True)


def _consistent_identity(rows: Sequence[Mapping[str, Any]], id_field: str) -> None:
    identities = {
        (str(row["Dataset"]), str(row["Location"]), str(row["WalkID"]),
         str(row[id_field]))
        for row in rows
    }
    if len(identities) != 1:
        raise TopologyIOError(f"{id_field} rows disagree on dataset/location/walk identity")


def _validate_theta_points(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        _validate_common_point(row, "GraphID")
        strand = str(row["StrandID"])
        if strand not in _THETA_STRANDS:
            raise TopologyIOError(f"invalid theta StrandID {strand!r}")
        order = int(str(row["PointOrder"]))
        key = (str(row["GraphID"]), strand, order)
        if key in keys:
            raise TopologyIOError(f"duplicate theta point key {key!r}")
        keys.add(key)
        _text(row["PointRole"], "PointRole")
        _text(row["CoordinateSource"], "CoordinateSource")
        _integer(row["TPIndex"], "TPIndex", optional=True)
        _bit(row["Observed"], "Observed")
        _bit(row["InferredFork"], "InferredFork")
        grouped[str(row["GraphID"])].append(row)

    for graph_id, graph_rows in grouped.items():
        _consistent_identity(graph_rows, "GraphID")
        strands = {str(row["StrandID"]) for row in graph_rows}
        if strands != set(_THETA_STRANDS):
            raise TopologyIOError(
                f"GraphID {graph_id!r} must contain exactly {_THETA_STRANDS}")
        ordered: dict[str, list[dict[str, Any]]] = {}
        for strand in _THETA_STRANDS:
            part = sorted(
                (row for row in graph_rows if row["StrandID"] == strand),
                key=lambda row: int(str(row["PointOrder"])),
            )
            if len(part) < 3:
                raise TopologyIOError(f"GraphID {graph_id!r} strand {strand} has fewer than 3 points")
            orders = [int(str(row["PointOrder"])) for row in part]
            if orders != list(range(len(part))):
                raise TopologyIOError(
                    f"GraphID {graph_id!r} strand {strand} PointOrder must be contiguous from 0")
            if part[0]["PointRole"] != "fork_1" or part[-1]["PointRole"] != "fork_2":
                raise TopologyIOError(
                    f"GraphID {graph_id!r} strand {strand} must run fork_1 to fork_2")
            for index, row in enumerate(part):
                endpoint = index in (0, len(part) - 1)
                if _bit(row["InferredFork"], "InferredFork") != int(endpoint):
                    raise TopologyIOError("InferredFork must mark exactly the two theta endpoints")
                if _bit(row["Observed"], "Observed") != int(not endpoint):
                    raise TopologyIOError("Observed must be 0 at forks and 1 at measured theta points")
            ordered[strand] = part

        starts = [_point_xyz(ordered[strand][0]) for strand in _THETA_STRANDS]
        ends = [_point_xyz(ordered[strand][-1]) for strand in _THETA_STRANDS]
        if not all(np.array_equal(starts[0], point) for point in starts[1:]):
            raise TopologyIOError(f"GraphID {graph_id!r} does not have an exactly shared first fork")
        if not all(np.array_equal(ends[0], point) for point in ends[1:]):
            raise TopologyIOError(f"GraphID {graph_id!r} does not have an exactly shared second fork")
        sister_tp = [
            [(row["TP"], str(row["TPIndex"])) for row in ordered[strand][1:-1]]
            for strand in ("sister_A", "sister_B")
        ]
        if sister_tp[0] != sister_tp[1]:
            raise TopologyIOError(f"GraphID {graph_id!r} sister paths use different locus orders")


def _validate_link_points(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        _validate_common_point(row, "LinkID")
        loop = str(row["LoopID"])
        if loop not in _LINK_LOOPS:
            raise TopologyIOError(f"invalid link LoopID {loop!r}")
        order = int(str(row["PointOrder"]))
        key = (str(row["LinkID"]), loop, order)
        if key in keys:
            raise TopologyIOError(f"duplicate link point key {key!r}")
        keys.add(key)
        _text(row["PointRole"], "PointRole")
        _text(row["TP"], "TP")
        _integer(row["TPIndex"], "TPIndex")
        _text(row["CoordinateSource"], "CoordinateSource")
        _bit(row["Closed"], "Closed")
        grouped[str(row["LinkID"])].append(row)

    for link_id, link_rows in grouped.items():
        _consistent_identity(link_rows, "LinkID")
        loops = {str(row["LoopID"]) for row in link_rows}
        if loops != set(_LINK_LOOPS):
            raise TopologyIOError(f"LinkID {link_id!r} must contain exactly {_LINK_LOOPS}")
        ordered: dict[str, list[dict[str, Any]]] = {}
        for loop in _LINK_LOOPS:
            part = sorted(
                (row for row in link_rows if row["LoopID"] == loop),
                key=lambda row: int(str(row["PointOrder"])),
            )
            if len(part) < 3:
                raise TopologyIOError(f"LinkID {link_id!r} loop {loop} has fewer than 3 vertices")
            orders = [int(str(row["PointOrder"])) for row in part]
            if orders != list(range(len(part))):
                raise TopologyIOError(
                    f"LinkID {link_id!r} loop {loop} PointOrder must be contiguous from 0")
            if any(_bit(row["Closed"], "Closed") != 1 for row in part):
                raise TopologyIOError(f"LinkID {link_id!r} loop {loop} must have Closed=1")
            if np.array_equal(_point_xyz(part[0]), _point_xyz(part[-1])):
                raise TopologyIOError(
                    f"LinkID {link_id!r} loop {loop} duplicates its closing vertex in the CSV")
            ordered[loop] = part
        locus_orders = [
            [(row["TP"], str(row["TPIndex"])) for row in ordered[loop]]
            for loop in _LINK_LOOPS
        ]
        if locus_orders[0] != locus_orders[1]:
            raise TopologyIOError(f"LinkID {link_id!r} sister loops use different locus orders")


def _normalise(
    theta_summaries: Iterable[Mapping[str, Any]],
    theta_points: Iterable[Mapping[str, Any]],
    link_summaries: Iterable[Mapping[str, Any]],
    link_points: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], ...]:
    theta_summary = _exact_rows(theta_summaries, THETA_SUMMARY_FIELDS, "theta summary")
    theta_point = _exact_rows(theta_points, THETA_POINT_FIELDS, "theta points")
    link_summary = _exact_rows(link_summaries, LINK_SUMMARY_FIELDS, "link summary")
    link_point = _exact_rows(link_points, LINK_POINT_FIELDS, "link points")
    _validate_summary(
        theta_summary, id_field="GraphID", status_field="ThetaStatus",
        count_field="NumDoubledLoci",
        optional_numbers=("AssignmentCost", "AssignmentMargin"),
        optional_bits=("ProjectionStable", "PerturbationStable", "ForkModelStable"),
    )
    _validate_summary(
        link_summary, id_field="LinkID", status_field="LinkStatus",
        count_field="NumLoci",
        optional_numbers=("AssignmentCost", "AssignmentMargin", "LinkingNumber",
                          "MinInterLoopDistanceNm"),
        optional_bits=("AssignmentStable", "PerturbationStable", "DisjointStable"),
    )
    _validate_theta_points(theta_point)
    _validate_link_points(link_point)
    theta_ids = {str(row["GraphID"]) for row in theta_summary}
    point_theta_ids = {str(row["GraphID"]) for row in theta_point}
    if not point_theta_ids <= theta_ids:
        raise TopologyIOError(
            f"theta points lack summary rows for {sorted(point_theta_ids - theta_ids)}")
    theta_reconstructable = {
        str(row["GraphID"]) for row in theta_summary
        if row["ThetaStatus"] in {"nominal_theta_graph", "theta_unstable"}
    }
    if not theta_reconstructable <= point_theta_ids:
        raise TopologyIOError(
            "reconstructable theta summaries lack point rows for "
            f"{sorted(theta_reconstructable - point_theta_ids)}")
    link_ids = {str(row["LinkID"]) for row in link_summary}
    point_link_ids = {str(row["LinkID"]) for row in link_point}
    if not point_link_ids <= link_ids:
        raise TopologyIOError(
            f"link points lack summary rows for {sorted(point_link_ids - link_ids)}")
    link_reconstructable = {
        str(row["LinkID"]) for row in link_summary
        if row["LinkStatus"] in {
            "nominal_linked_polygon", "nominal_unlinked_polygon", "link_unstable"}
    }
    if not link_reconstructable <= point_link_ids:
        raise TopologyIOError(
            "reconstructable link summaries lack point rows for "
            f"{sorted(link_reconstructable - point_link_ids)}")

    theta_summary.sort(key=lambda row: str(row["GraphID"]))
    theta_point.sort(key=lambda row: (
        str(row["GraphID"]), _THETA_STRANDS.index(str(row["StrandID"])),
        int(str(row["PointOrder"]))))
    link_summary.sort(key=lambda row: str(row["LinkID"]))
    link_point.sort(key=lambda row: (
        str(row["LinkID"]), _LINK_LOOPS.index(str(row["LoopID"])),
        int(str(row["PointOrder"]))))
    return theta_summary, theta_point, link_summary, link_point


def _serialise(row: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field, value in row.items():
        if value is None or value == "":
            result[field] = ""
        elif isinstance(value, bool):
            result[field] = "1" if value else "0"
        elif isinstance(value, (float, np.floating)):
            number = float(value)
            if not math.isfinite(number):
                raise TopologyIOError(f"{field} must be finite")
            result[field] = format(number, ".17g")
        else:
            result[field] = str(value)
    return result


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            writer.writerows(_serialise(row) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_topology_outputs(
    output_dir: str | os.PathLike[str], *,
    theta_summaries: Iterable[Mapping[str, Any]],
    theta_points: Iterable[Mapping[str, Any]],
    link_summaries: Iterable[Mapping[str, Any]],
    link_points: Iterable[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate and atomically write the five public topology deliverables.

    The manifest is written last and contains hashes of all four CSVs. Callers
    may place input hashes and policy/software metadata inside ``provenance``.
    """
    if not isinstance(provenance, Mapping):
        raise TopologyIOError("provenance must be a mapping")
    # Validate JSON serialisability and reject NaN/Inf before touching outputs.
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TopologyIOError(f"provenance is not finite JSON: {error}") from error
    theta_summary, theta_point, link_summary, link_point = _normalise(
        theta_summaries, theta_points, link_summaries, link_points)
    root = Path(output_dir)
    paths = {key: root / name for key, name in OUTPUT_NAMES.items()}
    _atomic_csv(paths["theta_summary"], THETA_SUMMARY_FIELDS, theta_summary)
    _atomic_csv(paths["theta_points"], THETA_POINT_FIELDS, theta_point)
    _atomic_csv(paths["link_summary"], LINK_SUMMARY_FIELDS, link_summary)
    _atomic_csv(paths["link_points"], LINK_POINT_FIELDS, link_point)
    csv_keys = ("theta_summary", "theta_points", "link_summary", "link_points")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": COORDINATE_FRAME,
        "files": {
            key: {
                "name": paths[key].name,
                "rows": len({
                    "theta_summary": theta_summary,
                    "theta_points": theta_point,
                    "link_summary": link_summary,
                    "link_points": link_point,
                }[key]),
                "sha256": _sha256(paths[key]),
            }
            for key in csv_keys
        },
        "provenance": dict(provenance),
    }
    _atomic_json(paths["manifest"], manifest)
    return paths


def _read_strict(path: str | os.PathLike[str], fields: Sequence[str], table: str) -> list[dict[str, str]]:
    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise TopologyIOError(
                    f"{source}: {table} header must be exactly {list(fields)!r}; "
                    f"got {reader.fieldnames!r}")
            return [dict(row) for row in reader]
    except OSError as error:
        raise TopologyIOError(f"cannot read {source}: {error}") from error


def load_theta_yamada(
    points_csv: str | os.PathLike[str], graph_id: str,
) -> list[np.ndarray]:
    """Return ``[unreplicated, sister_A, sister_B]`` as Yamada-ready XYZ arrays."""
    rows = _read_strict(points_csv, THETA_POINT_FIELDS, "theta points")
    _validate_theta_points(rows)
    selected = [row for row in rows if row["GraphID"] == str(graph_id)]
    if not selected:
        raise TopologyIOError(f"GraphID {graph_id!r} is absent from {points_csv}")
    result = []
    for strand in _THETA_STRANDS:
        part = sorted(
            (row for row in selected if row["StrandID"] == strand),
            key=lambda row: int(row["PointOrder"]),
        )
        result.append(np.vstack([_point_xyz(row) for row in part]))
    return result


def load_link_loops(
    points_csv: str | os.PathLike[str], link_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two explicitly closed XYZ loops, appending vertex zero once."""
    rows = _read_strict(points_csv, LINK_POINT_FIELDS, "link points")
    _validate_link_points(rows)
    selected = [row for row in rows if row["LinkID"] == str(link_id)]
    if not selected:
        raise TopologyIOError(f"LinkID {link_id!r} is absent from {points_csv}")
    result = []
    for loop in _LINK_LOOPS:
        part = sorted(
            (row for row in selected if row["LoopID"] == loop),
            key=lambda row: int(row["PointOrder"]),
        )
        vertices = np.vstack([_point_xyz(row) for row in part])
        result.append(np.vstack([vertices, vertices[0]]))
    return result[0], result[1]
