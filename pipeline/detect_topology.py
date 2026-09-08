#!/usr/bin/env python3
"""Reconstruct theta-graph and two-loop geometry from replication calls.

This is deliberately a downstream topology stage.  It does not reclassify
spots or interpolate missing loci.  Input two-component centres are the raw
target-channel pixel centres from replication QC; they are put into the same
drift-corrected XYZ-nm frame as the one-component trajectory before tracing.

Single Python orchestrator.  Every numeric step (theta pattern, sister
assignment, strand validation, Yamada polynomial, linking number) runs in the
co-located C library ``topology_engine.so``::

    gcc -O3 -fopenmp -shared -fPIC -o topology_engine.so topology_engine.c -lm

Input CSVs are streamed in row chunks (``--chunk-rows``); only the columns
needed are retained and only for walks that can reach a topology calculation,
so memory is bounded by the candidate count rather than by the evidence file.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter, namedtuple
from pathlib import Path

ENGINE_SO = Path(__file__).with_name("topology_engine.so")
GEOMETRY_TOL_NM = 1e-6
LINK_MIN_VALID_PROJECTIONS = 16
LINK_MIN_PROJECTION_STABILITY = 0.95
LINK_PROJECTION_SEED = 0
# numpy.random.default_rng(0).uniform(0, 2*pi): keeps the projection sweep
# identical to earlier runs without importing numpy.
LINK_PROJECTION_PHASE = 4.002148315014479
DEFAULT_MAX_CROSSINGS = 16
DEFAULT_CHUNK_ROWS = 100_000

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
THETA_STRANDS = ("unreplicated", "sister_A", "sister_B")
LINK_LOOPS = ("sister_A", "sister_B")

PATTERN_STATUSES = (
    "not_theta_ori_not_doubled", "not_theta_one_sided",
    "fully_doubled_not_theta", "theta_pattern_supported",
    "theta_pattern_discordant",
)
TRACE_STATUSES = (
    "theta_traceable", "theta_geometry_incomplete", "theta_not_traceable")
THETA_CANDIDATE_STATUSES = {"theta_pattern_supported", "theta_pattern_discordant"}
LINK_CANDIDATE_STATUS = "fully_doubled_not_theta"

THETA_VALIDATION_MESSAGES = {
    1: "a theta graph requires exactly 3 strands",
    2: "theta strand {i} needs at least two points",
    3: "theta fork vertices must be distinct",
    4: "theta strand {i} does not share the exact ordered fork endpoints",
    5: "theta strand {i} contains a zero-length segment",
    6: "theta strand {i} self-intersects or touches",
    7: "theta strands overlap beyond fork 1",
    8: "theta strands overlap beyond fork 2",
    9: "theta strands intersect or touch away from a fork",
}
LOOP_VALIDATION_MESSAGES = {
    1: "loop {n} needs three distinct loop vertices",
    2: "loop {n} contains a zero/near-zero segment",
    3: "loop {n} self-intersects or touches within 1e-06 nm",
}

Layout = namedtuple("Layout", "order ori ter index")
Config = namedtuple(
    "Config",
    "layout xy_pixel_nm z_step_nm genomic_bp power_lcb_min exploratory_q family_id")
Pattern = namedtuple(
    "Pattern", "status trace_status replicated_path unreplicated_path")
Evidence = namedtuple("Evidence", "source component_1 component_2")
Coordinate = namedtuple("Coordinate", "xyz_nm raw_xyz_px drift_zyx_px xyz_se_nm")
Yamada = namedtuple("Yamada", "raw_terms normalized_terms crossings vertices seed")
Linking = namedtuple(
    "Linking",
    "linking_number projections_valid projection_stability stable min_distance_nm")


class TopologyDetectionError(ValueError):
    """Input data cannot support the requested topology analysis."""


class TopologyValidationError(ValueError):
    """Coordinates do not define the requested topology."""


class CrossingsExceeded(TopologyValidationError):
    """Every generic projection needs more crossings than allowed."""


class Assignment(namedtuple(
        "Assignment",
        "loci sister_a sister_b state_path total_cost runner_up_cost")):
    @property
    def cost_margin(self):
        if self.runner_up_cost is None:
            return None
        return self.runner_up_cost - self.total_cost


# ---------------------------------------------------------------------------
# C engine binding
# ---------------------------------------------------------------------------

class Engine:
    """ctypes binding to ``topology_engine.so``."""

    def __init__(self, path=ENGINE_SO):
        self.path = Path(path)
        if not self.path.is_file():
            raise TopologyDetectionError(
                f"{self.path} is missing; build it with: gcc -O3 -fopenmp "
                "-shared -fPIC -o topology_engine.so topology_engine.c -lm")
        lib = ctypes.CDLL(str(self.path))
        D = ctypes.POINTER(ctypes.c_double)
        I = ctypes.POINTER(ctypes.c_int)
        c_int, c_uint, c_double = ctypes.c_int, ctypes.c_uint, ctypes.c_double
        lib.compute_yamada.argtypes = [
            D, I, c_int, c_double, c_int, ctypes.c_ulonglong, c_int, I, I, I, I]
        lib.validate_theta_strands.argtypes = [D, I, c_int, c_double, I]
        lib.assign_sisters.argtypes = [D, D, c_int, c_int, c_int, I, D, D, D, D]
        lib.theta_pattern.argtypes = [c_int, c_int, c_int, c_uint, c_uint] + [I] * 8
        lib.compute_linking.argtypes = [
            D, c_int, D, c_int, c_int, c_double, c_double, I, I, I, I, D]
        for name in ("compute_yamada", "validate_theta_strands", "assign_sisters",
                     "theta_pattern", "compute_linking", "get_poly_size",
                     "get_offset", "get_link_nongeneric_sentinel", "get_max_loci",
                     "has_openmp", "get_memo_budget_mb"):
            getattr(lib, name).restype = c_int
        lib.set_num_threads.argtypes = [c_int]
        lib.set_num_threads.restype = None
        self.lib = lib
        self.poly_size = lib.get_poly_size()
        self.offset = lib.get_offset()
        self.max_loci = lib.get_max_loci()
        self.openmp = bool(lib.has_openmp())
        self.memo_budget_mb = lib.get_memo_budget_mb()

    def set_threads(self, count):
        self.lib.set_num_threads(int(count))

    @staticmethod
    def _flat(points):
        values = [float(v) for point in points for v in point]
        return (ctypes.c_double * len(values))(*values)

    def _strands(self, strands):
        lens = (ctypes.c_int * len(strands))(*[len(s) for s in strands])
        return self._flat([p for s in strands for p in s]), lens

    def theta_pattern(self, layout, doubled_mask, evaluable_mask):
        n = len(layout.order)
        status, trace = ctypes.c_int(), ctypes.c_int()
        n_rep, n_unrep = ctypes.c_int(), ctypes.c_int()
        n_disc, n_uneval = ctypes.c_int(), ctypes.c_int()
        replicated = (ctypes.c_int * n)()
        unreplicated = (ctypes.c_int * n)()
        rc = self.lib.theta_pattern(
            n, layout.index[layout.ori], layout.index[layout.ter],
            doubled_mask, evaluable_mask, ctypes.byref(status), ctypes.byref(trace),
            replicated, ctypes.byref(n_rep), unreplicated, ctypes.byref(n_unrep),
            ctypes.byref(n_disc), ctypes.byref(n_uneval))
        if rc:
            raise TopologyDetectionError(
                "theta pattern: every doubled locus must also be evaluable")
        return Pattern(
            PATTERN_STATUSES[status.value], TRACE_STATUSES[trace.value],
            tuple(layout.order[i] for i in replicated[:n_rep.value]),
            tuple(layout.order[i] for i in unreplicated[:n_unrep.value]))

    def assign(self, loci, pairs, *, circular, dims=(0, 1, 2)):
        n = len(pairs)
        minimum = 3 if circular else 2
        if n < minimum:
            raise TopologyDetectionError(
                f"{'circular' if circular else 'open'} sister assignment "
                f"requires at least {minimum} loci")
        if n > self.max_loci:
            raise TopologyDetectionError(
                f"sister assignment supports at most {self.max_loci} loci")
        comp_1 = self._flat([pair[0] for pair in pairs])
        comp_2 = self._flat([pair[1] for pair in pairs])
        path = (ctypes.c_int * n)()
        total, runner_up = ctypes.c_double(), ctypes.c_double()
        sister_a = (ctypes.c_double * (3 * n))()
        sister_b = (ctypes.c_double * (3 * n))()
        rc = self.lib.assign_sisters(
            comp_1, comp_2, n, int(circular), sum(1 << d for d in dims), path,
            ctypes.byref(total), ctypes.byref(runner_up), sister_a, sister_b)
        if rc:
            raise TopologyDetectionError("no valid sister assignment")
        return Assignment(
            tuple(loci),
            tuple(tuple(sister_a[3 * i:3 * i + 3]) for i in range(n)),
            tuple(tuple(sister_b[3 * i:3 * i + 3]) for i in range(n)),
            tuple(path), total.value,
            None if math.isnan(runner_up.value) else runner_up.value)

    def validate_theta(self, strands):
        points, lens = self._strands(strands)
        which = ctypes.c_int(-1)
        rc = self.lib.validate_theta_strands(
            points, lens, len(strands), GEOMETRY_TOL_NM, ctypes.byref(which))
        if rc:
            raise TopologyValidationError(
                THETA_VALIDATION_MESSAGES[rc].format(i=which.value))

    def yamada(self, strands, *, tries, seed, max_crossings):
        points, lens = self._strands(strands)
        raw = (ctypes.c_int * self.poly_size)()
        norm = (ctypes.c_int * self.poly_size)()
        crossings, vertices = ctypes.c_int(), ctypes.c_int()
        rc = self.lib.compute_yamada(
            points, lens, len(strands), GEOMETRY_TOL_NM, int(tries), int(seed),
            int(max_crossings), raw, norm, ctypes.byref(crossings),
            ctypes.byref(vertices))
        if rc == -1:
            raise TopologyValidationError(
                "no generic projection found for the theta graph")
        if rc == -2:
            raise CrossingsExceeded(
                f"every generic projection needs more than {max_crossings} crossings")
        if rc < 0:
            raise TopologyValidationError("theta graph exceeds engine size limits")

        def terms(coefficients):
            return tuple(sorted(
                ((i - self.offset, coefficients[i])
                 for i in range(self.poly_size) if coefficients[i]),
                reverse=True))

        return Yamada(terms(raw), terms(norm), rc, vertices.value, int(seed))

    def linking(self, loop_a, loop_b, *, n_projections):
        a, b = self._flat(loop_a), self._flat(loop_b)
        values = (ctypes.c_int * n_projections)()
        number, valid, modal = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        distance = ctypes.c_double()
        rc = self.lib.compute_linking(
            a, len(loop_a), b, len(loop_b), int(n_projections),
            LINK_PROJECTION_PHASE, GEOMETRY_TOL_NM, values, ctypes.byref(number),
            ctypes.byref(valid), ctypes.byref(modal), ctypes.byref(distance))
        if rc in (-1, -2, -3):
            raise TopologyValidationError(
                LOOP_VALIDATION_MESSAGES[-rc].format(n="A"))
        if rc in (-11, -12, -13):
            raise TopologyValidationError(
                LOOP_VALIDATION_MESSAGES[-rc - 10].format(n="B"))
        if rc == -20:
            raise TopologyValidationError(
                "classical linking number requires mutually disjoint loops; "
                f"minimum distance is {distance.value:g} nm")
        if rc == -21:
            raise TopologyValidationError(
                "no generic projection found for the two loops")
        if rc:
            raise TopologyValidationError("loops exceed engine size limits")
        stability = modal.value / valid.value
        stable = (valid.value >= LINK_MIN_VALID_PROJECTIONS and
                  stability >= LINK_MIN_PROJECTION_STABILITY)
        return Linking(number.value, valid.value, stability, stable, distance.value)


def _format_laurent(terms, variable="A"):
    pieces = []
    for exponent, coefficient in terms:
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        if exponent == 0:
            body = str(magnitude)
        else:
            power = variable if exponent == 1 else f"{variable}^{exponent}"
            body = power if magnitude == 1 else f"{magnitude}*{power}"
        pieces.append((sign, body))
    if not pieces:
        return "0"
    first_sign, first_body = pieces[0]
    text = ("-" if first_sign == "-" else "") + first_body
    return text + "".join(f" {sign} {body}" for sign, body in pieces[1:])


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TopologyDetectionError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise TopologyDetectionError(f"{label} must be finite, got {value!r}")
    return result


def _positive(value, label):
    result = _finite(value, label)
    if result <= 0:
        raise TopologyDetectionError(f"{label} must be positive")
    return result


def _optional_finite(value, label):
    if value is None or not str(value).strip():
        return None
    return _finite(value, label)


def _walk_id(value, label):
    text = str(value).strip()
    try:
        result = int(text)
    except ValueError as exc:
        raise TopologyDetectionError(f"{label} must be an integer, got {value!r}") from exc
    if text not in {str(result), f"+{result}"} or result < 0:
        raise TopologyDetectionError(f"{label} must be a non-negative integer")
    return result


def _bit(value, label):
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    raise TopologyDetectionError(f"{label} must be 0 or 1, got {value!r}")


def _nonblank(value, label):
    result = str(value).strip()
    if not result:
        raise TopologyDetectionError(f"{label} must not be blank")
    return result


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TopologyDetectionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path):
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
    if len(order) < 3:
        raise TopologyDetectionError("chromosome_order must contain at least three loci")
    if len(order) > 32:
        raise TopologyDetectionError("chromosome_order supports at most 32 loci")
    if any(not isinstance(tp, str) or not tp.strip() for tp in order):
        raise TopologyDetectionError("chromosome_order must contain non-empty strings")
    if len(set(order)) != len(order):
        raise TopologyDetectionError("chromosome_order contains duplicate loci")
    if ori not in order:
        raise TopologyDetectionError("ori is outside chromosome_order")
    if ter not in order:
        raise TopologyDetectionError("ter is outside chromosome_order")
    if ori == ter:
        raise TopologyDetectionError("ori and ter must be different loci")
    layout = Layout(order, ori, ter, {tp: i for i, tp in enumerate(order)})
    bp_payload = payload.get("genomic_bp", {})
    if not isinstance(bp_payload, dict):
        raise TopologyDetectionError("genomic_bp must be a TP-to-number object")
    outside = set(bp_payload) - set(order)
    if outside:
        raise TopologyDetectionError(
            f"genomic_bp contains loci outside chromosome_order: {sorted(outside)}")
    genomic_bp = {str(tp): _finite(value, f"genomic_bp.{tp}")
                  for tp, value in bp_payload.items()}
    if not 0.0 <= power_lcb_min <= 1.0:
        raise TopologyDetectionError("power_lcb_min must lie in [0, 1]")
    if not 0.0 < exploratory_q < 1.0:
        raise TopologyDetectionError("exploratory_bh_q must lie in (0, 1)")
    return Config(layout, xy, z, genomic_bp, power_lcb_min, exploratory_q,
                  str(payload.get("family_id", "")).strip())


def _geometry_id(config):
    payload = {"xy_pixel_nm": config.xy_pixel_nm, "z_step_nm": config.z_step_nm}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chunked CSV streaming
# ---------------------------------------------------------------------------

def _header(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle), None)
    except OSError as exc:
        raise TopologyDetectionError(f"cannot read {path}: {exc}") from exc
    if not fields:
        raise TopologyDetectionError(f"{path}: missing CSV header")
    if len(fields) != len(set(fields)):
        raise TopologyDetectionError(f"{path}: duplicate CSV columns")
    return fields


def _iter_chunks(path, columns, chunk_rows):
    """Yield lists of ``(line_number, values)`` holding only ``columns``.

    A file with at most ``chunk_rows`` data rows arrives as a single chunk;
    larger files are streamed so that no more than ``chunk_rows`` compact
    tuples are resident at once.  Columns absent from the header read as "".
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            position = {name: i for i, name in enumerate(header)}
            picks = tuple(position.get(name, -1) for name in columns)
            width = len(header)
            chunk = []
            for number, row in enumerate(reader, 2):
                if not row:
                    continue
                if len(row) != width:
                    raise TopologyDetectionError(
                        f"{path}:{number}: expected {width} fields, got {len(row)}")
                chunk.append((number, tuple("" if i < 0 else row[i] for i in picks)))
                if len(chunk) >= chunk_rows:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    except OSError as exc:
        raise TopologyDetectionError(f"cannot read {path}: {exc}") from exc


def _tp_column(fields, path):
    for name in ("TP", "tp"):
        if name in fields:
            return name
    raise TopologyDetectionError(f"{path}: need a TP or tp column")


def _locus_key(path, number, dataset, location, walk, original, tp):
    """Location-local walk identity.

    Comprehensive trace-salvage files renumber ``WalkID`` globally and retain
    the location-local identifier in ``OriginalWalkID``.  Replication evidence
    is location-local, so topology joins use the latter when present.
    """
    walk_field, walk_text = (("OriginalWalkID", original) if original.strip()
                             else ("WalkID", walk))
    return (
        _nonblank(dataset, f"{path}:{number} Dataset"),
        _nonblank(location, f"{path}:{number} Location"),
        _walk_id(walk_text, f"{path}:{number} {walk_field}"),
        _nonblank(tp, f"{path}:{number} TP"),
    )


def _update_state(states, walk, bit, evaluable, doubled, confident, duplicate):
    state = states.setdefault(walk, [0, 0, 0, 0])
    if state[0] & bit:
        raise TopologyDetectionError(duplicate)
    state[0] |= bit
    if evaluable:
        state[1] |= bit
    if doubled:
        state[2] |= bit
    if confident:
        state[3] |= bit


# ---------------------------------------------------------------------------
# Pass 1: per-walk locus-state bit masks (seen, evaluable, doubled, confident)
# ---------------------------------------------------------------------------

def scan_legacy_states(path, config, chunk_rows):
    """Normalize legacy trace-salvage evidence for exploratory topology.

    This mode deliberately consumes the already-declared ``Raw*`` evidence
    columns.  It does not reinterpret fit statistics, invent power, or claim
    publication-v1 calibration.
    """
    fields = _header(path)
    required = {
        "Dataset", "Location", "WalkID", "RawTwoComponentCall",
        "RawEvaluable", "RawConfidentSingle", "x_raw", "y_raw", "z_raw",
        "m2_c1_z", "m2_c1_y", "m2_c1_x", "m2_c2_z", "m2_c2_y", "m2_c2_x",
    }
    missing = sorted(required - set(fields))
    if not ({"TP", "tp"} & set(fields)):
        missing.append("TP or tp")
    if missing:
        raise TopologyDetectionError(
            f"{path}: not a trace-salvage full-schema evidence file; "
            f"missing {missing}")
    columns = ("Dataset", "Location", "WalkID", "OriginalWalkID",
               _tp_column(fields, path), "RawEvaluable", "RawTwoComponentCall",
               "RawConfidentSingle")
    states, rows = {}, 0
    for chunk in _iter_chunks(path, columns, chunk_rows):
        for number, values in chunk:
            rows += 1
            key = _locus_key(path, number, *values[:5])
            index = config.layout.index.get(key[3])
            if index is None:
                continue
            label = f"{path}:{number}"
            evaluable = _bit(values[5], f"{label} RawEvaluable")
            doubled = _bit(values[6], f"{label} RawTwoComponentCall")
            confident = _bit(values[7], f"{label} RawConfidentSingle")
            if doubled and not evaluable:
                raise TopologyDetectionError(
                    f"{label}: RawTwoComponentCall cannot be set when "
                    "RawEvaluable is zero")
            if confident and (doubled or not evaluable):
                raise TopologyDetectionError(
                    f"{label}: RawConfidentSingle conflicts with the raw "
                    "evaluability/two-component verdict")
            _update_state(states, key[:3], 1 << index, evaluable, doubled,
                          confident, f"{label}: duplicate exploratory locus key {key}")
    if rows == 0:
        raise TopologyDetectionError(f"{path}: no locus rows")
    if not states:
        raise TopologyDetectionError(
            f"{path}: no rows overlap configured chromosome_order")
    return states


def scan_calibrated_states(path, config, chunk_rows):
    fields = _header(path)
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
            f"{path}: not a calibrated v3 locus-results file; missing {missing}")
    columns = ("Dataset", "Location", "WalkID", "TP", "v3_schema_version",
               "v3_reduction_phase", "v3_locus_evaluable",
               "ResolvedLateralTwoComponentPrimary", "v3_profile_hash",
               "v3_profile_id", "v3_geometry_id", "v3_power_lcb_effective",
               "v3_locus_holm_p")
    states, rows, witnessed = {}, 0, False
    witnesses = {"v3_profile_hash": set(), "v3_profile_id": set(),
                 "v3_geometry_id": set()}
    for chunk in _iter_chunks(path, columns, chunk_rows):
        for number, values in chunk:
            rows += 1
            (dataset, location, walk, tp, schema, phase, evaluable, doubled,
             profile_hash, profile_id, geometry_id, power, holm_p) = values
            label = f"{path}:{number}"
            key = _locus_key(path, number, dataset, location, walk, "", tp)
            schema = schema.strip()
            if schema:
                if schema != "replication_qc_v3":
                    raise TopologyDetectionError(
                        f"{label}: unexpected v3_schema_version {schema!r}")
                witnessed = True
            if phase.strip() != "final":
                raise TopologyDetectionError(
                    f"{label}: v3_reduction_phase must be 'final', got {phase.strip()!r}")
            evaluable = _bit(evaluable, f"{label} v3_locus_evaluable")
            doubled = _bit(doubled, f"{label} ResolvedLateralTwoComponentPrimary")
            if doubled and not evaluable:
                raise TopologyDetectionError(
                    f"{label}: a doubled locus cannot be unevaluable")
            if (evaluable or doubled) and schema != "replication_qc_v3":
                raise TopologyDetectionError(
                    f"{label}: evaluable v3 locus has no v3 schema witness")
            for field, value in (("v3_profile_hash", profile_hash),
                                 ("v3_profile_id", profile_id),
                                 ("v3_geometry_id", geometry_id)):
                value = value.strip()
                if value:
                    witnesses[field].add(value)
                elif evaluable or doubled:
                    raise TopologyDetectionError(
                        f"{label}: evaluable v3 locus lacks {field}")
            index = config.layout.index.get(key[3])
            if index is None:
                continue
            confident = 0
            if evaluable and not doubled:
                power = _optional_finite(power, f"{key} v3_power_lcb_effective")
                holm_p = _finite(holm_p, f"{key} v3_locus_holm_p")
                confident = int(power is not None and power >= config.power_lcb_min
                                and holm_p > config.exploratory_q)
            _update_state(states, key[:3], 1 << index, evaluable, doubled,
                          confident, f"{label}: duplicate locus key {key}")
    if rows == 0:
        raise TopologyDetectionError(f"{path}: no locus rows")
    if not witnessed:
        raise TopologyDetectionError(
            f"{path}: no row identifies replication_qc_v3 evidence")
    for field, name in (("v3_profile_hash", "profile hashes"),
                        ("v3_profile_id", "profile IDs"),
                        ("v3_geometry_id", "geometry IDs")):
        if len(witnesses[field]) != 1:
            raise TopologyDetectionError(
                f"{path}: mixed v3 {name}: {sorted(witnesses[field])}")
    if witnesses["v3_geometry_id"] != {_geometry_id(config)}:
        raise TopologyDetectionError(
            f"{path}: v3 geometry {next(iter(witnesses['v3_geometry_id']))!r} "
            f"does not match configuration geometry {_geometry_id(config)!r}")
    if not states:
        raise TopologyDetectionError(
            f"{path}: no rows overlap configured chromosome_order")
    return states


def validate_reduction_manifest(path, *, config_path, locus_results, config, states):
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
    datasets = {key[0] for key in states}
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
    if inputs.get("config_sha256") != _sha256(config_path):
        raise TopologyDetectionError(
            f"{source}: configuration hash differs from final reduction")
    if outputs.get("locus_results_sha256") != _sha256(locus_results):
        raise TopologyDetectionError(
            f"{source}: locus-results hash differs from final reduction")
    return payload


# ---------------------------------------------------------------------------
# Pass 2 and 3: raw fitter evidence and drift-corrected coordinates for
# candidate walks only
# ---------------------------------------------------------------------------

def load_evidence(path, config, candidates, evidence_mode, chunk_rows):
    fields = _header(path)
    if evidence_mode == "legacy-full-schema":
        columns = ("Dataset", "Location", "WalkID", "OriginalWalkID",
                   _tp_column(fields, path), "x_raw", "y_raw", "z_raw",
                   "m2_c1_z", "m2_c1_y", "m2_c1_x", "m2_c2_z", "m2_c2_y", "m2_c2_x")
    else:
        columns = ("Dataset", "Location", "WalkID", "", "TP", "x", "y", "z",
                   "v3_m2_c1_z", "v3_m2_c1_y", "v3_m2_c1_x",
                   "v3_m2_c2_z", "v3_m2_c2_y", "v3_m2_c2_x")
    evidence = {}
    for chunk in _iter_chunks(path, columns, chunk_rows):
        for number, values in chunk:
            key = _locus_key(path, number, *values[:5])
            if key[3] not in config.layout.index or key[:3] not in candidates:
                continue
            evidence.setdefault(key[:3], {})[key[3]] = Evidence(
                values[5:8], values[8:11], values[11:14])
    return evidence


def load_coordinates(path, config, candidates, *, require_ownership, chunk_rows):
    fields = set(_header(path))
    if not {"Location", "WalkID"} <= fields or not ({"TP", "tp"} & fields):
        raise TopologyDetectionError(
            f"{path}: coordinates need Location, WalkID, and TP or tp")
    if not {"x_raw", "y_raw", "z_raw"} <= fields:
        raise TopologyDetectionError(
            f"{path}: coordinates need x_raw, y_raw, and z_raw to bind them "
            "to the v3 localization source")
    original_names = ("OriginalXNm", "OriginalYNm", "OriginalZNm")
    coordinate_names = (original_names if set(original_names) <= fields
                        else ("x_nm", "y_nm", "z_nm"))
    if not set(coordinate_names) <= fields:
        raise TopologyDetectionError(
            f"{path}: coordinates need OriginalXNm/OriginalYNm/OriginalZNm "
            "or x_nm/y_nm/z_nm")
    if require_ownership and "CenterOwnershipValid" not in fields:
        raise TopologyDetectionError(
            f"{path}: coordinates lack CenterOwnershipValid; strict-owned "
            "one-spot coordinates are required")
    check_family = bool(config.family_id) and "v3_family_id" in fields
    columns = ("Dataset", "Location", "WalkID", "OriginalWalkID",
               _tp_column(fields, path), *coordinate_names, "x_raw", "y_raw",
               "z_raw", "drift_z", "DriftZ", "drift_y", "DriftY", "drift_x",
               "DriftX", "x_err", "y_err", "z_err", "CenterOwnershipValid",
               "v3_family_id")

    alias_name = {"drift_z": "DriftZ", "drift_y": "DriftY", "drift_x": "DriftX"}

    def pick_drift(primary, alias, name):
        if primary.strip():
            return primary
        if alias.strip():
            return alias
        raise TopologyDetectionError(f"missing {name}; tried {[name, alias_name[name]]}")
    result, drift_by_acquisition = {}, {}
    for chunk in _iter_chunks(path, columns, chunk_rows):
        for number, values in chunk:
            xyz_values = values[5:8]
            populated = [bool(value.strip()) for value in xyz_values]
            if any(populated) and not all(populated):
                raise TopologyDetectionError(
                    f"{path}:{number}: partially populated XYZ-nm coordinates")
            if not any(populated):
                # Audit-only rejected attempts deliberately have no coordinates.
                continue
            key = _locus_key(path, number, *values[:5])
            if key[3] not in config.layout.index or key[:3] not in candidates:
                continue
            label = f"{path}:{number}"
            if check_family and values[21].strip() != config.family_id:
                raise TopologyDetectionError(
                    f"{label}: coordinate family {values[21].strip()!r} does not "
                    f"match configuration family {config.family_id!r}")
            if key in result:
                raise TopologyDetectionError(f"{label}: duplicate coordinate key {key}")
            xyz = tuple(_finite(value, f"{label} {name}")
                        for value, name in zip(xyz_values, coordinate_names))
            raw_xyz = tuple(_finite(value, f"{label} {name}")
                            for value, name in zip(values[8:11], ("x_raw", "y_raw", "z_raw")))
            drift = tuple(
                _finite(pick_drift(values[i], values[i + 1], name), f"{label} {name}")
                for i, name in ((11, "drift_z"), (13, "drift_y"), (15, "drift_x")))
            acquisition = key[:2] + (key[3],)
            if drift_by_acquisition.setdefault(acquisition, drift) != drift:
                raise TopologyDetectionError(
                    f"{label}: drift differs within acquisition {acquisition}")
            if require_ownership and _bit(
                    values[20], f"{label} CenterOwnershipValid") != 1:
                raise TopologyDetectionError(
                    f"{label}: usable coordinate is not strict-owned")
            error_values = values[17:20]
            error_populated = [bool(value.strip()) for value in error_values]
            if any(error_populated) and not all(error_populated):
                raise TopologyDetectionError(
                    f"{label}: partially populated x_err/y_err/z_err")
            xyz_se = None
            if all(error_populated):
                raw_errors = tuple(_finite(value, f"{label} localization error")
                                   for value in error_values)
                if all(value >= 0.0 for value in raw_errors):
                    xyz_se = (raw_errors[0] * config.xy_pixel_nm,
                              raw_errors[1] * config.xy_pixel_nm,
                              raw_errors[2] * config.z_step_nm)
            result[key] = Coordinate(xyz, raw_xyz, drift, xyz_se)
    if not result and candidates:
        raise TopologyDetectionError(f"{path}: no usable coordinate rows")
    return result


# ---------------------------------------------------------------------------
# Per-walk geometry
# ---------------------------------------------------------------------------

def _component_pair(evidence, coordinate, config, label):
    dz, dy, dx = coordinate.drift_zyx_px
    components = []
    for index, centre in ((1, evidence.component_1), (2, evidence.component_2)):
        z = _finite(centre[0], f"{label} component {index} z")
        y = _finite(centre[1], f"{label} component {index} y")
        x = _finite(centre[2], f"{label} component {index} x")
        components.append(((x - dx) * config.xy_pixel_nm,
                           (y - dy) * config.xy_pixel_nm,
                           (z - dz) * config.z_step_nm))
    if components[0] == components[1]:
        raise TopologyDetectionError(f"{label}: coincident two-component centres")
    return components[0], components[1]


def _verify_coordinate_source(evidence, coordinate, config, label):
    source_xyz = tuple(_finite(value, f"{label} source {axis}")
                       for value, axis in zip(evidence.source, ("x", "y", "z")))
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


def _component_id(point, original):
    if point == original[0]:
        return 1
    if point == original[1]:
        return 2
    raise TopologyDetectionError("assigned coordinate is not an input component")


def _mean(first, second):
    return tuple((a + b) / 2.0 for a, b in zip(first, second))


def _theta_strands(assignment, singles, *, boundary_midpoint):
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


def _blank_uncertainty():
    return {"XSEnm": "", "YSEnm": "", "ZSEnm": ""}


def _theta_points(key, graph_id, config, pattern, assignment, singles,
                  single_errors, component_pairs):
    strands = _theta_strands(assignment, singles, boundary_midpoint=True)
    strand_loci = {
        "unreplicated": pattern.unreplicated_path,
        "sister_A": pattern.replicated_path,
        "sister_B": pattern.replicated_path,
    }
    result = []
    for strand_id, points in zip(THETA_STRANDS, strands):
        loci = strand_loci[strand_id]
        for index, point in enumerate(points):
            endpoint = index in (0, len(points) - 1)
            locus = "" if endpoint else loci[index - 1]
            component = ""
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
                "TPIndex": "" if endpoint else config.layout.index[locus],
                "GenomicBP": "" if endpoint else config.genomic_bp.get(locus, ""),
                "CoordinateSource": source, "ComponentID": component,
                "XNm": point[0], "YNm": point[1], "ZNm": point[2],
                **uncertainty,
                "Observed": int(not endpoint), "InferredFork": int(endpoint),
            })
    return result


def _link_points(key, link_id, config, assignment, component_pairs):
    result = []
    for loop_id, points in zip(LINK_LOOPS, (assignment.sister_a, assignment.sister_b)):
        for index, (locus, point) in enumerate(zip(assignment.loci, points)):
            result.append({
                "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                "LinkID": link_id, "LoopID": loop_id, "PointOrder": index,
                "PointRole": "locus", "TP": locus,
                "TPIndex": config.layout.index[locus],
                "GenomicBP": config.genomic_bp.get(locus, ""),
                "CoordinateSource": "two_component_drift_corrected",
                "ComponentID": _component_id(point, component_pairs[locus]),
                "XNm": point[0], "YNm": point[1], "ZNm": point[2],
                **_blank_uncertainty(), "Closed": 1,
            })
    return result


def _failure_summary(key, identifier, *, theta, count, status, reason):
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


def _evaluate_theta(engine, strands, *, seeds, tries, max_crossings):
    """Yamada invariant across projection seeds; stable if all agree."""
    results = tuple(engine.yamada(strands, tries=tries, seed=seed,
                                  max_crossings=max_crossings) for seed in seeds)
    reference = results[0].normalized_terms
    stable = all(result.normalized_terms == reference for result in results[1:])
    return results, stable


def _log(message):
    print(f"[detect_topology] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect(
    *, locus_results, coordinates_csv, config_path, reduction_manifest,
    output_dir, yamada_seeds=(0, 1, 2, 3), yamada_tries=60,
    max_crossings=DEFAULT_MAX_CROSSINGS, link_projections=96,
    evidence_mode="calibrated-v3", evaluate_yamada_polynomial=True,
    chunk_rows=DEFAULT_CHUNK_ROWS, engine_path=ENGINE_SO, threads=None,
):
    started = time.time()
    config = load_config(config_path)
    engine = Engine(engine_path)
    if threads:
        engine.set_threads(threads)
    layout = config.layout
    locus_path, coordinate_path = Path(locus_results), Path(coordinates_csv)
    config_source = Path(config_path)
    yamada_seeds = tuple(int(seed) for seed in yamada_seeds)
    if not yamada_seeds:
        raise TopologyDetectionError("at least one projection seed is required")

    if evidence_mode == "calibrated-v3":
        if reduction_manifest is None:
            raise TopologyDetectionError(
                "--reduction-manifest is required in calibrated-v3 mode")
        states = scan_calibrated_states(locus_path, config, chunk_rows)
        validate_reduction_manifest(
            reduction_manifest, config_path=config_source,
            locus_results=locus_path, config=config, states=states)
        require_ownership = True
    elif evidence_mode == "legacy-full-schema":
        states = scan_legacy_states(locus_path, config, chunk_rows)
        require_ownership = False
    else:
        raise TopologyDetectionError(f"unsupported evidence_mode {evidence_mode!r}")

    pattern_counts = Counter()
    candidates = {}
    for key, (_seen, _evaluable, doubled, confident) in states.items():
        pattern = engine.theta_pattern(layout, doubled, doubled | confident)
        pattern_counts[pattern.status] += 1
        if pattern.status in THETA_CANDIDATE_STATUSES or pattern.status == LINK_CANDIDATE_STATUS:
            candidates[key] = (pattern, bin(doubled).count("1"))
    walks_total = len(states)
    del states
    _log(f"{walks_total} walks scanned, {len(candidates)} topology candidates "
         f"({time.time() - started:.0f}s)")
    evidence = load_evidence(locus_path, config, candidates, evidence_mode, chunk_rows)
    coordinates = load_coordinates(
        coordinate_path, config, candidates, require_ownership=require_ownership,
        chunk_rows=chunk_rows)
    _log(f"evidence and coordinates loaded for candidates ({time.time() - started:.0f}s)")

    theta_summaries, theta_points, link_summaries, link_points = [], [], [], []
    crossings_seen = Counter()
    for done, key in enumerate(sorted(candidates), 1):
        if done % 100 == 0:
            _log(f"{done}/{len(candidates)} candidates ({time.time() - started:.0f}s)")
        pattern, n_doubled = candidates[key]
        by_tp = evidence.get(key, {})
        is_theta = pattern.status in THETA_CANDIDATE_STATUSES
        identifier = ("theta" if is_theta else "link") + ":" + ":".join(
            (key[0], key[1], str(key[2])))

        if is_theta and pattern.trace_status != "theta_traceable":
            theta_summaries.append(_failure_summary(
                key, identifier, theta=True, count=n_doubled,
                status="pattern_not_traceable",
                reason=("discordant or unevaluable loci prevent a coherent "
                        "three-strand reconstruction")))
            continue

        def locus_inputs(tp):
            label = f"{key + (tp,)}"
            row = by_tp.get(tp)
            coordinate = coordinates.get(key + (tp,))
            if row is None:
                raise TopologyDetectionError(f"missing evidence row for {label}")
            if coordinate is None:
                raise TopologyDetectionError(f"missing coordinate for {label}")
            _verify_coordinate_source(row, coordinate, config, label)
            return row, coordinate, label

        try:
            needed_double = pattern.replicated_path if is_theta else layout.order
            component_pairs, pairs = {}, []
            for tp in needed_double:
                row, coordinate, label = locus_inputs(tp)
                component_pairs[tp] = _component_pair(row, coordinate, config, label)
                pairs.append(component_pairs[tp])
            assignment = engine.assign(needed_double, pairs, circular=not is_theta)
            xy_assignment = engine.assign(
                needed_double, pairs, circular=not is_theta, dims=(0, 1))
            margin, xy_margin = assignment.cost_margin, xy_assignment.cost_margin
            margin_clear = (
                margin is not None and
                margin > 1e-9 * max(1.0, assignment.total_cost) and
                xy_margin is not None and
                xy_margin > 1e-9 * max(1.0, xy_assignment.total_cost))
            assignment_stable = (
                assignment.state_path == xy_assignment.state_path and margin_clear)
        except TopologyDetectionError as exc:
            (theta_summaries if is_theta else link_summaries).append(_failure_summary(
                key, identifier, theta=is_theta,
                count=n_doubled if is_theta else len(layout.order),
                status="geometry_incomplete", reason=str(exc)))
            continue

        reasons = []
        if assignment.state_path != xy_assignment.state_path:
            reasons.append("XY and XYZ sister assignments disagree")
        if not margin_clear:
            reasons.append("sister assignment has no numerical cost margin")

        if is_theta:
            try:
                singles, single_errors = [], []
                for tp in pattern.unreplicated_path:
                    _row, coordinate, _label = locus_inputs(tp)
                    singles.append(coordinate.xyz_nm)
                    single_errors.append(coordinate.xyz_se_nm)
                primary = _theta_strands(assignment, singles, boundary_midpoint=True)
                alternate = _theta_strands(assignment, singles, boundary_midpoint=False)
                engine.validate_theta(primary)
                engine.validate_theta(alternate)
                raw_polynomial = normalized_polynomial = ""
                projection_stable = fork_stable = ""
                if not evaluate_yamada_polynomial:
                    status = ("nominal_theta_geometry" if not reasons
                              else "theta_geometry_unstable")
                else:
                    try:
                        primary_results, primary_stable = _evaluate_theta(
                            engine, primary, seeds=yamada_seeds, tries=yamada_tries,
                            max_crossings=max_crossings)
                        alternate_results, alternate_stable = _evaluate_theta(
                            engine, alternate, seeds=yamada_seeds, tries=yamada_tries,
                            max_crossings=max_crossings)
                    except CrossingsExceeded as exc:
                        crossings_seen["exceeded"] += 1
                        reasons.append(str(exc))
                        status = "theta_crossings_exceeded"
                    else:
                        crossings_seen[primary_results[0].crossings] += 1
                        fork_stable = int(
                            alternate_stable and
                            primary_results[0].normalized_terms ==
                            alternate_results[0].normalized_terms)
                        if not primary_stable:
                            reasons.append("Yamada polynomial changes across projections")
                        if not fork_stable:
                            reasons.append("Yamada polynomial changes with fork model")
                        if any(result.vertices != 2
                               for result in primary_results + alternate_results):
                            reasons.append("Yamada reconstruction has extra graph vertices")
                        status = "nominal_theta_graph" if not reasons else "theta_unstable"
                        raw_polynomial = _format_laurent(primary_results[0].raw_terms)
                        normalized_polynomial = _format_laurent(
                            primary_results[0].normalized_terms)
                        projection_stable = int(primary_stable)
                theta_summaries.append({
                    "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                    "GraphID": identifier, "ThetaStatus": status,
                    "FailureReason": "; ".join(reasons),
                    "NumDoubledLoci": n_doubled,
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
            except (TopologyDetectionError, TopologyValidationError) as exc:
                theta_summaries.append(_failure_summary(
                    key, identifier, theta=True, count=n_doubled,
                    status="geometry_invalid", reason=str(exc)))
        else:
            try:
                evaluation = engine.linking(
                    assignment.sister_a, assignment.sister_b,
                    n_projections=link_projections)
                if not evaluation.stable:
                    reasons.append("linking number changes across projections")
                if reasons:
                    status = "link_unstable"
                elif evaluation.linking_number:
                    status = "nominal_linked_polygon"
                else:
                    status = "nominal_unlinked_polygon"
                link_summaries.append({
                    "Dataset": key[0], "Location": key[1], "WalkID": key[2],
                    "LinkID": identifier, "LinkStatus": status,
                    "FailureReason": "; ".join(reasons),
                    "NumLoci": len(layout.order),
                    "AssignmentCost": assignment.total_cost,
                    "AssignmentMargin": assignment.cost_margin,
                    "LinkingNumber": evaluation.linking_number,
                    "MinInterLoopDistanceNm": evaluation.min_distance_nm,
                    "AssignmentStable": int(assignment_stable),
                    "PerturbationStable": "", "DisjointStable": "",
                })
                link_points.extend(_link_points(
                    key, identifier, config, assignment, component_pairs))
            except TopologyValidationError as exc:
                link_summaries.append(_failure_summary(
                    key, identifier, theta=False, count=len(layout.order),
                    status="geometry_invalid", reason=str(exc)))

    software = {"detect_topology.py": _sha256(Path(__file__)),
                engine.path.name: _sha256(engine.path)}
    engine_source = engine.path.with_suffix(".c")
    if engine_source.is_file():
        software[engine_source.name] = _sha256(engine_source)
    provenance = {
        "detector": "pipeline.detect_topology",
        "software_sha256": software,
        "engine": {"library": str(engine.path.resolve()), "openmp": engine.openmp,
                   "threads": threads, "memo_budget_mb_per_thread": engine.memo_budget_mb},
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
        "walks_reaching_topology_geometry": len(candidates),
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "theta_candidates": len(theta_summaries),
        "link_candidates": len(link_summaries),
        "theta_status_counts": dict(sorted(Counter(
            row["ThetaStatus"] for row in theta_summaries).items())),
        "link_status_counts": dict(sorted(Counter(
            row["LinkStatus"] for row in link_summaries).items())),
        "yamada_seeds": list(yamada_seeds),
        "yamada_tries": yamada_tries,
        "yamada_evaluated": evaluate_yamada_polynomial,
        "yamada_max_crossings": max_crossings,
        "yamada_crossing_counts": {str(k): v for k, v in sorted(
            crossings_seen.items(), key=lambda item: str(item[0]))},
        "link_projections": link_projections,
        "link_projection_seed": LINK_PROJECTION_SEED,
        "link_min_valid_projections": LINK_MIN_VALID_PROJECTIONS,
        "link_min_projection_stability": LINK_MIN_PROJECTION_STABILITY,
        "geometry_intersection_tolerance_nm": GEOMETRY_TOL_NM,
        "csv_chunk_rows": chunk_rows,
    }
    if reduction_manifest is not None:
        provenance["input_sha256"]["reduction_manifest"] = _sha256(reduction_manifest)
        provenance["input_paths"]["reduction_manifest"] = str(
            Path(reduction_manifest).resolve())
    paths = write_topology_outputs(
        output_dir, theta_summaries=theta_summaries, theta_points=theta_points,
        link_summaries=link_summaries, link_points=link_points,
        provenance=provenance)
    _log(f"theta {provenance['theta_status_counts']}")
    _log(f"link {provenance['link_status_counts']}")
    _log(f"done ({time.time() - started:.0f}s)")
    return paths


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def _serialise(row):
    result = {}
    for field, value in row.items():
        if value is None or value == "":
            result[field] = ""
        elif isinstance(value, bool):
            result[field] = "1" if value else "0"
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise TopologyDetectionError(f"{field} must be finite")
            result[field] = format(value, ".17g")
        else:
            result[field] = str(value)
    return result


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path, fields, rows, table):
    expected = set(fields)
    for number, row in enumerate(rows, 1):
        if set(row) != expected:
            raise TopologyDetectionError(
                f"{table} row {number} has wrong schema; "
                f"missing={sorted(expected - set(row))}, extra={sorted(set(row) - expected)}")

    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(_serialise(row) for row in rows)

    _atomic_write(path, write)


def write_topology_outputs(output_dir, *, theta_summaries, theta_points,
                           link_summaries, link_points, provenance):
    """Write the four CSV deliverables and, last, the manifest hashing them."""
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TopologyDetectionError(f"provenance is not finite JSON: {exc}") from exc
    theta_summaries = sorted(theta_summaries, key=lambda row: str(row["GraphID"]))
    theta_points = sorted(theta_points, key=lambda row: (
        str(row["GraphID"]), THETA_STRANDS.index(row["StrandID"]), int(row["PointOrder"])))
    link_summaries = sorted(link_summaries, key=lambda row: str(row["LinkID"]))
    link_points = sorted(link_points, key=lambda row: (
        str(row["LinkID"]), LINK_LOOPS.index(row["LoopID"]), int(row["PointOrder"])))
    root = Path(output_dir)
    paths = {key: root / name for key, name in OUTPUT_NAMES.items()}
    tables = {
        "theta_summary": (THETA_SUMMARY_FIELDS, theta_summaries),
        "theta_points": (THETA_POINT_FIELDS, theta_points),
        "link_summary": (LINK_SUMMARY_FIELDS, link_summaries),
        "link_points": (LINK_POINT_FIELDS, link_points),
    }
    for key, (fields, rows) in tables.items():
        _atomic_csv(paths[key], fields, rows, key)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": COORDINATE_FRAME,
        "files": {
            key: {"name": paths[key].name, "rows": len(rows), "sha256": _sha256(paths[key])}
            for key, (_fields, rows) in tables.items()
        },
        "provenance": dict(provenance),
    }

    def write(handle):
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    _atomic_write(paths["manifest"], write)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _seed_list(text):
    try:
        result = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locus-results", required=True,
                        help=("calibrated v3 locus results, or a trace-salvage "
                              "full-schema CSV in legacy-full-schema mode"))
    parser.add_argument("--coordinates-csv", required=True,
                        help="combined comprehensive or unfiltered drift-corrected coordinates")
    parser.add_argument("--config", required=True,
                        help="dataset config with chromosome order and geometry")
    parser.add_argument(
        "--evidence-mode", choices=("calibrated-v3", "legacy-full-schema"),
        default="calibrated-v3",
        help=("calibrated-v3 enforces the publication reduction contract; "
              "legacy-full-schema uses explicit Raw* calls for exploratory analysis"))
    parser.add_argument(
        "--reduction-manifest",
        help="final v3 publication_v1_manifest.json (required for calibrated-v3)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--skip-yamada", action="store_true",
        help=("reconstruct and validate theta geometry without evaluating the "
              "Yamada polynomial; link detection is unchanged"))
    parser.add_argument("--yamada-seeds", type=_seed_list, default=(0, 1, 2, 3))
    parser.add_argument("--yamada-tries", type=int, default=60)
    parser.add_argument(
        "--max-crossings", type=int, default=DEFAULT_MAX_CROSSINGS,
        help=("report theta_crossings_exceeded instead of evaluating a theta whose "
              "best projection has more crossings (skein states grow as 3^c)"))
    parser.add_argument("--link-projections", type=int, default=96)
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
        help="CSV rows resident per chunk; smaller files load as one chunk")
    parser.add_argument(
        "--threads", type=int,
        help="OpenMP threads for the engine (default: SLURM_CPUS_PER_TASK or OpenMP default)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.yamada_tries < 1 or args.link_projections < LINK_MIN_VALID_PROJECTIONS
            or args.max_crossings < 1 or args.chunk_rows < 1):
        print("error: --yamada-tries, --max-crossings and --chunk-rows must be "
              f"positive and --link-projections >= {LINK_MIN_VALID_PROJECTIONS}",
              file=sys.stderr)
        return 2
    threads = args.threads
    if threads is None and os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit():
        threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        paths = detect(
            locus_results=args.locus_results, coordinates_csv=args.coordinates_csv,
            config_path=args.config, output_dir=args.outdir,
            reduction_manifest=args.reduction_manifest,
            yamada_seeds=args.yamada_seeds, yamada_tries=args.yamada_tries,
            max_crossings=args.max_crossings, link_projections=args.link_projections,
            evidence_mode=args.evidence_mode,
            evaluate_yamada_polynomial=not args.skip_yamada,
            chunk_rows=args.chunk_rows, threads=threads)
    except (TopologyDetectionError, TopologyValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
