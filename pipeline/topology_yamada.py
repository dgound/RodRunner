#!/usr/bin/env python3
"""Small, deterministic geometry adapter for theta graphs and two-loop links.

Coordinates in this module are always ``(x, y, z)`` in nanometres.  The
module deliberately does not infer strands, forks, or sister identities; it
only validates already reconstructed geometry and evaluates its topology.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Mapping, Sequence

import numpy as np


class TopologyValidationError(ValueError):
    """Raised when coordinates do not define the requested topology."""


@dataclass(frozen=True)
class YamadaResult:
    """Normalized Yamada polynomial and projection diagnostics."""

    raw_terms: tuple[tuple[int, int], ...]
    raw_polynomial: str
    normalized_terms: tuple[tuple[int, int], ...]
    normalized_polynomial: str
    crossings: int
    vertices: int
    tries: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        """Return a stable, compact serialization suitable for CSV/JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ThetaEvaluation:
    """Yamada result compared across deterministic projection seeds."""

    normalized_terms: tuple[tuple[int, int], ...]
    normalized_polynomial: str
    seed_results: tuple[YamadaResult, ...]
    stable: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LinkingResult:
    """Classical linking number with deterministic projection diagnostics."""

    linking_number: int
    absolute_linking_number: int
    projection_values: tuple[int, ...]
    projections_requested: int
    projections_valid: int
    projection_stability: float
    stable: bool
    min_segment_distance_nm: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _xyz_array(points: Sequence[Sequence[float]], name: str,
               min_points: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise TopologyValidationError(
            f"{name} must have shape (n, 3) in XYZ-nm order; got {arr.shape}")
    if len(arr) < min_points:
        raise TopologyValidationError(
            f"{name} requires at least {min_points} points; got {len(arr)}")
    if not np.isfinite(arr).all():
        raise TopologyValidationError(f"{name} contains non-finite coordinates")
    return np.ascontiguousarray(arr)


def validate_theta_strands(
        strands_xyz_nm: Sequence[Sequence[Sequence[float]]],
        intersection_tolerance_nm: float = 1e-6) -> tuple[np.ndarray, ...]:
    """Validate the exact three-strand representation expected by Yamada.

    All three observed polylines must contain at least two points and must use
    numerically identical first and last fork coordinates.  Approximate fork
    merging is intentionally forbidden: graph construction must make its
    inferred fork choice explicit before calling this adapter.
    """
    if len(strands_xyz_nm) != 3:
        raise TopologyValidationError(
            f"a theta graph requires exactly 3 strands; got {len(strands_xyz_nm)}")
    strands = tuple(
        _xyz_array(points, f"theta strand {index}", min_points=2)
        for index, points in enumerate(strands_xyz_nm)
    )
    start, end = strands[0][0], strands[0][-1]
    if np.array_equal(start, end):
        raise TopologyValidationError("theta fork vertices must be distinct")
    for index, strand in enumerate(strands[1:], start=1):
        if not np.array_equal(strand[0], start) or not np.array_equal(strand[-1], end):
            raise TopologyValidationError(
                f"theta strand {index} does not share the exact ordered fork endpoints")
    for index, strand in enumerate(strands):
        if np.any(np.linalg.norm(np.diff(strand, axis=0), axis=1) == 0.0):
            raise TopologyValidationError(
                f"theta strand {index} contains a zero-length segment")
        for first in range(len(strand) - 1):
            for second in range(first + 2, len(strand) - 1):
                if _segment_distance(
                        strand[first], strand[first + 1],
                        strand[second], strand[second + 1]
                ) <= intersection_tolerance_nm:
                    raise TopologyValidationError(
                        f"theta strand {index} self-intersects or touches")
    for first in range(3):
        for second in range(first + 1, 3):
            a, b = strands[first], strands[second]
            for a_segment in range(len(a) - 1):
                for b_segment in range(len(b) - 1):
                    # The first segments share fork 1; the last segments share
                    # fork 2. Those are the only permitted contacts.
                    if a_segment == 0 and b_segment == 0:
                        if _overlap_from_shared_endpoint(
                                a[0], a[1], b[1], intersection_tolerance_nm):
                            raise TopologyValidationError(
                                "theta strands overlap beyond fork 1")
                        continue
                    if (a_segment == len(a) - 2 and
                            b_segment == len(b) - 2):
                        if _overlap_from_shared_endpoint(
                                a[-1], a[-2], b[-2], intersection_tolerance_nm):
                            raise TopologyValidationError(
                                "theta strands overlap beyond fork 2")
                        continue
                    if _segment_distance(
                            a[a_segment], a[a_segment + 1],
                            b[b_segment], b[b_segment + 1]
                    ) <= intersection_tolerance_nm:
                        raise TopologyValidationError(
                            "theta strands intersect or touch away from a fork")
    return strands


def _overlap_from_shared_endpoint(
        common: np.ndarray, end_a: np.ndarray, end_b: np.ndarray,
        tolerance: float) -> bool:
    """Whether two segments sharing ``common`` overlap past that endpoint."""
    first, second = end_a - common, end_b - common
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    cross = float(np.linalg.norm(np.cross(first, second)))
    collinear = cross <= tolerance * max(first_length, second_length, 1.0)
    same_ray = float(first @ second) > 0.0
    return collinear and same_ray


def _resolve_yamada_file(yamada_path: str | Path) -> Path:
    path = Path(yamada_path).expanduser().resolve()
    if path.is_dir():
        path = path / "yamada.py"
    if not path.is_file():
        raise FileNotFoundError(f"Yamada implementation not found: {path}")
    return path


def load_yamada(yamada_path: str | Path) -> ModuleType:
    """Load a supplied ``yamada.py`` without modifying ``sys.path``.

    Loading Python source necessarily executes that source.  This helper's
    safety guarantee is namespace isolation: it loads exactly the resolved
    file supplied by the caller under a path-derived private module name and
    verifies the small API used below.
    """
    path = _resolve_yamada_file(yamada_path)
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(f"_fish_yamada_{digest}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot construct an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    # This Yamada checkout imports a sibling ``_geom`` module whose import-time
    # convenience path compiles a shared object into the checkout.  Supply its
    # documented pure-NumPy interface while loading so adapting the repository
    # is read-only and does not alter global import search paths.
    shim = _yamada_geometry_shim()
    previous_geom = sys.modules.get("_geom")
    sys.modules["_geom"] = shim
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_geom is None:
            sys.modules.pop("_geom", None)
        else:
            sys.modules["_geom"] = previous_geom
    for name in ("yamada", "normalize"):
        if not callable(getattr(module, name, None)):
            raise ImportError(f"{path} does not provide callable {name}()")
    return module


def _yamada_geometry_shim() -> ModuleType:
    """Return the public pure-NumPy crossing API used by Yamada 0.1."""
    shim = ModuleType("_geom")

    class NonGeneric(Exception):
        pass

    def find_crossings(segments, metadata, maxout=4096):  # noqa: ARG001
        segments = np.ascontiguousarray(segments, dtype=np.float64)
        metadata = np.ascontiguousarray(metadata, dtype=np.int32)
        count = segments.shape[0]
        first, second = np.triu_indices(count, k=1)
        meta_first, meta_second = metadata[first], metadata[second]
        shared = ((meta_first[:, 1:2] == meta_second[:, 1:]).any(1) |
                  (meta_first[:, 2:3] == meta_second[:, 1:]).any(1))
        first, second = first[~shared], second[~shared]
        a, b = segments[first], segments[second]
        da, db = a[:, 3:5] - a[:, 0:2], b[:, 3:5] - b[:, 0:2]
        denominator = da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0]
        nonparallel = np.abs(denominator) >= 1e-9
        first, second = first[nonparallel], second[nonparallel]
        a, b = a[nonparallel], b[nonparallel]
        da, db, denominator = da[nonparallel], db[nonparallel], denominator[nonparallel]
        offset = b[:, 0:2] - a[:, 0:2]
        ta = (offset[:, 0] * db[:, 1] - offset[:, 1] * db[:, 0]) / denominator
        tb = (offset[:, 0] * da[:, 1] - offset[:, 1] * da[:, 0]) / denominator
        interior = ((ta > 1e-7) & (ta < 1.0 - 1e-7) &
                    (tb > 1e-7) & (tb < 1.0 - 1e-7))
        near = ((~interior) & (ta >= -1e-7) & (ta <= 1.0 + 1e-7) &
                (tb >= -1e-7) & (tb <= 1.0 + 1e-7))
        if near.any():
            raise NonGeneric()
        first, second = first[interior], second[interior]
        a, b, ta, tb = a[interior], b[interior], ta[interior], tb[interior]
        za = a[:, 2] * (1.0 - ta) + a[:, 5] * ta
        zb = b[:, 2] * (1.0 - tb) + b[:, 5] * tb
        if np.any(np.abs(za - zb) < 1e-9):
            raise NonGeneric()
        px = a[:, 0] + (a[:, 3] - a[:, 0]) * ta
        py = a[:, 1] + (a[:, 4] - a[:, 1]) * ta
        if len(px) > 1:
            separation2 = ((px[:, None] - px[None, :]) ** 2 +
                           (py[:, None] - py[None, :]) ** 2)
            np.fill_diagonal(separation2, np.inf)
            if separation2.min() < 1e-12:
                raise NonGeneric()
        return np.column_stack((first, second, ta, tb, za, zb)).astype(np.float64)

    shim.find_crossings = find_crossings
    shim.NonGeneric = NonGeneric
    shim.USING_C = False
    return shim


def _polynomial_terms(polynomial: Mapping) -> tuple[tuple[int, int], ...]:
    terms = []
    for exponent, coefficient in polynomial.items():
        if isinstance(exponent, bool) or isinstance(coefficient, bool):
            raise TypeError("Yamada polynomial terms must be integer pairs")
        exponent_int, coefficient_int = int(exponent), int(coefficient)
        if exponent_int != exponent or coefficient_int != coefficient:
            raise TypeError("Yamada polynomial terms must be integer pairs")
        if coefficient_int:
            terms.append((exponent_int, coefficient_int))
    return tuple(sorted(terms, reverse=True))


def _format_laurent(terms: Iterable[tuple[int, int]], variable: str = "A") -> str:
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


def compute_yamada(
        strands_xyz_nm: Sequence[Sequence[Sequence[float]]],
        yamada_path: str | Path,
        *,
        tries: int = 60,
        seed: int = 0) -> YamadaResult:
    """Compute the normalized Yamada invariant for an exact theta graph."""
    if tries < 1:
        raise ValueError("tries must be positive")
    strands = validate_theta_strands(strands_xyz_nm)
    module = load_yamada(yamada_path)
    polynomial, info = module.yamada(
        [strand.copy() for strand in strands], tries=int(tries), seed=int(seed),
        return_info=True)
    normalized = module.normalize(polynomial)
    raw_terms = _polynomial_terms(polynomial)
    terms = _polynomial_terms(normalized)
    return YamadaResult(
        raw_terms=raw_terms,
        raw_polynomial=_format_laurent(raw_terms),
        normalized_terms=terms,
        normalized_polynomial=_format_laurent(terms),
        crossings=int(info["crossings"]),
        vertices=int(info["vertices"]),
        tries=int(tries),
        seed=int(seed),
    )


def evaluate_theta(
        strands_xyz_nm: Sequence[Sequence[Sequence[float]]],
        yamada_path: str | Path,
        *,
        seeds: Sequence[int] = (0, 1, 2, 3),
        tries: int = 60) -> ThetaEvaluation:
    """Evaluate a theta graph and test its invariant across projection seeds."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one projection seed is required")
    results = tuple(
        compute_yamada(strands_xyz_nm, yamada_path, tries=tries, seed=seed)
        for seed in seeds
    )
    reference = results[0]
    stable = all(result.normalized_terms == reference.normalized_terms
                 for result in results[1:])
    return ThetaEvaluation(
        normalized_terms=reference.normalized_terms,
        normalized_polynomial=reference.normalized_polynomial,
        seed_results=results,
        stable=stable,
    )


def _segment_distance(p0: np.ndarray, p1: np.ndarray,
                      q0: np.ndarray, q1: np.ndarray) -> float:
    """Minimum Euclidean distance between two finite 3D segments."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denominator = a * c - b * b
    small = np.finfo(float).eps * max(a * c, 1.0) * 32.0
    s_num, s_den = denominator, denominator
    t_num, t_den = denominator, denominator
    if denominator <= small:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = e, c
    else:
        s_num, t_num = b * e - c * d, a * e - b * d
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, e, c
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, e + b, c
    if t_num < 0.0:
        t_num = 0.0
        s_num = min(max(-d, 0.0), a)
        s_den = a
    elif t_num > t_den:
        t_num = t_den
        s_num = min(max(b - d, 0.0), a)
        s_den = a
    s = 0.0 if abs(s_num) <= small else s_num / s_den
    t = 0.0 if abs(t_num) <= small else t_num / t_den
    return float(np.linalg.norm(w + s * u - t * v))


def _prepare_loop(points: Sequence[Sequence[float]], name: str,
                  intersection_tolerance_nm: float) -> np.ndarray:
    raw = _xyz_array(points, name, min_points=3)
    if np.array_equal(raw[0], raw[-1]):
        raw = raw[:-1]
    if len(raw) < 3:
        raise TopologyValidationError(f"{name} needs three distinct loop vertices")
    closed = np.vstack((raw, raw[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(lengths <= intersection_tolerance_nm):
        raise TopologyValidationError(f"{name} contains a zero/near-zero segment")
    n_segments = len(raw)
    for first in range(n_segments):
        for second in range(first + 1, n_segments):
            if second == first + 1 or (first == 0 and second == n_segments - 1):
                continue
            distance = _segment_distance(
                closed[first], closed[first + 1], closed[second], closed[second + 1])
            if distance <= intersection_tolerance_nm:
                raise TopologyValidationError(
                    f"{name} self-intersects or touches within "
                    f"{intersection_tolerance_nm:g} nm")
    return closed


def min_interloop_segment_distance(loop_a_closed: np.ndarray,
                                   loop_b_closed: np.ndarray) -> float:
    """Return the exact minimum segment-to-segment distance between loops."""
    best = np.inf
    for first in range(len(loop_a_closed) - 1):
        for second in range(len(loop_b_closed) - 1):
            best = min(best, _segment_distance(
                loop_a_closed[first], loop_a_closed[first + 1],
                loop_b_closed[second], loop_b_closed[second + 1]))
    return float(best)


def validate_link_loops(
        loop_a_xyz_nm: Sequence[Sequence[float]],
        loop_b_xyz_nm: Sequence[Sequence[float]],
        *,
        intersection_tolerance_nm: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate two simple, closed, mutually disjoint polygonal loops.

    Inputs may omit the repeated closing point; closure is part of the declared
    loop semantics of this function.  Returned arrays include that point.
    """
    if not np.isfinite(intersection_tolerance_nm) or intersection_tolerance_nm < 0:
        raise ValueError("intersection_tolerance_nm must be finite and non-negative")
    loop_a = _prepare_loop(loop_a_xyz_nm, "loop A", intersection_tolerance_nm)
    loop_b = _prepare_loop(loop_b_xyz_nm, "loop B", intersection_tolerance_nm)
    distance = min_interloop_segment_distance(loop_a, loop_b)
    if distance <= intersection_tolerance_nm:
        raise TopologyValidationError(
            "classical linking number requires mutually disjoint loops; "
            f"minimum distance is {distance:g} nm")
    return loop_a, loop_b, distance


def deterministic_projection_directions(count: int, seed: int = 0) -> np.ndarray:
    """Generate reproducible, approximately uniform unit directions."""
    if count < 1:
        raise ValueError("projection count must be positive")
    rng = np.random.default_rng(int(seed))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = phase + index * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


class _NonGenericProjection(RuntimeError):
    pass


def _projection_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = axis - direction * float(axis @ direction)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


def _linking_in_projection(loop_a: np.ndarray, loop_b: np.ndarray,
                           direction: np.ndarray,
                           relative_tolerance: float = 1e-10) -> int:
    first, second = _projection_basis(direction)
    projected_a = np.column_stack((loop_a @ first, loop_a @ second))
    projected_b = np.column_stack((loop_b @ first, loop_b @ second))
    depth_a, depth_b = loop_a @ direction, loop_b @ direction
    span = max(float(np.ptp(np.vstack((loop_a, loop_b)), axis=0).max()), 1.0)
    signed_sum = 0
    for i in range(len(loop_a) - 1):
        pa, ea = projected_a[i], projected_a[i + 1] - projected_a[i]
        tangent_a = loop_a[i + 1] - loop_a[i]
        for j in range(len(loop_b) - 1):
            pb, eb = projected_b[j], projected_b[j + 1] - projected_b[j]
            tangent_b = loop_b[j + 1] - loop_b[j]
            denominator = ea[0] * eb[1] - ea[1] * eb[0]
            scale = max(float(np.linalg.norm(ea) * np.linalg.norm(eb)), 1.0)
            delta = pb - pa
            if abs(denominator) <= relative_tolerance * scale:
                # Collinear projected segments make this projection nongeneric;
                # merely parallel separated segments do not.
                if abs(delta[0] * ea[1] - delta[1] * ea[0]) <= relative_tolerance * scale:
                    raise _NonGenericProjection
                continue
            along_a = (delta[0] * eb[1] - delta[1] * eb[0]) / denominator
            along_b = (delta[0] * ea[1] - delta[1] * ea[0]) / denominator
            edge_tol = relative_tolerance * 100.0
            if (-edge_tol <= along_a <= edge_tol or
                    1.0 - edge_tol <= along_a <= 1.0 + edge_tol or
                    -edge_tol <= along_b <= edge_tol or
                    1.0 - edge_tol <= along_b <= 1.0 + edge_tol):
                if (-edge_tol <= along_a <= 1.0 + edge_tol and
                        -edge_tol <= along_b <= 1.0 + edge_tol):
                    raise _NonGenericProjection
                continue
            if not (0.0 < along_a < 1.0 and 0.0 < along_b < 1.0):
                continue
            da = depth_a[i] + along_a * (depth_a[i + 1] - depth_a[i])
            db = depth_b[j] + along_b * (depth_b[j + 1] - depth_b[j])
            if abs(da - db) <= relative_tolerance * span:
                raise _NonGenericProjection
            if da > db:
                orientation = float(np.cross(tangent_a, tangent_b) @ direction)
            else:
                orientation = float(np.cross(tangent_b, tangent_a) @ direction)
            if abs(orientation) <= relative_tolerance * span * span:
                raise _NonGenericProjection
            signed_sum += 1 if orientation > 0 else -1
    if signed_sum % 2:
        raise _NonGenericProjection
    return signed_sum // 2


def compute_linking_number(
        loop_a_xyz_nm: Sequence[Sequence[float]],
        loop_b_xyz_nm: Sequence[Sequence[float]],
        *,
        n_projections: int = 96,
        seed: int = 0,
        min_valid_projections: int = 16,
        min_projection_stability: float = 0.95,
        intersection_tolerance_nm: float = 1e-6) -> LinkingResult:
    """Evaluate the oriented classical linking number of two polygonal loops.

    A modal value is reported together with the fraction of generic
    deterministic projections that agree.  ``stable`` is false rather than
    silently asserting topology when too few projections are usable or their
    values disagree.
    """
    if min_valid_projections < 1 or min_valid_projections > n_projections:
        raise ValueError("min_valid_projections must be within n_projections")
    if not 0.0 <= min_projection_stability <= 1.0:
        raise ValueError("min_projection_stability must lie in [0, 1]")
    loop_a, loop_b, distance = validate_link_loops(
        loop_a_xyz_nm, loop_b_xyz_nm,
        intersection_tolerance_nm=intersection_tolerance_nm)
    values = []
    for direction in deterministic_projection_directions(n_projections, seed):
        try:
            values.append(_linking_in_projection(loop_a, loop_b, direction))
        except _NonGenericProjection:
            continue
    if not values:
        raise TopologyValidationError("no generic projection found for the two loops")
    counts = Counter(values)
    linking_number, count = min(
        counts.items(), key=lambda item: (-item[1], abs(item[0]), item[0]))
    stability = count / len(values)
    stable = (len(values) >= min_valid_projections and
              stability >= min_projection_stability)
    return LinkingResult(
        linking_number=int(linking_number),
        absolute_linking_number=abs(int(linking_number)),
        projection_values=tuple(int(value) for value in values),
        projections_requested=int(n_projections),
        projections_valid=len(values),
        projection_stability=float(stability),
        stable=bool(stable),
        min_segment_distance_nm=float(distance),
    )


def evaluate_link(
        loop_a_xyz_nm: Sequence[Sequence[float]],
        loop_b_xyz_nm: Sequence[Sequence[float]],
        **kwargs) -> LinkingResult:
    """Public workflow name for :func:`compute_linking_number`."""
    return compute_linking_number(loop_a_xyz_nm, loop_b_xyz_nm, **kwargs)


__all__ = [
    "LinkingResult",
    "ThetaEvaluation",
    "TopologyValidationError",
    "YamadaResult",
    "compute_linking_number",
    "compute_yamada",
    "deterministic_projection_directions",
    "evaluate_link",
    "evaluate_theta",
    "load_yamada",
    "min_interloop_segment_distance",
    "validate_link_loops",
    "validate_theta_strands",
]
