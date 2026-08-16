#!/usr/bin/env python3
"""Per-cell axial support for 2.5D segmentation.

The 2D label image is the authoritative XY ownership map.  This module only
estimates the inclusive z interval over which each label is supported at each
DAPI time point.  It deliberately has no image-reader or Cellpose dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


SCHEMA_VERSION = "1.0"
ALGORITHM_VERSION = "1.0"

SMOOTH_SIGMA_PLANES = 1.0
BASELINE_PERCENTILE = 10.0
ACTIVE_THRESHOLD = 0.20
MIN_GATE_WIDTH_PLANES = 3
GATE_PADDING_PLANES = 1
AMBIGUOUS_MIN_WIDTH_PLANES = 2
AMBIGUOUS_SIGNAL_RATIO = 0.50
WEAK_RELATIVE_EPSILON = 1e-6

STATUS_GOOD = "good"
STATUS_EDGE_CLIPPED = "edge_clipped"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_WEAK = "weak"
STATUS_NONFINITE = "nonfinite"

FALLBACK_NONE = "none"
FALLBACK_CONSENSUS = "consensus"
FALLBACK_FULL_Z = "full_z"
FALLBACK_DISALLOWED = "disallowed"

FAILURE_POLICIES = frozenset(("quarantine", "fail"))
VALID_STATUSES = frozenset(
    (
        STATUS_GOOD,
        STATUS_EDGE_CLIPPED,
        STATUS_AMBIGUOUS,
        STATUS_WEAK,
        STATUS_NONFINITE,
    )
)
VALID_FALLBACKS = frozenset(
    (FALLBACK_NONE, FALLBACK_CONSENSUS, FALLBACK_FULL_Z, FALLBACK_DISALLOWED)
)
# Renamed to carry its subject in its name. This is the Z-GATE vocabulary -- was
# the DAPI z interval usable for this cell/TP -- which is a completely different
# question from the identically-named set in replication_qc, which asks whether
# the one-vs-two-spot test was answered. One name for two vocabularies invited
# exactly that conflation.
Z_GATE_EVALUABLE_STATUSES = frozenset((STATUS_GOOD, STATUS_EDGE_CLIPPED))


def _unicode_array(values, width=32):
    return np.asarray(values, dtype=f"<U{int(width)}")


@dataclass(frozen=True)
class ZSupportTimepoint:
    """One exact time point selected from :class:`ZSupport`."""

    tp_name: str
    label_ids: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    status: np.ndarray
    quarantined: np.ndarray
    evaluable: np.ndarray
    fit_allowed: np.ndarray
    fallback: np.ndarray
    peak_plane: np.ndarray
    peak_strength: np.ndarray
    primary_signal: np.ndarray
    competing_signal_ratio: np.ndarray
    edge_clipped: np.ndarray
    nz: int
    mask_shape_yx: Tuple[int, int]
    failure_policy: str


@dataclass(frozen=True)
class ZSupport:
    """Versioned time-point-by-label z-support table."""

    tp_names: np.ndarray
    label_ids: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    status: np.ndarray
    quarantined: np.ndarray
    evaluable: np.ndarray
    fit_allowed: np.ndarray
    fallback: np.ndarray
    peak_plane: np.ndarray
    peak_strength: np.ndarray
    primary_signal: np.ndarray
    competing_signal_ratio: np.ndarray
    edge_clipped: np.ndarray
    nz: int
    mask_shape_yx: Tuple[int, int]
    failure_policy: str
    mask_sha256: str
    schema_version: str = SCHEMA_VERSION
    algorithm_version: str = ALGORITHM_VERSION

    def select_tp(self, tp_name: str) -> ZSupportTimepoint:
        """Select one time point by exact name; positional guessing is forbidden."""
        matches = np.flatnonzero(self.tp_names == tp_name)
        if matches.size != 1:
            if matches.size == 0:
                raise KeyError(f"z-support has no exact time point named {tp_name!r}")
            raise ValueError(f"z-support contains duplicate time point {tp_name!r}")
        i = int(matches[0])
        return ZSupportTimepoint(
            tp_name=str(self.tp_names[i]),
            label_ids=self.label_ids,
            z_min=self.z_min[i],
            z_max=self.z_max[i],
            status=self.status[i],
            quarantined=self.quarantined[i],
            evaluable=self.evaluable[i],
            fit_allowed=self.fit_allowed[i],
            fallback=self.fallback[i],
            peak_plane=self.peak_plane[i],
            peak_strength=self.peak_strength[i],
            primary_signal=self.primary_signal[i],
            competing_signal_ratio=self.competing_signal_ratio[i],
            edge_clipped=self.edge_clipped[i],
            nz=self.nz,
            mask_shape_yx=self.mask_shape_yx,
            failure_policy=self.failure_policy,
        )


@dataclass(frozen=True)
class _RawGate:
    z_min: int
    z_max: int
    status: str
    peak_plane: int
    peak_strength: float
    primary_signal: float
    competing_signal_ratio: float
    edge_clipped: bool


def _validate_labels_2d(labels_2d):
    labels = np.asarray(labels_2d)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label image, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"label image must be integer-valued, got {labels.dtype}")
    if np.any(labels < 0):
        raise ValueError("label image cannot contain negative values")
    return labels


def _mask_hash(labels_2d):
    canonical = np.ascontiguousarray(labels_2d, dtype="<i8")
    digest = sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _fill_single_plane_gaps(active):
    active = np.asarray(active, dtype=bool).copy()
    if active.size >= 3:
        gaps = (~active[1:-1]) & active[:-2] & active[2:]
        active[1:-1][gaps] = True
    return active


def _components(active):
    padded = np.pad(np.asarray(active, dtype=np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    return list(zip(starts.tolist(), stops.tolist()))


def _expand_interval(start, stop, minimum_width, nz):
    width = stop - start + 1
    missing = max(0, min(int(minimum_width), nz) - width)
    add_before = missing // 2
    add_after = missing - add_before
    start -= add_before
    stop += add_after
    if start < 0:
        stop = min(nz - 1, stop - start)
        start = 0
    if stop >= nz:
        start = max(0, start - (stop - nz + 1))
        stop = nz - 1
    return int(start), int(stop)


def _derive_gate(profile):
    profile = np.asarray(profile, dtype=np.float64)
    nz = int(profile.size)
    if nz < 1:
        raise ValueError("a z profile must contain at least one plane")
    if not np.all(np.isfinite(profile)):
        return _RawGate(0, nz - 1, STATUS_NONFINITE, -1, np.nan, 0.0, np.nan, False)

    smoothed = gaussian_filter1d(profile, SMOOTH_SIGMA_PLANES, mode="nearest")
    baseline = float(np.percentile(smoothed, BASELINE_PERCENTILE))
    corrected = np.maximum(smoothed - baseline, 0.0)
    peak_plane = int(np.argmax(corrected))
    peak_strength = float(corrected[peak_plane])
    scale = max(1.0, float(np.max(np.abs(smoothed))))
    if not np.isfinite(peak_strength) or peak_strength <= WEAK_RELATIVE_EPSILON * scale:
        return _RawGate(0, nz - 1, STATUS_WEAK, peak_plane, peak_strength, 0.0, 0.0, False)

    normalized = corrected / peak_strength
    active = _fill_single_plane_gaps(normalized >= ACTIVE_THRESHOLD)
    components = _components(active)
    primary_index = next(
        i for i, (start, stop) in enumerate(components) if start <= peak_plane <= stop
    )
    primary_start, primary_stop = components[primary_index]
    primary_signal = float(corrected[primary_start : primary_stop + 1].sum())

    competitor_signals = []
    for i, (start, stop) in enumerate(components):
        if i == primary_index or stop - start + 1 < AMBIGUOUS_MIN_WIDTH_PLANES:
            continue
        competitor_signals.append(float(corrected[start : stop + 1].sum()))
    strongest_competitor = max(competitor_signals, default=0.0)
    competitor_ratio = strongest_competitor / primary_signal if primary_signal > 0 else np.inf
    ambiguous = competitor_ratio >= AMBIGUOUS_SIGNAL_RATIO

    start, stop = _expand_interval(
        primary_start, primary_stop, MIN_GATE_WIDTH_PLANES, nz
    )
    unclipped_start = start - GATE_PADDING_PLANES
    unclipped_stop = stop + GATE_PADDING_PLANES
    edge_clipped = unclipped_start < 0 or unclipped_stop >= nz
    start = max(0, unclipped_start)
    stop = min(nz - 1, unclipped_stop)
    status = (
        STATUS_AMBIGUOUS
        if ambiguous
        else STATUS_EDGE_CLIPPED
        if edge_clipped
        else STATUS_GOOD
    )
    return _RawGate(
        start,
        stop,
        status,
        peak_plane,
        peak_strength,
        primary_signal,
        float(competitor_ratio),
        edge_clipped,
    )


def _profiles_for_stack(labels, label_ids, stack):
    stack = np.asarray(stack)
    expected_yx = tuple(int(v) for v in labels.shape)
    if stack.ndim != 3 or tuple(stack.shape[1:]) != expected_yx:
        raise ValueError(
            "DAPI stack must have shape (z, y, x) matching labels; "
            f"got {stack.shape} for labels {labels.shape}"
        )
    if not np.issubdtype(stack.dtype, np.number):
        raise ValueError(f"DAPI stack must be numeric, got {stack.dtype}")
    nz = int(stack.shape[0])
    if nz < 1:
        raise ValueError("DAPI stack must contain at least one z plane")

    flat_labels = labels.ravel()
    background_mask = flat_labels == 0
    counts = np.bincount(flat_labels.astype(np.int64, copy=False))
    label_counts = counts[label_ids]
    profiles = np.empty((label_ids.size, nz), dtype=np.float64)
    backgrounds = np.empty(nz, dtype=np.float64)
    max_label = int(label_ids.max()) if label_ids.size else 0
    for z in range(nz):
        plane = np.asarray(stack[z], dtype=np.float64).ravel()
        if np.any(background_mask):
            backgrounds[z] = float(np.median(plane[background_mask]))
        else:
            backgrounds[z] = np.nan
        if label_ids.size:
            sums = np.bincount(flat_labels, weights=plane, minlength=max_label + 1)
            profiles[:, z] = sums[label_ids] / label_counts
    profiles -= backgrounds[None, :]
    return profiles


def compute_z_support(
    labels_2d,
    timepoint_stacks: Iterable[Tuple[str, np.ndarray]],
    failure_policy="quarantine",
):
    """Compute effective z support for every label at every DAPI time point.

    ``timepoint_stacks`` must yield exact ``(tp_name, stack_zyx)`` pairs.  Weak,
    non-finite, and ambiguous gates are either quarantined with a per-label
    consensus/full-z fallback or marked as disallowed, according to
    ``failure_policy``.
    """
    labels = _validate_labels_2d(labels_2d)
    if failure_policy not in FAILURE_POLICIES:
        raise ValueError(
            f"failure_policy must be one of {sorted(FAILURE_POLICIES)}, got {failure_policy!r}"
        )
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0].astype(np.int64, copy=False)

    names = []
    profiles_by_tp = []
    nz = None
    for item in timepoint_stacks:
        try:
            tp_name, stack = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timepoint_stacks must yield (exact_name, stack_zyx) pairs"
            ) from exc
        if not isinstance(tp_name, str) or not tp_name:
            raise ValueError(f"time point name must be a non-empty string, got {tp_name!r}")
        if tp_name in names:
            raise ValueError(f"duplicate time point name {tp_name!r}")
        stack = np.asarray(stack)
        if nz is None:
            if stack.ndim < 1:
                raise ValueError(f"DAPI stack must be 3D, got shape {stack.shape}")
            nz = int(stack.shape[0])
        elif stack.ndim < 1 or int(stack.shape[0]) != nz:
            raise ValueError(
                f"all DAPI stacks must have the same z depth {nz}; got shape {stack.shape}"
            )
        profiles_by_tp.append(_profiles_for_stack(labels, label_ids, stack))
        names.append(tp_name)
    if not names:
        raise ValueError("at least one DAPI time point is required")

    nt, nl = len(names), int(label_ids.size)
    shape = (nt, nl)
    z_min = np.zeros(shape, dtype=np.int32)
    z_max = np.full(shape, nz - 1, dtype=np.int32)
    status = np.full(shape, STATUS_WEAK, dtype="<U16")
    fallback = np.full(shape, FALLBACK_NONE, dtype="<U16")
    quarantined = np.zeros(shape, dtype=bool)
    evaluable = np.zeros(shape, dtype=bool)
    fit_allowed = np.ones(shape, dtype=bool)
    peak_plane = np.full(shape, -1, dtype=np.int32)
    peak_strength = np.full(shape, np.nan, dtype=np.float64)
    primary_signal = np.zeros(shape, dtype=np.float64)
    competing_signal_ratio = np.full(shape, np.nan, dtype=np.float64)
    edge_clipped = np.zeros(shape, dtype=bool)

    for t, profiles in enumerate(profiles_by_tp):
        for j in range(nl):
            gate = _derive_gate(profiles[j])
            z_min[t, j] = gate.z_min
            z_max[t, j] = gate.z_max
            status[t, j] = gate.status
            peak_plane[t, j] = gate.peak_plane
            peak_strength[t, j] = gate.peak_strength
            primary_signal[t, j] = gate.primary_signal
            competing_signal_ratio[t, j] = gate.competing_signal_ratio
            edge_clipped[t, j] = gate.edge_clipped
            evaluable[t, j] = gate.status in Z_GATE_EVALUABLE_STATUSES

    for j in range(nl):
        valid = evaluable[:, j]
        invalid = ~valid
        if not np.any(invalid):
            continue
        if failure_policy == "fail":
            fit_allowed[invalid, j] = False
            fallback[invalid, j] = FALLBACK_DISALLOWED
            continue
        quarantined[invalid, j] = True
        if np.any(valid):
            consensus_min = int(np.rint(np.median(z_min[valid, j])))
            consensus_max = int(np.rint(np.median(z_max[valid, j])))
            z_min[invalid, j] = consensus_min
            z_max[invalid, j] = consensus_max
            fallback[invalid, j] = FALLBACK_CONSENSUS
        else:
            z_min[invalid, j] = 0
            z_max[invalid, j] = nz - 1
            fallback[invalid, j] = FALLBACK_FULL_Z

    support = ZSupport(
        tp_names=_unicode_array(names, max(1, max(map(len, names)))),
        label_ids=label_ids,
        z_min=z_min,
        z_max=z_max,
        status=status,
        quarantined=quarantined,
        evaluable=evaluable,
        fit_allowed=fit_allowed,
        fallback=fallback,
        peak_plane=peak_plane,
        peak_strength=peak_strength,
        primary_signal=primary_signal,
        competing_signal_ratio=competing_signal_ratio,
        edge_clipped=edge_clipped,
        nz=int(nz),
        mask_shape_yx=tuple(int(v) for v in labels.shape),
        failure_policy=str(failure_policy),
        mask_sha256=_mask_hash(labels),
    )
    _validate_support(support, labels)
    return support


def _validate_support(
    support,
    labels_2d,
    expected_tp_names: Optional[Sequence[str]] = None,
    expected_nz: Optional[int] = None,
    expected_failure_policy: Optional[str] = None,
):
    labels = _validate_labels_2d(labels_2d)
    if support.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported z-support schema {support.schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    if support.failure_policy not in FAILURE_POLICIES:
        raise ValueError(f"invalid z-support failure policy {support.failure_policy!r}")
    if expected_failure_policy is not None and support.failure_policy != expected_failure_policy:
        raise ValueError(
            "z-support failure policy mismatch: "
            f"expected {expected_failure_policy!r}, got {support.failure_policy!r}"
        )
    tp_name_array = np.asarray(support.tp_names)
    if tp_name_array.ndim != 1 or tp_name_array.dtype.kind != "U":
        raise ValueError("z-support tp_names must be a one-dimensional Unicode array")
    tp_names = [str(v) for v in tp_name_array.tolist()]
    if not tp_names or any(not name for name in tp_names):
        raise ValueError("z-support must contain non-empty time point names")
    if len(set(tp_names)) != len(tp_names):
        raise ValueError("z-support contains duplicate time point names")
    if expected_tp_names is not None:
        expected = list(expected_tp_names)
        if len(set(expected)) != len(expected):
            raise ValueError("expected time point names contain duplicates")
        if tp_names != expected:
            raise ValueError(
                f"z-support time points mismatch: expected {expected!r}, got {tp_names!r}"
            )

    label_ids = np.asarray(support.label_ids)
    if label_ids.ndim != 1 or not np.issubdtype(label_ids.dtype, np.integer):
        raise ValueError("z-support label_ids must be a one-dimensional integer array")
    if label_ids.size and (np.any(label_ids <= 0) or np.unique(label_ids).size != label_ids.size):
        raise ValueError("z-support label IDs must be unique and positive")
    expected_labels = np.unique(labels)
    expected_labels = expected_labels[expected_labels > 0].astype(np.int64, copy=False)
    if not np.array_equal(label_ids, expected_labels):
        raise ValueError(
            f"z-support label identity mismatch: expected {expected_labels.tolist()}, "
            f"got {label_ids.tolist()}"
        )
    if tuple(support.mask_shape_yx) != tuple(labels.shape):
        raise ValueError(
            f"z-support mask shape mismatch: expected {labels.shape}, "
            f"got {support.mask_shape_yx}"
        )
    if support.mask_sha256 != _mask_hash(labels):
        raise ValueError("z-support XY ownership map hash does not match labels_2d")

    nz = int(support.nz)
    if nz < 1:
        raise ValueError(f"z-support nz must be positive, got {nz}")
    if expected_nz is not None and nz != int(expected_nz):
        raise ValueError(f"z-support z depth mismatch: expected {expected_nz}, got {nz}")
    shape = (len(tp_names), label_ids.size)
    required_shapes = {
        "z_min": support.z_min,
        "z_max": support.z_max,
        "status": support.status,
        "quarantined": support.quarantined,
        "evaluable": support.evaluable,
        "fit_allowed": support.fit_allowed,
        "fallback": support.fallback,
        "peak_plane": support.peak_plane,
        "peak_strength": support.peak_strength,
        "primary_signal": support.primary_signal,
        "competing_signal_ratio": support.competing_signal_ratio,
        "edge_clipped": support.edge_clipped,
    }
    for name, values in required_shapes.items():
        if np.asarray(values).shape != shape:
            raise ValueError(
                f"z-support {name} has shape {np.asarray(values).shape}; expected {shape}"
            )
    for name in ("z_min", "z_max", "peak_plane"):
        if not np.issubdtype(np.asarray(required_shapes[name]).dtype, np.integer):
            raise ValueError(f"z-support {name} must be integer-valued")
    for name in ("quarantined", "evaluable", "fit_allowed", "edge_clipped"):
        if np.asarray(required_shapes[name]).dtype.kind != "b":
            raise ValueError(f"z-support {name} must be a boolean array")
    for name in ("status", "fallback"):
        if np.asarray(required_shapes[name]).dtype.kind != "U":
            raise ValueError(f"z-support {name} must be a Unicode array")
    if np.any(support.z_min < 0) or np.any(support.z_max >= nz) or np.any(
        support.z_min > support.z_max
    ):
        raise ValueError("z-support contains invalid or out-of-range z bounds")
    if not set(np.asarray(support.status).ravel().tolist()) <= VALID_STATUSES:
        raise ValueError("z-support contains an unknown status code")
    if not set(np.asarray(support.fallback).ravel().tolist()) <= VALID_FALLBACKS:
        raise ValueError("z-support contains an unknown fallback code")
    expected_eval = np.isin(support.status, tuple(Z_GATE_EVALUABLE_STATUSES))
    if not np.array_equal(np.asarray(support.evaluable, dtype=bool), expected_eval):
        raise ValueError("z-support evaluable flags disagree with status codes")
    if np.any(np.asarray(support.quarantined) & np.asarray(support.evaluable)):
        raise ValueError("an evaluable z gate cannot be quarantined")
    invalid = ~np.asarray(support.evaluable)
    if support.failure_policy == "quarantine":
        if np.any(~np.asarray(support.fit_allowed)) or np.any(
            invalid & ~np.asarray(support.quarantined)
        ):
            raise ValueError("quarantine-policy flags are internally inconsistent")
    elif np.any(invalid & np.asarray(support.fit_allowed)) or np.any(
        np.asarray(support.quarantined)
    ):
        raise ValueError("fail-policy flags are internally inconsistent")
    return support


def save_z_support(
    support,
    npz_path,
    json_path,
    provenance: Optional[Mapping] = None,
):
    """Save a pickle-free NPZ sidecar and a human-readable JSON summary."""
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": _unicode_array([support.schema_version])[0],
        "algorithm_version": _unicode_array([support.algorithm_version])[0],
        "tp_names": np.asarray(support.tp_names),
        "label_ids": np.asarray(support.label_ids),
        "z_min": np.asarray(support.z_min),
        "z_max": np.asarray(support.z_max),
        "status": np.asarray(support.status),
        "quarantined": np.asarray(support.quarantined),
        "evaluable": np.asarray(support.evaluable),
        "fit_allowed": np.asarray(support.fit_allowed),
        "fallback": np.asarray(support.fallback),
        "peak_plane": np.asarray(support.peak_plane),
        "peak_strength": np.asarray(support.peak_strength),
        "primary_signal": np.asarray(support.primary_signal),
        "competing_signal_ratio": np.asarray(support.competing_signal_ratio),
        "edge_clipped": np.asarray(support.edge_clipped),
        "nz": np.asarray(support.nz, dtype=np.int32),
        "mask_shape_yx": np.asarray(support.mask_shape_yx, dtype=np.int64),
        "failure_policy": _unicode_array([support.failure_policy])[0],
        "mask_sha256": _unicode_array([support.mask_sha256], 64)[0],
    }
    with npz_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    status_values, status_counts = np.unique(support.status, return_counts=True)
    fallback_values, fallback_counts = np.unique(support.fallback, return_counts=True)
    summary = {
        "schema_version": support.schema_version,
        "algorithm_version": support.algorithm_version,
        "failure_policy": support.failure_policy,
        "time_points": [str(v) for v in support.tp_names.tolist()],
        "label_ids": [int(v) for v in support.label_ids.tolist()],
        "n_time_points": int(support.tp_names.size),
        "n_labels": int(support.label_ids.size),
        "stack_shape_zyx": [int(support.nz), *map(int, support.mask_shape_yx)],
        "mask_sha256": support.mask_sha256,
        "gate_parameters": {
            "smooth_sigma_planes": SMOOTH_SIGMA_PLANES,
            "baseline_percentile": BASELINE_PERCENTILE,
            "active_threshold": ACTIVE_THRESHOLD,
            "fill_internal_gap_planes": 1,
            "minimum_gate_width_planes": MIN_GATE_WIDTH_PLANES,
            "padding_planes": GATE_PADDING_PLANES,
            "ambiguous_min_component_width_planes": AMBIGUOUS_MIN_WIDTH_PLANES,
            "ambiguous_signal_ratio": AMBIGUOUS_SIGNAL_RATIO,
            "weak_relative_epsilon": WEAK_RELATIVE_EPSILON,
        },
        "status_counts": {
            str(value): int(count)
            for value, count in zip(status_values.tolist(), status_counts.tolist())
        },
        "fallback_counts": {
            str(value): int(count)
            for value, count in zip(fallback_values.tolist(), fallback_counts.tolist())
        },
        "quarantined_count": int(np.count_nonzero(support.quarantined)),
        "evaluable_count": int(np.count_nonzero(support.evaluable)),
        "fit_allowed_count": int(np.count_nonzero(support.fit_allowed)),
        "provenance": dict(provenance or {}),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _load_scalar(npz, name, cast):
    values = np.asarray(npz[name])
    if values.ndim != 0:
        raise ValueError(f"z-support {name} must be a scalar")
    return cast(values.item())


def load_z_support(
    npz_path,
    labels_2d,
    expected_tp_names=None,
    expected_nz=None,
    expected_failure_policy=None,
):
    """Load and strictly validate a z-support NPZ without enabling pickle."""
    required = {
        "schema_version",
        "algorithm_version",
        "tp_names",
        "label_ids",
        "z_min",
        "z_max",
        "status",
        "quarantined",
        "evaluable",
        "fit_allowed",
        "fallback",
        "peak_plane",
        "peak_strength",
        "primary_signal",
        "competing_signal_ratio",
        "edge_clipped",
        "nz",
        "mask_shape_yx",
        "failure_policy",
        "mask_sha256",
    }
    with np.load(npz_path, allow_pickle=False) as npz:
        missing = sorted(required - set(npz.files))
        if missing:
            raise ValueError(f"z-support sidecar is missing fields: {', '.join(missing)}")
        for name in ("schema_version", "algorithm_version", "failure_policy", "mask_sha256"):
            if np.asarray(npz[name]).dtype.kind != "U":
                raise ValueError(f"z-support {name} must be a Unicode scalar")
        for name in ("tp_names", "status", "fallback"):
            if np.asarray(npz[name]).dtype.kind != "U":
                raise ValueError(f"z-support {name} must be a Unicode array")
        mask_shape = np.asarray(npz["mask_shape_yx"])
        if mask_shape.shape != (2,):
            raise ValueError("z-support mask_shape_yx must have shape (2,)")
        support = ZSupport(
            tp_names=np.asarray(npz["tp_names"]).copy(),
            label_ids=np.asarray(npz["label_ids"]).copy(),
            z_min=np.asarray(npz["z_min"]).copy(),
            z_max=np.asarray(npz["z_max"]).copy(),
            status=np.asarray(npz["status"]).copy(),
            quarantined=np.asarray(npz["quarantined"]).copy(),
            evaluable=np.asarray(npz["evaluable"]).copy(),
            fit_allowed=np.asarray(npz["fit_allowed"]).copy(),
            fallback=np.asarray(npz["fallback"]).copy(),
            peak_plane=np.asarray(npz["peak_plane"]).copy(),
            peak_strength=np.asarray(npz["peak_strength"]).copy(),
            primary_signal=np.asarray(npz["primary_signal"]).copy(),
            competing_signal_ratio=np.asarray(npz["competing_signal_ratio"]).copy(),
            edge_clipped=np.asarray(npz["edge_clipped"]).copy(),
            nz=_load_scalar(npz, "nz", int),
            mask_shape_yx=tuple(int(v) for v in mask_shape.tolist()),
            failure_policy=_load_scalar(npz, "failure_policy", str),
            mask_sha256=_load_scalar(npz, "mask_sha256", str),
            schema_version=_load_scalar(npz, "schema_version", str),
            algorithm_version=_load_scalar(npz, "algorithm_version", str),
        )
    return _validate_support(
        support,
        labels_2d,
        expected_tp_names=expected_tp_names,
        expected_nz=expected_nz,
        expected_failure_policy=expected_failure_policy,
    )


def materialize_labels(labels_2d, nz, z_min, z_max, fit_allowed=None,
                       label_ids=None):
    """Materialize one time point's gated 3D labels without changing XY ownership."""
    labels = _validate_labels_2d(labels_2d)
    nz = int(nz)
    if nz < 1:
        raise ValueError(f"nz must be positive, got {nz}")
    if label_ids is None:
        label_ids = np.unique(labels)
        label_ids = label_ids[label_ids > 0]
    else:
        label_ids = np.asarray(label_ids)
    z_min = np.asarray(z_min)
    z_max = np.asarray(z_max)
    expected_shape = (label_ids.size,)
    if z_min.shape != expected_shape or z_max.shape != expected_shape:
        raise ValueError(
            f"z bounds must have shape {expected_shape}, got {z_min.shape} and {z_max.shape}"
        )
    if not np.issubdtype(z_min.dtype, np.integer) or not np.issubdtype(
        z_max.dtype, np.integer
    ):
        raise ValueError("z bounds must be integer-valued")
    if np.any(z_min < 0) or np.any(z_max >= nz) or np.any(z_min > z_max):
        raise ValueError("z bounds are invalid or outside the requested stack depth")
    if fit_allowed is None:
        fit_allowed = np.ones(expected_shape, dtype=bool)
    else:
        fit_allowed = np.asarray(fit_allowed)
        if fit_allowed.shape != expected_shape:
            raise ValueError(
                f"fit_allowed must have shape {expected_shape}, got {fit_allowed.shape}"
            )
        if fit_allowed.dtype.kind != "b":
            raise ValueError("fit_allowed must be a boolean array")

    # Map every foreground pixel to its row in the sorted label axis once.
    # Broadcasting the resulting per-pixel bounds is O(nz * y * x), whereas
    # repeatedly evaluating ``labels == label_id`` is O(nlabels * y * x).
    foreground = labels > 0
    label_index = np.zeros(labels.shape, dtype=np.int64)
    label_index[foreground] = np.searchsorted(label_ids, labels[foreground])
    lo_by_pixel = np.zeros(labels.shape, dtype=np.int64)
    hi_by_pixel = np.full(labels.shape, -1, dtype=np.int64)
    allowed_by_pixel = np.zeros(labels.shape, dtype=bool)
    lo_by_pixel[foreground] = z_min[label_index[foreground]]
    hi_by_pixel[foreground] = z_max[label_index[foreground]]
    allowed_by_pixel[foreground] = fit_allowed[label_index[foreground]]
    z = np.arange(nz, dtype=np.int64)[:, None, None]
    active = (
        foreground[None, :, :]
        & allowed_by_pixel[None, :, :]
        & (z >= lo_by_pixel[None, :, :])
        & (z <= hi_by_pixel[None, :, :])
    )
    return np.where(active, labels[None, :, :], 0).astype(labels.dtype, copy=False)
