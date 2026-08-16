#!/usr/bin/env python3
"""Chromatic channel registration for the widefield sequential-FISH pipeline.

Scope is deliberately narrow: read the acquisition's own Vutara ``widefield.json``
chromatic calibration and represent the DAPI (405) label mask on another
channel's pixel grid.  This is not a general image-registration framework; it
does one composition and one categorical resample.

Convention, verified against rep3 Location-01 before adoption
-------------------------------------------------------------
``ChromaticOffset.Transforms[i]`` is a ``PlaneTransformation`` whose
``Matrix``/``Matrix1x`` entries form an affine acting on **calibrated physical XY
coordinates in nanometres**, mapping the common/reference plane INTO channel
``i``'s raw plane.  Translations are physical, not pixel, values.

Transform index ``i`` is bound to a wavelength through
``Configuration.Probes[i].WidefieldLaser`` -- never by assuming an order.  On
rep3 that yields ``['640', '488', '405']`` with ``T_640`` exactly identity, and
``RMSErrors[0] == 0``, both consistent with 640 being the reference plane.

To express the DAPI mask on target-channel grid ``c``, each target pixel is
inverse-mapped into the DAPI image:

    p_405 = T_405 @ inverse(T_c) @ p_c

Measured on rep3 Location-01, 22,511 sampled one-spot fits, fraction of fitted
640 centres inside their own DAPI label:

    unregistered                              71.38 %
    p_405 = T_405 @ inv(T_640) @ p_640        81.89 %   <- adopted
    inverse direction                         42.58 %
    translations read as pixels                0.00 %

The wrong direction and the pixel reading are both catastrophic, so the
convention is not merely plausible but discriminated by the data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

PROVENANCE_SCHEMA = "chromatic_registration_v1"
TRANSFORM_DIRECTION = "reference_plane_to_channel_plane"
INTERPOLATION = "nearest_neighbour_categorical"

#: Pipeline probe directory index -> Vutara ``WidefieldLaser`` label, WIDEFIELD
#: ONLY. Kept for backward compatibility; do NOT use it to infer a wavelength
#: from a probe index. Confocal probe_0 is 561 nm, not 640 -- that is precisely
#: the assumption this mapping used to bake in wrongly.
PROBE_WAVELENGTH = {0: "640", 1: "488", 2: "405"}

#: Segmentation/reference channel. Same for every modality here (DAPI).
DAPI_WAVELENGTH = "405"

#: Per-modality defaults. Callers pass wavelengths EXPLICITLY; these only supply
#: a default when a CLI flag is omitted.
MODALITY_DEFAULTS = {
    "widefield": {"reference": "405", "probe": "640", "fiducial": "488"},
    "confocal":  {"reference": "405", "probe": "561", "fiducial": "488"},
}


def modality_default(modality: str, role: str) -> str:
    """Default wavelength for (modality, role), e.g. ('confocal', 'probe')."""
    try:
        return MODALITY_DEFAULTS[modality][role]
    except KeyError:
        raise ChromaticRegistrationError(
            f"no default wavelength for modality={modality!r} role={role!r}; "
            f"known modalities {sorted(MODALITY_DEFAULTS)}, "
            f"roles {sorted(next(iter(MODALITY_DEFAULTS.values())))}") from None

#: A non-zero axial chromatic offset would need a z-dependent correction that
#: this helper does not implement; it must fail loudly rather than be ignored.
Z_OFFSET_TOLERANCE_NM = 1e-9


class ChromaticRegistrationError(ValueError):
    """Raised when calibration metadata is missing, malformed, or unsupported."""


@dataclass(frozen=True)
class ChromaticCalibration:
    """Parsed, validated chromatic calibration for one acquisition."""

    source_path: str
    pixel_size_nm: tuple[float, float]          # (X, Y) from the metadata itself
    matrices: Mapping[str, np.ndarray]          # wavelength -> 3x3, physical nm
    z_offsets_nm: Mapping[str, float]
    rms_errors_nm: Mapping[str, float]
    transform_index: Mapping[str, int]          # wavelength -> Transforms index

    def matrix(self, wavelength: str) -> np.ndarray:
        try:
            return self.matrices[str(wavelength)]
        except KeyError:
            raise ChromaticRegistrationError(
                f"no chromatic transform for wavelength {wavelength!r}; "
                f"metadata provides {sorted(self.matrices)}") from None


def _require(node, key, path):
    if not isinstance(node, dict) or key not in node:
        raise ChromaticRegistrationError(f"{path}: missing {key!r}")
    return node[key]


def _unwrap(node, key, path):
    """Vutara JSON wraps payloads as {"type": ..., "value": ...}."""
    return _require(_require(node, key, path), "value", f"{path}.{key}")


def load_calibration(widefield_json_path: str | os.PathLike[str]) -> ChromaticCalibration:
    """Parse and validate ``widefield.json``.

    Fails clearly rather than degrading to identity: a caller that asked for
    metadata registration must not silently get an unregistered result.
    """
    path = os.fspath(widefield_json_path)
    if not os.path.isfile(path):
        raise ChromaticRegistrationError(
            f"chromatic registration requested but metadata is absent: {path}")
    try:
        with open(path, encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChromaticRegistrationError(
            f"chromatic metadata unreadable: {path}: {exc}") from exc

    root = raw.get("value", raw)
    cfg = _require(root, "Configuration", path)
    chrom = _require(root, "ChromaticOffset", path)

    try:
        px_x = float(_require(cfg, "PixelSizeX", f"{path}.Configuration"))
        px_y = float(_require(cfg, "PixelSizeY", f"{path}.Configuration"))
    except (TypeError, ValueError) as exc:
        raise ChromaticRegistrationError(
            f"{path}: non-numeric Configuration pixel size") from exc
    if not (np.isfinite(px_x) and np.isfinite(px_y)) or px_x <= 0 or px_y <= 0:
        raise ChromaticRegistrationError(
            f"{path}: invalid pixel size ({px_x}, {px_y})")

    probes = _unwrap(cfg, "Probes", f"{path}.Configuration")
    transforms = _unwrap(chrom, "Transforms", f"{path}.ChromaticOffset")
    if not isinstance(probes, list) or not isinstance(transforms, list):
        raise ChromaticRegistrationError(f"{path}: Probes/Transforms are not lists")
    if len(probes) != len(transforms):
        raise ChromaticRegistrationError(
            f"{path}: {len(probes)} probes but {len(transforms)} chromatic "
            "transforms; cannot bind transforms to wavelengths")

    # ZOffsets: if the field is present at all it must cover EVERY referenced
    # plane. Silently substituting 0.0 for a missing entry would turn an
    # unspecified axial offset into an asserted zero -- exactly the class of
    # error this module is meant to fail on. A wholly absent field is the
    # documented "no axial calibration" case and is treated as all-zero.
    z_present = isinstance(chrom, dict) and "ZOffsets" in chrom
    if z_present:
        z_raw = _unwrap(chrom, "ZOffsets", f"{path}.ChromaticOffset")
        if z_raw is None:
            z_present = False
            z_raw = []
        elif not isinstance(z_raw, list):
            raise ChromaticRegistrationError(
                f"{path}: ChromaticOffset.ZOffsets is {type(z_raw).__name__}, "
                "expected a list")
        elif len(z_raw) != len(transforms):
            raise ChromaticRegistrationError(
                f"{path}: ChromaticOffset.ZOffsets has {len(z_raw)} entries for "
                f"{len(transforms)} transforms. An incomplete axial calibration "
                "cannot be completed with zeros; refusing to guess.")
    else:
        z_raw = []
    rms_raw = _unwrap(chrom, "RMSErrors", f"{path}.ChromaticOffset")
    if rms_raw is None or not isinstance(rms_raw, list):
        rms_raw = []

    matrices: dict[str, np.ndarray] = {}
    z_offsets: dict[str, float] = {}
    rms: dict[str, float] = {}
    index: dict[str, int] = {}

    for i, (probe, entry) in enumerate(zip(probes, transforms)):
        pv = _require(probe, "value", f"{path}.Probes[{i}]")
        laser = pv.get("WidefieldLaser")
        if laser is None or str(laser).strip() in ("", "None"):
            raise ChromaticRegistrationError(
                f"{path}: Probes[{i}] has no usable WidefieldLaser; cannot map "
                "the transform to a wavelength")
        laser = str(laser).strip()
        if laser in matrices:
            raise ChromaticRegistrationError(
                f"{path}: wavelength {laser!r} appears more than once in Probes")

        tv = _require(entry, "value", f"{path}.Transforms[{i}]")
        try:
            m = np.array(
                [[float(tv["Matrix00"]), float(tv["Matrix01"]), float(tv["Matrix02"])],
                 [float(tv["Matrix10"]), float(tv["Matrix11"]), float(tv["Matrix12"])],
                 [0.0, 0.0, 1.0]], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ChromaticRegistrationError(
                f"{path}: Transforms[{i}] has no usable affine matrix") from exc
        if not np.all(np.isfinite(m)):
            raise ChromaticRegistrationError(
                f"{path}: Transforms[{i}] matrix is not finite")
        if abs(np.linalg.det(m[:2, :2])) < 1e-9:
            raise ChromaticRegistrationError(
                f"{path}: Transforms[{i}] linear part is singular; not invertible")

        nonlinear = tv.get("NonlinearTransform")
        if isinstance(nonlinear, dict):
            nv = nonlinear.get("value", {})
            dim_x, dim_y = int(nv.get("DimX", 0) or 0), int(nv.get("DimY", 0) or 0)
            coeffs = nv.get("Coefficients") or []
            if dim_x or dim_y or len(coeffs):
                raise ChromaticRegistrationError(
                    f"{path}: Transforms[{i}] carries a non-empty nonlinear term "
                    f"(DimX={dim_x}, DimY={dim_y}, {len(coeffs)} coefficients). "
                    "This helper implements the affine part only; refusing to "
                    "silently drop the nonlinear correction.")

        matrices[laser] = m
        index[laser] = i
        if z_present:
            try:
                z_offsets[laser] = float(z_raw[i])
            except (TypeError, ValueError) as exc:
                raise ChromaticRegistrationError(
                    f"{path}: ZOffsets[{i}] is not numeric ({z_raw[i]!r}); a "
                    "malformed axial calibration is a hard failure") from exc
            if not np.isfinite(z_offsets[laser]):
                raise ChromaticRegistrationError(
                    f"{path}: ZOffsets[{i}] is not finite ({z_raw[i]!r})")
        else:
            # Field legitimately absent: no axial calibration exists, which this
            # XY-only helper supports. Recorded as 0.0 and reported in provenance.
            z_offsets[laser] = 0.0
        rms[laser] = float(rms_raw[i]) if i < len(rms_raw) else float("nan")

    offenders = {w: z for w, z in z_offsets.items()
                 if abs(z) > Z_OFFSET_TOLERANCE_NM}
    if offenders:
        raise ChromaticRegistrationError(
            f"{path}: non-zero axial chromatic ZOffsets {offenders}. Correcting "
            "these needs an axial term this helper does not implement; refusing "
            "to ignore them silently.")

    if DAPI_WAVELENGTH not in matrices:
        raise ChromaticRegistrationError(
            f"{path}: no {DAPI_WAVELENGTH} nm (DAPI) transform; cannot register "
            f"the segmentation mask. Present: {sorted(matrices)}")

    return ChromaticCalibration(
        source_path=os.path.abspath(path),
        pixel_size_nm=(px_x, px_y),
        matrices=matrices,
        z_offsets_nm=z_offsets,
        rms_errors_nm=rms,
        transform_index=index,
    )


def lookup_matrix(calib: ChromaticCalibration, target_wavelength: str,
                  reference_wavelength: str = DAPI_WAVELENGTH) -> np.ndarray:
    """``T_ref @ inverse(T_target)``: target-channel XY -> reference XY, physical nm.

    Verified convention (see the module docstring): each ``Transforms[i]`` maps the
    common/reference plane INTO channel ``i``'s raw plane, so to express a
    reference-frame image on the target grid every TARGET pixel is inverse-mapped
    through ``T_target`` and then forward-mapped through ``T_ref``:

        p_reference = T_reference @ inverse(T_target) @ p_target

    Both wavelengths are explicit -- nothing here infers a channel from a probe
    directory index.
    """
    t_target = calib.matrix(target_wavelength)
    t_reference = calib.matrix(reference_wavelength)
    return t_reference @ np.linalg.inv(t_target)


def dapi_lookup_matrix(calib: ChromaticCalibration, target_wavelength: str) -> np.ndarray:
    """Backward-compatible alias: reference is DAPI (405 nm)."""
    return lookup_matrix(calib, target_wavelength, DAPI_WAVELENGTH)


def apply_matrix_px(matrix: np.ndarray, x_px, y_px,
                    pixel_size_nm: tuple[float, float]):
    """Apply a physical-nm affine to pixel coordinates, returning pixels.

    Pixel centres are taken at integer indices (origin at index 0), matching the
    convention validated on rep3.
    """
    px_x, px_y = pixel_size_nm
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    x_nm = x_px * px_x
    y_nm = y_px * px_y
    out_x = matrix[0, 0] * x_nm + matrix[0, 1] * y_nm + matrix[0, 2]
    out_y = matrix[1, 0] * x_nm + matrix[1, 1] * y_nm + matrix[1, 2]
    return out_x / px_x, out_y / px_y


def resample_labels_to_channel(labels: np.ndarray,
                               calib: ChromaticCalibration,
                               target_wavelength: str,
                               reference_wavelength: str = DAPI_WAVELENGTH
                               ) -> np.ndarray:
    """Represent a DAPI-frame label volume on ``target_wavelength``'s pixel grid.

    Strictly categorical: nearest-neighbour index lookup, so label IDs and the 0
    background are preserved exactly and no interpolated value can appear.  XY
    only -- the axial axis is untouched (validated zero ``ZOffsets``).
    """
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3):
        raise ChromaticRegistrationError(
            f"label array must be 2D or 3D, got shape {labels.shape}")

    matrix = lookup_matrix(calib, target_wavelength, reference_wavelength)
    ny, nx = labels.shape[-2:]

    # Inverse-map every TARGET pixel into the DAPI image (a pull resample).
    yy, xx = np.meshgrid(np.arange(ny, dtype=float),
                         np.arange(nx, dtype=float), indexing="ij")
    src_x, src_y = apply_matrix_px(matrix, xx, yy, calib.pixel_size_nm)
    xi = np.rint(src_x).astype(np.intp)
    yi = np.rint(src_y).astype(np.intp)
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    xi_safe = np.where(inside, xi, 0)
    yi_safe = np.where(inside, yi, 0)

    if labels.ndim == 2:
        out = np.where(inside, labels[yi_safe, xi_safe], 0)
        return out.astype(labels.dtype, copy=False)

    out = np.empty_like(labels)
    for z in range(labels.shape[0]):
        plane = labels[z]
        out[z] = np.where(inside, plane[yi_safe, xi_safe], 0)
    return out


def labels_in_probe_frame(labels: np.ndarray,
                          calib: ChromaticCalibration | None,
                          probe_index: int) -> np.ndarray:
    """DAPI-frame labels represented in ``probe_index``'s pixel frame.

    The convenience wrapper the pipeline and every figure should use, so the
    target frame is named by the probe being displayed rather than by a
    wavelength string chosen at the call site.  ``calib=None`` (registration
    off) returns the array unchanged, and the DAPI probe itself is a no-op --
    resampling the mask into 640 and then drawing it over the 405 image would
    misregister it by exactly the correction being applied.
    """
    if calib is None:
        return np.asarray(labels)
    wavelength = PROBE_WAVELENGTH[int(probe_index)]
    if wavelength == DAPI_WAVELENGTH:
        return np.asarray(labels)
    return resample_labels_to_channel(labels, calib, wavelength)


def check_display_frame(image_probe_index: int, mask_probe_index: int,
                        coordinates_drift_corrected: bool) -> None:
    """Refuse a figure that mixes coordinate frames.

    Two mistakes are silent and both look plausible on screen:

    1. drawing a mask registered into one probe's frame over a different
       probe's image;
    2. drawing drift-corrected coordinates on a raw image.  Masks come from the
       time-averaged DAPI stack and images are per-acquisition raw stacks, so
       both live in the RAW acquisition frame.  Drift correction is applied to
       coordinates only -- never to the images -- so a drift-corrected point
       does not belong on a raw image.
    """
    if int(image_probe_index) != int(mask_probe_index):
        raise ChromaticRegistrationError(
            f"frame mismatch: mask is in the probe_{mask_probe_index} pixel "
            f"frame but the image is probe_{image_probe_index}. Register the "
            "mask into the frame of the image being displayed.")
    if coordinates_drift_corrected:
        raise ChromaticRegistrationError(
            "frame mismatch: drift-corrected coordinates cannot be plotted on a "
            "raw image. Use the pre-drift localizer coordinates, which share the "
            "raw acquisition frame with both the image and the mask.")


def is_identity(calib: ChromaticCalibration, target_wavelength: str,
                reference_wavelength: str = DAPI_WAVELENGTH,
                atol: float = 1e-12) -> bool:
    """True when no resample is needed for this reference->target pair."""
    return np.allclose(lookup_matrix(calib, target_wavelength, reference_wavelength),
                       np.eye(3), atol=atol)


def calibration_sha256(path: str | os.PathLike[str]) -> str:
    """Hash of the calibration file, so a provenance record pins its input."""
    import hashlib
    try:
        with open(os.fspath(path), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def provenance(calib: ChromaticCalibration,
               targets: Mapping[str, str] | None = None,
               mode: str = "metadata",
               notes: str | None = None,
               stage: str | None = None,
               reference_wavelength: str = DAPI_WAVELENGTH,
               ownership_policy: str | None = None,
               cli_args: Mapping | None = None) -> dict:
    """Recordable description of exactly what was applied, by which stage.

    ``stage`` is mandatory in practice: each stage writes its own provenance file
    so no stage can overwrite another's record.
    """
    targets = dict(targets or {})
    record = {
        "schema_version": PROVENANCE_SCHEMA,
        "stage": stage,
        "mode": mode,
        "reference_wavelength_nm": reference_wavelength,
        "ownership_policy": ownership_policy,
        "source_calibration_sha256": calibration_sha256(calib.source_path),
        "cli_args": {str(k): str(v) for k, v in dict(cli_args or {}).items()},
        "source_metadata_path": calib.source_path,
        "transform_direction": TRANSFORM_DIRECTION,
        "mask_lookup_composition": "p_405 = T_405 @ inverse(T_target) @ p_target",
        "interpolation": INTERPOLATION,
        "translation_units": "nanometre_physical",
        "pixel_size_nm": {"x": calib.pixel_size_nm[0], "y": calib.pixel_size_nm[1]},
        "pixel_origin": "pixel_centre_at_integer_index_0",
        "channel_map": {
            wavelength: {
                "transform_index": calib.transform_index[wavelength],
                "z_offset_nm": calib.z_offsets_nm[wavelength],
                "rms_error_nm": calib.rms_errors_nm[wavelength],
                "matrix": calib.matrix(wavelength).tolist(),
            }
            for wavelength in sorted(calib.matrices)
        },
        "probe_wavelength_map": {str(k): v for k, v in PROBE_WAVELENGTH.items()},
        "applied": {
            str(name): {
                "reference_wavelength": reference_wavelength,
                "target_wavelength": wavelength,
                "matrix": lookup_matrix(calib, wavelength,
                                        reference_wavelength).tolist(),
                "identity": bool(is_identity(calib, wavelength,
                                             reference_wavelength)),
            }
            for name, wavelength in targets.items()
        },
    }
    if notes:
        record["notes"] = notes
    return record


def disabled_provenance(reason: str, stage: str | None = None,
                        ownership_policy: str | None = None,
                        cli_args: Mapping | None = None) -> dict:
    """Provenance for an explicitly unregistered run."""
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "stage": stage,
        "ownership_policy": ownership_policy,
        "cli_args": {str(k): str(v) for k, v in dict(cli_args or {}).items()},
        "mode": "off",
        "transform_direction": None,
        "interpolation": None,
        "warning": reason,
    }


def write_provenance(path: str | os.PathLike[str], record: Mapping) -> None:
    """Write a stage's provenance, refusing to overwrite another stage's record.

    The localizer and replication_qc.py previously wrote the SAME filename, so
    whichever ran last silently destroyed the other's provenance.
    """
    path = os.fspath(path)
    stage = (record or {}).get("stage")
    if stage and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError):
            existing = {}
        other = existing.get("stage")
        if other and other != stage:
            raise ChromaticRegistrationError(
                f"refusing to overwrite provenance from stage {other!r} with "
                f"stage {stage!r} at {path}. Each stage must use its own "
                "provenance filename.")
    tmp = f"{os.fspath(path)}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, os.fspath(path))
