"""
wave_optics.py

High-level scalar wave-optics operations on Wavefront2D objects.

This module wraps fresnel.py so that propagation and optical operations preserve
Wavefront2D metadata, pixel scale, wavelength, and optional plane geometry.

It deliberately does not modify beam_trace.py.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from . import fresnel
from .wavefront import Wavefront2D


PropagationMethod = Literal["angular_spectrum", "fresnel"]


# ============================================================
# Internal helpers
# ============================================================

def _unit_vector(v: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a 3-vector."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / n


def _append_history(wf: Wavefront2D, entry: dict) -> None:
    """
    Append an operation record to wf.metadata["history"] in-place.
    """
    history = wf.metadata.setdefault("history", [])
    history.append(dict(entry))


def _copy_with_new_field(
    wf: Wavefront2D,
    field: np.ndarray,
    label: Optional[str] = None,
    history_entry: Optional[dict] = None,
) -> Wavefront2D:
    """
    Return a copied Wavefront2D with a replaced field and optional history.
    """
    out = wf.with_field(field, label=label)

    if history_entry is not None:
        _append_history(out, history_entry)

    return out


# ============================================================
# Propagation
# ============================================================

def propagate(
    wf: Wavefront2D,
    z: float,
    method: PropagationMethod = "angular_spectrum",
    include_global_phase: bool = True,
    bandlimit: bool = True,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Propagate a Wavefront2D by distance z.

    Parameters
    ----------
    wf:
        Input wavefront.
    z:
        Propagation distance [m].
    method:
        "angular_spectrum" or "fresnel".
    include_global_phase:
        Whether to include exp(i k z) propagation phase.
    bandlimit:
        Only used for angular-spectrum propagation.
    label:
        Optional output label.

    Returns
    -------
    Wavefront2D
        Propagated wavefront.

    Notes
    -----
    This keeps the same sampling dx, dy. For long propagation distances where
    the field expands significantly, ensure the computational window is large
    enough to avoid wrap-around.
    """
    field = fresnel.propagate(
        wf.field,
        wavelength=wf.wavelength,
        dx=wf.dx,
        dy=wf.dy,
        z=z,
        method=method,
        include_global_phase=include_global_phase,
        bandlimit=bandlimit,
    )

    out = _copy_with_new_field(
        wf,
        field,
        label=label,
        history_entry={
            "op": "propagate",
            "z_m": float(z),
            "method": method,
            "include_global_phase": bool(include_global_phase),
            "bandlimit": bool(bandlimit),
        },
    )

    # If the wavefront has a known plane, advance it along the plane normal.
    if out.plane_point is not None and out.plane_normal is not None:
        n = _unit_vector(out.plane_normal)
        out.plane_point = out.plane_point + float(z) * n

    return out


def propagate_to_lens_focus(
    wf: Wavefront2D,
    focal_length: float,
    method: PropagationMethod = "angular_spectrum",
    include_global_phase: bool = True,
    bandlimit: bool = True,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply a thin lens and propagate by one focal length.

    This is a convenience helper for quick focus-plane simulations.
    """
    after_lens = apply_lens(
        wf,
        focal_length=focal_length,
        label=None if label is None else f"{label}: after lens",
    )

    return propagate(
        after_lens,
        z=focal_length,
        method=method,
        include_global_phase=include_global_phase,
        bandlimit=bandlimit,
        label=label,
    )


# ============================================================
# Phase and amplitude operations
# ============================================================

def apply_phase(
    wf: Wavefront2D,
    phase_rad: np.ndarray,
    mask: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply a phase map in radians to a Wavefront2D.
    """
    phase_rad = np.asarray(phase_rad, dtype=float)

    if phase_rad.shape != wf.shape:
        raise ValueError("phase_rad must have same shape as wf.field.")

    phasor = np.exp(1j * phase_rad)

    if mask is None:
        field = wf.field * phasor
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != wf.shape:
            raise ValueError("mask must have same shape as wf.field.")
        field = np.where(mask, wf.field * phasor, wf.field)

    return _copy_with_new_field(
        wf,
        field,
        label=label,
        history_entry={"op": "apply_phase"},
    )


def apply_opd(
    wf: Wavefront2D,
    opd_map_m: np.ndarray,
    mask: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply an optical path difference map to a Wavefront2D.

    Parameters
    ----------
    opd_map_m:
        OPD map [m].
    mask:
        Optional boolean mask selecting where OPD is applied.
    """
    opd_map_m = np.asarray(opd_map_m, dtype=float)

    if opd_map_m.shape != wf.shape:
        raise ValueError("opd_map_m must have same shape as wf.field.")

    phase_rad = 2.0 * np.pi * opd_map_m / wf.wavelength

    out = apply_phase(wf, phase_rad, mask=mask, label=label)
    _append_history(out, {"op": "apply_opd", "wavelength_m": wf.wavelength})
    return out


def apply_amplitude(
    wf: Wavefront2D,
    amplitude: np.ndarray,
    mask: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Multiply a Wavefront2D by an amplitude transmission map.
    """
    amplitude = np.asarray(amplitude, dtype=float)

    if amplitude.shape != wf.shape:
        raise ValueError("amplitude must have same shape as wf.field.")

    if mask is None:
        field = wf.field * amplitude
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != wf.shape:
            raise ValueError("mask must have same shape as wf.field.")
        field = np.where(mask, wf.field * amplitude, wf.field)

    return _copy_with_new_field(
        wf,
        field,
        label=label,
        history_entry={"op": "apply_amplitude"},
    )


def apply_gaussian_amplitude(
    wf,
    waist_radius_m,
    center_m=(0.0, 0.0),
    truncate=True,
    label=None,
):
    """
    Apply a Gaussian field-amplitude envelope to a Wavefront2D.

    This modifies only the complex field amplitude; it does not change the
    phase, wavelength, sampling, or coordinate system of the input wavefront.

    The Gaussian is defined as a *field amplitude* profile,

        A(r) = exp[-(r / w)^2]

    where ``w = waist_radius_m`` is the 1/e field-amplitude radius. Therefore
    the corresponding intensity profile is

        I(r) = |A(r)|^2 = exp[-2 (r / w)^2]

    so ``waist_radius_m`` is also the usual 1/e^2 intensity radius.

    Parameters
    ----------
    wf : Wavefront2D
        Input wavefront. Its ``x`` and ``y`` coordinate grids are used to define
        the Gaussian envelope.
    waist_radius_m : float
        Gaussian waist radius ``w`` in metres. This is the 1/e field-amplitude
        radius, equivalently the 1/e^2 intensity radius.
    center_m : tuple of float, optional
        Gaussian centre ``(x0, y0)`` in metres, expressed in the same local
        coordinate system as ``wf.x`` and ``wf.y``. The default is centred on
        the wavefront coordinate origin.
    truncate : bool, optional
        If True, preserve the existing aperture support by multiplying the
        Gaussian by ``wf.amplitude > 0``. This models a Gaussian beam clipped by
        the sampled pupil/beam stop. If False, the Gaussian is applied across
        the full computational grid, including pixels outside the original
        aperture.
    label : str, optional
        Label for the returned wavefront. If None, the input label is preserved.

    Returns
    -------
    Wavefront2D
        New wavefront with field

            E_out(x, y) = E_in(x, y) * exp[-((x-x0)^2 + (y-y0)^2) / w^2]

        and, if ``truncate=True``, additionally clipped to the original
        nonzero-amplitude support.

    Notes
    -----
    This function does not renormalise optical power. Applying a Gaussian
    envelope generally changes the total power of the wavefront, especially
    when ``truncate=True``. If constant power is required for a comparison,
    renormalise the returned field explicitly after applying this function.

    Common choices
    --------------
    If ``beam.diameter`` is the physical clipped beam diameter:

    - ``waist_radius_m = 0.5 * beam.diameter`` gives intensity = 1/e^2 at the
      nominal beam edge.
    - ``waist_radius_m > 0.5 * beam.diameter`` gives a weakly tapered,
      nearly top-hat beam.
    - ``waist_radius_m < 0.5 * beam.diameter`` gives a strongly tapered beam.
    """

def apply_tilt(
    wf: Wavefront2D,
    angle_x_rad: float = 0.0,
    angle_y_rad: float = 0.0,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply a small-angle plane-wave tilt.

    For small angles:
        phase = k (x angle_x + y angle_y)

    This is useful for testing off-axis beams or injecting a known wavefront
    slope.
    """
    if wf.x is None or wf.y is None:
        X, Y = fresnel.make_coordinate_grid(wf.shape, wf.dx, wf.dy)
    else:
        X, Y = wf.x, wf.y

    k = 2.0 * np.pi / wf.wavelength
    phase = k * (X * float(angle_x_rad) + Y * float(angle_y_rad))

    out = apply_phase(wf, phase, label=label)
    _append_history(
        out,
        {
            "op": "apply_tilt",
            "angle_x_rad": float(angle_x_rad),
            "angle_y_rad": float(angle_y_rad),
        },
    )
    return out


# ============================================================
# Lenses and apertures
# ============================================================

def apply_lens(
    wf: Wavefront2D,
    focal_length: float,
    center: Optional[Tuple[float, float]] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply an ideal thin lens to a Wavefront2D.

    The lens phase is:
        exp[-i pi (x^2 + y^2) / (lambda f)]
    """
    field = fresnel.apply_thin_lens(
        wf.field,
        wavelength=wf.wavelength,
        dx=wf.dx,
        dy=wf.dy,
        focal_length=focal_length,
        center=center,
    )

    return _copy_with_new_field(
        wf,
        field,
        label=label,
        history_entry={
            "op": "apply_lens",
            "focal_length_m": float(focal_length),
            "center_m": None if center is None else tuple(float(c) for c in center),
        },
    )


def apply_aperture(
    wf: Wavefront2D,
    aperture: np.ndarray,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply an arbitrary aperture/transmission map.

    aperture may be boolean or floating point transmission.
    """
    aperture = np.asarray(aperture)

    if aperture.shape != wf.shape:
        raise ValueError("aperture must have same shape as wf.field.")

    field = wf.field * aperture

    return _copy_with_new_field(
        wf,
        field,
        label=label,
        history_entry={"op": "apply_aperture"},
    )


def apply_circular_aperture(
    wf: Wavefront2D,
    radius: float,
    center: Optional[Tuple[float, float]] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply a centred or shifted circular aperture.
    """
    aperture = fresnel.circular_aperture(
        wf.shape,
        dx=wf.dx,
        dy=wf.dy,
        radius=radius,
        center=center,
    )

    out = apply_aperture(wf, aperture, label=label)
    _append_history(
        out,
        {
            "op": "apply_circular_aperture",
            "radius_m": float(radius),
            "center_m": None if center is None else tuple(float(c) for c in center),
        },
    )
    return out


def apply_rectangular_aperture(
    wf: Wavefront2D,
    width: float,
    height: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Apply a rectangular aperture.
    """
    aperture = fresnel.rectangular_aperture(
        wf.shape,
        dx=wf.dx,
        dy=wf.dy,
        width=width,
        height=height,
        center=center,
    )

    out = apply_aperture(wf, aperture, label=label)
    _append_history(
        out,
        {
            "op": "apply_rectangular_aperture",
            "width_m": float(width),
            "height_m": None if height is None else float(height),
            "center_m": None if center is None else tuple(float(c) for c in center),
        },
    )
    return out


# ============================================================
# Diagnostics
# ============================================================

def power(wf: Wavefront2D) -> float:
    """
    Integrated wavefront intensity.
    """
    return float(np.sum(np.abs(wf.field) ** 2) * wf.dx * wf.dy)


def normalize_power(
    wf: Wavefront2D,
    target_power: float = 1.0,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Scale field amplitude so that integrated power equals target_power.
    """
    current = power(wf)

    if current <= 0:
        raise ValueError("Cannot normalize a zero-power wavefront.")

    scale = np.sqrt(float(target_power) / current)
    field = wf.field * scale

    return _copy_with_new_field(
        wf,
        field,
        label=label,
        history_entry={
            "op": "normalize_power",
            "target_power": float(target_power),
            "previous_power": float(current),
            "scale": float(scale),
        },
    )


def relative_l2_error(
    wf_reference: Wavefront2D,
    wf_test: Wavefront2D,
    remove_scale: bool = True,
) -> float:
    """
    Relative L2 error between two wavefront fields.
    """
    if wf_reference.shape != wf_test.shape:
        raise ValueError("Wavefronts must have the same shape.")

    return fresnel.relative_l2_error(
        wf_reference.field,
        wf_test.field,
        remove_scale=remove_scale,
    )


def assert_same_sampling(wf_a: Wavefront2D, wf_b: Wavefront2D) -> None:
    """
    Raise AssertionError if two Wavefront2D objects do not share sampling.
    """
    if wf_a.shape != wf_b.shape:
        raise AssertionError(f"Shape mismatch: {wf_a.shape} vs {wf_b.shape}")

    if not np.isclose(wf_a.dx, wf_b.dx):
        raise AssertionError(f"dx mismatch: {wf_a.dx} vs {wf_b.dx}")

    if not np.isclose(wf_a.dy, wf_b.dy):
        raise AssertionError(f"dy mismatch: {wf_a.dy} vs {wf_b.dy}")

    if not np.isclose(wf_a.wavelength, wf_b.wavelength):
        raise AssertionError(
            f"wavelength mismatch: {wf_a.wavelength} vs {wf_b.wavelength}"
        )
    



    # ============================================================
# Padding and scaled focal-plane propagation
# ============================================================

def pad_wavefront(
    wf: Wavefront2D,
    pad_factor: Optional[int] = None,
    pad_to: Optional[int] = None,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Zero-pad a Wavefront2D while preserving physical input-plane sampling.

    Parameters
    ----------
    wf:
        Input wavefront.
    pad_factor:
        Integer multiplicative padding factor.
    pad_to:
        Explicit final square array size.
    label:
        Optional output label.

    Notes
    -----
    Padding increases the computational window and improves Fourier/focal-plane
    sampling. It does not change dx or dy.
    """
    if pad_factor is None and pad_to is None:
        raise ValueError("Provide either pad_factor or pad_to.")

    ny, nx = wf.shape

    if pad_to is not None:
        new_ny = int(pad_to)
        new_nx = int(pad_to)
    else:
        if pad_factor is None or pad_factor < 1:
            raise ValueError("pad_factor must be >= 1.")
        new_ny = int(pad_factor * ny)
        new_nx = int(pad_factor * nx)

    if new_ny < ny or new_nx < nx:
        raise ValueError("Padded size must be >= original wavefront size.")

    py = new_ny - ny
    px = new_nx - nx

    field_pad = np.pad(
        wf.field,
        ((py // 2, py - py // 2), (px // 2, px - px // 2)),
        mode="constant",
    )

    y = (np.arange(new_ny) - new_ny // 2) * wf.dy
    x = (np.arange(new_nx) - new_nx // 2) * wf.dx
    xx, yy = np.meshgrid(x, y)

    out = Wavefront2D(
        field=field_pad,
        wavelength=wf.wavelength,
        dx=wf.dx,
        dy=wf.dy,
        x=xx,
        y=yy,
        plane_point=None if wf.plane_point is None else wf.plane_point.copy(),
        plane_normal=None if wf.plane_normal is None else wf.plane_normal.copy(),
        e1=None if wf.e1 is None else wf.e1.copy(),
        e2=None if wf.e2 is None else wf.e2.copy(),
        label=wf.label if label is None else label,
        metadata=dict(wf.metadata),
    )

    _append_history(
        out,
        {
            "op": "pad_wavefront",
            "old_shape": tuple(int(v) for v in wf.shape),
            "new_shape": tuple(int(v) for v in out.shape),
            "pad_factor": None if pad_factor is None else int(pad_factor),
            "pad_to": None if pad_to is None else int(pad_to),
        },
    )

    return out


def lens_focal_plane(
    wf: Wavefront2D,
    focal_length: float,
    include_global_phase: bool = False,
    label: Optional[str] = None,
) -> Wavefront2D:
    """
    Compute the properly sampled focal-plane field after an ideal thin lens.

    This uses Fourier-transform lens scaling rather than fixed-grid angular
    spectrum propagation.

    Output sampling is:

        dx_focal = wavelength * focal_length / (N_x * dx_pupil)
        dy_focal = wavelength * focal_length / (N_y * dy_pupil)

    This is usually the preferred way to compute a focal-plane PSF from a pupil
    field.
    """
    field_focal, dx_focal, dy_focal = fresnel.lens_focal_plane_field(
        wf.field,
        wavelength=wf.wavelength,
        dx=wf.dx,
        dy=wf.dy,
        focal_length=focal_length,
        include_global_phase=include_global_phase,
    )

    ny, nx = field_focal.shape
    y = (np.arange(ny) - ny // 2) * dy_focal
    x = (np.arange(nx) - nx // 2) * dx_focal
    xx, yy = np.meshgrid(x, y)

    plane_point = None
    if wf.plane_point is not None and wf.plane_normal is not None:
        n = _unit_vector(wf.plane_normal)
        plane_point = wf.plane_point + float(focal_length) * n

    out = Wavefront2D(
        field=field_focal,
        wavelength=wf.wavelength,
        dx=dx_focal,
        dy=dy_focal,
        x=xx,
        y=yy,
        plane_point=plane_point,
        plane_normal=None if wf.plane_normal is None else wf.plane_normal.copy(),
        e1=None if wf.e1 is None else wf.e1.copy(),
        e2=None if wf.e2 is None else wf.e2.copy(),
        label=wf.label if label is None else label,
        metadata=dict(wf.metadata),
    )

    _append_history(
        out,
        {
            "op": "lens_focal_plane",
            "focal_length_m": float(focal_length),
            "input_dx_m": float(wf.dx),
            "input_dy_m": float(wf.dy),
            "output_dx_m": float(dx_focal),
            "output_dy_m": float(dy_focal),
            "include_global_phase": bool(include_global_phase),
        },
    )

    return out