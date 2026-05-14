"""
wavefront.py

Wavefront2D dataclass and conversion helpers.

This module bridges the existing beam_trace.py sample dictionaries and the new
Fresnel / scalar wave-optics tools.

It deliberately does not import beam_trace.py, so it remains independent of the
geometric ray-tracing layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Wavefront2D:
    """
    Sampled scalar complex wavefront.

    Parameters
    ----------
    field:
        2D complex electric field.
    wavelength:
        Wavelength [m].
    dx, dy:
        Pixel scales [m/pixel].
    x, y:
        Optional coordinate grids [m].
    plane_point:
        Optional 3D point defining the physical plane.
    plane_normal:
        Optional 3D normal vector of the physical plane.
    e1, e2:
        Optional 3D local basis vectors spanning the plane.
    label:
        Human-readable label.
    metadata:
        Free-form metadata dictionary.
    """

    field: np.ndarray
    wavelength: float
    dx: float
    dy: Optional[float] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    plane_point: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    e1: Optional[np.ndarray] = None
    e2: Optional[np.ndarray] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.field = np.asarray(self.field, dtype=complex)
        self.wavelength = float(self.wavelength)
        self.dx = float(self.dx)

        if self.dy is None:
            self.dy = self.dx
        else:
            self.dy = float(self.dy)

        if self.field.ndim != 2:
            raise ValueError("field must be a 2D array.")

        if self.x is not None:
            self.x = np.asarray(self.x, dtype=float)
            if self.x.shape != self.field.shape:
                raise ValueError("x must have the same shape as field.")

        if self.y is not None:
            self.y = np.asarray(self.y, dtype=float)
            if self.y.shape != self.field.shape:
                raise ValueError("y must have the same shape as field.")

        if self.plane_point is not None:
            self.plane_point = np.asarray(self.plane_point, dtype=float).reshape(3)

        if self.plane_normal is not None:
            self.plane_normal = np.asarray(self.plane_normal, dtype=float).reshape(3)

        if self.e1 is not None:
            self.e1 = np.asarray(self.e1, dtype=float).reshape(3)

        if self.e2 is not None:
            self.e2 = np.asarray(self.e2, dtype=float).reshape(3)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.field.shape

    @property
    def ny(self) -> int:
        return self.field.shape[0]

    @property
    def nx(self) -> int:
        return self.field.shape[1]

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.field) ** 2

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.field)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.field)

    @property
    def power(self) -> float:
        return float(np.sum(np.abs(self.field) ** 2) * self.dx * self.dy)

    def copy(self) -> "Wavefront2D":
        return Wavefront2D(
            field=self.field.copy(),
            wavelength=self.wavelength,
            dx=self.dx,
            dy=self.dy,
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            plane_point=None if self.plane_point is None else self.plane_point.copy(),
            plane_normal=None if self.plane_normal is None else self.plane_normal.copy(),
            e1=None if self.e1 is None else self.e1.copy(),
            e2=None if self.e2 is None else self.e2.copy(),
            label=self.label,
            metadata=dict(self.metadata),
        )

    def with_field(self, new_field: np.ndarray, label: Optional[str] = None) -> "Wavefront2D":
        """
        Return a copy with a replaced complex field.
        """
        out = self.copy()
        new_field = np.asarray(new_field, dtype=complex)

        if new_field.shape != self.field.shape:
            raise ValueError("new_field must have the same shape as the original field.")

        out.field = new_field

        if label is not None:
            out.label = label

        return out


def make_empty_coordinate_grids(shape: Tuple[int, int], dx: float, dy: Optional[float] = None):
    """
    Create centred coordinate grids for a Wavefront2D.
    """
    if dy is None:
        dy = dx

    ny, nx = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy

    return np.meshgrid(x, y)


def wavefront_from_amplitude_phase(
    amplitude: np.ndarray,
    phase_rad: np.ndarray,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    label: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Wavefront2D:
    """
    Construct a Wavefront2D from amplitude and phase arrays.
    """
    amplitude = np.asarray(amplitude, dtype=float)
    phase_rad = np.asarray(phase_rad, dtype=float)

    if amplitude.shape != phase_rad.shape:
        raise ValueError("amplitude and phase_rad must have the same shape.")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != amplitude.shape:
            raise ValueError("mask must have the same shape as amplitude.")
        field = np.where(mask, amplitude * np.exp(1j * phase_rad), 0.0)
    else:
        field = amplitude * np.exp(1j * phase_rad)

    return Wavefront2D(
        field=field,
        wavelength=wavelength,
        dx=dx,
        dy=dy,
        x=x,
        y=y,
        label=label,
        metadata={} if metadata is None else dict(metadata),
    )


def wavefront_from_beam_sample(
    sample: Dict[str, Any],
    wavelength: float,
    label: str = "",
    keep_original_sample: bool = False,
) -> Wavefront2D:
    """
    Convert an existing beam_trace sample dictionary into a Wavefront2D.

    Expected keys from beam_trace.sample_beam_phase_amplitude_on_pupil_plane:

        sample["amplitude"]
        sample["phase_map_rad"]
        sample["mask"]
        sample["dx"]

    Optional keys:

        sample["xx"]
        sample["yy"]
        sample["opd_map_m"]

    Parameters
    ----------
    sample:
        beam_trace-style sample dictionary.
    wavelength:
        Wavelength [m].
    label:
        Optional wavefront label.
    keep_original_sample:
        If True, stores the original sample dictionary inside metadata.
        This is useful for debugging but can consume memory.
    """
    required = ["amplitude", "phase_map_rad", "mask", "dx"]
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"sample is missing required keys: {missing}")

    amp = np.asarray(sample["amplitude"], dtype=float)
    phase = np.nan_to_num(np.asarray(sample["phase_map_rad"], dtype=float), nan=0.0)
    mask = np.asarray(sample["mask"], dtype=bool)

    if amp.shape != phase.shape or amp.shape != mask.shape:
        raise ValueError("amplitude, phase_map_rad, and mask must have matching shapes.")

    field = np.where(mask, amp * np.exp(1j * phase), 0.0)

    metadata: Dict[str, Any] = {
        "source": "beam_trace_sample",
        "has_opd_map_m": "opd_map_m" in sample,
    }

    if keep_original_sample:
        metadata["sample"] = sample

    return Wavefront2D(
        field=field,
        wavelength=wavelength,
        dx=float(sample["dx"]),
        dy=float(sample.get("dy", sample["dx"])),
        x=sample.get("xx"),
        y=sample.get("yy"),
        label=label,
        metadata=metadata,
    )


def beam_sample_from_wavefront(
    wf: Wavefront2D,
    mask: Optional[np.ndarray] = None,
    include_unwrapped_phase: bool = False,
) -> Dict[str, Any]:
    """
    Convert Wavefront2D back to a beam_trace-like sample dictionary.

    This is useful so older analysis functions can consume propagated fields.

    Notes
    -----
    The returned phase is wrapped to [-pi, pi] because it comes from np.angle.
    """
    amp = np.abs(wf.field)
    phase = np.angle(wf.field)

    if mask is None:
        mask = amp > 0
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != wf.field.shape:
            raise ValueError("mask must have the same shape as wf.field.")

    if wf.x is None or wf.y is None:
        xx, yy = make_empty_coordinate_grids(wf.field.shape, wf.dx, wf.dy)
    else:
        xx, yy = wf.x, wf.y

    sample = {
        "xx": xx,
        "yy": yy,
        "mask": mask,
        "amplitude": amp,
        "opd_map_m": np.full_like(amp, np.nan, dtype=float),
        "phase_map_rad": np.where(mask, phase, np.nan),
        "dx": wf.dx,
        "dy": wf.dy,
    }

    if include_unwrapped_phase:
        sample["phase_unwrapped_rad"] = np.unwrap(np.unwrap(phase, axis=0), axis=1)

    return sample


def copy_wavefront_metadata(src: Wavefront2D, dst: Wavefront2D) -> Wavefront2D:
    """
    Copy metadata and geometric plane information from one wavefront to another.
    """
    out = dst.copy()
    out.metadata.update(src.metadata)

    out.plane_point = None if src.plane_point is None else src.plane_point.copy()
    out.plane_normal = None if src.plane_normal is None else src.plane_normal.copy()
    out.e1 = None if src.e1 is None else src.e1.copy()
    out.e2 = None if src.e2 is None else src.e2.copy()

    return out