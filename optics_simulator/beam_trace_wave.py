"""
beam_trace_wave.py

Bridge functions between the existing beam_trace.py module and the new
Wavefront2D/Fresnel propagation tools.

This module intentionally imports beam_trace only inside functions, so the new
optics_simulator package remains importable even when beam_trace.py is not on
the Python path.

Main purpose
------------
Use existing geometric beam tracing / phase-screen sampling to initialize a
sampled scalar wavefront:

    beam_trace beam + bench
        -> sample_beam_phase_amplitude_on_pupil_plane(...)
        -> Wavefront2D

This keeps beam_trace.py backwards compatible and untouched.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .wavefront import Wavefront2D, wavefront_from_beam_sample


def _get_beam_trace_module(beam_trace_module=None):
    """
    Return a beam_trace-like module.

    Allows dependency injection during testing, but defaults to importing the
    user's existing beam_trace.py from the current Python path.
    """
    if beam_trace_module is not None:
        return beam_trace_module

    try:
        import beam_trace as bt
    except ImportError as exc:
        raise ImportError(
            "Could not import beam_trace. Make sure beam_trace.py is on the "
            "Python path, or pass beam_trace_module explicitly."
        ) from exc

    return bt


def _as_3vector(value, name: str) -> np.ndarray:
    """
    Convert value to a 3-vector float array.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values.")

    return arr


def _unit_vector(value, name: str) -> np.ndarray:
    """
    Convert value to a normalized 3-vector.
    """
    arr = _as_3vector(value, name=name)
    norm = np.linalg.norm(arr)

    if norm == 0:
        raise ValueError(f"{name} must be non-zero.")

    return arr / norm


def _default_plane_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a stable orthonormal basis for a plane normal.

    This mirrors the style of beam_trace.orthonormal_basis_from_normal without
    importing beam_trace at module import time.
    """
    normal = _unit_vector(normal, name="normal")

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, ref)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)

    return e1, e2


def sample_wavefront_on_pupil_plane(
    beam,
    bench,
    pupil_point,
    t: float,
    npix: int = 192,
    diameter: Optional[float] = None,
    screen_labels: Optional[Sequence[str]] = None,
    plane_normal: Sequence[float] = (0.0, 0.0, 1.0),
    e1: Optional[Sequence[float]] = None,
    e2: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    keep_original_sample: bool = False,
    beam_trace_module=None,
) -> Wavefront2D:
    """
    Sample the existing beam_trace pupil-plane field as a Wavefront2D.

    Parameters
    ----------
    beam:
        Existing beam_trace.Beam3D instance.
    bench:
        Existing beam_trace.OpticalBench3D instance.
    pupil_point:
        3D point defining the pupil plane location.
    t:
        Time [s], passed through to beam_trace phase-screen sampling.
    npix:
        Number of pixels across the sampled beam grid.
    diameter:
        Optional sampled pupil diameter [m]. Defaults to beam.diameter in
        beam_trace.
    screen_labels:
        Optional list of phase-screen labels to include.
    plane_normal:
        Normal of the output wavefront plane. Default is global +z.
    e1, e2:
        Optional local basis vectors for the output plane.
    label:
        Optional wavefront label. Defaults to beam.label if present.
    keep_original_sample:
        If True, stores the original sample dictionary in wf.metadata.
    beam_trace_module:
        Optional injected beam_trace-like module.

    Returns
    -------
    Wavefront2D
        Complex wavefront built from the beam_trace sample dictionary.
    """
    bt = _get_beam_trace_module(beam_trace_module)

    pupil_point = _as_3vector(pupil_point, name="pupil_point")
    plane_normal = _unit_vector(plane_normal, name="plane_normal")

    if e1 is None or e2 is None:
        e1_arr, e2_arr = _default_plane_basis_from_normal(plane_normal)
    else:
        e1_arr = _unit_vector(e1, name="e1")
        e2_arr = _unit_vector(e2, name="e2")

    sample = bt.sample_beam_phase_amplitude_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=pupil_point,
        t=t,
        npix=npix,
        diameter=diameter,
        screen_labels=None if screen_labels is None else list(screen_labels),
    )

    wf_label = label
    if wf_label is None:
        wf_label = getattr(beam, "label", "")

    wavelength = getattr(beam, "wavelength", None)
    if wavelength is None:
        raise AttributeError("beam must have a wavelength attribute.")

    wf = wavefront_from_beam_sample(
        sample,
        wavelength=wavelength,
        label=wf_label,
        keep_original_sample=keep_original_sample,
    )

    wf.plane_point = pupil_point
    wf.plane_normal = plane_normal
    wf.e1 = e1_arr
    wf.e2 = e2_arr

    wf.metadata.update(
        {
            "bridge": "beam_trace_wave.sample_wavefront_on_pupil_plane",
            "beam_label": getattr(beam, "label", None),
            "beam_diameter_m": getattr(beam, "diameter", None),
            "time_s": float(t),
            "npix": int(npix),
            "diameter_override_m": None if diameter is None else float(diameter),
            "screen_labels": None if screen_labels is None else list(screen_labels),
            "pupil_point_m": tuple(float(v) for v in pupil_point),
            "plane_normal": tuple(float(v) for v in plane_normal),
        }
    )

    return wf


def sample_wavefront_and_beam_sample_on_pupil_plane(
    beam,
    bench,
    pupil_point,
    t: float,
    npix: int = 192,
    diameter: Optional[float] = None,
    screen_labels: Optional[Sequence[str]] = None,
    plane_normal: Sequence[float] = (0.0, 0.0, 1.0),
    e1: Optional[Sequence[float]] = None,
    e2: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    beam_trace_module=None,
) -> Tuple[Wavefront2D, Dict[str, Any]]:
    """
    Same as sample_wavefront_on_pupil_plane, but also return the original
    beam_trace sample dictionary.

    Useful while migrating old scripts, because you can compare old and new
    analysis paths side-by-side.
    """
    wf = sample_wavefront_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=pupil_point,
        t=t,
        npix=npix,
        diameter=diameter,
        screen_labels=screen_labels,
        plane_normal=plane_normal,
        e1=e1,
        e2=e2,
        label=label,
        keep_original_sample=True,
        beam_trace_module=beam_trace_module,
    )

    sample = wf.metadata["sample"]
    wf.metadata.pop("sample", None)

    return wf, sample


def wavefront_from_existing_beam_sample(
    sample: Dict[str, Any],
    wavelength: float,
    label: str = "",
    pupil_point: Optional[Sequence[float]] = None,
    plane_normal: Sequence[float] = (0.0, 0.0, 1.0),
    keep_original_sample: bool = False,
) -> Wavefront2D:
    """
    Convert an already-created beam_trace sample dictionary into Wavefront2D.

    This is useful when existing code already calls:

        sample = bt.sample_beam_phase_amplitude_on_pupil_plane(...)

    and you only want to enter the new wave-optics pipeline afterwards.
    """
    wf = wavefront_from_beam_sample(
        sample,
        wavelength=wavelength,
        label=label,
        keep_original_sample=keep_original_sample,
    )

    if pupil_point is not None:
        wf.plane_point = _as_3vector(pupil_point, name="pupil_point")

    wf.plane_normal = _unit_vector(plane_normal, name="plane_normal")
    wf.e1, wf.e2 = _default_plane_basis_from_normal(wf.plane_normal)

    wf.metadata.update(
        {
            "bridge": "beam_trace_wave.wavefront_from_existing_beam_sample",
            "pupil_point_m": None
            if pupil_point is None
            else tuple(float(v) for v in wf.plane_point),
            "plane_normal": tuple(float(v) for v in wf.plane_normal),
        }
    )

    return wf