"""
test_wavefront.py

Tests for optics_simulator.wavefront.

Run from repository root:

    python3 -m optics_simulator.tests.test_wavefront

or with pytest:

    PYTHONPATH=. pytest -q optics_simulator/tests/test_wavefront.py
"""

from __future__ import annotations

import numpy as np

from optics_simulator.wavefront import (
    Wavefront2D,
    beam_sample_from_wavefront,
    copy_wavefront_metadata,
    make_empty_coordinate_grids,
    wavefront_from_amplitude_phase,
    wavefront_from_beam_sample,
)


def test_wavefront_from_amplitude_phase_flat_field():
    n = 64
    dx = 10e-6
    wavelength = 633e-9

    amplitude = np.ones((n, n))
    phase = np.zeros((n, n))

    wf = wavefront_from_amplitude_phase(
        amplitude=amplitude,
        phase_rad=phase,
        wavelength=wavelength,
        dx=dx,
        label="flat",
    )

    assert wf.shape == (n, n)
    assert wf.nx == n
    assert wf.ny == n
    assert np.isclose(wf.wavelength, wavelength)
    assert np.isclose(wf.dx, dx)
    assert np.isclose(wf.dy, dx)
    assert wf.label == "flat"
    assert np.allclose(wf.amplitude, 1.0)
    assert np.allclose(wf.phase, 0.0)
    assert np.allclose(wf.intensity, 1.0)

    expected_power = n * n * dx * dx
    assert np.isclose(wf.power, expected_power)


def test_wavefront_from_amplitude_phase_with_mask():
    n = 64
    dx = 10e-6
    wavelength = 633e-9

    xx, yy = make_empty_coordinate_grids((n, n), dx)
    mask = xx**2 + yy**2 <= (0.2e-3) ** 2

    amplitude = np.ones((n, n))
    phase = 0.3 * np.ones((n, n))

    wf = wavefront_from_amplitude_phase(
        amplitude=amplitude,
        phase_rad=phase,
        wavelength=wavelength,
        dx=dx,
        mask=mask,
        x=xx,
        y=yy,
        label="masked",
    )

    assert wf.shape == (n, n)
    assert np.allclose(wf.field[~mask], 0.0)
    assert np.allclose(np.abs(wf.field[mask]), 1.0)
    assert np.allclose(np.angle(wf.field[mask]), 0.3)
    assert wf.x.shape == wf.shape
    assert wf.y.shape == wf.shape


def test_wavefront_copy_is_deep_enough():
    n = 32
    dx = 5e-6
    wavelength = 633e-9

    wf = wavefront_from_amplitude_phase(
        amplitude=np.ones((n, n)),
        phase_rad=np.zeros((n, n)),
        wavelength=wavelength,
        dx=dx,
        metadata={"a": 1},
    )

    wf2 = wf.copy()

    wf2.field[0, 0] = 99.0 + 0.0j
    wf2.metadata["a"] = 2

    assert wf.field[0, 0] != wf2.field[0, 0]
    assert wf.metadata["a"] == 1
    assert wf2.metadata["a"] == 2


def test_with_field_replaces_field_and_preserves_sampling():
    n = 32
    dx = 5e-6
    wavelength = 633e-9

    wf = wavefront_from_amplitude_phase(
        amplitude=np.ones((n, n)),
        phase_rad=np.zeros((n, n)),
        wavelength=wavelength,
        dx=dx,
        label="old",
    )

    new_field = 2.0 * np.ones((n, n), dtype=complex)
    wf2 = wf.with_field(new_field, label="new")

    assert wf2.shape == wf.shape
    assert wf2.label == "new"
    assert np.isclose(wf2.dx, wf.dx)
    assert np.isclose(wf2.dy, wf.dy)
    assert np.isclose(wf2.wavelength, wf.wavelength)
    assert np.allclose(wf2.field, new_field)


def test_wavefront_from_beam_sample_round_trip():
    n = 64
    dx = 10e-6
    wavelength = 633e-9

    xx, yy = make_empty_coordinate_grids((n, n), dx)
    mask = xx**2 + yy**2 <= (0.25e-3) ** 2

    amplitude = mask.astype(float)
    phase = np.where(mask, 0.2 * np.sin(2.0 * np.pi * xx / 0.2e-3), np.nan)

    sample = {
        "xx": xx,
        "yy": yy,
        "mask": mask,
        "amplitude": amplitude,
        "phase_map_rad": phase,
        "opd_map_m": np.full_like(amplitude, np.nan),
        "dx": dx,
    }

    wf = wavefront_from_beam_sample(
        sample,
        wavelength=wavelength,
        label="from_sample",
        keep_original_sample=True,
    )

    assert wf.shape == (n, n)
    assert wf.label == "from_sample"
    assert wf.metadata["source"] == "beam_trace_sample"
    assert "sample" in wf.metadata

    sample2 = beam_sample_from_wavefront(wf, mask=mask)

    assert sample2["xx"].shape == (n, n)
    assert sample2["yy"].shape == (n, n)
    assert sample2["mask"].shape == (n, n)
    assert np.allclose(sample2["amplitude"][mask], amplitude[mask])
    assert np.allclose(sample2["amplitude"][~mask], 0.0)
    assert np.allclose(sample2["phase_map_rad"][mask], np.angle(wf.field[mask]))


def test_coordinate_grid_centre_and_spacing():
    n = 10
    dx = 2.0
    dy = 3.0

    xx, yy = make_empty_coordinate_grids((n, n), dx, dy)

    assert xx.shape == (n, n)
    assert yy.shape == (n, n)

    assert np.isclose(xx[0, 1] - xx[0, 0], dx)
    assert np.isclose(yy[1, 0] - yy[0, 0], dy)

    assert np.isclose(xx[n // 2, n // 2], 0.0)
    assert np.isclose(yy[n // 2, n // 2], 0.0)


def test_copy_wavefront_metadata():
    n = 16
    dx = 1e-6
    wavelength = 633e-9

    src = Wavefront2D(
        field=np.ones((n, n), dtype=complex),
        wavelength=wavelength,
        dx=dx,
        plane_point=np.array([1.0, 2.0, 3.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        e1=np.array([1.0, 0.0, 0.0]),
        e2=np.array([0.0, 1.0, 0.0]),
        metadata={"source_key": "source_value"},
    )

    dst = Wavefront2D(
        field=2.0 * np.ones((n, n), dtype=complex),
        wavelength=wavelength,
        dx=dx,
        metadata={"dst_key": "dst_value"},
    )

    out = copy_wavefront_metadata(src, dst)

    assert np.allclose(out.field, dst.field)
    assert out.metadata["source_key"] == "source_value"
    assert out.metadata["dst_key"] == "dst_value"
    assert np.allclose(out.plane_point, src.plane_point)
    assert np.allclose(out.plane_normal, src.plane_normal)
    assert np.allclose(out.e1, src.e1)
    assert np.allclose(out.e2, src.e2)


if __name__ == "__main__":
    test_wavefront_from_amplitude_phase_flat_field()
    test_wavefront_from_amplitude_phase_with_mask()
    test_wavefront_copy_is_deep_enough()
    test_with_field_replaces_field_and_preserves_sampling()
    test_wavefront_from_beam_sample_round_trip()
    test_coordinate_grid_centre_and_spacing()
    test_copy_wavefront_metadata()

    print("All Wavefront2D tests passed.")