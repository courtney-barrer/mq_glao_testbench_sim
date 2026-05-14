"""
test_beam_trace_backwards_compat.py

Regression tests showing that the new optics_simulator wavefront/PSF path
reproduces the old beam_trace.py PSF path.

These tests use optics_simulator/beam_trace_ORIG.py explicitly so that the
current original implementation remains frozen and testable.
"""

from __future__ import annotations

import numpy as np

from optics_simulator import beam_trace as bt
from optics_simulator import beam_trace_wave as btw
from optics_simulator import psf_tools
from optics_simulator.wavefront import wavefront_from_beam_sample


def make_empty_bench_and_beam():
    bench = bt.OpticalBench3D()

    beam = bt.Beam3D.collimated_circular(
        radius=6.5e-3,
        nrings=1,
        nphi=4,
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        wavelength=633e-9,
        label="test_beam",
    )

    return bench, beam


def make_phase_screen_bench_and_beam():
    """
    Build a minimal deterministic bench with one phase screen.

    This avoids relying on external FITS files while still checking that
    phase-screen OPD sampling is preserved through the new Wavefront2D path.
    """
    bench = bt.OpticalBench3D()

    n = 128
    extent = 40e-3
    x = np.linspace(-extent / 2, extent / 2, n)
    y = np.linspace(-extent / 2, extent / 2, n)
    xx, yy = np.meshgrid(x, y)

    opd = (
        80e-9 * np.sin(2.0 * np.pi * xx / 8e-3)
        + 40e-9 * np.cos(2.0 * np.pi * yy / 10e-3)
    )

    screen = bt.RotatingPhaseScreen3D(
        point=[0.0, 0.0, -0.5],
        normal=[0.0, 0.0, 1.0],
        opd_map=opd,
        map_extent_m=extent,
        clear_radius=20e-3,
        angular_velocity=0.0,
        label="test_screen",
    )

    bench.add(screen)

    beam = bt.Beam3D.collimated_circular(
        radius=6.5e-3,
        nrings=1,
        nphi=4,
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        wavelength=633e-9,
        label="test_beam_with_screen",
    )

    return bench, beam


def compare_old_and_new_psf(sample):
    """
    Compare old beam_trace PSF path against new Wavefront2D PSF path.
    """
    old_pack = bt.psf_from_plane_sample(sample)
    old_psf = old_pack["psf"]

    wf = wavefront_from_beam_sample(
        sample,
        wavelength=633e-9,
        label="from_existing_sample",
    )

    new_psf = psf_tools.psf_from_wavefront(
        wf,
        pad_to=None,
        normalize="peak",
    )

    assert old_psf.shape == new_psf.shape
    assert np.allclose(old_psf, new_psf, rtol=1e-13, atol=1e-13)


def test_expected_beam_trace_api_exists():
    """
    Check that the original public API used by current scripts is available.
    """
    required_names = [
        "OpticalBench3D",
        "Beam3D",
        "RotatingPhaseScreen3D",
        "make_converging_beam_from_field_angles",
        "sample_beam_phase_amplitude_on_pupil_plane",
        "psf_from_plane_sample",
        "fit_2d_gaussian",
        "fit_2d_moffat",
        "gaussian_fwhm_and_ellipticity",
        "moffat_fwhm_and_ellipticity",
    ]

    missing = [name for name in required_names if not hasattr(bt, name)]
    assert not missing, f"Missing expected beam_trace API names: {missing}"


def test_empty_bench_sample_keys_and_wavefront_bridge():
    bench, beam = make_empty_bench_and_beam()

    wf = btw.sample_wavefront_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=64,
        beam_trace_module=bt,
    )

    assert wf.shape == (64, 64)
    assert wf.label == "test_beam"
    assert np.isclose(wf.wavelength, 633e-9)
    assert wf.plane_point.shape == (3,)
    assert wf.plane_normal.shape == (3,)
    assert wf.metadata["bridge"] == "beam_trace_wave.sample_wavefront_on_pupil_plane"
    assert wf.power > 0.0


def test_old_new_psf_match_empty_bench():
    bench, beam = make_empty_bench_and_beam()

    sample = bt.sample_beam_phase_amplitude_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=np.array([0.0, 0.0, 0.0]),
        t=0.0,
        npix=64,
    )

    compare_old_and_new_psf(sample)


def test_old_new_psf_match_with_phase_screen():
    bench, beam = make_phase_screen_bench_and_beam()

    sample = bt.sample_beam_phase_amplitude_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=np.array([0.0, 0.0, 0.0]),
        t=0.0,
        npix=64,
    )

    compare_old_and_new_psf(sample)


if __name__ == "__main__":
    test_expected_beam_trace_api_exists()
    test_empty_bench_sample_keys_and_wavefront_bridge()
    test_old_new_psf_match_empty_bench()
    test_old_new_psf_match_with_phase_screen()

    print("All backwards-compatibility tests passed.")