"""
test_hybrid_propagation.py

End-to-end tests for the hybrid geometry + wave-optics path.

Run from repository root:

    python3 -m optics_simulator.tests.test_hybrid_propagation

or with pytest:

    PYTHONPATH=. pytest -q optics_simulator/tests/test_hybrid_propagation.py
"""

from __future__ import annotations

import numpy as np

from optics_simulator import beam_trace as bt
from optics_simulator import beam_trace_wave as btw
from optics_simulator import psf_tools
from optics_simulator import wave_optics


def make_flat_bench_and_beam():
    bench = bt.OpticalBench3D()

    beam = bt.Beam3D.collimated_circular(
        radius=6.5e-3,
        nrings=1,
        nphi=4,
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        wavelength=633e-9,
        label="flat_test_beam",
    )

    return bench, beam


def make_phase_screen_bench_and_beam():
    bench = bt.OpticalBench3D()

    n = 256
    extent = 50e-3
    x = np.linspace(-extent / 2, extent / 2, n)
    y = np.linspace(-extent / 2, extent / 2, n)
    xx, yy = np.meshgrid(x, y)

    opd = (
        150e-9 * np.sin(2.0 * np.pi * xx / 8e-3)
        + 80e-9 * np.cos(2.0 * np.pi * yy / 11e-3)
    )

    screen = bt.RotatingPhaseScreen3D(
        point=[0.0, 0.0, -0.5],
        normal=[0.0, 0.0, 1.0],
        opd_map=opd,
        map_extent_m=extent,
        clear_radius=25e-3,
        angular_velocity=0.0,
        label="deterministic_screen",
    )

    bench.add(screen)

    beam = bt.Beam3D.collimated_circular(
        radius=6.5e-3,
        nrings=1,
        nphi=4,
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        wavelength=633e-9,
        label="phase_screen_test_beam",
    )

    return bench, beam


def test_bridge_to_wavefront_and_fixed_grid_propagation():
    bench, beam = make_flat_bench_and_beam()

    wf = btw.sample_wavefront_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=128,
        beam_trace_module=bt,
    )

    wf_pad = wave_optics.pad_wavefront(wf, pad_to=512)

    wf_prop = wave_optics.propagate(
        wf_pad,
        z=0.25,
        method="angular_spectrum",
        include_global_phase=False,
    )

    assert wf.shape == (128, 128)
    assert wf_pad.shape == (512, 512)
    assert wf_prop.shape == wf_pad.shape

    p0 = wave_optics.power(wf_pad)
    p1 = wave_optics.power(wf_prop)

    assert p0 > 0.0
    assert np.isclose(p0, p1, rtol=1e-12)

    assert wf.metadata["bridge"] == "beam_trace_wave.sample_wavefront_on_pupil_plane"
    assert wf_prop.metadata["history"][-1]["op"] == "propagate"


def test_bridge_to_scaled_lens_focal_plane():
    bench, beam = make_flat_bench_and_beam()

    wf = btw.sample_wavefront_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=256,
        beam_trace_module=bt,
    )

    wf_pad = wave_optics.pad_wavefront(wf, pad_to=1024)

    focal_length = 0.25
    wf_focus = wave_optics.lens_focal_plane(
        wf_pad,
        focal_length=focal_length,
        include_global_phase=False,
    )

    expected_dx = wf.wavelength * focal_length / (wf_pad.nx * wf_pad.dx)

    assert wf_focus.shape == wf_pad.shape
    assert np.isclose(wf_focus.dx, expected_dx, rtol=1e-14)
    assert np.isclose(wf_focus.dy, expected_dx, rtol=1e-14)

    p0 = wave_optics.power(wf_pad)
    p1 = wave_optics.power(wf_focus)

    assert np.isclose(p0, p1, rtol=1e-12)

    airy_radius = 1.22 * wf.wavelength * focal_length / beam.diameter
    pixels_per_airy_radius = airy_radius / wf_focus.dx

    assert pixels_per_airy_radius > 4.0

    intensity = wf_focus.intensity
    y_peak, x_peak = np.unravel_index(np.nanargmax(intensity), intensity.shape)

    assert abs(x_peak - intensity.shape[1] // 2) <= 1
    assert abs(y_peak - intensity.shape[0] // 2) <= 1


def test_phase_screen_reduces_fft_strehl():
    flat_bench, flat_beam = make_flat_bench_and_beam()
    screen_bench, screen_beam = make_phase_screen_bench_and_beam()

    wf_flat = btw.sample_wavefront_on_pupil_plane(
        beam=flat_beam,
        bench=flat_bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=256,
        beam_trace_module=bt,
    )

    wf_screen = btw.sample_wavefront_on_pupil_plane(
        beam=screen_beam,
        bench=screen_bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=256,
        beam_trace_module=bt,
    )

    psf_flat = psf_tools.psf_from_wavefront(
        wf_flat,
        pad_to=1024,
        normalize=None,
    )

    psf_screen = psf_tools.psf_from_wavefront(
        wf_screen,
        pad_to=1024,
        normalize=None,
    )

    strehl = psf_tools.strehl_from_psfs(psf_screen, psf_flat)

    phase = np.angle(wf_screen.field)
    mask = wf_screen.amplitude > 0

    assert np.nanstd(phase[mask]) > 0.01
    assert np.isfinite(strehl)
    assert 0.0 < strehl <= 1.05
    assert strehl < 0.95


def test_basic_psf_analysis_from_hybrid_wavefront():
    bench, beam = make_flat_bench_and_beam()

    wf = btw.sample_wavefront_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=128,
        beam_trace_module=bt,
    )

    psf_pack = psf_tools.psf_pack_from_wavefront(
        wf,
        pad_to=512,
        normalize="peak",
        pupil_diameter=beam.diameter,
    )

    psf = psf_pack["psf"]

    result = psf_tools.analyse_psf_basic(
        psf,
        radial_scale=1.0,
        r_max=50.0,
        n_radii=200,
    )

    assert psf.shape == (512, 512)
    assert np.isclose(result["peak"], 1.0)
    assert result["flux"] > 0.0
    assert abs(result["x_peak"] - 256) <= 1
    assert abs(result["y_peak"] - 256) <= 1
    assert np.isfinite(result["ee_radius"])


if __name__ == "__main__":
    test_bridge_to_wavefront_and_fixed_grid_propagation()
    test_bridge_to_scaled_lens_focal_plane()
    test_phase_screen_reduces_fft_strehl()
    test_basic_psf_analysis_from_hybrid_wavefront()

    print("All hybrid propagation tests passed.")