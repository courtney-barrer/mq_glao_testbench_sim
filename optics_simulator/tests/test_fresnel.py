"""
test_fresnel.py

Unit/regression tests for optics_simulator.fresnel.

Run from repository root:

    python3 -m optics_simulator.tests.test_fresnel

or with pytest:

    PYTHONPATH=. pytest -q optics_simulator/tests/test_fresnel.py
"""

from __future__ import annotations

import numpy as np

from optics_simulator import fresnel


def circular_pupil_field(n=256, dx=50e-6, diameter=13e-3):
    aperture = fresnel.circular_aperture(
        shape=(n, n),
        dx=dx,
        radius=0.5 * diameter,
    )
    return aperture.astype(complex)


def radial_profile(image, dx, centre=None, nbins=500):
    image = np.asarray(image, dtype=float)

    ny, nx = image.shape
    yy, xx = np.indices(image.shape)

    if centre is None:
        yc, xc = np.unravel_index(np.nanargmax(image), image.shape)
    else:
        xc, yc = centre

    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2) * dx

    r_max = np.nanmax(r)
    bins = np.linspace(0.0, r_max, nbins + 1)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    prof = np.full(nbins, np.nan)

    for i in range(nbins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(m):
            prof[i] = np.nanmean(image[m])

    return r_mid, prof


def estimate_first_airy_minimum_radius(intensity, dx_focal, expected_radius):
    """
    Estimate first Airy minimum from radial profile.

    This deliberately searches near the theoretical first-null location rather
    than globally, because sampled/log-ring sidelobes can create small local
    minima elsewhere.
    """
    r, prof = radial_profile(intensity, dx=dx_focal, nbins=800)

    prof = prof / np.nanmax(prof)

    search = (
        (r > 0.55 * expected_radius)
        & (r < 1.45 * expected_radius)
        & np.isfinite(prof)
    )

    if not np.any(search):
        raise AssertionError("No valid samples in Airy-minimum search region.")

    r_search = r[search]
    p_search = prof[search]

    return float(r_search[np.nanargmin(p_search)])


def test_plane_wave_preserved_without_global_phase():
    n = 128
    dx = 10e-6
    wavelength = 633e-9
    z = 0.1

    field0 = np.ones((n, n), dtype=complex)

    field1 = fresnel.angular_spectrum_propagate(
        field0,
        wavelength=wavelength,
        dx=dx,
        z=z,
        include_global_phase=False,
    )

    assert np.allclose(np.abs(field1), 1.0, rtol=1e-14, atol=1e-14)
    assert np.std(np.angle(field1)) < 1e-14

    e0 = fresnel.energy(field0, dx)
    e1 = fresnel.energy(field1, dx)
    assert np.isclose(e0, e1, rtol=1e-14)


def test_fresnel_and_angular_spectrum_agree_for_paraxial_case():
    n = 128
    dx = 8e-6
    wavelength = 633e-9
    z = 0.02

    x, y = fresnel.make_coordinate_grid((n, n), dx)
    w0 = 0.25e-3
    field0 = np.exp(-(x**2 + y**2) / w0**2)

    field_as = fresnel.angular_spectrum_propagate(
        field0,
        wavelength=wavelength,
        dx=dx,
        z=z,
        include_global_phase=False,
    )

    field_fr = fresnel.fresnel_transfer_function_propagate(
        field0,
        wavelength=wavelength,
        dx=dx,
        z=z,
        include_global_phase=False,
    )

    err = fresnel.relative_l2_error(field_as, field_fr, remove_scale=True)

    assert err < 1e-3


def test_round_trip_angular_spectrum():
    n = 128
    dx = 10e-6
    wavelength = 633e-9
    z = 0.15

    x, y = fresnel.make_coordinate_grid((n, n), dx)
    # field0 = np.exp(-(x**2 + y**2) / (0.3e-3) ** 2)
    # field0 *= np.exp(1j * 0.5 * np.sin(2.0 * np.pi * x / 0.25e-3))
    field0 = np.exp(-(x**2 + y**2) / (0.3e-3) ** 2).astype(complex)
    field0 *= np.exp(1j * 0.5 * np.sin(2.0 * np.pi * x / 0.25e-3))

    field1 = fresnel.angular_spectrum_propagate(
        field0,
        wavelength=wavelength,
        dx=dx,
        z=z,
        include_global_phase=False,
    )

    field2 = fresnel.angular_spectrum_propagate(
        field1,
        wavelength=wavelength,
        dx=dx,
        z=-z,
        include_global_phase=False,
    )

    err = fresnel.relative_l2_error(field0, field2, remove_scale=True)

    assert err < 1e-12


def test_lens_focal_plane_sampling_and_power():
    n = 256
    pad_to = 1024
    dx = 50e-6
    wavelength = 633e-9
    diameter = 13e-3
    focal_length = 0.25

    field = circular_pupil_field(n=n, dx=dx, diameter=diameter)

    pad = pad_to - n
    field_pad = np.pad(
        field,
        ((pad // 2, pad - pad // 2), (pad // 2, pad - pad // 2)),
        mode="constant",
    )

    field_focal, dx_focal, dy_focal = fresnel.lens_focal_plane_field(
        field_pad,
        wavelength=wavelength,
        dx=dx,
        focal_length=focal_length,
    )

    expected_dx = wavelength * focal_length / (pad_to * dx)

    assert np.isclose(dx_focal, expected_dx, rtol=1e-14)
    assert np.isclose(dy_focal, expected_dx, rtol=1e-14)

    e0 = fresnel.energy(field_pad, dx)
    e1 = fresnel.energy(field_focal, dx_focal)

    assert np.isclose(e0, e1, rtol=1e-12)


def test_lens_focal_plane_airy_first_null_scale():
    n = 256
    pad_to = 2048
    dx = 50e-6
    wavelength = 633e-9
    diameter = 13e-3
    focal_length = 0.25

    field = circular_pupil_field(n=n, dx=dx, diameter=diameter)

    pad = pad_to - n
    field_pad = np.pad(
        field,
        ((pad // 2, pad - pad // 2), (pad // 2, pad - pad // 2)),
        mode="constant",
    )

    field_focal, dx_focal, _ = fresnel.lens_focal_plane_field(
        field_pad,
        wavelength=wavelength,
        dx=dx,
        focal_length=focal_length,
    )

    intensity = np.abs(field_focal) ** 2
    intensity /= np.nanmax(intensity)

    expected_first_null = 1.22 * wavelength * focal_length / diameter
    measured_first_null = estimate_first_airy_minimum_radius(
        intensity,
        dx_focal=dx_focal,
        expected_radius=expected_first_null,
    )

    rel_err = abs(measured_first_null - expected_first_null) / expected_first_null

    # The aperture edge is pixelated, so do not make this unrealistically tight.
    assert rel_err < 0.10


if __name__ == "__main__":
    test_plane_wave_preserved_without_global_phase()
    test_fresnel_and_angular_spectrum_agree_for_paraxial_case()
    test_round_trip_angular_spectrum()
    test_lens_focal_plane_sampling_and_power()
    test_lens_focal_plane_airy_first_null_scale()

    print("All Fresnel tests passed.")