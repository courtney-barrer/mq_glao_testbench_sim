"""
test_phase_tools.py

Run from repository root:

    python3 -m optics_simulator.tests.test_phase_tools
"""

from __future__ import annotations

import numpy as np

from optics_simulator import phase_tools


def test_phase_opd_round_trip():
    wavelength = 1.2e-6
    phase = np.linspace(-5.0, 5.0, 100)

    opd = phase_tools.phase_to_opd(phase, wavelength)
    phase2 = phase_tools.opd_to_phase(opd, wavelength)

    assert np.allclose(phase, phase2)


def test_remove_piston():
    arr = np.ones((10, 10)) * 5.0
    arr[3, 4] += 2.0

    corrected, piston = phase_tools.remove_piston(arr)

    assert np.isclose(piston, np.mean(arr))
    assert np.isclose(np.mean(corrected), 0.0)


def test_remove_piston_with_mask():
    arr = np.zeros((10, 10))
    arr[2:8, 2:8] = 10.0

    mask = np.zeros_like(arr, dtype=bool)
    mask[2:8, 2:8] = True

    corrected, piston = phase_tools.remove_piston(arr, mask=mask)

    assert np.isclose(piston, 10.0)
    assert np.allclose(corrected[mask], 0.0)
    assert np.all(np.isnan(corrected[~mask]))


def test_fit_and_remove_piston_tip_tilt():
    n = 64
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)

    coeff_true = np.array([2.0, 3.0, -4.0])
    arr = coeff_true[0] + coeff_true[1] * xx + coeff_true[2] * yy

    mask = xx**2 + yy**2 <= 1.0

    corrected, coeff, model = phase_tools.remove_piston_tip_tilt(
        arr,
        xx=xx,
        yy=yy,
        mask=mask,
    )

    assert np.allclose(coeff, coeff_true, atol=1e-12)
    assert np.nanstd(corrected[mask]) < 1e-12


def test_marechal_strehl_from_phase():
    rng = np.random.default_rng(1)
    sigma = 0.2
    phase = rng.normal(0.0, sigma, size=(256, 256))

    strehl = phase_tools.marechal_strehl_from_phase(phase)

    assert np.isclose(strehl, np.exp(-np.var(phase)), rtol=1e-12)


def test_phase_rms_report():
    n = 64
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    mask = xx**2 + yy**2 <= 1.0

    phase = 1.0 + 0.2 * xx - 0.1 * yy + 0.05 * np.sin(8.0 * xx)

    report = phase_tools.phase_rms_report(
        phase,
        mask=mask,
        xx=xx,
        yy=yy,
    )

    assert "raw_std_rad" in report
    assert "piston_removed_rms_rad" in report
    assert "ptt_removed_rms_rad" in report
    assert report["ptt_removed_rms_rad"] < report["piston_removed_rms_rad"]


if __name__ == "__main__":
    test_phase_opd_round_trip()
    test_remove_piston()
    test_remove_piston_with_mask()
    test_fit_and_remove_piston_tip_tilt()
    test_marechal_strehl_from_phase()
    test_phase_rms_report()

    print("All phase_tools tests passed.")
    