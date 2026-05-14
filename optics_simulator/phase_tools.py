"""
phase_tools.py

Small utilities for OPD/phase conversion and low-order phase removal.

These are intentionally independent of beam_trace.py and Wavefront2D so they
can be used by both old analysis scripts and the new optics_simulator pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def opd_to_phase(opd_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    """
    Convert optical path difference [m] to phase [rad].
    """
    return (2.0 * np.pi / float(wavelength_m)) * np.asarray(opd_m, dtype=float)


def phase_to_opd(phase_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """
    Convert phase [rad] to optical path difference [m].
    """
    return np.asarray(phase_rad, dtype=float) * float(wavelength_m) / (2.0 * np.pi)


def masked_values(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return finite values inside mask, or all finite values if mask is None.
    """
    arr = np.asarray(arr, dtype=float)

    if mask is None:
        return arr[np.isfinite(arr)]

    mask = np.asarray(mask, dtype=bool)

    if mask.shape != arr.shape:
        raise ValueError("mask must have same shape as arr.")

    return arr[mask & np.isfinite(arr)]


def rms(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    RMS of finite values, optionally inside mask.
    """
    vals = masked_values(arr, mask)

    if vals.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(vals**2)))


def std(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Standard deviation of finite values, optionally inside mask.
    """
    vals = masked_values(arr, mask)

    if vals.size == 0:
        return np.nan

    return float(np.std(vals))


def remove_piston(
    arr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, float]:
    """
    Remove piston/mean value from an array.

    Parameters
    ----------
    arr:
        Input OPD or phase map.
    mask:
        Optional boolean mask defining valid pupil pixels.
    fill_value:
        Value used outside mask in returned array.

    Returns
    -------
    corrected:
        arr with mean value subtracted inside mask.
    piston:
        Mean value that was subtracted.
    """
    arr = np.asarray(arr, dtype=float)

    if mask is None:
        valid = np.isfinite(arr)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("mask must have same shape as arr.")
        valid = mask & np.isfinite(arr)

    if not np.any(valid):
        return np.full_like(arr, fill_value, dtype=float), np.nan

    piston = float(np.mean(arr[valid]))

    corrected = np.full_like(arr, fill_value, dtype=float)
    corrected[valid] = arr[valid] - piston

    return corrected, piston


def fit_piston_tip_tilt(
    arr: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit piston + tip + tilt plane to arr.

    Model:
        arr ~= c0 + cx * x + cy * y

    Parameters
    ----------
    arr:
        OPD or phase map.
    xx, yy:
        Coordinate grids matching arr.
    mask:
        Optional valid-pixel mask.

    Returns
    -------
    coeff:
        Array [c0, cx, cy].
    model:
        Best-fit plane evaluated over full grid.
    """
    arr = np.asarray(arr, dtype=float)
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)

    if arr.shape != xx.shape or arr.shape != yy.shape:
        raise ValueError("arr, xx, and yy must have matching shapes.")

    if mask is None:
        valid = np.isfinite(arr) & np.isfinite(xx) & np.isfinite(yy)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("mask must have same shape as arr.")
        valid = mask & np.isfinite(arr) & np.isfinite(xx) & np.isfinite(yy)

    if np.count_nonzero(valid) < 3:
        coeff = np.array([np.nan, np.nan, np.nan])
        model = np.full_like(arr, np.nan, dtype=float)
        return coeff, model

    A = np.column_stack(
        [
            np.ones(np.count_nonzero(valid)),
            xx[valid],
            yy[valid],
        ]
    )

    b = arr[valid]

    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)

    model = coeff[0] + coeff[1] * xx + coeff[2] * yy

    return coeff, model


def remove_piston_tip_tilt(
    arr: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove best-fit piston + tip + tilt plane from an OPD or phase map.

    Returns
    -------
    corrected:
        Low-order-removed map.
    coeff:
        Plane coefficients [c0, cx, cy].
    model:
        Best-fit plane over full grid.
    """
    arr = np.asarray(arr, dtype=float)

    coeff, model = fit_piston_tip_tilt(arr, xx, yy, mask=mask)

    if mask is None:
        valid = np.isfinite(arr)
    else:
        mask = np.asarray(mask, dtype=bool)
        valid = mask & np.isfinite(arr)

    corrected = np.full_like(arr, fill_value, dtype=float)
    corrected[valid] = arr[valid] - model[valid]

    return corrected, coeff, model


def marechal_strehl_from_phase(
    phase_rad: np.ndarray,
    mask: Optional[np.ndarray] = None,
    remove_mean: bool = True,
) -> float:
    """
    Maréchal/Mahajan Strehl approximation:

        S ~= exp(-sigma_phi^2)

    Parameters
    ----------
    phase_rad:
        Phase map [rad].
    mask:
        Optional pupil mask.
    remove_mean:
        If True, subtract piston before computing variance.
    """
    vals = masked_values(phase_rad, mask)

    if vals.size == 0:
        return np.nan

    if remove_mean:
        vals = vals - np.mean(vals)

    return float(np.exp(-np.var(vals)))


def phase_rms_report(
    phase_rad: np.ndarray,
    mask: Optional[np.ndarray] = None,
    xx: Optional[np.ndarray] = None,
    yy: Optional[np.ndarray] = None,
) -> dict:
    """
    Convenience report for raw, piston-removed, and optionally TT-removed phase.

    Returns values in radians.
    """
    out = {
        "raw_std_rad": std(phase_rad, mask),
        "raw_rms_rad": rms(phase_rad, mask),
    }

    phase_piston_removed, piston = remove_piston(phase_rad, mask=mask)

    out.update(
        {
            "piston_rad": piston,
            "piston_removed_std_rad": std(phase_piston_removed, mask),
            "piston_removed_rms_rad": rms(phase_piston_removed, mask),
            "marechal_strehl_piston_removed": marechal_strehl_from_phase(
                phase_piston_removed,
                mask=mask,
                remove_mean=False,
            ),
        }
    )

    if xx is not None and yy is not None:
        phase_tt_removed, coeff, model = remove_piston_tip_tilt(
            phase_rad,
            xx=xx,
            yy=yy,
            mask=mask,
        )

        out.update(
            {
                "ptt_coeff_rad": coeff,
                "ptt_removed_std_rad": std(phase_tt_removed, mask),
                "ptt_removed_rms_rad": rms(phase_tt_removed, mask),
                "marechal_strehl_ptt_removed": marechal_strehl_from_phase(
                    phase_tt_removed,
                    mask=mask,
                    remove_mean=False,
                ),
            }
        )

    return out




# ============================================================
# Gaussian / Moffat model fitting
# ============================================================

def extract_fit_parameter(fit_obj, candidate_keys):
    """
    Safely extract a scalar parameter from a fit object.

    Supports dict-like fit results, attribute-style objects, and lmfit-like
    objects with .params.
    """
    if fit_obj is None:
        return np.nan

    if isinstance(fit_obj, dict):
        for key in candidate_keys:
            if key in fit_obj:
                try:
                    return float(fit_obj[key])
                except Exception:
                    return np.nan

    for key in candidate_keys:
        if hasattr(fit_obj, key):
            try:
                return float(getattr(fit_obj, key))
            except Exception:
                pass

    if hasattr(fit_obj, "params"):
        params = getattr(fit_obj, "params")
        for key in candidate_keys:
            if key in params:
                try:
                    return float(params[key].value)
                except Exception:
                    pass

    return np.nan


def fit_psf_models(
    psf_crop,
    x_ld,
    y_ld,
    fit_module,
    moffat_fit_region="all",
):
    """
    Fit Gaussian and Moffat models to a cropped PSF.

    Parameters
    ----------
    psf_crop:
        Cropped PSF image.
    x_ld, y_ld:
        2D coordinate grids in lambda/D units.
    fit_module:
        Module that provides:
            fit_2d_gaussian
            gaussian_fwhm_and_ellipticity
            fit_2d_moffat
            moffat_fwhm_and_ellipticity

        For the current code, pass:
            fit_module=bt
        where bt is optics_simulator.beam_trace_ORIG.
    moffat_fit_region:
        "all" or "wings", passed to fit_2d_moffat.
    """
    result = {
        "gaussian_fwhm": np.nan,
        "gaussian_ell": np.nan,
        "moffat_fwhm": np.nan,
        "moffat_ell": np.nan,
        "moffat_alpha": np.nan,
        "moffat_beta": np.nan,
        "moffat_amplitude": np.nan,
        "moffat_bg": np.nan,
        "gaussian_fit": None,
        "moffat_fit": None,
    }

    psf_crop = np.asarray(psf_crop, dtype=float)
    peak = np.nanmax(psf_crop)

    if not np.isfinite(peak) or peak <= 0:
        return result

    psf_crop_norm = psf_crop / peak

    try:
        gfit = fit_module.fit_2d_gaussian(x_ld, y_ld, psf_crop_norm)
        gmetrics = fit_module.gaussian_fwhm_and_ellipticity(gfit)

        result["gaussian_fwhm"] = gmetrics.get("fwhm_major", np.nan)
        result["gaussian_ell"] = gmetrics.get("ellipticity", np.nan)
        result["gaussian_fit"] = gfit
    except Exception:
        pass

    try:
        mfit = fit_module.fit_2d_moffat(
            x_ld,
            y_ld,
            psf_crop_norm,
            fit_region=moffat_fit_region,
        )
        mmetrics = fit_module.moffat_fwhm_and_ellipticity(mfit)

        result["moffat_fwhm"] = mmetrics.get("fwhm_major", np.nan)
        result["moffat_ell"] = mmetrics.get("ellipticity", np.nan)
        result["moffat_alpha"] = extract_fit_parameter(
            mfit,
            ["alpha", "alpha_x", "scale", "r0"],
        )
        result["moffat_beta"] = extract_fit_parameter(
            mfit,
            ["beta", "power"],
        )
        result["moffat_amplitude"] = extract_fit_parameter(
            mfit,
            ["amplitude", "amp", "A"],
        )
        result["moffat_bg"] = extract_fit_parameter(
            mfit,
            ["background", "bg", "offset", "c0"],
        )
        result["moffat_fit"] = mfit
    except Exception:
        pass

    return result


def analyse_psf_with_fits(
    psf,
    perfect_psf,
    angular_pixel_scale_ld,
    fit_module,
    crop_half_width_ld=6.0,
    r_max_ld=10.0,
    n_radii=400,
    moffat_fit_region="all",
):
    """
    Analyse PSF with Strehl, EE80, Gaussian fit, and Moffat fit.

    This mirrors the old psf_analysis.py behaviour but lives in the new
    psf_tools layer.
    """
    psf = np.asarray(psf, dtype=float)
    perfect_psf = np.asarray(perfect_psf, dtype=float)

    result = {
        "strehl": np.nan,
        "ee80": np.nan,
        "ee_radii": np.array([]),
        "ee_curve": np.array([]),
        "psf_crop": psf,
        "x_ld_crop": None,
        "y_ld_crop": None,
        "gaussian_fwhm": np.nan,
        "gaussian_ell": np.nan,
        "moffat_fwhm": np.nan,
        "moffat_ell": np.nan,
        "moffat_alpha": np.nan,
        "moffat_beta": np.nan,
        "moffat_amplitude": np.nan,
        "moffat_bg": np.nan,
    }

    total = np.nansum(psf)
    perfect_total = np.nansum(perfect_psf)

    if total <= 0 or perfect_total <= 0:
        return result

    result["strehl"] = strehl_from_psfs(psf, perfect_psf)

    # Smooth slightly for robust peak finding, same spirit as old script.
    try:
        from scipy.ndimage import gaussian_filter

        psf_smooth = gaussian_filter(psf, sigma=2)
        peak_y, peak_x = np.unravel_index(np.nanargmax(psf_smooth), psf.shape)
    except Exception:
        peak_x, peak_y = peak_location(psf)

    lim_pix = max(2, int(crop_half_width_ld / angular_pixel_scale_ld))

    y_s = max(0, peak_y - lim_pix)
    y_e = min(psf.shape[0], peak_y + lim_pix + 1)
    x_s = max(0, peak_x - lim_pix)
    x_e = min(psf.shape[1], peak_x + lim_pix + 1)

    psf_crop = psf[y_s:y_e, x_s:x_e]

    y_idx, x_idx = np.indices(psf_crop.shape)
    x_ld = (x_idx - (peak_x - x_s)) * angular_pixel_scale_ld
    y_ld = (y_idx - (peak_y - y_s)) * angular_pixel_scale_ld

    fit_result = fit_psf_models(
        psf_crop,
        x_ld,
        y_ld,
        fit_module=fit_module,
        moffat_fit_region=moffat_fit_region,
    )

    radii = np.linspace(0.0, r_max_ld, n_radii)
    ee = encircled_energy(
        psf,
        radii=radii,
        x0=float(peak_x),
        y0=float(peak_y),
        radial_scale=angular_pixel_scale_ld,
    )
    ee80 = encircled_energy_radius(radii, ee, fraction=0.8)

    result.update(fit_result)
    result.update(
        {
            "ee80": ee80,
            "ee_radii": radii,
            "ee_curve": ee,
            "psf_crop": psf_crop,
            "x_ld_crop": x_ld,
            "y_ld_crop": y_ld,
            "x_peak": int(peak_x),
            "y_peak": int(peak_y),
        }
    )

    return result