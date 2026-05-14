"""
PSF and image-plane analysis utilities for Wavefront2D objects.

This module deliberately does not import beam_trace.py. It provides a clean
wave-optics PSF path while old scripts can continue using their existing
beam_trace / psf_analysis helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    curve_fit = None
    SCIPY_AVAILABLE = False

from . import fresnel
from .wavefront import Wavefront2D


# ============================================================
# Basic PSF generation
# ============================================================

def psf_from_wavefront(
    wf: Wavefront2D,
    pad_to: Optional[int] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute a Fraunhofer-style FFT PSF from a Wavefront2D.

    Parameters
    ----------
    wf:
        Input pupil-plane wavefront.
    pad_to:
        Optional square padded size.
    normalize:
        None, "peak", or "sum".

    Returns
    -------
    psf:
        Shifted intensity PSF.
    """
    return fresnel.fft_psf(wf.field, pad_to=pad_to, normalize=normalize)


def psf_pack_from_wavefront(
    wf: Wavefront2D,
    pad_to: Optional[int] = None,
    normalize: Optional[str] = None,
    pupil_diameter: Optional[float] = None,
    focal_length: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute PSF and useful coordinate systems.

    Returns a dictionary containing:
        psf
        x_pix, y_pix
        optionally x_ld, y_ld if pupil_diameter is provided
        optionally x_focal_m, y_focal_m if focal_length is provided
    """
    psf = psf_from_wavefront(wf, pad_to=pad_to, normalize=normalize)
    ny, nx = psf.shape

    x_pix = np.arange(nx) - nx // 2
    y_pix = np.arange(ny) - ny // 2
    X_pix, Y_pix = np.meshgrid(x_pix, y_pix)

    out: Dict[str, np.ndarray] = {
        "psf": psf,
        "x_pix": X_pix,
        "y_pix": Y_pix,
    }

    if pupil_diameter is not None:
        X_ld, Y_ld = fresnel.lambda_over_d_coordinates(
            psf.shape,
            dx_pupil=wf.dx,
            pupil_diameter=pupil_diameter,
        )
        out["x_ld"] = X_ld
        out["y_ld"] = Y_ld

    if focal_length is not None:
        X_foc, Y_foc = fresnel.focal_plane_coordinates(
            psf.shape,
            dx_pupil=wf.dx,
            wavelength=wf.wavelength,
            focal_length=focal_length,
        )
        out["x_focal_m"] = X_foc
        out["y_focal_m"] = Y_foc

    return out


# ============================================================
# Basic metrics
# ============================================================

def total_flux(psf: np.ndarray) -> float:
    """Return total PSF flux."""
    return float(np.nansum(np.asarray(psf, dtype=float)))


def peak_value(psf: np.ndarray) -> float:
    """Return peak PSF value."""
    return float(np.nanmax(np.asarray(psf, dtype=float)))


def strehl_from_psfs(
    psf: np.ndarray,
    perfect_psf: np.ndarray,
    flux_normalize: bool = True,
) -> float:
    """
    Estimate Strehl ratio from measured and perfect/reference PSFs.

    If flux_normalize=True, uses:

        S = (peak(psf) / sum(psf)) / (peak(perfect) / sum(perfect))
    """
    psf = np.asarray(psf, dtype=float)
    perfect_psf = np.asarray(perfect_psf, dtype=float)

    if psf.size == 0 or perfect_psf.size == 0:
        return np.nan

    if flux_normalize:
        f = np.nansum(psf)
        f0 = np.nansum(perfect_psf)

        if f <= 0 or f0 <= 0:
            return np.nan

        return float((np.nanmax(psf) / f) / (np.nanmax(perfect_psf) / f0))

    denom = np.nanmax(perfect_psf)
    if denom <= 0:
        return np.nan

    return float(np.nanmax(psf) / denom)


def marechal_strehl_from_phase(
    phase_rad: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Maréchal/Mahajan approximation:

        S ≈ exp(-var(phi))

    phase_rad should be in radians.
    """
    phase_rad = np.asarray(phase_rad, dtype=float)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != phase_rad.shape:
            raise ValueError("mask must have same shape as phase_rad.")
        vals = phase_rad[mask]
    else:
        vals = phase_rad[np.isfinite(phase_rad)]

    if vals.size == 0:
        return np.nan

    return float(np.exp(-np.nanvar(vals)))


def centroid(
    image: np.ndarray,
    threshold_fraction: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Flux-weighted centroid of an image.

    Returns
    -------
    x0, y0:
        Centroid in pixel coordinates.
    """
    image = np.asarray(image, dtype=float)

    work = np.nan_to_num(image, nan=0.0)
    work = np.maximum(work, 0.0)

    if threshold_fraction is not None:
        peak = np.max(work)
        if peak > 0:
            work = np.where(work >= threshold_fraction * peak, work, 0.0)

    flux = np.sum(work)
    if flux <= 0:
        return np.nan, np.nan

    yy, xx = np.indices(work.shape)
    x0 = np.sum(xx * work) / flux
    y0 = np.sum(yy * work) / flux

    return float(x0), float(y0)


def peak_location(image: np.ndarray) -> Tuple[int, int]:
    """
    Return integer peak location as (x_peak, y_peak).
    """
    image = np.asarray(image, dtype=float)

    if image.size == 0 or not np.any(np.isfinite(image)):
        return -1, -1

    y, x = np.unravel_index(np.nanargmax(image), image.shape)
    return int(x), int(y)


# ============================================================
# Cropping
# ============================================================

def crop_around_pixel(
    image: np.ndarray,
    x0: float,
    y0: float,
    half_width_pix: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image around a pixel coordinate.

    Returns
    -------
    crop:
        Cropped image.
    bounds:
        (x_start, x_end, y_start, y_end)
    """
    image = np.asarray(image)
    ny, nx = image.shape

    x0 = int(round(x0))
    y0 = int(round(y0))
    half_width_pix = int(half_width_pix)

    xs = max(0, x0 - half_width_pix)
    xe = min(nx, x0 + half_width_pix + 1)
    ys = max(0, y0 - half_width_pix)
    ye = min(ny, y0 + half_width_pix + 1)

    return image[ys:ye, xs:xe], (xs, xe, ys, ye)


def crop_around_peak(
    image: np.ndarray,
    half_width_pix: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image around its brightest pixel.
    """
    x0, y0 = peak_location(image)
    if x0 < 0 or y0 < 0:
        return np.asarray(image), (0, image.shape[1], 0, image.shape[0])

    return crop_around_pixel(image, x0, y0, half_width_pix)


def crop_psf_lambda_over_d(
    psf: np.ndarray,
    x_ld: np.ndarray,
    y_ld: np.ndarray,
    half_width_ld: float,
    centre: str = "peak",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop a PSF to a square region in lambda/D coordinates.

    Parameters
    ----------
    psf:
        PSF image.
    x_ld, y_ld:
        2D lambda/D coordinate grids matching psf.
    half_width_ld:
        Half-width of cropped coordinate region.
    centre:
        "origin" or "peak".

    Returns
    -------
    psf_crop, x_ld_crop, y_ld_crop
    """
    psf = np.asarray(psf)
    x_ld = np.asarray(x_ld)
    y_ld = np.asarray(y_ld)

    if psf.shape != x_ld.shape or psf.shape != y_ld.shape:
        raise ValueError("psf, x_ld, and y_ld must have matching shapes.")

    if centre == "origin":
        x0_ld = 0.0
        y0_ld = 0.0
    elif centre == "peak":
        xp, yp = peak_location(psf)
        x0_ld = x_ld[yp, xp]
        y0_ld = y_ld[yp, xp]
    else:
        raise ValueError("centre must be 'origin' or 'peak'.")

    keep = (
        (x_ld >= x0_ld - half_width_ld)
        & (x_ld <= x0_ld + half_width_ld)
        & (y_ld >= y0_ld - half_width_ld)
        & (y_ld <= y0_ld + half_width_ld)
    )

    rows = np.where(np.any(keep, axis=1))[0]
    cols = np.where(np.any(keep, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return psf, x_ld, y_ld

    ys, ye = rows[0], rows[-1] + 1
    xs, xe = cols[0], cols[-1] + 1

    return psf[ys:ye, xs:xe], x_ld[ys:ye, xs:xe], y_ld[ys:ye, xs:xe]


# ============================================================
# Encircled energy / radial profiles
# ============================================================

def radial_coordinates_pix(
    shape: Tuple[int, int],
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return radius map in pixels."""
    yy, xx = np.indices(shape)
    return np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)


def encircled_energy(
    psf: np.ndarray,
    radii: np.ndarray,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    radial_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute encircled energy curve.

    Parameters
    ----------
    psf:
        PSF image.
    radii:
        Radii at which to evaluate encircled energy, in units of
        radial_scale * pixel.
    x0, y0:
        Centre in pixel coordinates. If None, uses peak location.
    radial_scale:
        Multiplicative scale from pixels to desired radial units.

        Example:
            if one pixel = 0.05 lambda/D, radial_scale=0.05 and radii are
            interpreted in lambda/D.
    """
    psf = np.nan_to_num(np.asarray(psf, dtype=float), nan=0.0)
    psf = np.maximum(psf, 0.0)
    radii = np.asarray(radii, dtype=float)

    if x0 is None or y0 is None:
        xp, yp = peak_location(psf)
        x0, y0 = float(xp), float(yp)

    total = np.sum(psf)
    if total <= 0:
        return np.full_like(radii, np.nan, dtype=float)

    r = radial_coordinates_pix(psf.shape, x0=x0, y0=y0) * radial_scale

    ee = np.empty_like(radii, dtype=float)
    for i, radius in enumerate(radii):
        ee[i] = np.sum(psf[r <= radius]) / total

    return ee


def encircled_energy_radius(
    radii: np.ndarray,
    ee: np.ndarray,
    fraction: float = 0.8,
) -> float:
    """
    Interpolate radius at which encircled energy reaches a target fraction.
    """
    radii = np.asarray(radii, dtype=float)
    ee = np.asarray(ee, dtype=float)

    valid = np.isfinite(radii) & np.isfinite(ee)
    if np.count_nonzero(valid) < 2:
        return np.nan

    radii = radii[valid]
    ee = ee[valid]

    order = np.argsort(radii)
    radii = radii[order]
    ee = ee[order]

    if np.nanmax(ee) < fraction:
        return np.nan

    return float(np.interp(fraction, ee, radii))


def radial_profile(
    image: np.ndarray,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    radial_scale: float = 1.0,
    nbins: int = 200,
    r_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally averaged radial profile.

    Returns
    -------
    r_centres, profile
    """
    image = np.asarray(image, dtype=float)

    if x0 is None or y0 is None:
        xp, yp = peak_location(image)
        x0, y0 = float(xp), float(yp)

    r = radial_coordinates_pix(image.shape, x0=x0, y0=y0) * radial_scale

    if r_max is None:
        r_max = float(np.nanmax(r))

    bins = np.linspace(0.0, r_max, nbins + 1)
    profile = np.full(nbins, np.nan, dtype=float)
    centres = 0.5 * (bins[:-1] + bins[1:])

    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            profile[i] = np.nanmean(image[mask])

    return centres, profile


# ============================================================
# Convenience analysis wrapper
# ============================================================

def analyse_psf_basic(
    psf: np.ndarray,
    perfect_psf: Optional[np.ndarray] = None,
    radial_scale: float = 1.0,
    ee_fraction: float = 0.8,
    r_max: float = 10.0,
    n_radii: int = 400,
) -> Dict[str, object]:
    """
    Basic PSF analysis without model fitting.

    Parameters
    ----------
    radial_scale:
        Scale from pixels to chosen radial units.
        For example, lambda/D per pixel.

    Returns
    -------
    result:
        Dictionary with peak, flux, centroid, Strehl if perfect_psf is given,
        and encircled-energy metrics.
    """
    psf = np.asarray(psf, dtype=float)

    x_peak, y_peak = peak_location(psf)
    x_cent, y_cent = centroid(psf)

    radii = np.linspace(0.0, r_max, n_radii)
    ee = encircled_energy(
        psf,
        radii=radii,
        x0=float(x_peak),
        y0=float(y_peak),
        radial_scale=radial_scale,
    )

    result: Dict[str, object] = {
        "peak": peak_value(psf),
        "flux": total_flux(psf),
        "x_peak": x_peak,
        "y_peak": y_peak,
        "x_centroid": x_cent,
        "y_centroid": y_cent,
        "ee_radii": radii,
        "ee_curve": ee,
        "ee_radius": encircled_energy_radius(radii, ee, fraction=ee_fraction),
        "ee_fraction": ee_fraction,
    }

    if perfect_psf is not None:
        result["strehl"] = strehl_from_psfs(psf, perfect_psf)
    else:
        result["strehl"] = np.nan

    return result




# ============================================================
# Local PSF fitting helpers
# These use the existing fit functions inside beam_trace_ORIG.
# ============================================================

def extract_fit_parameter(fit_obj, candidate_keys):
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


def fit_psf_models(psf_crop, x_ld, y_ld, fit_module=None, moffat_fit_region="all") -> Dict[str, Any]:
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

    if fit_module is None:
        fit_gaussian = fit_2d_gaussian
        gaussian_metrics = gaussian_fwhm_and_ellipticity
        fit_moffat = fit_2d_moffat
        moffat_metrics = moffat_fwhm_and_ellipticity
    else:
        fit_gaussian = fit_module.fit_2d_gaussian
        gaussian_metrics = fit_module.gaussian_fwhm_and_ellipticity
        fit_moffat = fit_module.fit_2d_moffat
        moffat_metrics = fit_module.moffat_fwhm_and_ellipticity

    psf_crop = np.asarray(psf_crop, dtype=float)
    peak = np.nanmax(psf_crop)

    if not np.isfinite(peak) or peak <= 0:
        return result

    psf_crop_norm = psf_crop / peak

    try:
        gfit = fit_gaussian(x_ld, y_ld, psf_crop_norm)
        gmetrics = gaussian_metrics(gfit)

        result["gaussian_fwhm"] = gmetrics.get("fwhm_major", np.nan)
        result["gaussian_ell"] = gmetrics.get("ellipticity", np.nan)
        result["gaussian_fit"] = gfit
    except Exception:
        pass

    try:
        mfit = fit_moffat(
            x_ld,
            y_ld,
            psf_crop_norm,
            fit_region=moffat_fit_region,
        )
        mmetrics = moffat_metrics(mfit)

        result["moffat_fwhm"] = mmetrics.get("fwhm_major", np.nan)
        result["moffat_ell"] = mmetrics.get("ellipticity", np.nan)
        result["moffat_alpha"] = extract_fit_parameter(mfit, ["alpha", "alpha_x", "scale", "r0"])
        result["moffat_beta"] = extract_fit_parameter(mfit, ["beta", "power"])
        result["moffat_amplitude"] = extract_fit_parameter(mfit, ["amplitude", "amp", "A"])
        result["moffat_bg"] = extract_fit_parameter(mfit, ["background", "bg", "offset", "c0"])
        result["moffat_fit"] = mfit
    except Exception:
        pass

    return result


def analyse_psf_with_fits(
    psf: np.ndarray,
    perfect_psf: np.ndarray,
    angular_pixel_scale_ld: float,
    fit_module=None,
    crop_half_width_ld: float = 6.0,
    r_max_ld: float = 10.0,
    n_radii: int = 400,
    moffat_fit_region: str = "all",
) -> Dict[str, Any]:
    """
    Analyse PSF with Strehl, EE80, Gaussian fit, and Moffat fit.

    This mirrors the old psf_analysis.py logic but is self-contained here.
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
        "x_peak": np.nan,
        "y_peak": np.nan,
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

    total = np.nansum(psf)
    perfect_total = np.nansum(perfect_psf)

    if total <= 0 or perfect_total <= 0:
        return result

    result["strehl"] = strehl_from_psfs(psf, perfect_psf)

    psf_smooth = gaussian_filter(psf, sigma=2)
    peak_y, peak_x = np.unravel_index(np.nanargmax(psf_smooth), psf.shape)

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




##################################
# Transfering from beam_trace.py.. Need to refactor because probably overlap of functions within this module 

# ============================================================
# 2D Gaussian fitting for PSF
# ============================================================

def gaussian2d_rotated(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    ct = np.cos(theta)
    st = np.sin(theta)

    xp = ct * (x - x0) + st * (y - y0)
    yp = -st * (x - x0) + ct * (y - y0)

    g = offset + amp * np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))
    return g.ravel()


def moffat2d_rotated(coords, amp, x0, y0, alpha_x, alpha_y, beta, theta, offset):
    x, y = coords
    ct = np.cos(theta)
    st = np.sin(theta)

    xp = ct * (x - x0) + st * (y - y0)
    yp = -st * (x - x0) + ct * (y - y0)

    rr = (xp / alpha_x) ** 2 + (yp / alpha_y) ** 2
    m = offset + amp * (1.0 + rr) ** (-beta)
    return m.ravel()


def second_moment_initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    z = np.asarray(z, dtype=float)
    z = z - np.nanmin(z)
    if np.nansum(z) <= 0:
        return None

    z = z / np.nansum(z)
    x0 = np.nansum(x * z)
    y0 = np.nansum(y * z)

    xx = np.nansum((x - x0) ** 2 * z)
    yy = np.nansum((y - y0) ** 2 * z)
    xy = np.nansum((x - x0) * (y - y0) * z)

    cov = np.array([[xx, xy], [xy, yy]])
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)

    sigma_minor = np.sqrt(evals[0])
    sigma_major = np.sqrt(evals[1])
    vec_major = evecs[:, 1]
    theta = np.arctan2(vec_major[1], vec_major[0])

    amp = np.nanmax(z)
    offset = 0.0
    return amp, x0, y0, sigma_major, sigma_minor, theta, offset

#update with diffractiuon limits
def fit_2d_gaussian(x: np.ndarray, y: np.ndarray, z: np.ndarray, minx=0.1, miny=0.1) -> Dict[str, Any]:
    z = np.asarray(z, dtype=float)
    z_fit = z - np.nanmin(z)
    if np.nanmax(z_fit) > 0:
        z_fit = z_fit / np.nanmax(z_fit)

    guess = second_moment_initial_guess(x, y, z_fit)
    if guess is None:
        return {"success": False}

    if not SCIPY_AVAILABLE:
        amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "sigma_x": np.max( sigma_x, minx ),
            "sigma_y": np.max(sigma_y, miny),
            "theta": theta,
            "offset": offset,
            "method": "moments",
        }

    p0 = guess
    lower = [0.0, np.nanmin(x), np.nanmin(y), 1e-6, 1e-6, -np.pi, -0.5]
    upper = [2.0, np.nanmax(x), np.nanmax(y), 10.0, 10.0, np.pi, 1.0]

    try:
        popt, _ = curve_fit(
            gaussian2d_rotated,
            (x.ravel(), y.ravel()),
            z_fit.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        amp, x0, y0, sigma_x, sigma_y, theta, offset = popt
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "sigma_x": np.max( sigma_x, minx ),
            "sigma_y": np.max(sigma_y, miny),
            "theta": theta,
            "offset": offset,
            "method": "curve_fit",
        }
    except Exception:
        amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
            "offset": offset,
            "method": "moments_fallback",
        }


def fit_2d_moffat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    minx=1.0,
    miny=1.0,
    fit_region: str = "all",
    wing_quantile: float = 0.70,
) -> Dict[str, Any]:
    z = np.asarray(z, dtype=float)
    z_fit = z - np.nanmin(z)
    if np.nanmax(z_fit) > 0:
        z_fit = z_fit / np.nanmax(z_fit)

    guess = second_moment_initial_guess(x, y, z_fit)
    if guess is None:
        return {"success": False}

    _, x0_g, y0_g, sigma_x_g, sigma_y_g, theta_g, offset_g = guess
    amp_g = max(float(np.nanmax(z_fit) - np.nanmin(z_fit)), 1e-6)
    alpha_x_g = max(float(sigma_x_g), 1e-6)
    alpha_y_g = max(float(sigma_y_g), 1e-6)
    beta_g = 2.5

    if fit_region not in ("all", "wings"):
        raise ValueError("fit_region must be 'all' or 'wings'.")

    if fit_region == "wings":
        q = float(np.clip(wing_quantile, 0.0, 0.99))
        threshold = np.nanquantile(z_fit, q)
        fit_mask = z_fit <= threshold
        if np.count_nonzero(fit_mask) < 20:
            fit_mask = np.isfinite(z_fit)
    else:
        fit_mask = np.isfinite(z_fit)

    x_data = x[fit_mask]
    y_data = y[fit_mask]
    z_data = z_fit[fit_mask]

    if not SCIPY_AVAILABLE:
        return {
            "success": True,
            "amp": amp_g,
            "x0": x0_g,
            "y0": y0_g,
            "alpha_x": np.max([alpha_x_g, minx]),
            "alpha_y": np.max([alpha_y_g, miny]),
            "beta": beta_g,
            "theta": theta_g,
            "offset": offset_g,
            "method": "moments",
            "fit_region": fit_region,
        }

    p0 = [amp_g, x0_g, y0_g, max(alpha_x_g, minx), max(alpha_y_g, miny), beta_g, theta_g, offset_g]
    lower = [0.0, np.nanmin(x), np.nanmin(y), 1e-6, 1e-6, 1.05, -np.pi, -0.5]
    upper = [3.0, np.nanmax(x), np.nanmax(y), 20.0, 20.0, 20.0, np.pi, 1.0]

    try:
        popt, _ = curve_fit(
            moffat2d_rotated,
            (x_data.ravel(), y_data.ravel()),
            z_data.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=30000,
        )
        amp, x0, y0, alpha_x, alpha_y, beta, theta, offset = popt
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "alpha_x": np.max([alpha_x, minx]),
            "alpha_y": np.max([alpha_y, miny]),
            "beta": beta,
            "theta": theta,
            "offset": offset,
            "method": "curve_fit",
            "fit_region": fit_region,
        }
    except Exception:
        return {
            "success": True,
            "amp": amp_g,
            "x0": x0_g,
            "y0": y0_g,
            "alpha_x": np.max([alpha_x_g, minx]),
            "alpha_y": np.max([alpha_y_g, miny]),
            "beta": beta_g,
            "theta": theta_g,
            "offset": offset_g,
            "method": "moments_fallback",
            "fit_region": fit_region,
        }


def gaussian_fwhm_and_ellipticity(fit: Dict[str, Any]) -> Dict[str, float]:
    if not fit.get("success", False):
        return {
            "fwhm_major": np.nan,
            "fwhm_minor": np.nan,
            "ellipticity": np.nan,
        }

    sigma1 = max(fit["sigma_x"], fit["sigma_y"])
    sigma2 = min(fit["sigma_x"], fit["sigma_y"])
    factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    fwhm_major = factor * sigma1
    fwhm_minor = factor * sigma2
    ellipticity = 1.0 - fwhm_minor / fwhm_major if fwhm_major > 0 else np.nan

    return {
        "fwhm_major": float(fwhm_major),
        "fwhm_minor": float(fwhm_minor),
        "ellipticity": float(ellipticity),
    }


def moffat_fwhm_and_ellipticity(fit: Dict[str, Any]) -> Dict[str, float]:
    if not fit.get("success", False):
        return {
            "fwhm_major": np.nan,
            "fwhm_minor": np.nan,
            "ellipticity": np.nan,
        }

    alpha1 = max(fit["alpha_x"], fit["alpha_y"])
    alpha2 = min(fit["alpha_x"], fit["alpha_y"])
    beta = max(float(fit["beta"]), 1.000001)
    factor = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)

    fwhm_major = factor * alpha1
    fwhm_minor = factor * alpha2
    ellipticity = 1.0 - fwhm_minor / fwhm_major if fwhm_major > 0 else np.nan

    return {
        "fwhm_major": float(fwhm_major),
        "fwhm_minor": float(fwhm_minor),
        "ellipticity": float(ellipticity),
    }


def gaussian_halfmax_contour(fit: Dict[str, Any], npts: int = 300):
    if not fit.get("success", False):
        return None, None

    sigma_x = fit["sigma_x"]
    sigma_y = fit["sigma_y"]
    theta = fit["theta"]
    x0 = fit["x0"]
    y0 = fit["y0"]

    t = np.linspace(0.0, 2.0 * np.pi, npts)
    a = np.sqrt(2.0 * np.log(2.0)) * sigma_x
    b = np.sqrt(2.0 * np.log(2.0)) * sigma_y

    xp = a * np.cos(t)
    yp = b * np.sin(t)

    ct = np.cos(theta)
    st = np.sin(theta)

    xc = x0 + ct * xp - st * yp
    yc = y0 + st * xp + ct * yp
    return xc, yc




##### FFT HELPERS TAKEN FROM beam_trace.py , so may need additional refactoring




def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    psf = np.abs(ef) ** 2
    if np.max(psf) > 0:
        psf = psf / np.max(psf)

    ny, nx = field.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    return {"psf": psf, "fx": fx, "fy": fy}



def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
    x_ld = psf_pack["fx"] * pupil_diameter_m
    y_ld = psf_pack["fy"] * pupil_diameter_m
    return x_ld, y_ld




def crop_psf_to_lambda_over_d(
    psf: np.ndarray,
    x_ld: np.ndarray,
    y_ld: np.ndarray,
    half_width_ld: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
    iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
    if len(ix) == 0 or len(iy) == 0:
        return psf, x_ld, y_ld
    return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]





def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    mask = sample["mask"]
    phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
    amp = np.where(mask, sample["amplitude"], 0.0)
    field = amp * np.exp(1j * phase)
    out = fft_psf_from_pupil_field(field, sample["dx"])
    return {**sample, **out}