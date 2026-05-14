"""
hybrid_glao_movie_demo.py

Animated hybrid GLAO testbench demo.

This reproduces the old test_movie.py workflow using the refactored
optics_simulator modules where possible.

It visualises:
    - geometric 3D beam / phase-screen layout
    - rotating phase screens with LGS/science footprints
    - LGS mean reconstructed OPD
    - DM correction OPD
    - residual science OPD
    - science PSFs before and after GLAO switches on
    - instantaneous Moffat FWHM time series
    - cumulative long-exposure Moffat FWHM time series

Run from repository root:

    python3 -m optics_simulator.examples.hybrid_glao_movie_demo

Short interactive test:

    python3 -m optics_simulator.examples.hybrid_glao_movie_demo \
        --exposure-s 1.0 \
        --dt 0.2 \
        --ao-start-time 0.4 \
        --no-save \
        --show

Save movie:

    python3 -m optics_simulator.examples.hybrid_glao_movie_demo \
        --exposure-s 5.6 \
        --dt 0.1 \
        --ao-start-time 2.0 \
        --output glao_telemetry_ao_switch
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

from optics_simulator import beam_trace as bt
from optics_simulator import beam_trace_wave as btw
from optics_simulator import phase_tools
from optics_simulator import psf_tools
from optics_simulator import wave_optics


# ============================================================
# Phase-screen / correction helpers
# ============================================================

def get_rotating_screen_image(elem, t: float, npix: int = 256) -> np.ndarray:
    """
    Return OPD image of a rotating phase screen for visualisation.

    Output units are metres OPD.
    """
    r_max = elem.clear_radius
    u = np.linspace(-r_max, r_max, npix)
    uu, vv = np.meshgrid(u, u)
    uv_grid = np.stack([uu, vv], axis=-1)

    opd, valid = elem.sample_uv(uv_grid.reshape(-1, 2), t=t)
    opd = opd.reshape(uu.shape)
    valid = valid.reshape(uu.shape)

    return np.where(valid, opd, np.nan)


def apply_dm_correction_opd(opd_map_m: np.ndarray, acts_across: int, mask: np.ndarray) -> np.ndarray:
    """
    DM correction proxy in OPD space [m].

    The DM is modelled as a spatially limited corrector that removes only the
    low-spatial-frequency content of the reconstructed OPD.
    """
    opd_map_m = np.asarray(opd_map_m, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if acts_across <= 0 or not np.any(mask):
        return np.zeros_like(opd_map_m)

    avg_opd = np.nanmean(opd_map_m[mask])
    work_opd = np.where(mask, opd_map_m, avg_opd)

    sigma = (opd_map_m.shape[0] / float(acts_across)) * 0.5
    low_spatial = gaussian_filter(work_opd, sigma=sigma, mode="reflect")

    return np.where(mask, low_spatial, 0.0)


def robust_abs_percentile(values: np.ndarray, percentile: float = 99.0, default: float = 1.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return default

    out = np.nanpercentile(np.abs(vals), percentile)

    if not np.isfinite(out) or out <= 0:
        return default

    return float(out)


# ============================================================
# Bench / beam construction
# ============================================================

def build_bench_from_fits_movie_style(
    fits_path: Path,
    opd_scale_m_per_rad: float,
    fa_scale: float = 0.6,
    fa_x_m: float = 26e-3,
    gl_x_m: float = 34e-3,
):
    """
    Build a phase-screen bench from the FITS phase-screen file.

    The old movie script placed the FA screen around x=26 mm and the GL screens
    around x=34 mm. These remain defaults here for visual continuity.
    """
    layers = [
        {"label": "FA",  "z": -2.50,  "hz": 0.4, "scale": fa_scale, "x": fa_x_m},
        {"label": "GL3", "z": -0.060, "hz": 1.4, "scale": 1.0,      "x": gl_x_m},
        {"label": "GL2", "z": -0.030, "hz": 1.0, "scale": 1.0,      "x": gl_x_m},
        {"label": "GL1", "z": -0.001, "hz": 0.7, "scale": 1.0,      "x": gl_x_m},
    ]

    bench = bt.OpticalBench3D()

    with fits.open(fits_path) as hdul:
        pix_scale = float(hdul[0].header["PIXSCALE"])

        for layer in layers:
            label = layer["label"]

            # FITS values are treated as phase-like values in radians and
            # converted to OPD [m].
            opd = np.asarray(hdul[label].data, dtype=float) * opd_scale_m_per_rad
            opd = layer["scale"] * opd

            bench.add(
                bt.RotatingPhaseScreen3D(
                    point=[layer["x"], 0.0, layer["z"]],
                    normal=[0.0, 0.0, 1.0],
                    opd_map=opd,
                    map_extent_m=opd.shape[0] * pix_scale,
                    angular_velocity=2.0 * np.pi * layer["hz"],
                    label=label,
                )
            )

    return bench


def make_lgs_beams(
    wavelength: float,
    beam_diameter: float,
    source_plane_z: float,
    pupil_point: np.ndarray,
    lgs_coords_arcmin: List[Tuple[float, float]],
    nrings: int,
    nphi: int,
):
    beams = []

    for i, (x_arcmin, y_arcmin) in enumerate(lgs_coords_arcmin):
        beam = bt.make_converging_beam_from_field_angles(
            np.deg2rad(x_arcmin / 60.0),
            np.deg2rad(y_arcmin / 60.0),
            source_plane_z,
            pupil_point,
            beam_diameter,
            wavelength,
            f"LGS{i}",
            nrings,
            nphi,
        )
        beams.append(beam)

    return beams


def make_science_beams(
    science_angles_arcmin: List[float],
    wavelength: float,
    beam_diameter: float,
    source_plane_z: float,
    pupil_point: np.ndarray,
    nrings: int,
    nphi: int,
):
    beams = []

    for theta_arcmin in science_angles_arcmin:
        beam = bt.make_converging_beam_from_field_angles(
            np.deg2rad(theta_arcmin / 60.0),
            0.0,
            source_plane_z,
            pupil_point,
            beam_diameter,
            wavelength,
            f"Sci_{theta_arcmin:.1f}arcmin",
            nrings,
            nphi,
        )
        beams.append(beam)

    return beams


# ============================================================
# Science performance frame
# ============================================================

def simulate_sci_performance(
    beam,
    dm_opd_corr: np.ndarray,
    bench,
    t: float,
    pad_size: int,
    npix_pupil: int,
):
    """
    Simulate residual science pupil and PSF using Wavefront2D.
    """
    wf_sci, sample_sci = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=float(t),
        npix=npix_pupil,
        beam_trace_module=bt,
    )

    mask = np.asarray(sample_sci["mask"], dtype=bool)

    raw_opd = np.nan_to_num(sample_sci["opd_map_m"], nan=0.0)
    residual_opd = raw_opd - dm_opd_corr

    raw_phase = phase_tools.opd_to_phase(raw_opd, beam.wavelength)
    residual_phase = phase_tools.opd_to_phase(residual_opd, beam.wavelength)

    wf_corr = wave_optics.apply_opd(
        wf_sci,
        opd_map_m=-dm_opd_corr,
        mask=mask,
        label=f"{wf_sci.label}_movie_corrected",
    )

    psf = psf_tools.psf_from_wavefront(
        wf_corr,
        pad_to=pad_size,
        normalize=None,
    )

    return {
        "sample": sample_sci,
        "mask": mask,
        "wf_raw": wf_sci,
        "wf_corr": wf_corr,
        "raw_opd": raw_opd,
        "residual_opd": residual_opd,
        "raw_phase": raw_phase,
        "residual_phase": residual_phase,
        "psf": psf,
    }


def make_perfect_reference_psf(
    science_beam,
    npix_pupil: int,
    pad_size: int,
):
    """
    Build a perfect reference PSF with the same sampled pupil support.
    """
    empty_bench = bt.OpticalBench3D()

    wf_ref = btw.sample_wavefront_on_pupil_plane(
        beam=science_beam,
        bench=empty_bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=npix_pupil,
        beam_trace_module=bt,
    )

    wf_ref = wf_ref.with_field(wf_ref.amplitude.astype(complex), label="perfect_reference")

    return psf_tools.psf_from_wavefront(
        wf_ref,
        pad_to=pad_size,
        normalize=None,
    )


def analyse_single_psf(
    psf: np.ndarray,
    perfect_psf: np.ndarray,
    angular_pixel_scale_ld: float,
    fit_half_width_ld: float,
) -> Dict[str, float]:
    """
    Analyse one instantaneous or cumulative PSF.

    Returns at least Strehl and Moffat FWHM. If the Moffat fit fails, returns NaN.
    """
    try:
        result = psf_tools.analyse_psf_with_fits(
            psf,
            perfect_psf,
            angular_pixel_scale_ld=angular_pixel_scale_ld,
            crop_half_width_ld=fit_half_width_ld,
            moffat_fit_region="all",
        )
    except TypeError:
        # Compatibility with older local versions that still require fit_module.
        result = psf_tools.analyse_psf_with_fits(
            psf,
            perfect_psf,
            angular_pixel_scale_ld=angular_pixel_scale_ld,
            fit_module=bt,
            crop_half_width_ld=fit_half_width_ld,
            moffat_fit_region="all",
        )

    return {
        "strehl": float(result.get("strehl", np.nan)),
        "moffat_fwhm": float(result.get("moffat_fwhm", np.nan)),
        "moffat_beta": float(result.get("moffat_beta", np.nan)),
    }


# ============================================================
# Telemetry generation
# ============================================================

def run_glao_movie_telemetry(
    fits_path: Path,
    exposure_s: float = 5.6,
    dt: float = 0.1,
    ao_start_time: float = 2.0,
    fa_scale: float = 0.6,
    dm_acts_across: int = 35,
    wfs_wavelength: float = 589e-9,
    science_wavelength: float = 0.589e-6,
    beam_diameter: float = 13e-3,
    npix_pupil: int = 256,
    pad_size: int = 2048,
    source_plane_z: float = -3.25,
    opd_scale_m_per_rad: float = 500e-9 / (2.0 * np.pi),
    fa_x_m: float = 26e-3,
    gl_x_m: float = 34e-3,
    lgs_radius_arcmin: float = 10.0,
    science_offsets_arcmin: Tuple[float, ...] = (0.0, 5.0, 10.0),
    fit_half_width_ld: float = 6.0,
):
    """
    Run the animated GLAO telemetry simulation.

    Important:
        LGS reconstruction is performed using sample["opd_map_m"], i.e. the
        unwrapped OPD from the geometric phase-screen sampling.
    """
    pupil_point = np.array([0.0, 0.0, 0.0])

    bench = build_bench_from_fits_movie_style(
        fits_path=fits_path,
        opd_scale_m_per_rad=opd_scale_m_per_rad,
        fa_scale=fa_scale,
        fa_x_m=fa_x_m,
        gl_x_m=gl_x_m,
    )

    lgs_coords = [
        (+lgs_radius_arcmin, +lgs_radius_arcmin),
        (-lgs_radius_arcmin, +lgs_radius_arcmin),
        (+lgs_radius_arcmin, -lgs_radius_arcmin),
        (-lgs_radius_arcmin, -lgs_radius_arcmin),
    ]

    lgs_beams = make_lgs_beams(
        wavelength=wfs_wavelength,
        beam_diameter=beam_diameter,
        source_plane_z=source_plane_z,
        pupil_point=pupil_point,
        lgs_coords_arcmin=lgs_coords,
        nrings=3,
        nphi=12,
    )

    lgs_trace_beams = make_lgs_beams(
        wavelength=wfs_wavelength,
        beam_diameter=beam_diameter,
        source_plane_z=source_plane_z,
        pupil_point=pupil_point,
        lgs_coords_arcmin=lgs_coords,
        nrings=1,
        nphi=4,
    )

    sci_beams = make_science_beams(
        science_angles_arcmin=list(science_offsets_arcmin),
        wavelength=science_wavelength,
        beam_diameter=beam_diameter,
        source_plane_z=source_plane_z,
        pupil_point=pupil_point,
        nrings=3,
        nphi=12,
    )

    perfect_psf = make_perfect_reference_psf(
        science_beam=sci_beams[0],
        npix_pupil=npix_pupil,
        pad_size=pad_size,
    )

    # More accurate FFT lambda/D pixel scale:
    # lambda/D coordinate per FFT pixel = D / (N_pad * dx_pupil)
    wf_for_scale = btw.sample_wavefront_on_pupil_plane(
        beam=sci_beams[0],
        bench=bt.OpticalBench3D(),
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=npix_pupil,
        beam_trace_module=bt,
    )
    angular_pixel_scale_ld = beam_diameter / (pad_size * wf_for_scale.dx)

    times = np.arange(0.0, exposure_s, dt)

    telemetry = {
        "times": times,
        "bench": bench,
        "frames": [],
        "sci_angles": list(science_offsets_arcmin),
        "lgs_beams": lgs_beams,
        "lgs_trace_beams": lgs_trace_beams,
        "sci_beams": sci_beams,
        "wfs_wavelength": wfs_wavelength,
        "science_wavelength": science_wavelength,
        "beam_diameter": beam_diameter,
        "npix_pupil": npix_pupil,
        "pad_size": pad_size,
        "dm_acts_across": dm_acts_across,
        "ao_start_time": ao_start_time,
        "fa_scale": fa_scale,
        "perfect_psf": perfect_psf,
        "angular_pixel_scale_ld": angular_pixel_scale_ld,
        "fit_half_width_ld": fit_half_width_ld,
    }

    cumulative_psf_sum = [
        np.zeros((pad_size, pad_size), dtype=float)
        for _ in sci_beams
    ]

    print("Generating movie telemetry")
    print("  FITS path:", fits_path)
    print("  frames:", len(times))
    print("  exposure [s]:", exposure_s)
    print("  dt [s]:", dt)
    print("  AO start [s]:", ao_start_time)
    print("  WFS wavelength [nm]:", wfs_wavelength * 1e9)
    print("  science wavelength [um]:", science_wavelength * 1e6)
    print("  DM acts across:", dm_acts_across)
    print("  FA scale:", fa_scale)

    for idx, t in enumerate(times):
        frame = {"t": float(t)}

        lgs_samples = []

        for beam in lgs_beams:
            _, sample_lgs = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
                beam=beam,
                bench=bench,
                pupil_point=pupil_point,
                t=float(t),
                npix=npix_pupil,
                beam_trace_module=bt,
            )
            lgs_samples.append(sample_lgs)

        lgs_mask = np.asarray(lgs_samples[0]["mask"], dtype=bool)

        # Use unwrapped OPD maps from the beam_trace sample.
        lgs_opd_maps = [
            np.nan_to_num(s["opd_map_m"], nan=0.0)
            for s in lgs_samples
        ]

        avg_lgs_opd = np.nanmean(lgs_opd_maps, axis=0)
        avg_lgs_opd = np.where(lgs_mask, avg_lgs_opd, 0.0)

        ao_on = bool(t >= ao_start_time)

        if ao_on:
            dm_opd = apply_dm_correction_opd(
                avg_lgs_opd,
                acts_across=dm_acts_across,
                mask=lgs_mask,
            )
        else:
            dm_opd = np.zeros_like(avg_lgs_opd)

        frame["ao_on"] = ao_on
        frame["recon_opd"] = avg_lgs_opd
        frame["dm_opd"] = dm_opd
        frame["recon_phase_wfs"] = phase_tools.opd_to_phase(avg_lgs_opd, wfs_wavelength)
        frame["dm_phase_wfs"] = phase_tools.opd_to_phase(dm_opd, wfs_wavelength)
        frame["lgs_mask"] = lgs_mask

        frame["sci"] = [
            simulate_sci_performance(
                beam=b,
                dm_opd_corr=dm_opd,
                bench=bench,
                t=float(t),
                pad_size=pad_size,
                npix_pupil=npix_pupil,
            )
            for b in sci_beams
        ]

        frame["strehl"] = []
        frame["inst_moffat_fwhm"] = []
        frame["inst_moffat_beta"] = []
        frame["cum_moffat_fwhm"] = []
        frame["cum_moffat_beta"] = []
        frame["cum_strehl"] = []

        for i, sci in enumerate(frame["sci"]):
            inst_metrics = analyse_single_psf(
                sci["psf"],
                perfect_psf,
                angular_pixel_scale_ld=angular_pixel_scale_ld,
                fit_half_width_ld=fit_half_width_ld,
            )

            cumulative_psf_sum[i] += sci["psf"]
            cumulative_psf = cumulative_psf_sum[i] / float(idx + 1)

            cum_metrics = analyse_single_psf(
                cumulative_psf,
                perfect_psf,
                angular_pixel_scale_ld=angular_pixel_scale_ld,
                fit_half_width_ld=fit_half_width_ld,
            )

            frame["strehl"].append(inst_metrics["strehl"])
            frame["inst_moffat_fwhm"].append(inst_metrics["moffat_fwhm"])
            frame["inst_moffat_beta"].append(inst_metrics["moffat_beta"])
            frame["cum_moffat_fwhm"].append(cum_metrics["moffat_fwhm"])
            frame["cum_moffat_beta"].append(cum_metrics["moffat_beta"])
            frame["cum_strehl"].append(cum_metrics["strehl"])

        telemetry["frames"].append(frame)

        if (idx + 1) % max(1, len(times) // 10) == 0 or idx == len(times) - 1:
            print(f"  frame {idx + 1}/{len(times)}")

    return telemetry


# ============================================================
# Movie / visualisation
# ============================================================

def draw_static_3d_panel(ax3d, tel):
    """
    Draw the static geometric bench panel.
    """
    bench = tel["bench"]

    ax3d.set_title("Geometric testbench")

    for beam in tel["lgs_trace_beams"]:
        paths, _ = bench.trace_beam(beam, s_end=0.1, t=0.0)
        for path in paths:
            ax3d.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                lw=0.8,
                alpha=0.45,
                color="orange",
            )

    for beam in tel["sci_beams"]:
        chief = beam.chief_ray
        p0 = chief.r
        p1 = p0 + 3.5 * chief.d
        ax3d.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            lw=1.0,
            alpha=0.8,
            color="tab:blue",
        )

    for elem in bench.elements:
        e1, e2 = elem.plane_basis()
        rad = elem.clear_radius if isinstance(elem, bt.RotatingPhaseScreen3D) else 1e-3
        circ = np.linspace(0.0, 2.0 * np.pi, 160)
        ring = elem.point[None, :] + rad * (
            np.cos(circ)[:, None] * e1[None, :]
            + np.sin(circ)[:, None] * e2[None, :]
        )
        ax3d.plot(ring[:, 0], ring[:, 1], ring[:, 2], "k-", lw=1.3)
        ax3d.text(elem.point[0], elem.point[1], elem.point[2], elem.label)

    ax3d.scatter([0.0], [0.0], [0.0], marker="x", s=60, color="k")
    ax3d.text(0.0, 0.0, 0.0, " pupil")

    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_box_aspect([1.0, 1.0, 1.5])

    try:
        ax3d.view_init(elev=18.0, azim=-70.0)
    except Exception:
        pass


def crop_psf_for_display(psf: np.ndarray, half_width_pix: int):
    """
    Crop PSF around its brightest pixel.
    """
    psf = np.asarray(psf, dtype=float)

    if np.nanmax(psf) <= 0:
        cy, cx = np.array(psf.shape) // 2
    else:
        cy, cx = np.unravel_index(np.nanargmax(psf), psf.shape)

    y0 = max(0, cy - half_width_pix)
    y1 = min(psf.shape[0], cy + half_width_pix + 1)
    x0 = max(0, cx - half_width_pix)
    x1 = min(psf.shape[1], cx + half_width_pix + 1)

    return psf[y0:y1, x0:x1]


def psf_display_image(
    psf_crop: np.ndarray,
    perfect_peak: float,
    stretch: str,
    vmax: Optional[float],
):
    """
    Convert raw PSF crop to display image.

    The crop is normalized by the perfect PSF peak, not by its own peak, so
    Strehl/peak improvement remains visible.
    """
    if perfect_peak > 0:
        crop_rel = psf_crop / perfect_peak
    else:
        crop_rel = psf_crop.copy()

    if stretch == "linear":
        image = crop_rel
        vmin = 0.0
        vmax_use = 1.0 if vmax is None else vmax

    elif stretch == "sqrt":
        vmax_base = 1.0 if vmax is None else vmax
        image = np.sqrt(np.maximum(crop_rel, 0.0))
        vmin = 0.0
        vmax_use = np.sqrt(max(vmax_base, 1e-12))

    elif stretch == "asinh":
        scale = 1.0 if vmax is None else vmax
        image = np.arcsinh(crop_rel / max(scale, 1e-12))
        vmin = 0.0
        vmax_use = np.arcsinh(1.0)

    elif stretch == "log":
        vmax_base = 1.0 if vmax is None else vmax
        image = np.log10(np.maximum(crop_rel, 1e-6))
        vmin = -6.0
        vmax_use = np.log10(max(vmax_base, 1e-6))

    else:
        raise ValueError("stretch must be 'linear', 'sqrt', 'asinh', or 'log'.")

    return image, vmin, vmax_use


def collect_metric_array(tel: Dict, metric_name: str) -> np.ndarray:
    """
    Return array with shape (n_frames, n_science_fields) for a frame metric.
    """
    arr = []
    for frame in tel["frames"]:
        arr.append(frame[metric_name])
    return np.asarray(arr, dtype=float)


def make_movie(
    tel: Dict,
    base_filename: str = "glao_telemetry_ao_switch",
    save: bool = True,
    show: bool = False,
    interval_ms: int = 250,
    fps: int = 8,
    screen_npix: int = 256,
    psf_half_width_pix: int = 48,
    psf_stretch: str = "linear",
    psf_vmax: Optional[float] = None,
):
    """
    Create and optionally save the movie.

    If ffmpeg is unavailable, falls back to GIF/pillow.
    """
    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(4, 4, width_ratios=[1.35, 1.0, 1.0, 1.0])

    try:
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        draw_static_3d_panel(ax3d, tel)
    except Exception as exc:
        ax3d = fig.add_subplot(gs[:, 0])
        ax3d.text(
            0.5,
            0.5,
            f"3D projection unavailable\n{exc}",
            ha="center",
            va="center",
            transform=ax3d.transAxes,
        )
        ax3d.axis("off")

    ax_scr = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    ax_pup = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    ax_psf = [fig.add_subplot(gs[i, 3]) for i in range(3)]

    ax_inst_fwhm = fig.add_subplot(gs[3, 2])
    ax_cum_fwhm = fig.add_subplot(gs[3, 3])

    # --------------------------------------------------------
    # Precompute common colour scales and metric arrays
    # --------------------------------------------------------
    subset = tel["frames"][::max(1, len(tel["frames"]) // 10)]

    screen_vals = []
    recon_vals = []
    psf_vals_rel = []

    perfect_peak = np.nanmax(tel["perfect_psf"])

    for frame in subset:
        for elem in tel["bench"].elements:
            img = get_rotating_screen_image(elem, frame["t"], npix=screen_npix) * 1e9
            screen_vals.append(img[np.isfinite(img)])

        mask = frame["lgs_mask"]
        recon_vals.append(frame["recon_opd"][mask] * 1e9)
        recon_vals.append(frame["dm_opd"][mask] * 1e9)

        for sci in frame["sci"]:
            crop = crop_psf_for_display(sci["psf"], psf_half_width_pix)
            if perfect_peak > 0:
                psf_vals_rel.append((crop / perfect_peak).ravel())

    screen_vmax_nm = robust_abs_percentile(np.concatenate(screen_vals), 99.0, default=500.0)
    recon_vmax_nm = robust_abs_percentile(np.concatenate(recon_vals), 99.0, default=500.0)

    if psf_vmax is None and psf_vals_rel:
        candidate = np.nanpercentile(np.concatenate(psf_vals_rel), 99.8)
        if np.isfinite(candidate) and candidate > 0:
            psf_vmax = min(1.0, max(0.1, 1.1 * candidate))
        else:
            psf_vmax = 1.0

    times = np.asarray(tel["times"], dtype=float)
    sci_angles = tel["sci_angles"]

    inst_fwhm = collect_metric_array(tel, "inst_moffat_fwhm")
    cum_fwhm = collect_metric_array(tel, "cum_moffat_fwhm")

    all_fwhm = np.concatenate(
        [
            inst_fwhm[np.isfinite(inst_fwhm)],
            cum_fwhm[np.isfinite(cum_fwhm)],
        ]
    )

    if all_fwhm.size > 0:
        fwhm_ymax = 1.15 * np.nanpercentile(all_fwhm, 95.0)
        if not np.isfinite(fwhm_ymax) or fwhm_ymax <= 0:
            fwhm_ymax = 5.0
    else:
        fwhm_ymax = 5.0

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def draw_time_series_axis(ax, metric, idx, title, ylabel):
        ax.clear()

        for j, angle in enumerate(sci_angles):
            color = colors[j % len(colors)]
            ax.plot(
                times[: idx + 1],
                metric[: idx + 1, j],
                "-",
                lw=1.8,
                color=color,
                label=f"{angle:.1f}'",
            )

        if tel["ao_start_time"] is not None:
            if times[0] <= tel["ao_start_time"] <= times[-1]:
                ax.axvline(
                    tel["ao_start_time"],
                    color="k",
                    ls="--",
                    lw=1.2,
                    alpha=0.8,
                    label="AO on" if idx == 0 else None,
                )

        ax.axvline(times[idx], color="0.4", ls=":", lw=1.2)

        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(0.0, fwhm_ymax)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.legend(fontsize=8, loc="upper right")

    def update(idx):
        frame = tel["frames"][idx]
        t = frame["t"]

        for ax in ax_scr + ax_pup + ax_psf:
            ax.clear()

        # ----------------------------------------------------
        # Column 2: rotating screens and footprints
        # ----------------------------------------------------
        for i, elem in enumerate(tel["bench"].elements):
            img_nm = get_rotating_screen_image(elem, t, npix=screen_npix) * 1e9
            r_mm = elem.clear_radius * 1e3

            ax_scr[i].imshow(
                img_nm,
                cmap="RdBu_r",
                origin="lower",
                extent=[-r_mm, r_mm, -r_mm, r_mm],
                vmin=-screen_vmax_nm,
                vmax=screen_vmax_nm,
                interpolation="nearest",
            )
            ax_scr[i].set_title(f"{elem.label} OPD [nm]")

            # LGS footprints
            for beam in tel["lgs_beams"]:
                inter = tel["bench"].trace_chief_intersections(beam, t=t)
                if elem.label in inter:
                    u, v = elem.local_coordinates(inter[elem.label]["point"])
                    ax_scr[i].add_patch(
                        plt.Circle(
                            (u * 1e3, v * 1e3),
                            0.5 * tel["beam_diameter"] * 1e3,
                            color="red",
                            fill=False,
                            lw=1.0,
                            alpha=0.9,
                        )
                    )

            # Science footprints
            for beam in tel["sci_beams"]:
                inter = tel["bench"].trace_chief_intersections(beam, t=t)
                if elem.label in inter:
                    u, v = elem.local_coordinates(inter[elem.label]["point"])
                    ax_scr[i].scatter(
                        u * 1e3,
                        v * 1e3,
                        marker="x",
                        color="white",
                        s=25,
                        linewidths=1.2,
                    )

            ax_scr[i].set_xlabel("u [mm]")
            ax_scr[i].set_ylabel("v [mm]")

        # ----------------------------------------------------
        # Column 3: pupil diagnostics
        # ----------------------------------------------------
        mask = frame["lgs_mask"]

        recon_nm = np.where(mask, frame["recon_opd"] * 1e9, np.nan)
        dm_nm = np.where(mask, frame["dm_opd"] * 1e9, np.nan)

        sci_display = frame["sci"][-1]
        sci_mask = sci_display["mask"]
        residual_nm = np.where(sci_mask, sci_display["residual_opd"] * 1e9, np.nan)

        pupil_maps = [
            ("LGS mean recon OPD [nm]", recon_nm),
            ("DM correction OPD [nm]", dm_nm),
            (f"Residual science OPD ({tel['sci_angles'][-1]:.1f}') [nm]", residual_nm),
        ]

        for ax, (title, image) in zip(ax_pup, pupil_maps):
            ax.imshow(
                image,
                origin="lower",
                cmap="RdBu_r",
                vmin=-recon_vmax_nm,
                vmax=recon_vmax_nm,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.axis("off")

        # ----------------------------------------------------
        # Column 4: science PSFs
        # ----------------------------------------------------
        for i, sci in enumerate(frame["sci"]):
            psf = sci["psf"]

            crop = crop_psf_for_display(psf, psf_half_width_pix)

            crop_show, vmin, vmax = psf_display_image(
                crop,
                perfect_peak=perfect_peak,
                stretch=psf_stretch,
                vmax=psf_vmax,
            )

            ax_psf[i].imshow(
                crop_show,
                cmap="magma",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            strehl = frame["strehl"][i]
            fwhm = frame["inst_moffat_fwhm"][i]
            ax_psf[i].set_title(
                f"PSF {tel['sci_angles'][i]:.1f}'\n"
                f"S={strehl:.3f}, FWHM={fwhm:.2f} λ/D"
            )
            ax_psf[i].axis("off")

        # ----------------------------------------------------
        # Bottom row: Moffat FWHM time series
        # ----------------------------------------------------
        draw_time_series_axis(
            ax_inst_fwhm,
            inst_fwhm,
            idx,
            "Instantaneous Moffat FWHM",
            "FWHM [λ/D]",
        )

        draw_time_series_axis(
            ax_cum_fwhm,
            cum_fwhm,
            idx,
            "Cumulative long-exposure Moffat FWHM",
            "FWHM [λ/D]",
        )

        ao_state = "GLAO ON" if frame["ao_on"] else "AO OFF"

        fig.suptitle(
            (
                f"t = {t:.2f} s   |   {ao_state}   |   "
                f"DM acts = {tel['dm_acts_across']}   |   "
                f"FA scale = {tel['fa_scale']:.2f}   |   "
                f"PSF stretch = {psf_stretch}, vmax = {psf_vmax:.3g}"
            ),
            fontsize=15,
        )

        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(tel["frames"]),
        interval=interval_ms,
        blit=False,
    )

    if save:
        output_path = Path(base_filename)

        if output_path.suffix.lower() in [".mp4", ".gif"]:
            base = output_path.with_suffix("")
            suffix = output_path.suffix.lower()
        else:
            base = output_path
            suffix = ".mp4"

        if suffix == ".gif":
            gif_path = str(base) + ".gif"
            ani.save(gif_path, writer="pillow", fps=fps)
            print(f"Saved movie to {gif_path}")
        else:
            mp4_path = str(base) + ".mp4"
            try:
                ani.save(mp4_path, writer="ffmpeg", fps=fps)
                print(f"Saved movie to {mp4_path}")
            except Exception as exc:
                gif_path = str(base) + ".gif"
                print(f"ffmpeg failed with: {exc}")
                print("Falling back to GIF/pillow.")
                ani.save(gif_path, writer="pillow", fps=fps)
                print(f"Saved movie to {gif_path}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_fits = (
        repo_root
        / "phasescreens"
        / "scrns_2_order_v2"
        / "Testbench_phasescreens_20260506.fits"
    )

    parser = argparse.ArgumentParser(description="Animated hybrid GLAO movie demo.")

    parser.add_argument("--fits-path", type=Path, default=default_fits)
    parser.add_argument("--output", type=str, default="glao_telemetry_ao_switch")

    parser.add_argument("--exposure-s", type=float, default=5.6)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--ao-start-time", type=float, default=2.0)

    parser.add_argument("--fa-scale", type=float, default=0.6)
    parser.add_argument("--dm-acts-across", type=int, default=35)

    parser.add_argument("--wfs-wavelength", type=float, default=589e-9)
    parser.add_argument("--science-wavelength", type=float, default=0.589e-6)
    parser.add_argument("--beam-diameter", type=float, default=13e-3)
    parser.add_argument("--source-plane-z", type=float, default=-3.25)

    parser.add_argument("--npix-pupil", type=int, default=256)
    parser.add_argument("--pad-size", type=int, default=2048)

    parser.add_argument("--fa-x-mm", type=float, default=26.0)
    parser.add_argument("--gl-x-mm", type=float, default=34.0)

    parser.add_argument(
        "--opd-scale",
        type=float,
        default=500e-9 / (2.0 * np.pi),
        help="Conversion factor from FITS values to OPD [m].",
    )

    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--screen-npix", type=int, default=256)
    parser.add_argument("--psf-half-width-pix", type=int, default=48)
    parser.add_argument("--fit-half-width-ld", type=float, default=6.0)

    parser.add_argument(
        "--psf-stretch",
        type=str,
        default="linear",
        choices=["linear", "sqrt", "asinh", "log"],
        help="Display stretch for PSF panels.",
    )

    parser.add_argument(
        "--psf-vmax",
        type=float,
        default=None,
        help="Maximum PSF display value relative to perfect PSF peak.",
    )

    parser.add_argument("--save", dest="save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--show", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.fits_path.exists():
        raise FileNotFoundError(f"Could not find FITS file: {args.fits_path}")

    tel = run_glao_movie_telemetry(
        fits_path=args.fits_path,
        exposure_s=args.exposure_s,
        dt=args.dt,
        ao_start_time=args.ao_start_time,
        fa_scale=args.fa_scale,
        dm_acts_across=args.dm_acts_across,
        wfs_wavelength=args.wfs_wavelength,
        science_wavelength=args.science_wavelength,
        beam_diameter=args.beam_diameter,
        npix_pupil=args.npix_pupil,
        pad_size=args.pad_size,
        source_plane_z=args.source_plane_z,
        opd_scale_m_per_rad=args.opd_scale,
        fa_x_m=args.fa_x_mm * 1e-3,
        gl_x_m=args.gl_x_mm * 1e-3,
        fit_half_width_ld=args.fit_half_width_ld,
    )

    make_movie(
        tel,
        base_filename=args.output,
        save=args.save,
        show=args.show,
        interval_ms=args.interval_ms,
        fps=args.fps,
        screen_npix=args.screen_npix,
        psf_half_width_pix=args.psf_half_width_pix,
        psf_stretch=args.psf_stretch,
        psf_vmax=args.psf_vmax,
    )


if __name__ == "__main__":
    main()


# """
# hybrid_glao_movie_demo.py

# Animated hybrid GLAO testbench demo.

# This reproduces the old test_movie.py workflow using the refactored
# optics_simulator modules where possible.

# It visualises:
#     - geometric 3D beam / phase-screen layout
#     - rotating phase screens with LGS/science footprints
#     - LGS mean reconstructed OPD
#     - DM correction OPD
#     - residual science OPD/phase
#     - science PSFs before and after GLAO switches on

# Run from repository root:

#     python3 -m optics_simulator.examples.hybrid_glao_movie_demo

# Short test run:

#     python3 -m optics_simulator.examples.hybrid_glao_movie_demo \
#         --exposure-s 1.0 \
#         --dt 0.2 \
#         --ao-start-time 0.4 \
#         --output glao_movie_test \
#         --no-save \
#         --show

# Save movie:

#     python3 -m optics_simulator.examples.hybrid_glao_movie_demo \
#         --exposure-s 5.6 \
#         --dt 0.1 \
#         --ao-start-time 2.0 \
#         --output glao_telemetry_ao_switch
# """

# from __future__ import annotations

# import argparse
# import warnings
# from pathlib import Path
# from typing import Dict, List, Tuple

# import matplotlib.pyplot as plt
# import numpy as np
# from astropy.io import fits
# from matplotlib.animation import FuncAnimation
# from scipy.ndimage import gaussian_filter

# warnings.filterwarnings(
#     "ignore",
#     message="Unable to import Axes3D.*",
#     category=UserWarning,
# )

# from optics_simulator import beam_trace as bt
# from optics_simulator import beam_trace_wave as btw
# from optics_simulator import phase_tools
# from optics_simulator import psf_tools
# from optics_simulator import wave_optics


# # ============================================================
# # Phase-screen / correction helpers
# # ============================================================

# def get_rotating_screen_image(elem, t: float, npix: int = 256) -> np.ndarray:
#     """
#     Return OPD image of a rotating phase screen for visualisation.

#     Output units are metres OPD.
#     """
#     r_max = elem.clear_radius
#     u = np.linspace(-r_max, r_max, npix)
#     uu, vv = np.meshgrid(u, u)
#     uv_grid = np.stack([uu, vv], axis=-1)

#     opd, valid = elem.sample_uv(uv_grid.reshape(-1, 2), t=t)
#     opd = opd.reshape(uu.shape)
#     valid = valid.reshape(uu.shape)

#     return np.where(valid, opd, np.nan)


# def apply_dm_correction_opd(opd_map_m: np.ndarray, acts_across: int, mask: np.ndarray) -> np.ndarray:
#     """
#     DM correction proxy in OPD space [m].

#     The DM is modelled as a spatially limited corrector that removes only the
#     low spatial frequency content of the reconstructed OPD.
#     """
#     opd_map_m = np.asarray(opd_map_m, dtype=float)
#     mask = np.asarray(mask, dtype=bool)

#     if acts_across <= 0:
#         return np.zeros_like(opd_map_m)

#     if not np.any(mask):
#         return np.zeros_like(opd_map_m)

#     avg_opd = np.nanmean(opd_map_m[mask])
#     work_opd = np.where(mask, opd_map_m, avg_opd)

#     sigma = (opd_map_m.shape[0] / float(acts_across)) * 0.5
#     low_spatial = gaussian_filter(work_opd, sigma=sigma, mode="reflect")

#     return np.where(mask, low_spatial, 0.0)


# def robust_abs_percentile(values: np.ndarray, percentile: float = 99.0, default: float = 1.0) -> float:
#     vals = np.asarray(values, dtype=float)
#     vals = vals[np.isfinite(vals)]

#     if vals.size == 0:
#         return default

#     out = np.nanpercentile(np.abs(vals), percentile)

#     if not np.isfinite(out) or out <= 0:
#         return default

#     return float(out)


# # ============================================================
# # Bench / beam construction
# # ============================================================

# def build_bench_from_fits_movie_style(
#     fits_path: Path,
#     opd_scale_m_per_rad: float,
#     fa_scale: float = 0.6,
#     fa_x_m: float = 26e-3,
#     gl_x_m: float = 34e-3,
# ):
#     """
#     Build a phase-screen bench from the FITS phase-screen file.

#     The old movie script placed the FA screen around x=26 mm and the GL screens
#     around x=34 mm. These remain the defaults here for visual continuity.
#     """
#     layers = [
#         {"label": "FA",  "z": -2.50,  "hz": 0.4, "scale": fa_scale, "x": fa_x_m},
#         {"label": "GL3", "z": -0.060, "hz": 1.4, "scale": 1.0,      "x": gl_x_m},
#         {"label": "GL2", "z": -0.030, "hz": 1.0, "scale": 1.0,      "x": gl_x_m},
#         {"label": "GL1", "z": -0.001, "hz": 0.7, "scale": 1.0,      "x": gl_x_m},
#     ]

#     bench = bt.OpticalBench3D()

#     with fits.open(fits_path) as hdul:
#         pix_scale = float(hdul[0].header["PIXSCALE"])

#         for layer in layers:
#             label = layer["label"]

#             # FITS values are treated as phase-like values in radians and
#             # converted to OPD [m].
#             opd = np.asarray(hdul[label].data, dtype=float) * opd_scale_m_per_rad
#             opd = layer["scale"] * opd

#             bench.add(
#                 bt.RotatingPhaseScreen3D(
#                     point=[layer["x"], 0.0, layer["z"]],
#                     normal=[0.0, 0.0, 1.0],
#                     opd_map=opd,
#                     map_extent_m=opd.shape[0] * pix_scale,
#                     angular_velocity=2.0 * np.pi * layer["hz"],
#                     label=label,
#                 )
#             )

#     return bench


# def make_lgs_beams(
#     wavelength: float,
#     beam_diameter: float,
#     source_plane_z: float,
#     pupil_point: np.ndarray,
#     lgs_coords_arcmin: List[Tuple[float, float]],
#     nrings: int,
#     nphi: int,
# ):
#     beams = []

#     for i, (x_arcmin, y_arcmin) in enumerate(lgs_coords_arcmin):
#         beam = bt.make_converging_beam_from_field_angles(
#             np.deg2rad(x_arcmin / 60.0),
#             np.deg2rad(y_arcmin / 60.0),
#             source_plane_z,
#             pupil_point,
#             beam_diameter,
#             wavelength,
#             f"LGS{i}",
#             nrings,
#             nphi,
#         )
#         beams.append(beam)

#     return beams


# def make_science_beams(
#     science_angles_arcmin: List[float],
#     wavelength: float,
#     beam_diameter: float,
#     source_plane_z: float,
#     pupil_point: np.ndarray,
#     nrings: int,
#     nphi: int,
# ):
#     beams = []

#     for theta_arcmin in science_angles_arcmin:
#         beam = bt.make_converging_beam_from_field_angles(
#             np.deg2rad(theta_arcmin / 60.0),
#             0.0,
#             source_plane_z,
#             pupil_point,
#             beam_diameter,
#             wavelength,
#             f"Sci_{theta_arcmin:.1f}arcmin",
#             nrings,
#             nphi,
#         )
#         beams.append(beam)

#     return beams


# # ============================================================
# # Science performance frame
# # ============================================================

# def simulate_sci_performance(
#     beam,
#     dm_opd_corr: np.ndarray,
#     bench,
#     t: float,
#     pad_size: int,
#     npix_pupil: int,
# ):
#     """
#     Simulate residual science pupil and PSF using Wavefront2D.

#     Parameters
#     ----------
#     beam:
#         Science Beam3D.
#     dm_opd_corr:
#         DM correction in OPD [m].
#     bench:
#         OpticalBench3D.
#     t:
#         Time [s].
#     pad_size:
#         FFT pad size.
#     npix_pupil:
#         Pupil sampling.

#     Returns
#     -------
#     dict
#         Contains raw/residual OPD and phase, raw/corrected wavefronts,
#         PSF, Strehl placeholder fields, and sample dictionary.
#     """
#     wf_sci, sample_sci = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
#         beam=beam,
#         bench=bench,
#         pupil_point=(0.0, 0.0, 0.0),
#         t=float(t),
#         npix=npix_pupil,
#         beam_trace_module=bt,
#     )

#     mask = np.asarray(sample_sci["mask"], dtype=bool)

#     raw_opd = np.nan_to_num(sample_sci["opd_map_m"], nan=0.0)
#     residual_opd = raw_opd - dm_opd_corr

#     raw_phase = phase_tools.opd_to_phase(raw_opd, beam.wavelength)
#     residual_phase = phase_tools.opd_to_phase(residual_opd, beam.wavelength)

#     wf_corr = wave_optics.apply_opd(
#         wf_sci,
#         opd_map_m=-dm_opd_corr,
#         mask=mask,
#         label=f"{wf_sci.label}_movie_corrected",
#     )

#     psf = psf_tools.psf_from_wavefront(
#         wf_corr,
#         pad_to=pad_size,
#         normalize=None,
#     )

#     return {
#         "sample": sample_sci,
#         "mask": mask,
#         "wf_raw": wf_sci,
#         "wf_corr": wf_corr,
#         "raw_opd": raw_opd,
#         "residual_opd": residual_opd,
#         "raw_phase": raw_phase,
#         "residual_phase": residual_phase,
#         "psf": psf,
#     }


# def make_perfect_reference_psf(
#     science_beam,
#     npix_pupil: int,
#     pad_size: int,
# ):
#     """
#     Build a perfect reference PSF with the same sampled pupil support.
#     """
#     empty_bench = bt.OpticalBench3D()

#     wf_ref = btw.sample_wavefront_on_pupil_plane(
#         beam=science_beam,
#         bench=empty_bench,
#         pupil_point=(0.0, 0.0, 0.0),
#         t=0.0,
#         npix=npix_pupil,
#         beam_trace_module=bt,
#     )

#     wf_ref = wf_ref.with_field(wf_ref.amplitude.astype(complex), label="perfect_reference")

#     return psf_tools.psf_from_wavefront(
#         wf_ref,
#         pad_to=pad_size,
#         normalize=None,
#     )


# # ============================================================
# # Telemetry generation
# # ============================================================

# def run_glao_movie_telemetry(
#     fits_path: Path,
#     exposure_s: float = 5.6,
#     dt: float = 0.1,
#     ao_start_time: float = 2.0,
#     fa_scale: float = 0.6,
#     dm_acts_across: int = 35,
#     wfs_wavelength: float = 589e-9,
#     science_wavelength: float = 0.589e-6,
#     beam_diameter: float = 13e-3,
#     npix_pupil: int = 256,
#     pad_size: int = 2048,
#     source_plane_z: float = -3.25,
#     opd_scale_m_per_rad: float = 500e-9 / (2.0 * np.pi),
#     fa_x_m: float = 26e-3,
#     gl_x_m: float = 34e-3,
#     lgs_radius_arcmin: float = 10.0,
#     science_offsets_arcmin: Tuple[float, ...] = (0.0, 5.0, 10.0),
# ):
#     """
#     Run the animated GLAO telemetry simulation.

#     Important:
#         LGS reconstruction is performed using sample["opd_map_m"], i.e. the
#         unwrapped OPD from the geometric phase-screen sampling.
#     """
#     pupil_point = np.array([0.0, 0.0, 0.0])

#     bench = build_bench_from_fits_movie_style(
#         fits_path=fits_path,
#         opd_scale_m_per_rad=opd_scale_m_per_rad,
#         fa_scale=fa_scale,
#         fa_x_m=fa_x_m,
#         gl_x_m=gl_x_m,
#     )

#     lgs_coords = [
#         (+lgs_radius_arcmin, +lgs_radius_arcmin),
#         (-lgs_radius_arcmin, +lgs_radius_arcmin),
#         (+lgs_radius_arcmin, -lgs_radius_arcmin),
#         (-lgs_radius_arcmin, -lgs_radius_arcmin),
#     ]

#     lgs_beams = make_lgs_beams(
#         wavelength=wfs_wavelength,
#         beam_diameter=beam_diameter,
#         source_plane_z=source_plane_z,
#         pupil_point=pupil_point,
#         lgs_coords_arcmin=lgs_coords,
#         nrings=3,
#         nphi=12,
#     )

#     # Sparse beams for 3D display only.
#     lgs_trace_beams = make_lgs_beams(
#         wavelength=wfs_wavelength,
#         beam_diameter=beam_diameter,
#         source_plane_z=source_plane_z,
#         pupil_point=pupil_point,
#         lgs_coords_arcmin=lgs_coords,
#         nrings=1,
#         nphi=4,
#     )

#     sci_beams = make_science_beams(
#         science_angles_arcmin=list(science_offsets_arcmin),
#         wavelength=science_wavelength,
#         beam_diameter=beam_diameter,
#         source_plane_z=source_plane_z,
#         pupil_point=pupil_point,
#         nrings=3,
#         nphi=12,
#     )

#     perfect_psf = make_perfect_reference_psf(
#         science_beam=sci_beams[0],
#         npix_pupil=npix_pupil,
#         pad_size=pad_size,
#     )

#     times = np.arange(0.0, exposure_s, dt)

#     telemetry = {
#         "times": times,
#         "bench": bench,
#         "frames": [],
#         "sci_angles": list(science_offsets_arcmin),
#         "lgs_beams": lgs_beams,
#         "lgs_trace_beams": lgs_trace_beams,
#         "sci_beams": sci_beams,
#         "wfs_wavelength": wfs_wavelength,
#         "science_wavelength": science_wavelength,
#         "beam_diameter": beam_diameter,
#         "npix_pupil": npix_pupil,
#         "pad_size": pad_size,
#         "dm_acts_across": dm_acts_across,
#         "ao_start_time": ao_start_time,
#         "fa_scale": fa_scale,
#         "perfect_psf": perfect_psf,
#     }

#     print("Generating movie telemetry")
#     print("  FITS path:", fits_path)
#     print("  frames:", len(times))
#     print("  exposure [s]:", exposure_s)
#     print("  dt [s]:", dt)
#     print("  AO start [s]:", ao_start_time)
#     print("  WFS wavelength [nm]:", wfs_wavelength * 1e9)
#     print("  science wavelength [um]:", science_wavelength * 1e6)
#     print("  DM acts across:", dm_acts_across)
#     print("  FA scale:", fa_scale)

#     for idx, t in enumerate(times):
#         frame = {"t": float(t)}

#         lgs_samples = []

#         for beam in lgs_beams:
#             _, sample_lgs = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
#                 beam=beam,
#                 bench=bench,
#                 pupil_point=pupil_point,
#                 t=float(t),
#                 npix=npix_pupil,
#                 beam_trace_module=bt,
#             )
#             lgs_samples.append(sample_lgs)

#         lgs_mask = np.asarray(lgs_samples[0]["mask"], dtype=bool)

#         # Use unwrapped OPD maps from the beam_trace sample.
#         lgs_opd_maps = [
#             np.nan_to_num(s["opd_map_m"], nan=0.0)
#             for s in lgs_samples
#         ]

#         avg_lgs_opd = np.nanmean(lgs_opd_maps, axis=0)
#         avg_lgs_opd = np.where(lgs_mask, avg_lgs_opd, 0.0)

#         ao_on = bool(t >= ao_start_time)

#         if ao_on:
#             dm_opd = apply_dm_correction_opd(
#                 avg_lgs_opd,
#                 acts_across=dm_acts_across,
#                 mask=lgs_mask,
#             )
#         else:
#             dm_opd = np.zeros_like(avg_lgs_opd)

#         frame["ao_on"] = ao_on
#         frame["recon_opd"] = avg_lgs_opd
#         frame["dm_opd"] = dm_opd
#         frame["recon_phase_wfs"] = phase_tools.opd_to_phase(avg_lgs_opd, wfs_wavelength)
#         frame["dm_phase_wfs"] = phase_tools.opd_to_phase(dm_opd, wfs_wavelength)
#         frame["lgs_mask"] = lgs_mask

#         frame["sci"] = [
#             simulate_sci_performance(
#                 beam=b,
#                 dm_opd_corr=dm_opd,
#                 bench=bench,
#                 t=float(t),
#                 pad_size=pad_size,
#                 npix_pupil=npix_pupil,
#             )
#             for b in sci_beams
#         ]

#         frame["strehl"] = [
#             psf_tools.strehl_from_psfs(sci["psf"], perfect_psf)
#             for sci in frame["sci"]
#         ]

#         telemetry["frames"].append(frame)

#         if (idx + 1) % max(1, len(times) // 10) == 0 or idx == len(times) - 1:
#             print(f"  frame {idx + 1}/{len(times)}")

#     return telemetry


# # ============================================================
# # Movie / visualisation
# # ============================================================

# def draw_static_3d_panel(ax3d, tel):
#     """
#     Draw the static geometric bench panel.
#     """
#     bench = tel["bench"]

#     ax3d.set_title("Geometric testbench")

#     # LGS beams
#     for beam in tel["lgs_trace_beams"]:
#         paths, _ = bench.trace_beam(beam, s_end=0.1, t=0.0)
#         for path in paths:
#             ax3d.plot(
#                 path[:, 0],
#                 path[:, 1],
#                 path[:, 2],
#                 lw=0.8,
#                 alpha=0.45,
#                 color="orange",
#             )

#     # Science chief rays / sparse beams
#     for beam in tel["sci_beams"]:
#         # Draw only chief ray path by creating a one-ray beam-like copy.
#         chief = beam.chief_ray
#         p0 = chief.r
#         p1 = p0 + 3.5 * chief.d
#         ax3d.plot(
#             [p0[0], p1[0]],
#             [p0[1], p1[1]],
#             [p0[2], p1[2]],
#             lw=1.0,
#             alpha=0.8,
#             color="tab:blue",
#         )

#     for elem in bench.elements:
#         e1, e2 = elem.plane_basis()
#         rad = elem.clear_radius if isinstance(elem, bt.RotatingPhaseScreen3D) else 1e-3
#         circ = np.linspace(0.0, 2.0 * np.pi, 160)
#         ring = elem.point[None, :] + rad * (
#             np.cos(circ)[:, None] * e1[None, :]
#             + np.sin(circ)[:, None] * e2[None, :]
#         )
#         ax3d.plot(ring[:, 0], ring[:, 1], ring[:, 2], "k-", lw=1.3)
#         ax3d.text(elem.point[0], elem.point[1], elem.point[2], elem.label)

#     ax3d.scatter([0.0], [0.0], [0.0], marker="x", s=60, color="k")
#     ax3d.text(0.0, 0.0, 0.0, " pupil")

#     ax3d.set_xlabel("x [m]")
#     ax3d.set_ylabel("y [m]")
#     ax3d.set_zlabel("z [m]")
#     ax3d.set_box_aspect([1.0, 1.0, 1.5])

#     try:
#         ax3d.view_init(elev=18.0, azim=-70.0)
#     except Exception:
#         pass


# def crop_psf_for_display(psf: np.ndarray, half_width_pix: int):
#     """
#     Crop PSF around its brightest pixel.
#     """
#     psf = np.asarray(psf, dtype=float)

#     if np.nanmax(psf) <= 0:
#         cy, cx = np.array(psf.shape) // 2
#     else:
#         cy, cx = np.unravel_index(np.nanargmax(psf), psf.shape)

#     y0 = max(0, cy - half_width_pix)
#     y1 = min(psf.shape[0], cy + half_width_pix + 1)
#     x0 = max(0, cx - half_width_pix)
#     x1 = min(psf.shape[1], cx + half_width_pix + 1)

#     return psf[y0:y1, x0:x1]


# def make_movie(
#     tel: Dict,
#     base_filename: str = "glao_telemetry_ao_switch",
#     save: bool = True,
#     show: bool = False,
#     interval_ms: int = 250,
#     fps: int = 8,
#     screen_npix: int = 256,
#     psf_half_width_pix: int = 48,
# ):
#     """
#     Create and optionally save the movie.

#     If ffmpeg is unavailable, falls back to GIF/pillow.
#     """
#     fig = plt.figure(figsize=(22, 12))
#     gs = fig.add_gridspec(4, 4, width_ratios=[1.35, 1.0, 1.0, 1.0])

#     # Some environments lack 3D projection support. Fallback gracefully.
#     try:
#         ax3d = fig.add_subplot(gs[:, 0], projection="3d")
#         draw_static_3d_panel(ax3d, tel)
#     except Exception as exc:
#         ax3d = fig.add_subplot(gs[:, 0])
#         ax3d.text(
#             0.5,
#             0.5,
#             f"3D projection unavailable\n{exc}",
#             ha="center",
#             va="center",
#             transform=ax3d.transAxes,
#         )
#         ax3d.axis("off")

#     ax_scr = [fig.add_subplot(gs[i, 1]) for i in range(4)]
#     ax_pup = [fig.add_subplot(gs[i, 2]) for i in range(3)]
#     ax_psf = [fig.add_subplot(gs[i, 3]) for i in range(3)]

#     ax_unused_pup = fig.add_subplot(gs[3, 2])
#     ax_unused_psf = fig.add_subplot(gs[3, 3])
#     ax_unused_pup.axis("off")
#     ax_unused_psf.axis("off")

#     screen_vmax_nm = None
#     recon_vmax_nm = None
#     psf_vmax = None

#     # Precompute display colour scales from a subset of frames.
#     subset = tel["frames"][::max(1, len(tel["frames"]) // 10)]

#     screen_vals = []
#     recon_vals = []
#     psf_vals = []

#     for frame in subset:
#         for elem in tel["bench"].elements:
#             img = get_rotating_screen_image(elem, frame["t"], npix=screen_npix) * 1e9
#             screen_vals.append(img[np.isfinite(img)])

#         mask = frame["lgs_mask"]
#         recon_vals.append(frame["recon_opd"][mask] * 1e9)
#         recon_vals.append(frame["dm_opd"][mask] * 1e9)

#         for sci in frame["sci"]:
#             psf_vals.append(crop_psf_for_display(sci["psf"], psf_half_width_pix))

#     if screen_vals:
#         screen_vmax_nm = robust_abs_percentile(np.concatenate(screen_vals), 99.0, default=500.0)

#     if recon_vals:
#         recon_vmax_nm = robust_abs_percentile(np.concatenate(recon_vals), 99.0, default=500.0)

#     if psf_vals:
#         psf_vmax = np.nanpercentile(np.concatenate([p.ravel() for p in psf_vals]), 99.8)
#         if not np.isfinite(psf_vmax) or psf_vmax <= 0:
#             psf_vmax = 1.0

#     def update(idx):
#         frame = tel["frames"][idx]
#         t = frame["t"]

#         for ax in ax_scr + ax_pup + ax_psf:
#             ax.clear()

#         # ----------------------------------------------------
#         # Column 2: rotating screens and footprints
#         # ----------------------------------------------------
#         for i, elem in enumerate(tel["bench"].elements):
#             img_nm = get_rotating_screen_image(elem, t, npix=screen_npix) * 1e9
#             r_mm = elem.clear_radius * 1e3

#             im = ax_scr[i].imshow(
#                 img_nm,
#                 cmap="RdBu_r",
#                 origin="lower",
#                 extent=[-r_mm, r_mm, -r_mm, r_mm],
#                 vmin=-screen_vmax_nm,
#                 vmax=screen_vmax_nm,
#                 interpolation="nearest",
#             )
#             ax_scr[i].set_title(f"{elem.label} OPD [nm]")

#             # LGS footprints
#             for beam in tel["lgs_beams"]:
#                 inter = tel["bench"].trace_chief_intersections(beam, t=t)
#                 if elem.label in inter:
#                     u, v = elem.local_coordinates(inter[elem.label]["point"])
#                     ax_scr[i].add_patch(
#                         plt.Circle(
#                             (u * 1e3, v * 1e3),
#                             0.5 * tel["beam_diameter"] * 1e3,
#                             color="red",
#                             fill=False,
#                             lw=1.0,
#                             alpha=0.9,
#                         )
#                     )

#             # Science footprints
#             for beam in tel["sci_beams"]:
#                 inter = tel["bench"].trace_chief_intersections(beam, t=t)
#                 if elem.label in inter:
#                     u, v = elem.local_coordinates(inter[elem.label]["point"])
#                     ax_scr[i].scatter(
#                         u * 1e3,
#                         v * 1e3,
#                         marker="x",
#                         color="white",
#                         s=25,
#                         linewidths=1.2,
#                     )

#             ax_scr[i].set_xlabel("u [mm]")
#             ax_scr[i].set_ylabel("v [mm]")

#         # ----------------------------------------------------
#         # Column 3: pupil diagnostics
#         # ----------------------------------------------------
#         mask = frame["lgs_mask"]

#         recon_nm = np.where(mask, frame["recon_opd"] * 1e9, np.nan)
#         dm_nm = np.where(mask, frame["dm_opd"] * 1e9, np.nan)

#         # Show residual OPD for the widest science field.
#         sci_display = frame["sci"][-1]
#         sci_mask = sci_display["mask"]
#         residual_nm = np.where(sci_mask, sci_display["residual_opd"] * 1e9, np.nan)

#         pupil_maps = [
#             ("LGS mean recon OPD [nm]", recon_nm, "RdBu_r", -recon_vmax_nm, recon_vmax_nm),
#             ("DM correction OPD [nm]", dm_nm, "RdBu_r", -recon_vmax_nm, recon_vmax_nm),
#             (f"Residual science OPD ({tel['sci_angles'][-1]:.1f}') [nm]", residual_nm, "RdBu_r", -recon_vmax_nm, recon_vmax_nm),
#         ]

#         for ax, (title, image, cmap, vmin, vmax) in zip(ax_pup, pupil_maps):
#             ax.imshow(
#                 image,
#                 origin="lower",
#                 cmap=cmap,
#                 vmin=vmin,
#                 vmax=vmax,
#                 interpolation="nearest",
#             )
#             ax.set_title(title)
#             ax.axis("off")

#         # ----------------------------------------------------
#         # Column 4: science PSFs
#         # ----------------------------------------------------
#         for i, sci in enumerate(frame["sci"]):
#             psf = sci["psf"]

#             crop = crop_psf_for_display(psf, psf_half_width_pix)
#             crop_norm = crop.copy()
#             if np.nanmax(crop_norm) > 0:
#                 crop_norm = crop_norm / np.nanmax(crop_norm)

#             # Log display is more informative for halo structure.
#             crop_show = np.log10(np.maximum(crop_norm, 1e-5))

#             ax_psf[i].imshow(
#                 crop_show,
#                 cmap="magma",
#                 origin="lower",
#                 interpolation="nearest",
#                 vmin=-5.0,
#                 vmax=0.0,
#             )

#             strehl = frame["strehl"][i]
#             ax_psf[i].set_title(
#                 f"PSF {tel['sci_angles'][i]:.1f}'\nS={strehl:.3f}"
#             )
#             ax_psf[i].axis("off")

#         ao_state = "GLAO ON" if frame["ao_on"] else "AO OFF"

#         fig.suptitle(
#             (
#                 f"t = {t:.2f} s   |   {ao_state}   |   "
#                 f"DM acts = {tel['dm_acts_across']}   |   "
#                 f"FA scale = {tel['fa_scale']:.2f}"
#             ),
#             fontsize=16,
#         )

#         return []

#     ani = FuncAnimation(
#         fig,
#         update,
#         frames=len(tel["frames"]),
#         interval=interval_ms,
#         blit=False,
#     )

#     if save:
#         output_path = Path(base_filename)

#         if output_path.suffix.lower() in [".mp4", ".gif"]:
#             base = output_path.with_suffix("")
#             suffix = output_path.suffix.lower()
#         else:
#             base = output_path
#             suffix = ".mp4"

#         if suffix == ".gif":
#             gif_path = str(base) + ".gif"
#             ani.save(gif_path, writer="pillow", fps=fps)
#             print(f"Saved movie to {gif_path}")
#         else:
#             mp4_path = str(base) + ".mp4"
#             try:
#                 ani.save(mp4_path, writer="ffmpeg", fps=fps)
#                 print(f"Saved movie to {mp4_path}")
#             except Exception as exc:
#                 gif_path = str(base) + ".gif"
#                 print(f"ffmpeg failed with: {exc}")
#                 print("Falling back to GIF/pillow.")
#                 ani.save(gif_path, writer="pillow", fps=fps)
#                 print(f"Saved movie to {gif_path}")

#     if show:
#         plt.show()

#     plt.close(fig)


# # ============================================================
# # CLI
# # ============================================================

# def parse_args():
#     repo_root = Path(__file__).resolve().parents[2]
#     default_fits = (
#         repo_root
#         / "phasescreens"
#         / "scrns_2_order_v2"
#         / "Testbench_phasescreens_20260506.fits"
#     )

#     parser = argparse.ArgumentParser(description="Animated hybrid GLAO movie demo.")

#     parser.add_argument("--fits-path", type=Path, default=default_fits)
#     parser.add_argument("--output", type=str, default="glao_telemetry_ao_switch")

#     parser.add_argument("--exposure-s", type=float, default=5.6)
#     parser.add_argument("--dt", type=float, default=0.1)
#     parser.add_argument("--ao-start-time", type=float, default=2.0)

#     parser.add_argument("--fa-scale", type=float, default=0.6)
#     parser.add_argument("--dm-acts-across", type=int, default=35)

#     parser.add_argument("--wfs-wavelength", type=float, default=589e-9)
#     parser.add_argument("--science-wavelength", type=float, default=0.589e-6)
#     parser.add_argument("--beam-diameter", type=float, default=13e-3)
#     parser.add_argument("--source-plane-z", type=float, default=-3.25)

#     parser.add_argument("--npix-pupil", type=int, default=256)
#     parser.add_argument("--pad-size", type=int, default=2048)

#     parser.add_argument("--fa-x-mm", type=float, default=26.0)
#     parser.add_argument("--gl-x-mm", type=float, default=34.0)

#     parser.add_argument(
#         "--opd-scale",
#         type=float,
#         default=500e-9 / (2.0 * np.pi),
#         help="Conversion factor from FITS values to OPD [m].",
#     )

#     parser.add_argument("--interval-ms", type=int, default=250)
#     parser.add_argument("--fps", type=int, default=8)
#     parser.add_argument("--screen-npix", type=int, default=256)
#     parser.add_argument("--psf-half-width-pix", type=int, default=48)

#     parser.add_argument("--save", dest="save", action="store_true", default=True)
#     parser.add_argument("--no-save", dest="save", action="store_false")
#     parser.add_argument("--show", action="store_true")

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     if not args.fits_path.exists():
#         raise FileNotFoundError(f"Could not find FITS file: {args.fits_path}")

#     tel = run_glao_movie_telemetry(
#         fits_path=args.fits_path,
#         exposure_s=args.exposure_s,
#         dt=args.dt,
#         ao_start_time=args.ao_start_time,
#         fa_scale=args.fa_scale,
#         dm_acts_across=args.dm_acts_across,
#         wfs_wavelength=args.wfs_wavelength,
#         science_wavelength=args.science_wavelength,
#         beam_diameter=args.beam_diameter,
#         npix_pupil=args.npix_pupil,
#         pad_size=args.pad_size,
#         source_plane_z=args.source_plane_z,
#         opd_scale_m_per_rad=args.opd_scale,
#         fa_x_m=args.fa_x_mm * 1e-3,
#         gl_x_m=args.gl_x_mm * 1e-3,
#     )

#     make_movie(
#         tel,
#         base_filename=args.output,
#         save=args.save,
#         show=args.show,
#         interval_ms=args.interval_ms,
#         fps=args.fps,
#         screen_npix=args.screen_npix,
#         psf_half_width_pix=args.psf_half_width_pix,
#     )


# if __name__ == "__main__":
#     main()