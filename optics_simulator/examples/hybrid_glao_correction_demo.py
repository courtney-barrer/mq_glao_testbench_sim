"""
hybrid_glao_correction_demo.py

Hybrid GLAO correction demo using real phase screens.

This reproduces the core logic of the old psf_analysis.py workflow using:

    beam_trace_ORIG geometry
    -> Wavefront2D bridge
    -> OPD-space GLAO correction
    -> FFT PSF / Strehl / EE / Gaussian / Moffat analysis

Run from repository root:

    python3 -m optics_simulator.examples.hybrid_glao_correction_demo

Optional:

    python3 -m optics_simulator.examples.hybrid_glao_correction_demo \
        --fits-path phasescreens/scrns_2_order_v2/Testbench_phasescreens_20260506.fits \
        --exposure-time 0.3 \
        --dt 0.1
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
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
# Bench / beam construction
# ============================================================

def build_bench_from_fits(fits_path: Path, opd_scale_m_per_rad: float):
    layer_configs = [
        {"label": "FA",  "z": -2.50,  "hz": 0.2, "point_xy": (26e-3, 0.0)},
        {"label": "GL3", "z": -0.060, "hz": 1.4, "point_xy": (24e-3, 0.0)},
        {"label": "GL2", "z": -0.030, "hz": 1.0, "point_xy": (24e-3, 0.0)},
        {"label": "GL1", "z": -0.001, "hz": 1.7, "point_xy": (24e-3, 0.0)},
    ]

    bench = bt.OpticalBench3D()

    with fits.open(fits_path) as hdul:
        pix_scale = float(hdul[0].header["PIXSCALE"])

        for cfg in layer_configs:
            label = cfg["label"]
            opd = np.asarray(hdul[label].data, dtype=float) * opd_scale_m_per_rad
            map_extent_m = opd.shape[0] * pix_scale

            x0, y0 = cfg["point_xy"]

            bench.add(
                bt.RotatingPhaseScreen3D(
                    point=[x0, y0, cfg["z"]],
                    normal=[0.0, 0.0, 1.0],
                    opd_map=opd,
                    map_extent_m=map_extent_m,
                    angular_velocity=2.0 * np.pi * cfg["hz"],
                    label=label,
                )
            )

    return bench


def make_lgs_beams(
    wavelength: float,
    beam_diameter: float,
    source_plane_z: float,
    pupil_point: np.ndarray,
    lgs_coords_arcmin,
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
            f"L{i}",
            3,
            12,
        )
        beams.append(beam)

    return beams


def make_science_beams(
    science_angles_arcmin,
    wavelength: float,
    beam_diameter: float,
    source_plane_z: float,
    pupil_point: np.ndarray,
):
    beams = []

    for i, theta_arcmin in enumerate(science_angles_arcmin):
        beam = bt.make_converging_beam_from_field_angles(
            np.deg2rad(theta_arcmin / 60.0),
            0.0,
            source_plane_z,
            pupil_point,
            beam_diameter,
            wavelength,
            f"S{i}_{theta_arcmin:.2f}arcmin",
            3,
            12,
        )
        beams.append(beam)

    return beams


# ============================================================
# Correction / reference helpers
# ============================================================

def apply_dm_correction_opd(opd_map_m: np.ndarray, acts_across: int, mask: np.ndarray) -> np.ndarray:
    """
    Simple DM spatial filtering model in OPD space.

    The DM correction is approximated as the low-spatial-frequency part of the
    reconstructed ground-layer OPD.
    """
    opd_map_m = np.asarray(opd_map_m, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if acts_across <= 0:
        return np.zeros_like(opd_map_m)

    avg_opd = np.nanmean(opd_map_m[mask])
    work = np.where(mask, opd_map_m, avg_opd)

    sigma = (opd_map_m.shape[0] / float(acts_across)) * 0.5
    low_spatial = gaussian_filter(work, sigma=sigma, mode="reflect")

    return np.where(mask, low_spatial, 0.0)


def perfect_reference_wavefront(wf):
    """
    Same amplitude support as wf, but zero phase.
    """
    amp = wf.amplitude
    return wf.with_field(amp.astype(complex), label="perfect reference")




# ============================================================
# Plotting helpers
# ============================================================

def plot_image(ax, image, title, extent=None, log=False, cmap=None, vmin=None, vmax=None):
    image = np.asarray(image, dtype=float)

    if log:
        peak = np.nanmax(image)
        if peak > 0:
            image = np.log10(np.maximum(image / peak, 1e-8))
        else:
            image = np.zeros_like(image)

    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_correction_diagnostics(example):
    """
    Diagnostic maps for the central-field first time step.

    Shows the raw science OPD, LGS mean OPD, DM correction, corrected OPD,
    and applied correction, all after piston/tip/tilt removal.
    """
    wf_sci = example["wf_sci"]
    xx = wf_sci.x
    yy = wf_sci.y
    mask = wf_sci.amplitude > 0

    opd_raw = np.nan_to_num(example["sample_sci"]["opd_map_m"], nan=0.0)
    gl_opd = np.nan_to_num(example["gl_opd"], nan=0.0)
    dm_opd_corr = np.nan_to_num(example["dm_opd_corr"], nan=0.0)
    opd_corr = opd_raw - dm_opd_corr
    applied = dm_opd_corr

    def ptt_removed_nm(opd):
        out, _, _ = phase_tools.remove_piston_tip_tilt(
            opd,
            xx=xx,
            yy=yy,
            mask=mask,
        )
        return out * 1e9

    maps = [
        ("Science raw OPD\nPTT removed", ptt_removed_nm(opd_raw)),
        ("LGS mean GL OPD\nPTT removed", ptt_removed_nm(gl_opd)),
        ("DM correction OPD\nPTT removed", ptt_removed_nm(dm_opd_corr)),
        ("Science corrected OPD\nPTT removed", ptt_removed_nm(opd_corr)),
        ("Applied correction\nPTT removed", ptt_removed_nm(applied)),
    ]

    vals = np.concatenate([m[mask & np.isfinite(m)] for _, m in maps])
    vmax = np.nanpercentile(np.abs(vals), 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    extent_mm = [
        np.nanmin(xx) * 1e3,
        np.nanmax(xx) * 1e3,
        np.nanmin(yy) * 1e3,
        np.nanmax(yy) * 1e3,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax, (title, img) in zip(axes, maps):
        im = ax.imshow(
            img,
            origin="lower",
            extent=extent_mm,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="OPD [nm]")

    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main simulation
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid GLAO correction demo.")

    repo_root = Path(__file__).resolve().parents[2]
    default_fits = (
        repo_root
        / "phasescreens"
        / "scrns_2_order_v2"
        / "Testbench_phasescreens_20260506.fits"
    )

    parser.add_argument("--fits-path", type=Path, default=default_fits)
    parser.add_argument("--npix", type=int, default=256)
    parser.add_argument("--pad-to", type=int, default=1024)
    parser.add_argument("--exposure-time", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--wfs-wavelength", type=float, default=633e-9)
    parser.add_argument("--science-wavelength", type=float, default=1.2e-6)
    parser.add_argument("--beam-diameter", type=float, default=13e-3)
    parser.add_argument("--source-plane-z", type=float, default=-3.25)
    parser.add_argument("--dm-acts-across", type=int, default=11)
    parser.add_argument("--focal-length", type=float, default=0.25)
    parser.add_argument("--focus-half-width-um", type=float, default=160.0)
    parser.add_argument(
        "--opd-scale",
        type=float,
        default=500e-9 / (2.0 * np.pi),
        help="Conversion factor from FITS screen values to OPD [m].",
    )

    args = parser.parse_args()

    if not args.fits_path.exists():
        raise FileNotFoundError(f"Could not find FITS file: {args.fits_path}")

    pupil_point = np.array([0.0, 0.0, 0.0])

    science_angles = np.linspace(0.0, 10.0, 5)
    lgs_coords = [(10.0, 10.0), (-10.0, 10.0), (10.0, -10.0), (-10.0, -10.0)]
    times = np.arange(0.0, args.exposure_time, args.dt)

    bench = build_bench_from_fits(
        fits_path=args.fits_path,
        opd_scale_m_per_rad=args.opd_scale,
    )

    lgs_beams = make_lgs_beams(
        wavelength=args.wfs_wavelength,
        beam_diameter=args.beam_diameter,
        source_plane_z=args.source_plane_z,
        pupil_point=pupil_point,
        lgs_coords_arcmin=lgs_coords,
    )

    sci_beams = make_science_beams(
        science_angles_arcmin=science_angles,
        wavelength=args.science_wavelength,
        beam_diameter=args.beam_diameter,
        source_plane_z=args.source_plane_z,
        pupil_point=pupil_point,
    )

    # Reference PSF from the central science beam amplitude support.
    wf_ref0 = btw.sample_wavefront_on_pupil_plane(
        beam=sci_beams[0],
        bench=bt.OpticalBench3D(),
        pupil_point=pupil_point,
        t=0.0,
        npix=args.npix,
        beam_trace_module=bt,
    )
    wf_ref0 = perfect_reference_wavefront(wf_ref0)

    perfect_psf = psf_tools.psf_from_wavefront(
        wf_ref0,
        pad_to=args.pad_to,
        normalize=None,
    )

    accum_no_ao = [
        np.zeros((args.pad_to, args.pad_to), dtype=float)
        for _ in sci_beams
    ]
    accum_glao = [
        np.zeros((args.pad_to, args.pad_to), dtype=float)
        for _ in sci_beams
    ]

    example = {}

    print("Hybrid GLAO correction demo")
    print("  FITS path:", args.fits_path)
    print("  n times:", len(times))
    print("  exposure [s]:", args.exposure_time)
    print("  dt [s]:", args.dt)
    print("  WFS wavelength [nm]:", args.wfs_wavelength * 1e9)
    print("  science wavelength [um]:", args.science_wavelength * 1e6)
    print("  DM acts across:", args.dm_acts_across)

    for it, t in enumerate(times):
        lgs_samples = []

        for b in lgs_beams:
            _, sample_lgs = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
                beam=b,
                bench=bench,
                pupil_point=pupil_point,
                t=float(t),
                npix=args.npix,
                beam_trace_module=bt,
            )
            lgs_samples.append(sample_lgs)

        lgs_mask = np.asarray(lgs_samples[0]["mask"], dtype=bool)

        # Important: use unwrapped OPD from beam_trace sample, not np.angle(wf.field).
        lgs_opd_maps = [
            np.nan_to_num(sample["opd_map_m"], nan=0.0)
            for sample in lgs_samples
        ]

        gl_opd = np.mean(lgs_opd_maps, axis=0)
        gl_opd = np.where(lgs_mask, gl_opd, 0.0)

        dm_opd_corr = apply_dm_correction_opd(
            gl_opd,
            acts_across=args.dm_acts_across,
            mask=lgs_mask,
        )

        for i, beam in enumerate(sci_beams):
            wf_sci, sample_sci = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
                beam=beam,
                bench=bench,
                pupil_point=pupil_point,
                t=float(t),
                npix=args.npix,
                beam_trace_module=bt,
            )

            sci_mask = wf_sci.amplitude > 0

            psf_no = psf_tools.psf_from_wavefront(
                wf_sci,
                pad_to=args.pad_to,
                normalize=None,
            )

            wf_corr = wave_optics.apply_opd(
                wf_sci,
                opd_map_m=-dm_opd_corr,
                mask=sci_mask,
                label=f"{wf_sci.label}_glao_corrected",
            )

            psf_ao = psf_tools.psf_from_wavefront(
                wf_corr,
                pad_to=args.pad_to,
                normalize=None,
            )

            accum_no_ao[i] += psf_no
            accum_glao[i] += psf_ao

            if it == 0 and i == 0:
                example = {
                    "t": t,
                    "wf_sci": wf_sci,
                    "wf_corr": wf_corr,
                    "sample_sci": sample_sci,
                    "gl_opd": gl_opd,
                    "dm_opd_corr": dm_opd_corr,
                    "lgs_mask": lgs_mask,
                }

    mean_no_ao = [p / len(times) for p in accum_no_ao]
    mean_glao = [p / len(times) for p in accum_glao]

    # Basic lambda/D pixel scale for FFT PSFs.
    angular_pixel_scale_ld = 1.0 / (args.pad_to / (args.npix / 2.0))

    analysis_no = [
        psf_tools.analyse_psf_with_fits(
            p,
            perfect_psf,
            angular_pixel_scale_ld=angular_pixel_scale_ld,
            fit_module=bt,
            crop_half_width_ld=6.0,
            moffat_fit_region="all",
        )
        for p in mean_no_ao
    ]

    analysis_ao = [
        psf_tools.analyse_psf_with_fits(
            p,
            perfect_psf,
            angular_pixel_scale_ld=angular_pixel_scale_ld,
            fit_module=bt,
            crop_half_width_ld=6.0,
            moffat_fit_region="all",
        )
        for p in mean_glao
    ]

    strehl_no = [r["strehl"] for r in analysis_no]
    strehl_ao = [r["strehl"] for r in analysis_ao]

    ee80_no = [r["ee80"] for r in analysis_no]
    ee80_ao = [r["ee80"] for r in analysis_ao]

    gauss_fwhm_no = [r["gaussian_fwhm"] for r in analysis_no]
    gauss_fwhm_ao = [r["gaussian_fwhm"] for r in analysis_ao]

    moffat_fwhm_no = [r["moffat_fwhm"] for r in analysis_no]
    moffat_fwhm_ao = [r["moffat_fwhm"] for r in analysis_ao]

    moffat_ell_no = [r["moffat_ell"] for r in analysis_no]
    moffat_ell_ao = [r["moffat_ell"] for r in analysis_ao]

    moffat_beta_no = [r["moffat_beta"] for r in analysis_no]
    moffat_beta_ao = [r["moffat_beta"] for r in analysis_ao]

    # ========================================================
    # PSF image grid
    # ========================================================

    fig_grid, axes = plt.subplots(
        2,
        len(sci_beams),
        figsize=(3.5 * len(sci_beams), 7),
    )

    for i, theta in enumerate(science_angles):
        crop_no = np.asarray(analysis_no[i]["psf_crop"], dtype=float)
        crop_ao = np.asarray(analysis_ao[i]["psf_crop"], dtype=float)

        if np.nanmax(crop_no) > 0:
            crop_no = crop_no / np.nanmax(crop_no)
        if np.nanmax(crop_ao) > 0:
            crop_ao = crop_ao / np.nanmax(crop_ao)

        x_crop = analysis_no[i]["x_ld_crop"]
        y_crop = analysis_no[i]["y_ld_crop"]
        x_crop2 = analysis_ao[i]["x_ld_crop"]
        y_crop2 = analysis_ao[i]["y_ld_crop"]

        extent_no = [
            np.nanmin(x_crop),
            np.nanmax(x_crop),
            np.nanmin(y_crop),
            np.nanmax(y_crop),
        ]

        extent_ao = [
            np.nanmin(x_crop2),
            np.nanmax(x_crop2),
            np.nanmin(y_crop2),
            np.nanmax(y_crop2),
        ]

        plot_image(
            axes[0, i],
            crop_no,
            f"No AO {theta:.1f}'\nS={strehl_no[i]:.3f}",
            extent=extent_no,
            log=True,
        )

        plot_image(
            axes[1, i],
            crop_ao,
            f"GLAO {theta:.1f}'\nS={strehl_ao[i]:.3f}",
            extent=extent_ao,
            log=True,
        )

        axes[0, i].set_xlabel("x [λ/D]")
        axes[0, i].set_ylabel("y [λ/D]")
        axes[1, i].set_xlabel("x [λ/D]")
        axes[1, i].set_ylabel("y [λ/D]")

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Summary plots with Moffat/Gaussian diagnostics
    # ========================================================

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(science_angles, strehl_no, "o--", label="No AO")
    axes[0, 0].plot(science_angles, strehl_ao, "o-", label="GLAO")
    axes[0, 0].set_ylabel("FFT Strehl")

    axes[0, 1].plot(science_angles, ee80_no, "o--", label="No AO")
    axes[0, 1].plot(science_angles, ee80_ao, "o-", label="GLAO")
    axes[0, 1].set_ylabel("EE80 radius [λ/D]")

    axes[0, 2].plot(science_angles, moffat_fwhm_no, "o--", label="No AO")
    axes[0, 2].plot(science_angles, moffat_fwhm_ao, "o-", label="GLAO")
    axes[0, 2].set_ylabel("Moffat FWHM [λ/D]")

    axes[1, 0].plot(science_angles, gauss_fwhm_no, "o--", label="No AO")
    axes[1, 0].plot(science_angles, gauss_fwhm_ao, "o-", label="GLAO")
    axes[1, 0].set_ylabel("Gaussian FWHM [λ/D]")

    axes[1, 1].plot(science_angles, moffat_ell_no, "o--", label="No AO")
    axes[1, 1].plot(science_angles, moffat_ell_ao, "o-", label="GLAO")
    axes[1, 1].set_ylabel("Moffat ellipticity")

    axes[1, 2].plot(science_angles, moffat_beta_no, "o--", label="No AO")
    axes[1, 2].plot(science_angles, moffat_beta_ao, "o-", label="GLAO")
    axes[1, 2].set_ylabel("Moffat beta")

    for ax in axes.ravel():
        ax.set_xlabel("Field angle [arcmin]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Correction diagnostics
    # ========================================================

    plot_correction_diagnostics(example)

    opd_raw = np.nan_to_num(example["sample_sci"]["opd_map_m"], nan=0.0)
    opd_corr = opd_raw - example["dm_opd_corr"]
    mask = example["wf_sci"].amplitude > 0

    opd_raw_ptt, _, _ = phase_tools.remove_piston_tip_tilt(
        opd_raw,
        xx=example["wf_sci"].x,
        yy=example["wf_sci"].y,
        mask=mask,
    )

    opd_corr_ptt, _, _ = phase_tools.remove_piston_tip_tilt(
        opd_corr,
        xx=example["wf_corr"].x,
        yy=example["wf_corr"].y,
        mask=mask,
    )

    # ========================================================
    # Printed summary
    # ========================================================

    print()
    print("Summary")
    for i, theta in enumerate(science_angles):
        print(
            f"  field={theta:4.1f} arcmin | "
            f"S no AO={strehl_no[i]:.4f} | "
            f"S GLAO={strehl_ao[i]:.4f} | "
            f"EE80 no AO={ee80_no[i]:.3f} λ/D | "
            f"EE80 GLAO={ee80_ao[i]:.3f} λ/D | "
            f"Moffat FWHM no AO={moffat_fwhm_no[i]:.3f} λ/D | "
            f"Moffat FWHM GLAO={moffat_fwhm_ao[i]:.3f} λ/D | "
            f"Moffat β no AO={moffat_beta_no[i]:.2f} | "
            f"Moffat β GLAO={moffat_beta_ao[i]:.2f}"
        )

    print()
    print("Example central-field OPD diagnostics at first time step")
    print("  raw PTT-removed OPD std [nm]:", np.nanstd(opd_raw_ptt[mask]) * 1e9)
    print("  GLAO PTT-removed OPD std [nm]:", np.nanstd(opd_corr_ptt[mask]) * 1e9)


if __name__ == "__main__":
    main()




# """
# hybrid_glao_correction_demo.py

# Hybrid GLAO correction demo using real phase screens.

# This reproduces the core logic of the old psf_analysis.py workflow using:

#     beam_trace_ORIG geometry
#     -> Wavefront2D bridge
#     -> OPD-space GLAO correction
#     -> FFT PSF / Strehl analysis

# Run from repository root:

#     python3 -m optics_simulator.examples.hybrid_glao_correction_demo

# Optional:

#     python3 -m optics_simulator.examples.hybrid_glao_correction_demo \
#         --fits-path phasescreens/scrns_2_order_v2/Testbench_phasescreens_20260506.fits \
#         --exposure-time 5.0 \
#         --dt 0.1
# """

# from __future__ import annotations

# import argparse
# import warnings
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from astropy.io import fits
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
# from optics_simulator import fresnel


# def build_bench_from_fits(fits_path: Path, opd_scale_m_per_rad: float):
#     layer_configs = [
#         {"label": "FA",  "z": -2.50,  "hz": 0.2, "point_xy": (26e-3, 0.0)},
#         {"label": "GL3", "z": -0.060, "hz": 1.4, "point_xy": (24e-3, 0.0)},
#         {"label": "GL2", "z": -0.030, "hz": 1.0, "point_xy": (24e-3, 0.0)},
#         {"label": "GL1", "z": -0.001, "hz": 1.7, "point_xy": (24e-3, 0.0)},
#     ]

#     bench = bt.OpticalBench3D()

#     with fits.open(fits_path) as hdul:
#         pix_scale = float(hdul[0].header["PIXSCALE"])

#         for cfg in layer_configs:
#             label = cfg["label"]
#             opd = np.asarray(hdul[label].data, dtype=float) * opd_scale_m_per_rad
#             map_extent_m = opd.shape[0] * pix_scale

#             x0, y0 = cfg["point_xy"]

#             bench.add(
#                 bt.RotatingPhaseScreen3D(
#                     point=[x0, y0, cfg["z"]],
#                     normal=[0.0, 0.0, 1.0],
#                     opd_map=opd,
#                     map_extent_m=map_extent_m,
#                     angular_velocity=2.0 * np.pi * cfg["hz"],
#                     label=label,
#                 )
#             )

#     return bench


# def make_lgs_beams(
#     wavelength,
#     beam_diameter,
#     source_plane_z,
#     pupil_point,
#     lgs_coords_arcmin,
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
#             f"L{i}",
#             3,
#             12,
#         )
#         beams.append(beam)

#     return beams


# def make_science_beams(
#     science_angles_arcmin,
#     wavelength,
#     beam_diameter,
#     source_plane_z,
#     pupil_point,
# ):
#     beams = []

#     for i, theta_arcmin in enumerate(science_angles_arcmin):
#         beam = bt.make_converging_beam_from_field_angles(
#             np.deg2rad(theta_arcmin / 60.0),
#             0.0,
#             source_plane_z,
#             pupil_point,
#             beam_diameter,
#             wavelength,
#             f"S{i}_{theta_arcmin:.2f}arcmin",
#             3,
#             12,
#         )
#         beams.append(beam)

#     return beams


# def apply_dm_correction_opd(opd_map_m, acts_across, mask):
#     """
#     Simple DM spatial filtering model in OPD space.

#     This matches the spirit of the old psf_analysis.py approximation:
#     the DM correction is the low-spatial-frequency part of the reconstructed
#     ground-layer OPD.
#     """
#     opd_map_m = np.asarray(opd_map_m, dtype=float)
#     mask = np.asarray(mask, dtype=bool)

#     if acts_across <= 0:
#         return np.zeros_like(opd_map_m)

#     avg_opd = np.nanmean(opd_map_m[mask])
#     work = np.where(mask, opd_map_m, avg_opd)

#     sigma = (opd_map_m.shape[0] / float(acts_across)) * 0.5
#     low_spatial = gaussian_filter(work, sigma=sigma, mode="reflect")

#     return np.where(mask, low_spatial, 0.0)


# def perfect_reference_wavefront(wf):
#     """
#     Same amplitude support as wf, but zero phase.
#     """
#     amp = wf.amplitude
#     return wf.with_field(amp.astype(complex), label="perfect reference")


# def normalized_intensity(field_or_wf):
#     if hasattr(field_or_wf, "field"):
#         field = field_or_wf.field
#     else:
#         field = field_or_wf

#     intensity = np.abs(field) ** 2
#     peak = np.nanmax(intensity)

#     if peak > 0:
#         return intensity / peak

#     return intensity


# def crop_by_extent(image, x_grid, y_grid, half_width):
#     keep = (
#         (x_grid >= -half_width)
#         & (x_grid <= half_width)
#         & (y_grid >= -half_width)
#         & (y_grid <= half_width)
#     )

#     rows = np.where(np.any(keep, axis=1))[0]
#     cols = np.where(np.any(keep, axis=0))[0]

#     if rows.size == 0 or cols.size == 0:
#         return image, x_grid, y_grid

#     ys, ye = rows[0], rows[-1] + 1
#     xs, xe = cols[0], cols[-1] + 1

#     return image[ys:ye, xs:xe], x_grid[ys:ye, xs:xe], y_grid[ys:ye, xs:xe]


# def plot_image(ax, image, title, extent=None, log=False):
#     image = np.asarray(image, dtype=float)

#     if log:
#         peak = np.nanmax(image)
#         if peak > 0:
#             image = np.log10(np.maximum(image / peak, 1e-8))
#         else:
#             image = np.zeros_like(image)

#     im = ax.imshow(
#         image,
#         origin="lower",
#         extent=extent,
#         interpolation="nearest",
#     )
#     ax.set_title(title)
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# def main():
#     parser = argparse.ArgumentParser(description="Hybrid GLAO correction demo.")

#     repo_root = Path(__file__).resolve().parents[2]
#     default_fits = (
#         repo_root
#         / "phasescreens"
#         / "scrns_2_order_v2"
#         / "Testbench_phasescreens_20260506.fits"
#     )

#     parser.add_argument("--fits-path", type=Path, default=default_fits)
#     parser.add_argument("--npix", type=int, default=256)
#     parser.add_argument("--pad-to", type=int, default=1024)
#     parser.add_argument("--exposure-time", type=float, default=5.0)
#     parser.add_argument("--dt", type=float, default=0.1)
#     parser.add_argument("--wfs-wavelength", type=float, default=633e-9)
#     parser.add_argument("--science-wavelength", type=float, default=1.2e-6)
#     parser.add_argument("--beam-diameter", type=float, default=13e-3)
#     parser.add_argument("--source-plane-z", type=float, default=-3.25)
#     parser.add_argument("--dm-acts-across", type=int, default=11)
#     parser.add_argument("--focal-length", type=float, default=0.25)
#     parser.add_argument("--focus-half-width-um", type=float, default=160.0)
#     parser.add_argument(
#         "--opd-scale",
#         type=float,
#         default=500e-9 / (2.0 * np.pi),
#         help="Conversion factor from FITS screen values to OPD [m].",
#     )

#     args = parser.parse_args()

#     if not args.fits_path.exists():
#         raise FileNotFoundError(f"Could not find FITS file: {args.fits_path}")

#     pupil_point = np.array([0.0, 0.0, 0.0])

#     science_angles = np.linspace(0.0, 10.0, 5)
#     lgs_coords = [(10.0, 10.0), (-10.0, 10.0), (10.0, -10.0), (-10.0, -10.0)]

#     times = np.arange(0.0, args.exposure_time, args.dt)

#     bench = build_bench_from_fits(
#         fits_path=args.fits_path,
#         opd_scale_m_per_rad=args.opd_scale,
#     )

#     lgs_beams = make_lgs_beams(
#         wavelength=args.wfs_wavelength,
#         beam_diameter=args.beam_diameter,
#         source_plane_z=args.source_plane_z,
#         pupil_point=pupil_point,
#         lgs_coords_arcmin=lgs_coords,
#     )

#     sci_beams = make_science_beams(
#         science_angles_arcmin=science_angles,
#         wavelength=args.science_wavelength,
#         beam_diameter=args.beam_diameter,
#         source_plane_z=args.source_plane_z,
#         pupil_point=pupil_point,
#     )

#     # Reference PSF from the central science beam amplitude support.
#     wf_ref0 = btw.sample_wavefront_on_pupil_plane(
#         beam=sci_beams[0],
#         bench=bt.OpticalBench3D(),
#         pupil_point=pupil_point,
#         t=0.0,
#         npix=args.npix,
#         beam_trace_module=bt,
#     )
#     wf_ref0 = perfect_reference_wavefront(wf_ref0)

#     perfect_psf = psf_tools.psf_from_wavefront(
#         wf_ref0,
#         pad_to=args.pad_to,
#         normalize=None,
#     )

#     accum_no_ao = [
#         np.zeros((args.pad_to, args.pad_to), dtype=float)
#         for _ in sci_beams
#     ]
#     accum_glao = [
#         np.zeros((args.pad_to, args.pad_to), dtype=float)
#         for _ in sci_beams
#     ]

#     example = {}

#     print("Hybrid GLAO correction demo")
#     print("  FITS path:", args.fits_path)
#     print("  n times:", len(times))
#     print("  exposure [s]:", args.exposure_time)
#     print("  dt [s]:", args.dt)
#     print("  WFS wavelength [nm]:", args.wfs_wavelength * 1e9)
#     print("  science wavelength [um]:", args.science_wavelength * 1e6)
#     print("  DM acts across:", args.dm_acts_across)

#     for it, t in enumerate(times):



#         lgs_wfs = []
#         lgs_samples = []

#         for b in lgs_beams:
#             wf_lgs, sample_lgs = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
#                 beam=b,
#                 bench=bench,
#                 pupil_point=pupil_point,
#                 t=float(t),
#                 npix=args.npix,
#                 beam_trace_module=bt,
#             )
#             lgs_wfs.append(wf_lgs)
#             lgs_samples.append(sample_lgs)

#         lgs_mask = lgs_samples[0]["mask"]

#         # Important: use unwrapped OPD from beam_trace sample, not np.angle(wf.field)
#         lgs_opd_maps = [
#             np.nan_to_num(sample["opd_map_m"], nan=0.0)
#             for sample in lgs_samples
#         ]


#         # lgs_wfs = [
#         #     btw.sample_wavefront_on_pupil_plane(
#         #         beam=b,
#         #         bench=bench,
#         #         pupil_point=pupil_point,
#         #         t=float(t),
#         #         npix=args.npix,
#         #         beam_trace_module=bt,
#         #     )
#         #     for b in lgs_beams
#         # ]

#         # lgs_mask = lgs_wfs[0].amplitude > 0

#         # lgs_opd_maps = [
#         #     phase_tools.phase_to_opd(wf.phase, wf.wavelength)
#         #     for wf in lgs_wfs
#         # ]

#         gl_opd = np.mean(lgs_opd_maps, axis=0)
#         gl_opd = np.where(lgs_mask, gl_opd, 0.0)

#         dm_opd_corr = apply_dm_correction_opd(
#             gl_opd,
#             acts_across=args.dm_acts_across,
#             mask=lgs_mask,
#         )

#         for i, beam in enumerate(sci_beams):


#             wf_sci, sample_sci = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
#                 beam=beam,
#                 bench=bench,
#                 pupil_point=pupil_point,
#                 t=float(t),
#                 npix=args.npix,
#                 beam_trace_module=bt,
#             )
                        
#             # wf_sci = btw.sample_wavefront_on_pupil_plane(
#             #     beam=beam,
#             #     bench=bench,
#             #     pupil_point=pupil_point,
#             #     t=float(t),
#             #     npix=args.npix,
#             #     beam_trace_module=bt,
#             # )

#             psf_no = psf_tools.psf_from_wavefront(
#                 wf_sci,
#                 pad_to=args.pad_to,
#                 normalize=None,
#             )

#             wf_corr = wave_optics.apply_opd(
#                 wf_sci,
#                 opd_map_m=-dm_opd_corr,
#                 mask=lgs_mask,
#                 label=f"{wf_sci.label}_glao_corrected",
#             )

#             psf_ao = psf_tools.psf_from_wavefront(
#                 wf_corr,
#                 pad_to=args.pad_to,
#                 normalize=None,
#             )

#             accum_no_ao[i] += psf_no
#             accum_glao[i] += psf_ao

#             if it == 0 and i == 0:
#                 example = {
#                     "t": t,
#                     "wf_sci": wf_sci,
#                     "wf_corr": wf_corr,
#                     "sample_sci": sample_sci,
#                     "gl_opd": gl_opd,
#                     "dm_opd_corr": dm_opd_corr,
#                     "lgs_mask": lgs_mask,
#                 }
                                
#                 # example = {
#                 #     "t": t,
#                 #     "wf_sci": wf_sci,
#                 #     "wf_corr": wf_corr,
#                 #     "gl_opd": gl_opd,
#                 #     "dm_opd_corr": dm_opd_corr,
#                 #     "lgs_mask": lgs_mask,
#                 # }

#     mean_no_ao = [p / len(times) for p in accum_no_ao]
#     mean_glao = [p / len(times) for p in accum_glao]


#     # Basic lambda/D pixel scale for FFT PSFs.
#     angular_pixel_scale_ld = 1.0 / (args.pad_to / (args.npix / 2.0))

#     analysis_no = [
#         psf_tools.psf_tools.analyse_psf_with_fits(
#             p,
#             perfect_psf,
#             angular_pixel_scale_ld=angular_pixel_scale_ld,
#             fit_module=bt,
#             crop_half_width_ld=6.0,
#             moffat_fit_region="all",
#         )
#         for p in mean_no_ao
#     ]

#     analysis_ao = [
#         psf_tools.psf_tools.analyse_psf_with_fits(
#             p,
#             perfect_psf,
#             angular_pixel_scale_ld=angular_pixel_scale_ld,
#             fit_module=bt,
#             crop_half_width_ld=6.0,
#             moffat_fit_region="all",
#         )
#         for p in mean_glao
#     ]

#     strehl_no = [r["strehl"] for r in analysis_no]
#     strehl_ao = [r["strehl"] for r in analysis_ao]

#     ee80_no = [r["ee80"] for r in analysis_no]
#     ee80_ao = [r["ee80"] for r in analysis_ao]

#     gauss_fwhm_no = [r["gaussian_fwhm"] for r in analysis_no]
#     gauss_fwhm_ao = [r["gaussian_fwhm"] for r in analysis_ao]

#     gauss_ell_no = [r["gaussian_ell"] for r in analysis_no]
#     gauss_ell_ao = [r["gaussian_ell"] for r in analysis_ao]

#     moffat_fwhm_no = [r["moffat_fwhm"] for r in analysis_no]
#     moffat_fwhm_ao = [r["moffat_fwhm"] for r in analysis_ao]

#     moffat_ell_no = [r["moffat_ell"] for r in analysis_no]
#     moffat_ell_ao = [r["moffat_ell"] for r in analysis_ao]

#     moffat_beta_no = [r["moffat_beta"] for r in analysis_no]
#     moffat_beta_ao = [r["moffat_beta"] for r in analysis_ao]













#     # strehl_no = [
#     #     psf_tools.strehl_from_psfs(p, perfect_psf)
#     #     for p in mean_no_ao
#     # ]
#     # strehl_ao = [
#     #     psf_tools.strehl_from_psfs(p, perfect_psf)
#     #     for p in mean_glao
#     # ]

#     # # Basic EE80 in lambda/D pixel coordinates.
#     # angular_pixel_scale_ld = 1.0 / (args.pad_to / (args.npix / 2.0))

#     # ee80_no = []
#     # ee80_ao = []

#     # for p_no, p_ao in zip(mean_no_ao, mean_glao):
#     #     r = np.linspace(0.0, 10.0, 400)

#     #     xp, yp = psf_tools.peak_location(p_no)
#     #     ee = psf_tools.encircled_energy(
#     #         p_no,
#     #         r,
#     #         x0=xp,
#     #         y0=yp,
#     #         radial_scale=angular_pixel_scale_ld,
#     #     )
#     #     ee80_no.append(psf_tools.encircled_energy_radius(r, ee, fraction=0.8))

#     #     xp, yp = psf_tools.peak_location(p_ao)
#     #     ee = psf_tools.encircled_energy(
#     #         p_ao,
#     #         r,
#     #         x0=xp,
#     #         y0=yp,
#     #         radial_scale=angular_pixel_scale_ld,
#     #     )
#     #     ee80_ao.append(psf_tools.encircled_energy_radius(r, ee, fraction=0.8))

#     # Plot PSF crops.
#     fig_grid, axes = plt.subplots(2, len(sci_beams), figsize=(3.5 * len(sci_beams), 7))

#     for i, theta in enumerate(science_angles):
#         psf_no = mean_no_ao[i] / np.nanmax(mean_no_ao[i])
#         psf_ao = mean_glao[i] / np.nanmax(mean_glao[i])

#         pack_no = {
#             "psf": psf_no,
#         }

#         x_ld, y_ld = fresnel.lambda_over_d_coordinates( #psf_tools.fresnel.lambda_over_d_coordinates(
#             psf_no.shape,
#             dx_pupil=example["wf_sci"].dx,
#             pupil_diameter=args.beam_diameter,
#         )

#         crop_no, x_crop, y_crop = psf_tools.crop_psf_lambda_over_d(
#             psf_no,
#             x_ld,
#             y_ld,
#             half_width_ld=6.0,
#             centre="peak",
#         )

#         crop_ao, x_crop2, y_crop2 = psf_tools.crop_psf_lambda_over_d(
#             psf_ao,
#             x_ld,
#             y_ld,
#             half_width_ld=6.0,
#             centre="peak",
#         )

#         extent_no = [
#             np.nanmin(x_crop),
#             np.nanmax(x_crop),
#             np.nanmin(y_crop),
#             np.nanmax(y_crop),
#         ]

#         extent_ao = [
#             np.nanmin(x_crop2),
#             np.nanmax(x_crop2),
#             np.nanmin(y_crop2),
#             np.nanmax(y_crop2),
#         ]

#         plot_image(
#             axes[0, i],
#             crop_no,
#             f"No AO {theta:.1f}'\nS={strehl_no[i]:.3f}",
#             extent=extent_no,
#             log=True,
#         )

#         plot_image(
#             axes[1, i],
#             crop_ao,
#             f"GLAO {theta:.1f}'\nS={strehl_ao[i]:.3f}",
#             extent=extent_ao,
#             log=True,
#         )

#         axes[0, i].set_xlabel("x [λ/D]")
#         axes[0, i].set_ylabel("y [λ/D]")
#         axes[1, i].set_xlabel("x [λ/D]")
#         axes[1, i].set_ylabel("y [λ/D]")

#     plt.tight_layout()
#     plt.show()

#     # Summary plots.
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#     axes[0].plot(science_angles, strehl_no, "o--", label="No AO")
#     axes[0].plot(science_angles, strehl_ao, "o-", label="GLAO")
#     axes[0].set_xlabel("Field angle [arcmin]")
#     axes[0].set_ylabel("FFT Strehl")
#     axes[0].grid(True, alpha=0.3)
#     axes[0].legend()

#     axes[1].plot(science_angles, ee80_no, "o--", label="No AO")
#     axes[1].plot(science_angles, ee80_ao, "o-", label="GLAO")
#     axes[1].set_xlabel("Field angle [arcmin]")
#     axes[1].set_ylabel("EE80 radius [λ/D]")
#     axes[1].grid(True, alpha=0.3)
#     axes[1].legend()


#     opd_raw = np.nan_to_num(example["sample_sci"]["opd_map_m"], nan=0.0)
#     opd_corr = opd_raw - example["dm_opd_corr"]
#     # opd_raw = phase_tools.phase_to_opd(
#     #     example["wf_sci"].phase,
#     #     example["wf_sci"].wavelength,
#     # )

#     # opd_corr = phase_tools.phase_to_opd(
#     #     example["wf_corr"].phase,
#     #     example["wf_corr"].wavelength,
#     # )

#     mask = example["wf_sci"].amplitude > 0

#     opd_raw_ptt, _, _ = phase_tools.remove_piston_tip_tilt(
#         opd_raw,
#         xx=example["wf_sci"].x,
#         yy=example["wf_sci"].y,
#         mask=mask,
#     )

#     opd_corr_ptt, _, _ = phase_tools.remove_piston_tip_tilt(
#         opd_corr,
#         xx=example["wf_corr"].x,
#         yy=example["wf_corr"].y,
#         mask=mask,
#     )

#     axes[2].bar(
#         ["raw", "GLAO"],
#         [
#             np.nanstd(opd_raw_ptt[mask]) * 1e9,
#             np.nanstd(opd_corr_ptt[mask]) * 1e9,
#         ],
#     )
#     axes[2].set_ylabel("Example PTT-removed OPD std [nm]")
#     axes[2].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()

#     print()
#     print("Summary")
#     for i, theta in enumerate(science_angles):
#         print(
#             f"  field={theta:4.1f} arcmin | "
#             f"S no AO={strehl_no[i]:.4f} | "
#             f"S GLAO={strehl_ao[i]:.4f} | "
#             f"EE80 no AO={ee80_no[i]:.3f} λ/D | "
#             f"EE80 GLAO={ee80_ao[i]:.3f} λ/D"
#         )

#     print()
#     print("Example central-field OPD diagnostics at first time step")
#     print("  raw PTT-removed OPD std [nm]:", np.nanstd(opd_raw_ptt[mask]) * 1e9)
#     print("  GLAO PTT-removed OPD std [nm]:", np.nanstd(opd_corr_ptt[mask]) * 1e9)


# if __name__ == "__main__":
#     main()