"""
hybrid_glao_phase_screen_demo.py

Real phase-screen hybrid demo.

This uses:
    - optics_simulator.beam_trace_ORIG as the current geometric engine
    - optics_simulator.beam_trace_wave to create Wavefront2D objects
    - optics_simulator.wave_optics for scaled focal-plane propagation
    - optics_simulator.psf_tools for FFT PSF utilities

Run from repository root:

    python3 -m optics_simulator.examples.hybrid_glao_phase_screen_demo

or explicitly:

    python3 -m optics_simulator.examples.hybrid_glao_phase_screen_demo \
        --fits-path phasescreens/scrns_2_order_v2/Testbench_phasescreens_20260506.fits
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

# from optics_simulator import beam_trace as bt
# from optics_simulator import beam_trace_wave as btw
# from optics_simulator import psf_tools
# from optics_simulator import wave_optics

from optics_simulator import beam_trace as bt
from optics_simulator import beam_trace_wave as btw
from optics_simulator import psf_tools
from optics_simulator import wave_optics
from optics_simulator import phase_tools

def build_bench_from_fits(fits_path: Path, opd_scale_m_per_rad: float):
    """
    Build OpticalBench3D from the real FITS phase-screen file.

    The FITS screen data are assumed to be phase-like values in radians and are
    converted to OPD using opd = data * opd_scale_m_per_rad.
    """
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


def make_science_beams(
    science_angles_arcmin,
    wavelength,
    beam_diameter,
    source_plane_z,
    pupil_point,
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


def normalized_intensity_from_field(field):
    intensity = np.abs(field) ** 2
    peak = np.nanmax(intensity)
    if peak > 0:
        intensity = intensity / peak
    return intensity


def plot_image(ax, image, title, extent=None, log=False, cmap=None):
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
    )

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def crop_by_extent(image, x_grid, y_grid, half_width):
    keep = (
        (x_grid >= -half_width)
        & (x_grid <= half_width)
        & (y_grid >= -half_width)
        & (y_grid <= half_width)
    )

    rows = np.where(np.any(keep, axis=1))[0]
    cols = np.where(np.any(keep, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return image, x_grid, y_grid

    ys, ye = rows[0], rows[-1] + 1
    xs, xe = cols[0], cols[-1] + 1

    return image[ys:ye, xs:xe], x_grid[ys:ye, xs:xe], y_grid[ys:ye, xs:xe]


def main():
    parser = argparse.ArgumentParser(description="Hybrid GLAO phase-screen demo.")

    repo_root = Path(__file__).resolve().parents[2]
    default_fits = repo_root / "phasescreens" / "scrns_2_order_v2" / "Testbench_phasescreens_20260506.fits"

    parser.add_argument("--fits-path", type=Path, default=default_fits)
    parser.add_argument("--npix", type=int, default=256)
    parser.add_argument("--pad-to", type=int, default=1024)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--beam-index", type=int, default=0)
    parser.add_argument("--wavelength", type=float, default=1.2e-6)
    parser.add_argument("--beam-diameter", type=float, default=13e-3)
    parser.add_argument("--source-plane-z", type=float, default=-3.25)
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

    bench = build_bench_from_fits(
        fits_path=args.fits_path,
        opd_scale_m_per_rad=args.opd_scale,
    )

    sci_beams = make_science_beams(
        science_angles_arcmin=science_angles,
        wavelength=args.wavelength,
        beam_diameter=args.beam_diameter,
        source_plane_z=args.source_plane_z,
        pupil_point=pupil_point,
    )

    if args.beam_index < 0 or args.beam_index >= len(sci_beams):
        raise ValueError(f"--beam-index must be between 0 and {len(sci_beams)-1}")

    beam = sci_beams[args.beam_index]
    field_angle = science_angles[args.beam_index]

    wf, sample = btw.sample_wavefront_and_beam_sample_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=pupil_point,
        t=args.time,
        npix=args.npix,
        beam_trace_module=bt,
    )

    wf_pad = wave_optics.pad_wavefront(wf, pad_to=args.pad_to)

    wf_focus = wave_optics.lens_focal_plane(
        wf_pad,
        focal_length=args.focal_length,
        include_global_phase=False,
        label="scaled focal plane",
    )

    # Perfect/reference wavefront with same sampled pupil amplitude but no phase.
    wf_ref = wf.with_field(np.asarray(sample["amplitude"], dtype=complex), label="perfect reference")

    psf_pack = psf_tools.psf_pack_from_wavefront(
        wf,
        pad_to=args.pad_to,
        normalize="peak",
        pupil_diameter=args.beam_diameter,
    )

    psf_ref = psf_tools.psf_from_wavefront(
        wf_ref,
        pad_to=args.pad_to,
        normalize=None,
    )

    psf_raw = psf_tools.psf_from_wavefront(
        wf,
        pad_to=args.pad_to,
        normalize=None,
    )

    strehl_fft = psf_tools.strehl_from_psfs(psf_raw, psf_ref)

    psf = psf_pack["psf"]
    x_ld = psf_pack["x_ld"]
    y_ld = psf_pack["y_ld"]

    psf_crop, x_ld_crop, y_ld_crop = psf_tools.crop_psf_lambda_over_d(
        psf,
        x_ld,
        y_ld,
        half_width_ld=10.0,
        centre="origin",
    )

    extent_ld = [
        np.nanmin(x_ld_crop),
        np.nanmax(x_ld_crop),
        np.nanmin(y_ld_crop),
        np.nanmax(y_ld_crop),
    ]

    focus_intensity = normalized_intensity_from_field(wf_focus.field)

    focus_x_um = wf_focus.x * 1e6
    focus_y_um = wf_focus.y * 1e6

    focus_crop, focus_x_crop, focus_y_crop = crop_by_extent(
        focus_intensity,
        focus_x_um,
        focus_y_um,
        half_width=args.focus_half_width_um,
    )

    extent_focus_um = [
        np.nanmin(focus_x_crop),
        np.nanmax(focus_x_crop),
        np.nanmin(focus_y_crop),
        np.nanmax(focus_y_crop),
    ]

    opd_nm = np.asarray(sample["opd_map_m"], dtype=float) * 1e9
    phase_rad = np.asarray(sample["phase_map_rad"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_image(
        axes[0, 0],
        opd_nm,
        f"Pupil OPD, field={field_angle:.1f} arcmin, t={args.time:g}s",
        extent=[
            np.nanmin(sample["xx"]) * 1e3,
            np.nanmax(sample["xx"]) * 1e3,
            np.nanmin(sample["yy"]) * 1e3,
            np.nanmax(sample["yy"]) * 1e3,
        ],
        log=False,
        cmap="viridis",
    )
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("y [mm]")

    plot_image(
        axes[0, 1],
        phase_rad,
        "Pupil phase [rad]",
        extent=[
            np.nanmin(sample["xx"]) * 1e3,
            np.nanmax(sample["xx"]) * 1e3,
            np.nanmin(sample["yy"]) * 1e3,
            np.nanmax(sample["yy"]) * 1e3,
        ],
        log=False,
        cmap="twilight",
    )
    axes[0, 1].set_xlabel("x [mm]")
    axes[0, 1].set_ylabel("y [mm]")

    plot_image(
        axes[1, 0],
        psf_crop,
        f"FFT PSF, Strehl={strehl_fft:.3f}",
        extent=extent_ld,
        log=True,
    )
    axes[1, 0].set_xlabel("x [λ/D]")
    axes[1, 0].set_ylabel("y [λ/D]")

    plot_image(
        axes[1, 1],
        focus_crop,
        "Scaled focal-plane intensity",
        extent=extent_focus_um,
        log=True,
    )
    axes[1, 1].set_xlabel("x [µm]")
    axes[1, 1].set_ylabel("y [µm]")

    plt.tight_layout()
    plt.show()

    airy_radius_um = 1.22 * args.wavelength * args.focal_length / args.beam_diameter * 1e6

    mask = np.asarray(sample["mask"], dtype=bool)

    print("Hybrid GLAO phase-screen demo")
    print("  FITS path:", args.fits_path)
    print("  beam label:", beam.label)
    print("  field angle [arcmin]:", field_angle)
    print("  time [s]:", args.time)
    print("  wavelength [um]:", args.wavelength * 1e6)
    print("  input wavefront shape:", wf.shape)
    print("  input dx [um]:", wf.dx * 1e6)
    print("  padded shape:", wf_pad.shape)
    print("  focal dx [um]:", wf_focus.dx * 1e6)
    print("  Airy first-null radius [um]:", airy_radius_um)
    print("  pixels per Airy radius:", airy_radius_um / (wf_focus.dx * 1e6))
    print("  input power:", wave_optics.power(wf_pad))
    print("  focal power:", wave_optics.power(wf_focus))
    print(
        "  focal relative power error:",
        abs(wave_optics.power(wf_focus) - wave_optics.power(wf_pad))
        / wave_optics.power(wf_pad),
    )


    # print("  OPD RMS [nm]:", np.nanstd(opd_nm[mask]))
    # print("  phase RMS [rad]:", np.nanstd(phase_rad[mask]))
    # print("  FFT Strehl:", strehl_fft)

    phase_report = phase_tools.phase_rms_report(
        phase_rad,
        mask=mask,
        xx=sample["xx"],
        yy=sample["yy"],
    )

    opd_piston_removed, opd_piston = phase_tools.remove_piston(
        sample["opd_map_m"],
        mask=mask,
    )

    opd_ptt_removed, opd_ptt_coeff, _ = phase_tools.remove_piston_tip_tilt(
        sample["opd_map_m"],
        xx=sample["xx"],
        yy=sample["yy"],
        mask=mask,
    )

    print("  OPD std raw [nm]:", np.nanstd(sample["opd_map_m"][mask]) * 1e9)
    print("  OPD piston removed std [nm]:", np.nanstd(opd_piston_removed[mask]) * 1e9)
    print("  OPD PTT removed std [nm]:", np.nanstd(opd_ptt_removed[mask]) * 1e9)

    print("  phase std raw [rad]:", phase_report["raw_std_rad"])
    print("  phase piston removed RMS [rad]:", phase_report["piston_removed_rms_rad"])
    print("  phase PTT removed RMS [rad]:", phase_report["ptt_removed_rms_rad"])

    print("  Maréchal Strehl, piston removed:", phase_report["marechal_strehl_piston_removed"])
    print("  Maréchal Strehl, PTT removed:", phase_report["marechal_strehl_ptt_removed"])
    print("  FFT Strehl:", strehl_fft)



if __name__ == "__main__":
    main()