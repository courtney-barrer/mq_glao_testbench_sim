"""
fresnel_beam_trace_demo.py

Hybrid ray + wave-optics demo.

Run from repository root:

    python3 -m optics_simulator.examples.fresnel_beam_trace_demo

This example demonstrates:

    beam_trace geometry
        -> Wavefront2D
        -> fixed-grid Fresnel propagation
        -> properly sampled lens focal-plane transform
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from optics_simulator import beam_trace as bt
from optics_simulator import beam_trace_wave as btw
from optics_simulator import psf_tools
from optics_simulator import wave_optics


def build_demo_wavefront(npix: int = 256):
    """
    Build a flat circular pupil wavefront using beam_trace_ORIG.
    """
    bench = bt.OpticalBench3D()

    beam = bt.Beam3D.collimated_circular(
        radius=6.5e-3,
        nrings=1,
        nphi=4,
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        wavelength=633e-9,
        label="demo_flat_circular_beam",
    )

    wf = btw.sample_wavefront_on_pupil_plane(
        beam=beam,
        bench=bench,
        pupil_point=(0.0, 0.0, 0.0),
        t=0.0,
        npix=npix,
        beam_trace_module=bt,
    )

    return wf, beam


def normalised_intensity(wf):
    intensity = wf.intensity
    peak = np.nanmax(intensity)
    if peak > 0:
        return intensity / peak
    return intensity


def plot_image(ax, image, title, extent=None, log=False):
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
    )

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def crop_by_physical_extent(image, x_grid, y_grid, half_width):
    """
    Crop a 2D image and matching physical coordinate grids to +/- half_width.
    """
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
    parser = argparse.ArgumentParser(description="Hybrid beam_trace + Fresnel demo.")
    parser.add_argument("--npix", type=int, default=256, help="Input pupil grid size.")
    parser.add_argument("--pad-to", type=int, default=1024, help="Padded grid size.")
    parser.add_argument("--z", type=float, default=0.25, help="Free-space propagation distance [m].")
    parser.add_argument("--focal-length", type=float, default=0.25, help="Thin lens focal length [m].")
    parser.add_argument(
        "--focus-half-width-um",
        type=float,
        default=80.0,
        help="Half-width of focal-plane crop [um].",
    )
    parser.add_argument("--log", action="store_true", help="Use log display for propagated intensity.")
    args = parser.parse_args()

    wf0, beam = build_demo_wavefront(npix=args.npix)

    wf_pad = wave_optics.pad_wavefront(
        wf0,
        pad_to=args.pad_to,
        label="padded pupil",
    )

    wf_free = wave_optics.propagate(
        wf_pad,
        z=args.z,
        method="angular_spectrum",
        include_global_phase=False,
        label=f"free-space propagated z={args.z:g} m",
    )

    wf_focus = wave_optics.lens_focal_plane(
        wf_pad,
        focal_length=args.focal_length,
        include_global_phase=False,
        label="properly sampled lens focal plane",
    )

    psf_fft = psf_tools.psf_from_wavefront(
        wf0,
        pad_to=args.pad_to,
        normalize="peak",
    )

    psf_pack = psf_tools.psf_pack_from_wavefront(
        wf0,
        pad_to=args.pad_to,
        normalize="peak",
        pupil_diameter=beam.diameter,
    )

    psf_fft = psf_pack["psf"]
    x_ld = psf_pack["x_ld"]
    y_ld = psf_pack["y_ld"]

    psf_crop, x_ld_crop, y_ld_crop = psf_tools.crop_psf_lambda_over_d(
        psf_fft,
        x_ld,
        y_ld,
        half_width_ld=10.0,
        centre="origin",
    )

    pupil_intensity = normalised_intensity(wf_pad)
    free_intensity = normalised_intensity(wf_free)
    focus_intensity = normalised_intensity(wf_focus)

    # Pupil/free-space extents in mm
    x_mm = wf_pad.x[0, :] * 1e3
    y_mm = wf_pad.y[:, 0] * 1e3
    extent_mm = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    # FFT PSF crop extent in lambda/D
    extent_ld = [
        np.nanmin(x_ld_crop),
        np.nanmax(x_ld_crop),
        np.nanmin(y_ld_crop),
        np.nanmax(y_ld_crop),
    ]

    # Focal-plane crop in um
    focus_x_um = wf_focus.x * 1e6
    focus_y_um = wf_focus.y * 1e6

    focus_crop, focus_x_crop, focus_y_crop = crop_by_physical_extent(
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_image(
        axes[0, 0],
        pupil_intensity,
        "Input padded pupil intensity",
        extent=extent_mm,
        log=False,
    )
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("y [mm]")

    plot_image(
        axes[0, 1],
        psf_crop,
        "Fraunhofer FFT PSF",
        extent=extent_ld,
        log=True,
    )
    axes[0, 1].set_xlabel(r"x [$\lambda/D$]")
    axes[0, 1].set_ylabel(r"y [$\lambda/D$]")

    plot_image(
        axes[1, 0],
        free_intensity,
        f"Fixed-grid Fresnel propagation, z={args.z:g} m",
        extent=extent_mm,
        log=args.log,
    )
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("y [mm]")

    plot_image(
        axes[1, 1],
        focus_crop,
        f"Scaled lens focal plane, f={args.focal_length:g} m",
        extent=extent_focus_um,
        log=True,
    )
    axes[1, 1].set_xlabel("x [µm]")
    axes[1, 1].set_ylabel("y [µm]")

    plt.tight_layout()
    plt.show()

    airy_radius_m = 1.22 * wf0.wavelength * args.focal_length / beam.diameter

    print("Input wavefront")
    print("  shape:", wf0.shape)
    print("  dx [m]:", wf0.dx)
    print("  power:", wave_optics.power(wf0))

    print("Padded wavefront")
    print("  shape:", wf_pad.shape)
    print("  dx [m]:", wf_pad.dx)
    print("  power:", wave_optics.power(wf_pad))

    print("Free propagated wavefront")
    print("  shape:", wf_free.shape)
    print("  dx [m]:", wf_free.dx)
    print("  power:", wave_optics.power(wf_free))
    print(
        "  relative power error:",
        abs(wave_optics.power(wf_free) - wave_optics.power(wf_pad))
        / wave_optics.power(wf_pad),
    )

    print("Scaled lens focal plane")
    print("  shape:", wf_focus.shape)
    print("  dx [um]:", wf_focus.dx * 1e6)
    print("  dy [um]:", wf_focus.dy * 1e6)
    print("  power:", wave_optics.power(wf_focus))
    print(
        "  relative power error:",
        abs(wave_optics.power(wf_focus) - wave_optics.power(wf_pad))
        / wave_optics.power(wf_pad),
    )

    print("Nominal Airy first-null radius [um]:", airy_radius_m * 1e6)
    print("Pixels per Airy first-null radius:", airy_radius_m / wf_focus.dx)
    print("FFT PSF peak:", np.nanmax(psf_fft))
    print("Focal-plane peak-normalized intensity:", np.nanmax(focus_intensity))


if __name__ == "__main__":
    main()