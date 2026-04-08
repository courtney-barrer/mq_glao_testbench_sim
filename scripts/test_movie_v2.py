import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import beam_trace as bt

# ==========================================
# 1. ROBUST PHYSICS HELPERS
# ==========================================

def phase_to_opd(phase_rad, wavelength_m):
    """
    Convert phase [rad] to OPD [m]:
        OPD = phase * lambda / (2*pi)
    """
    return phase_rad * wavelength_m / (2.0 * np.pi)


def opd_to_phase(opd_m, wavelength_m):
    """
    Convert OPD [m] to phase [rad]:
        phase = 2*pi*OPD / lambda
    """
    return (2.0 * np.pi / wavelength_m) * opd_m


def get_rotating_screen_image(elem, t, npix=256):
    """
    Return OPD image of a rotating phase screen for visualization.
    """
    r_max = elem.clear_radius
    u = np.linspace(-r_max, r_max, npix)
    uu, vv = np.meshgrid(u, u)
    uv_grid = np.stack([uu, vv], axis=-1)
    opd, valid = elem.sample_uv(uv_grid.reshape(-1, 2), t=t)
    return np.where(valid.reshape(uu.shape), opd.reshape(uu.shape), 0.0)


def apply_dm_correction_opd(opd_map, acts, mask):
    """
    DM correction proxy in OPD space [m].

    The DM is modeled as a spatially limited corrector that removes only the
    low spatial frequency content. The returned map is an OPD map [m].
    """
    if acts == 0:
        return np.zeros_like(opd_map)

    avg_opd = np.mean(opd_map[mask])
    work_opd = np.where(mask, opd_map, avg_opd)

    sigma = (opd_map.shape[0] / acts) * 0.5
    low_spatial = gaussian_filter(work_opd, sigma=sigma, mode='reflect')

    return np.where(mask, low_spatial, 0.0)


def pad_field(field, pad_size):
    """
    Symmetrically pad a 2D complex field to pad_size x pad_size.
    """
    ny, nx = field.shape
    if ny != nx:
        raise ValueError(f"Expected square field, got shape {field.shape}")
    if pad_size < ny:
        raise ValueError(f"pad_size={pad_size} must be >= field size {ny}")

    pad_total = pad_size - ny
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    return np.pad(
        field,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode='constant'
    )


def simulate_sci_performance(beam, dm_opd_corr, bench, t, pad_size, npix_pupil, science_wavelength):
    """
    Simulate the residual science pupil and PSF.

    Parameters
    ----------
    beam : beam object
    dm_opd_corr : 2D ndarray
        DM correction in OPD units [m].
    bench : OpticalBench3D
    t : float
        Time [s]
    pad_size : int
    npix_pupil : int
    science_wavelength : float
        Wavelength of the science beam [m]

    Returns
    -------
    residual_phase : 2D ndarray
        Residual phase map [rad] at the science wavelength.
    psf_norm : 2D ndarray
        Peak-normalized PSF.
    """
    samp = bt.sample_beam_phase_amplitude_on_pupil_plane(beam, bench, [0, 0, 0], t, npix_pupil)

    raw_phase = np.nan_to_num(samp["phase_map_rad"])
    dm_phase_corr = opd_to_phase(dm_opd_corr, science_wavelength)

    residual_phase = np.where(samp["mask"], raw_phase - dm_phase_corr, 0.0)

    field = np.where(
        samp["mask"],
        samp["amplitude"] * np.exp(1j * residual_phase),
        0.0
    )

    padded = pad_field(field, pad_size)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))) ** 2

    if np.max(psf) > 0:
        psf_norm = psf #/ np.max(psf)
    else:
        psf_norm = psf

    return residual_phase, psf_norm


# ==========================================
# 2. TELEMETRY GENERATION
# ==========================================

def run_glao_telemetry(
    exposure_s=5.6,
    dt=0.1,
    ao_start_time=2.0,
    fa_scale=0.6,
    dm_acts_across=35,
):
    """
    Run a GLAO telemetry simulation.

    Features
    --------
    - First `ao_start_time` seconds are open loop (DM off).
    - After that, GLAO is enabled.
    - Free-atmosphere layer can be softened using `fa_scale` for presentation.

    Notes
    -----
    The DM reconstruction is performed in OPD [m], not phase [rad].
    LGS pupil phases are converted to OPD, averaged, filtered by the DM model,
    then converted back to science-wavelength phase only when applied to each
    science beam.
    """
    WAVELENGTH = 589e-9
    SCIENCE_WAVELENGTH = 0.589e-6
    D_BEAM = 0.013
    NPIX_PUPIL = 256
    PAD_SIZE = 2048

    SCI_OFFS = [0.0, 5.0, 10.0]

    # Update this path if needed
    FITS_PATH = "/home/bbarrer/mq_glao_testbench_sim/phasescreens/batch1_test/phasescreens_median_dmScaled-1_radialScaled-0.fits"

    with fits.open(FITS_PATH) as hdul:
        pix_scale = hdul[0].header["PIXSCALE"]
        bench = bt.OpticalBench3D()

        layers = [
            {"lbl": "FA",  "z": -2.50,  "hz": 0.4, "scale": fa_scale},
            {"lbl": "GL3", "z": -0.060, "hz": 1.4, "scale": 1.0},
            {"lbl": "GL2", "z": -0.030, "hz": 1.0, "scale": 1.0},
            {"lbl": "GL1", "z": -0.001, "hz": 0.7, "scale": 1.0},
        ]

        for l in layers:
            # FITS stored as phase [rad] at 500 nm -> convert to OPD [m]
            opd = (hdul[l["lbl"]].data * 500e-9) / (2.0 * np.pi)
            opd = l["scale"] * opd

            if l["lbl"] == "FA":
                bench.add(
                    bt.RotatingPhaseScreen3D(
                        point=[26e-3, 0, l["z"]],
                        normal=[0, 0, 1],
                        opd_map=opd,
                        map_extent_m=opd.shape[0] * pix_scale,
                        angular_velocity=2.0 * np.pi * l["hz"],
                        label=l["lbl"],
                    )
                )
            else:
                bench.add(
                    bt.RotatingPhaseScreen3D(
                        point=[34e-3, 0, l["z"]],
                        normal=[0, 0, 1],
                        opd_map=opd,
                        map_extent_m=opd.shape[0] * pix_scale,
                        angular_velocity=2.0 * np.pi * l["hz"],
                        label=l["lbl"],
                    )
                )

    lgs_coords = [(10, 10), (-10, 10), (10, -10), (-10, -10)]

    lgs_beams = [
        bt.make_converging_beam_from_field_angles(
            np.deg2rad(x / 60.0),
            np.deg2rad(y / 60.0),
            -3.25,
            [0, 0, 0],
            D_BEAM,
            WAVELENGTH,
            "LGS",
            3,
            12
        )
        for x, y in lgs_coords
    ]

    sci_beams = [
        bt.make_converging_beam_from_field_angles(
            np.deg2rad(th / 60.0),
            0.0,
            -3.25,
            [0, 0, 0],
            D_BEAM,
            SCIENCE_WAVELENGTH,
            "Sci",
            3,
            12
        )
        for th in SCI_OFFS
    ]

    telemetry = {
        "times": np.arange(0, exposure_s, dt),
        "bench": bench,
        "frames": [],
        "sci_angles": SCI_OFFS,
        "lgs_beams": lgs_beams,
        "sci_beams": sci_beams,
        "wfs_wavelength": WAVELENGTH,
        "science_wavelength": SCIENCE_WAVELENGTH,
        "npix_pupil": NPIX_PUPIL,
        "pad_size": PAD_SIZE,
        "dm_acts_across": dm_acts_across,
        "ao_start_time": ao_start_time,
        "fa_scale": fa_scale,
    }

    for t in telemetry["times"]:
        frame = {"t": t}

        # LGS pupil samples
        lgs_samps = [
            bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0, 0, 0], t, NPIX_PUPIL)
            for b in lgs_beams
        ]

        lgs_mask = lgs_samps[0]["mask"]

        # Reconstruct in OPD space
        lgs_opd_maps = [
            phase_to_opd(np.nan_to_num(s["phase_map_rad"]), WAVELENGTH)
            for s in lgs_samps
        ]
        avg_lgs_opd = np.nanmean(lgs_opd_maps, axis=0)

        # Open-loop before ao_start_time, closed-loop after
        ao_on = (t >= ao_start_time)
        if ao_on:
            dm_opd = apply_dm_correction_opd(avg_lgs_opd, acts=dm_acts_across, mask=lgs_mask)
        else:
            dm_opd = np.zeros_like(avg_lgs_opd)

        # For visualization, keep the same panels as before, shown in phase at WFS wavelength
        recon_phase_wfs = opd_to_phase(avg_lgs_opd, WAVELENGTH)
        dm_phase_wfs = opd_to_phase(dm_opd, WAVELENGTH)

        frame["ao_on"] = ao_on
        frame["recon"] = recon_phase_wfs
        frame["dm"] = dm_phase_wfs
        frame["recon_opd"] = avg_lgs_opd
        frame["dm_opd"] = dm_opd

        frame["sci"] = [
            simulate_sci_performance(
                b,
                dm_opd,
                bench,
                t,
                PAD_SIZE,
                NPIX_PUPIL,
                SCIENCE_WAVELENGTH
            )
            for b in sci_beams
        ]

        telemetry["frames"].append(frame)

    return telemetry


# ==========================================
# 3. FIXED MOVIE FUNCTION
# ==========================================

def make_movie(tel, base_filename="glao_telemetry"):
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(4, 4, width_ratios=[1.3, 1, 1, 1])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_scr = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    ax_pup = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    ax_psf = [fig.add_subplot(gs[i, 3]) for i in range(3)]

    # Draw 3D bench once
    for b in tel["lgs_beams"]:
        paths, _ = tel["bench"].trace_beam(b, s_end=0.1, t=0.0)
        for p in paths:
            ax3d.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.8, alpha=0.5, color="orange")

    for elem in tel["bench"].elements:
        e1, e2 = elem.plane_basis()
        rad = elem.clear_radius if isinstance(elem, bt.RotatingPhaseScreen3D) else 0.001
        circ = np.linspace(0, 2 * np.pi, 100)
        ring = elem.point[None, :] + rad * (
            np.cos(circ)[:, None] * e1 + np.sin(circ)[:, None] * e2
        )
        ax3d.plot(ring[:, 0], ring[:, 1], ring[:, 2], "k-", lw=1.5)
        ax3d.text(elem.point[0], elem.point[1], elem.point[2], elem.label)

    ax3d.set_box_aspect([1, 1, 1.5])

    def update(idx):
        f = tel["frames"][idx]

        for ax in ax_scr + ax_pup + ax_psf:
            ax.clear()

        # Col 2: rotating screens
        for i, elem in enumerate(tel["bench"].elements):
            img = get_rotating_screen_image(elem, f["t"])
            r_mm = elem.clear_radius * 1000.0

            ax_scr[i].imshow(
                img,
                cmap="RdBu",
                origin="lower",
                extent=[-r_mm, r_mm, -r_mm, r_mm]
            )
            ax_scr[i].set_title(elem.label)

            for b in tel["lgs_beams"]:
                inter = tel["bench"].trace_chief_intersections(b, t=f["t"])
                if elem.label in inter:
                    u, v = elem.local_coordinates(inter[elem.label]["point"])
                    ax_scr[i].add_patch(
                        plt.Circle((u * 1000.0, v * 1000.0), 6.5, color="red", fill=False)
                    )

            for b in tel["sci_beams"]:
                inter = tel["bench"].trace_chief_intersections(b, t=f["t"])
                if elem.label in inter:
                    u, v = elem.local_coordinates(inter[elem.label]["point"])
                    ax_scr[i].scatter(u * 1000.0, v * 1000.0, marker="x", color="white", s=20)

            ax_scr[i].axis("off")

        # Col 3: pupil-plane diagnostics
        mask = np.isfinite(f["recon"]) & np.isfinite(f["dm"])
        if np.sum(mask) == 0:
            mask = np.ones_like(f["recon"], dtype=bool)

        ax_pup[0].imshow(np.where(mask, f["recon"], np.nan), cmap="viridis")
        ax_pup[0].set_title("Recon")

        ax_pup[1].imshow(np.where(mask, f["dm"], np.nan), cmap="viridis")
        ax_pup[1].set_title("DM Shape")

        ax_pup[2].imshow(np.where(mask, f["sci"][2][0], np.nan), cmap="viridis")
        ax_pup[2].set_title("Resid (10')")

        for ax in ax_pup:
            ax.axis("off")

        # Col 4: science PSFs
        lim = int(6 * 8)
        for i, (residual, psf) in enumerate(f["sci"]):
            cy, cx = np.array(psf.shape) // 2
            if np.max(psf) > 0:
                cy, cx = np.unravel_index(np.argmax(psf), psf.shape)

            y0 = max(0, cy - lim)
            y1 = min(psf.shape[0], cy + lim)
            x0 = max(0, cx - lim)
            x1 = min(psf.shape[1], cx + lim)

            crop = psf[y0:y1, x0:x1]

            ax_psf[i].imshow(
                crop, #np.log10(np.maximum(crop, 1e-4)),
                cmap="magma",
                origin="lower",
                vmin=0,
                vmax=np.max( f["sci"][0][1] ),
            )
            ax_psf[i].set_title(f"PSF {tel['sci_angles'][i]}'")
            ax_psf[i].axis("off")

        ao_state = "GLAO ON" if f["ao_on"] else "AO OFF"
        fig.suptitle(
            f"t = {f['t']:.2f} s   |   {ao_state}  " # |   FA scale = {tel['fa_scale']:.2f}"
        )

    ani = FuncAnimation(fig, update, frames=len(tel["frames"]), interval=250)

    try:
        ani.save(f"{base_filename}.mp4", writer="ffmpeg")
        print(f"Saved movie to {base_filename}.mp4")
    except Exception:
        ani.save(f"{base_filename}.gif", writer="pillow")
        print(f"Saved movie to {base_filename}.gif")

    plt.close(fig)


# ==========================================
# 4. RUN
# ==========================================

if __name__ == "__main__":
    tel_data = run_glao_telemetry(
        exposure_s=5.6,     # 2.0 s open-loop + 3.6 s with GLAO
        dt=0.1,
        ao_start_time=2.0,
        fa_scale=0.6,       # reduce free atmosphere for cleaner presentation PSFs
        dm_acts_across=35,
    )
    make_movie(tel_data, base_filename="glao_telemetry_ao_switch")