import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.patches as patches
import beam_trace as bt  # Assumes beam_trace.py is in the same directory

# ==========================================
# 1. CORE PHYSICS & ANALYSIS FUNCTIONS
# ==========================================

def calculate_marechal_strehl(phase_map_rad, mask):
    """Maréchal/Mahajan approximation: S ~ exp(-sigma_phi^2)."""
    if not np.any(mask):
        return 0.0
    phase_var = np.var(phase_map_rad[mask])
    return np.exp(-phase_var)


def phase_to_opd(phase_rad, wavelength_m):
    """Convert phase [rad] to OPD [m]."""
    return phase_rad * wavelength_m / (2.0 * np.pi)


def opd_to_phase(opd_m, wavelength_m):
    """Convert OPD [m] to phase [rad]."""
    return (2.0 * np.pi / wavelength_m) * opd_m


def apply_dm_correction_opd(opd_map, acts, mask):
    """
    Simulates DM spatial filtering on an OPD map [m].

    Returns the low-spatial-frequency content that the DM is assumed
    capable of correcting.
    """
    if acts == 0:
        return np.zeros_like(opd_map)

    avg_opd = np.mean(opd_map[mask])
    work_opd = np.where(mask, opd_map, avg_opd)

    sigma = (opd_map.shape[0] / acts) * 0.5
    low_spatial = gaussian_filter(work_opd, sigma=sigma, mode='reflect')

    return np.where(mask, low_spatial, 0.0)


def pad_and_fft_psf(sample, pad_to=2048):
    """
    Generates high-resolution raw intensity PSF.
    """
    mask = sample["mask"]
    phase = np.nan_to_num(sample["phase_map_rad"])
    amp = sample["amplitude"]

    field = np.where(mask, amp * np.exp(1j * phase), 0.0)

    if pad_to < field.shape[0]:
        raise ValueError(f"pad_to={pad_to} must be >= field size {field.shape[0]}")

    pad_total = pad_to - field.shape[0]
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    padded = np.pad(
        field,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode='constant'
    )

    ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return np.abs(ef) ** 2


def compute_encircled_energy(psf, peak_x, peak_y, angular_pixel_scale,
                             r_max_ld=10.0, n_radii=400):
    """
    Compute encircled energy on a regular radial grid in lambda/D.
    """
    total_flux = np.sum(psf)
    if total_flux <= 0:
        return np.array([]), np.array([]), np.nan

    yy, xx = np.indices(psf.shape)
    r_ld = np.sqrt((xx - peak_x)**2 + (yy - peak_y)**2) * angular_pixel_scale

    ee_radii = np.linspace(0.0, r_max_ld, n_radii)
    ee_curve = np.empty_like(ee_radii)

    for i, r in enumerate(ee_radii):
        ee_curve[i] = np.sum(psf[r_ld <= r]) / total_flux

    if np.max(ee_curve) < 0.80:
        ee80 = np.nan
    else:
        ee80 = np.interp(0.80, ee_curve, ee_radii)

    return ee_radii, ee_curve, ee80


def extract_moffat_parameter(fit_obj, candidate_keys):
    """
    Safely extract a parameter from the returned Moffat fit object.

    Supports dict-like fit results and some common object styles.
    Returns np.nan if not found.
    """
    if fit_obj is None:
        return np.nan

    # Dict-like
    if isinstance(fit_obj, dict):
        for key in candidate_keys:
            if key in fit_obj:
                try:
                    return float(fit_obj[key])
                except Exception:
                    return np.nan

    # Object with attributes
    for key in candidate_keys:
        if hasattr(fit_obj, key):
            try:
                return float(getattr(fit_obj, key))
            except Exception:
                pass

    # lmfit-like params
    if hasattr(fit_obj, "params"):
        params = getattr(fit_obj, "params")
        for key in candidate_keys:
            if key in params:
                try:
                    return float(params[key].value)
                except Exception:
                    pass

    return np.nan


def fit_psf_models(psf_crop, x_ld, y_ld):
    """
    Fit Gaussian and Moffat models to the normalized cropped PSF.
    Returns a dict of fit-derived metrics.
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

    peak = np.max(psf_crop)
    if not np.isfinite(peak) or peak <= 0:
        return result

    psf_crop_norm = psf_crop / peak

    # Gaussian fit
    try:
        gfit = bt.fit_2d_gaussian(x_ld, y_ld, psf_crop_norm)
        gmetrics = bt.gaussian_fwhm_and_ellipticity(gfit)
        result["gaussian_fwhm"] = gmetrics.get("fwhm_major", np.nan)
        result["gaussian_ell"] = gmetrics.get("ellipticity", np.nan)
        result["gaussian_fit"] = gfit
    except Exception:
        pass

    # Moffat fit
    try:
        mfit = bt.fit_2d_moffat(x_ld, y_ld, psf_crop_norm, fit_region="all")
        mmetrics = bt.moffat_fwhm_and_ellipticity(mfit)

        result["moffat_fwhm"] = mmetrics.get("fwhm_major", np.nan)
        result["moffat_ell"] = mmetrics.get("ellipticity", np.nan)
        result["moffat_alpha"] = extract_moffat_parameter(
            mfit, ["alpha", "alpha_x", "scale", "r0"]
        )
        result["moffat_beta"] = extract_moffat_parameter(
            mfit, ["beta", "power"]
        )
        result["moffat_amplitude"] = extract_moffat_parameter(
            mfit, ["amplitude", "amp", "A"]
        )
        result["moffat_bg"] = extract_moffat_parameter(
            mfit, ["background", "bg", "offset", "c0"]
        )
        result["moffat_fit"] = mfit
    except Exception:
        pass

    return result


def analyze_psf(psf, perfect_psf, angular_pixel_scale):
    """Robust PSF analysis with Gaussian + Moffat characterization."""
    total_flux = np.sum(psf)
    if total_flux <= 0:
        return {
            "strehl": np.nan,
            "ee80": np.nan,
            "psf_crop": psf,
            "ee_curve": np.array([]),
            "ee_radii": np.array([]),
            "gaussian_fwhm": np.nan,
            "gaussian_ell": np.nan,
            "moffat_fwhm": np.nan,
            "moffat_ell": np.nan,
            "moffat_alpha": np.nan,
            "moffat_beta": np.nan,
            "moffat_amplitude": np.nan,
            "moffat_bg": np.nan,
        }

    perfect_total_flux = np.sum(perfect_psf)
    strehl_fft = (np.max(psf) / total_flux) / (np.max(perfect_psf) / perfect_total_flux)

    # Robust peak detection
    psf_smooth = gaussian_filter(psf, sigma=2)
    peak_y, peak_x = np.unravel_index(np.argmax(psf_smooth), psf.shape)

    # Dynamic crop +/- 6 lambda/D
    lim_pix = max(2, int(6.0 / angular_pixel_scale))
    y_s, y_e = max(0, peak_y - lim_pix), min(psf.shape[0], peak_y + lim_pix)
    x_s, x_e = max(0, peak_x - lim_pix), min(psf.shape[1], peak_x + lim_pix)
    psf_crop = psf[y_s:y_e, x_s:x_e]

    # Coordinates for model fitting
    y_idx, x_idx = np.indices(psf_crop.shape)
    x_ld = (x_idx - (peak_x - x_s)) * angular_pixel_scale
    y_ld = (y_idx - (peak_y - y_s)) * angular_pixel_scale

    fit_result = fit_psf_models(psf_crop, x_ld, y_ld)

    ee_radii, ee_curve, ee80 = compute_encircled_energy(
        psf, peak_x, peak_y, angular_pixel_scale, r_max_ld=10.0, n_radii=400
    )

    return {
        "strehl": strehl_fft,
        "ee80": ee80,
        "psf_crop": psf_crop,
        "ee_curve": ee_curve,
        "ee_radii": ee_radii,
        "gaussian_fwhm": fit_result["gaussian_fwhm"],
        "gaussian_ell": fit_result["gaussian_ell"],
        "moffat_fwhm": fit_result["moffat_fwhm"],
        "moffat_ell": fit_result["moffat_ell"],
        "moffat_alpha": fit_result["moffat_alpha"],
        "moffat_beta": fit_result["moffat_beta"],
        "moffat_amplitude": fit_result["moffat_amplitude"],
        "moffat_bg": fit_result["moffat_bg"],
    }


# new 
def plot_beam_footprints_on_screens(bench, lgs_beams, sci_beams, t=0.0):
    """
    Plots the full unrotated phase screen maps and overlays the cross-section
    contours of the LGS and Science beams at a specific time t.
    """
    # Extract only the phase screen elements from the bench
    screens = [e for e in bench.elements if isinstance(e, bt.RotatingPhaseScreen3D)]
    n_screens = len(screens)
    
    fig, axes = plt.subplots(1, n_screens, figsize=(5 * n_screens, 5))
    if n_screens == 1:
        axes = [axes]
        
    for ax, screen in zip(axes, screens):
        extent = screen.map_extent_m / 2.0
        
        # 1. Plot the background OPD map (converted to nanometers for visibility)
        im = ax.imshow(
            screen.opd_map * 1e9, 
            origin="lower", 
            extent=[-extent, extent, -extent, extent],
            cmap="viridis"
        )
        plt.colorbar(im, ax=ax, label="OPD [nm]", fraction=0.046, pad=0.04)
        
        # Draw the clear aperture of the phase screen
        clear_ap = patches.Circle((0, 0), screen.clear_radius, color='white', fill=False, ls=':', lw=1, label="Clear Aperture")
        ax.add_patch(clear_ap)

        # Because screens rotate in the simulation, map the fixed 3D spatial intersections 
        # backwards onto the raw (unrotated) phase map array.
        angle = screen.current_rotation_angle(t)
        rot_mat = bt.rotation_matrix_2d(-angle)

        # 2. Overlay LGS beams (Red Dashed)
        for i, beam in enumerate(lgs_beams):
            intersections = bench.trace_chief_intersections(beam, t=t)
            if screen.label in intersections:
                p = intersections[screen.label]["point"]
                u, v = screen.local_coordinates(p)
                uv_rot = rot_mat @ np.array([u, v])
                
                label = "LGS Beams" if i == 0 else None
                circle = patches.Circle((uv_rot[0], uv_rot[1]), beam.radius, 
                                        edgecolor='red', facecolor='none', lw=1.5, ls='--', label=label)
                ax.add_patch(circle)
        
        # 3. Overlay Science beams (Cyan Solid)
        for i, beam in enumerate(sci_beams):
            intersections = bench.trace_chief_intersections(beam, t=t)
            if screen.label in intersections:
                p = intersections[screen.label]["point"]
                u, v = screen.local_coordinates(p)
                uv_rot = rot_mat @ np.array([u, v])
                
                label = "Sci Beams" if i == 0 else None
                circle = patches.Circle((uv_rot[0], uv_rot[1]), beam.radius, 
                                        edgecolor='cyan', facecolor='none', lw=1.5, label=label)
                ax.add_patch(circle)

        ax.set_title(f"{screen.label} (z={screen.point[2]:.3f} m)\nt = {t}s")
        ax.set_xlabel("u [m]")
        ax.set_ylabel("v [m]")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

# ==========================================
# 2. SETUP & DATA LOADING
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
FITS_PATH = os.path.join(
    #script_dir, "../phasescreens/batch1_test/", "phasescreens_median_dmScaled-1_radialScaled-0.fits"
    #script_dir, "../phasescreens/scrns_2_order_v1/testbench_phasescreens_20260429_strong.fits"
    script_dir, "../phasescreens/scrns_2_order_v2/Testbench_phasescreens_20260506.fits"
)

with fits.open(FITS_PATH) as hdul:
    layer_configs = [
        {"label": "FA",  "z": -2.50,  "hz": 0.2},
        {"label": "GL3", "z": -0.060, "hz": 1.4},
        {"label": "GL2", "z": -0.030, "hz": 1.0},
        {"label": "GL1", "z": -0.001, "hz": 1.7},
    ]

    bench = bt.OpticalBench3D()
    pix_scale = hdul[0].header["PIXSCALE"]

    for cfg in layer_configs:
        opd = (hdul[cfg["label"]].data * 500e-9) / (2.0 * np.pi)

        if cfg["label"] == "FA":
            bench.add(
                bt.RotatingPhaseScreen3D(
                    point=[26e-3, 0, cfg["z"]],
                    normal=[0, 0, 1],
                    opd_map=opd,
                    map_extent_m=opd.shape[0] * pix_scale,
                    angular_velocity=2.0 * np.pi * cfg["hz"],
                    label=cfg["label"],
                )
            )
        else:
            bench.add(
                bt.RotatingPhaseScreen3D(
                    point=[24e-3, 0, cfg["z"]],#[34e-3, 0, cfg["z"]],
                    normal=[0, 0, 1],
                    opd_map=opd,
                    map_extent_m=opd.shape[0] * pix_scale,
                    angular_velocity=2.0 * np.pi * cfg["hz"],
                    label=cfg["label"],
                )
            )

# Constants
WAVELENGTH = 633e-9
SCIENCE_WAVELENGTH = 1.2e-6 #2.2e-6
D_BEAM = 0.013
NPIX_PUPIL = 256
PAD_SIZE = 2048

# Existing lambda/D pixel scaling convention
ANGULAR_SCALE = 1.0 / (PAD_SIZE / (NPIX_PUPIL / 2.0))

EXPOSURE_TIME = 5.0
DT = 0.1
times = np.arange(0, EXPOSURE_TIME, DT)

science_angles = np.linspace(0, 10, 5)
lgs_coords = [(10, 10), (-10, 10), (10, -10), (-10, -10)]

lgs_beams = [
    bt.make_converging_beam_from_field_angles(
        np.deg2rad(x / 60.0),
        np.deg2rad(y / 60.0),
        -3.25,
        [0, 0, 0],
        D_BEAM,
        WAVELENGTH,
        f"L{i}",
        3,
        12,
    )
    for i, (x, y) in enumerate(lgs_coords)
]

sci_beams = [
    bt.make_converging_beam_from_field_angles(
        np.deg2rad(th / 60.0),
        0.0,
        -3.25,
        [0, 0, 0],
        D_BEAM,
        SCIENCE_WAVELENGTH,
        f"S{i}",
        3,
        12,
    )
    for i, th in enumerate(science_angles)
]

# Perfect baseline at science wavelength
ref_beam = bt.make_converging_beam_from_field_angles(
    0.0, 0.0, -3.25, [0, 0, 0], D_BEAM, SCIENCE_WAVELENGTH, "ref", 3, 12
)
perf_sample = bt.sample_beam_phase_amplitude_on_pupil_plane(
    ref_beam, bench, [0, 0, 0], 0.0, NPIX_PUPIL
)
perf_sample = copy.deepcopy(perf_sample)
perf_sample["phase_map_rad"] = np.zeros_like(perf_sample["phase_map_rad"])
perfect_psf = pad_and_fft_psf(perf_sample, PAD_SIZE)

# ==========================================
# 3. GLAO SIMULATION LOOP
# ==========================================

accum_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE), dtype=float) for i in range(len(sci_beams))}
accum_no_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE), dtype=float) for i in range(len(sci_beams))}

DM_ACTS_ACROSS = 11

print(f"Exposure: {EXPOSURE_TIME}s | dt: {DT}s")
print(f"WFS wavelength = {WAVELENGTH * 1e9:.1f} nm")
print(f"Science wavelength = {SCIENCE_WAVELENGTH * 1e9:.1f} nm")
print(f"DM correction reconstructed in OPD space with {DM_ACTS_ACROSS} acts across pupil")

for t in times:
    lgs_samples = [
        bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0, 0, 0], t, NPIX_PUPIL)
        for b in lgs_beams
    ]

    lgs_mask = lgs_samples[0]["mask"]

    lgs_opd_maps = [
        phase_to_opd(np.nan_to_num(s["phase_map_rad"]), WAVELENGTH)
        for s in lgs_samples
    ]

    gl_opd = np.mean(lgs_opd_maps, axis=0)
    dm_opd_corr = apply_dm_correction_opd(gl_opd, acts=DM_ACTS_ACROSS, mask=lgs_mask)

    for i, beam in enumerate(sci_beams):
        s_samp = bt.sample_beam_phase_amplitude_on_pupil_plane(
            beam, bench, [0, 0, 0], t, NPIX_PUPIL
        )

        accum_no_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)

        dm_phase_corr_sci = opd_to_phase(dm_opd_corr, SCIENCE_WAVELENGTH)

        s_samp_corr = copy.deepcopy(s_samp)
        s_samp_corr["phase_map_rad"] = np.nan_to_num(s_samp_corr["phase_map_rad"]) - dm_phase_corr_sci

        accum_ao[i] += pad_and_fft_psf(s_samp_corr, PAD_SIZE)

# ==========================================
# 4. OUTPUTS & PLOTTING
# ==========================================


# --- Display beam footprints across all screens ---
print("Plotting beam footprints on phase screens...")
plot_beam_footprints_on_screens(bench, lgs_beams, sci_beams, t=0.0)


res_ao = [analyze_psf(accum_ao[i] / len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]
res_no = [analyze_psf(accum_no_ao[i] / len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]

USE_LOG = False

# PSF image grid
fig_grid, axes_grid = plt.subplots(2, 5, figsize=(18, 8))

for i in range(5):
    p_no = res_no[i]["psf_crop"]
    p_ao = res_ao[i]["psf_crop"]

    if USE_LOG:
        img_no = np.log10(np.maximum(p_no / np.max(p_no), 1e-5))
        img_ao = np.log10(np.maximum(p_ao / np.max(p_ao), 1e-5))
    else:
        img_no = p_no
        img_ao = p_ao

    extent = [-6, 6, -6, 6]
    axes_grid[0, i].imshow(img_no, origin="lower", extent=extent, cmap="magma")
    axes_grid[1, i].imshow(img_ao, origin="lower", extent=extent, cmap="magma")

    axes_grid[0, i].set_title(f"Uncorr {science_angles[i]}'")
    axes_grid[1, i].set_title(f"GLAO {science_angles[i]}'")

for ax in axes_grid.flatten():
    ax.set_xlabel(r"[$\lambda/D$]")
    ax.set_ylabel(r"[$\lambda/D$]")

plt.tight_layout()
plt.show()

# Main diagnostics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(science_angles, [r["strehl"] for r in res_no], "ro--", label="No AO")
axes[0, 0].plot(science_angles, [r["strehl"] for r in res_ao], "bo-", label="GLAO")

axes[0, 1].plot(science_angles, [r["gaussian_fwhm"] for r in res_no], "ro--", label="No AO")
axes[0, 1].plot(science_angles, [r["gaussian_fwhm"] for r in res_ao], "bo-", label="GLAO")

axes[1, 0].plot(science_angles, [r["gaussian_ell"] for r in res_no], "ro--", label="No AO")
axes[1, 0].plot(science_angles, [r["gaussian_ell"] for r in res_ao], "bo-", label="GLAO")

axes[1, 1].plot(res_no[0]["ee_radii"], res_no[0]["ee_curve"], "r--", label="No AO Center")
axes[1, 1].plot(res_ao[0]["ee_radii"], res_ao[0]["ee_curve"], "b-", label="GLAO Center")
axes[1, 1].set_xlim(0, 10)

axes[0, 0].set_ylabel("Strehl")
axes[0, 1].set_ylabel("Gaussian FWHM [λ/D]")
axes[1, 0].set_ylabel("Gaussian Ellipticity")
axes[1, 1].set_ylabel("Enc. Energy")

axes[0, 0].set_xlabel("Field Angle [arcmin]")
axes[0, 1].set_xlabel("Field Angle [arcmin]")
axes[1, 0].set_xlabel("Field Angle [arcmin]")
axes[1, 1].set_xlabel("Radius [λ/D]")

for ax in axes.flatten():
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# New Moffat characterization figure
fig_moffat, axes_m = plt.subplots(2, 2, figsize=(14, 10))

axes_m[0, 0].plot(science_angles, [r["moffat_fwhm"] for r in res_no], "ro--", label="No AO")
axes_m[0, 0].plot(science_angles, [r["moffat_fwhm"] for r in res_ao], "bo-", label="GLAO")
axes_m[0, 0].set_ylabel("Moffat FWHM [λ/D]")
axes_m[0, 0].set_xlabel("Field Angle [arcmin]")

axes_m[0, 1].plot(science_angles, [r["moffat_ell"] for r in res_no], "ro--", label="No AO")
axes_m[0, 1].plot(science_angles, [r["moffat_ell"] for r in res_ao], "bo-", label="GLAO")
axes_m[0, 1].set_ylabel("Moffat Ellipticity")
axes_m[0, 1].set_xlabel("Field Angle [arcmin]")

axes_m[1, 0].plot(science_angles, [r["moffat_alpha"] for r in res_no], "ro--", label="No AO")
axes_m[1, 0].plot(science_angles, [r["moffat_alpha"] for r in res_ao], "bo-", label="GLAO")
axes_m[1, 0].set_ylabel("Moffat Alpha [λ/D]")
axes_m[1, 0].set_xlabel("Field Angle [arcmin]")

axes_m[1, 1].plot(science_angles, [r["moffat_beta"] for r in res_no], "ro--", label="No AO")
axes_m[1, 1].plot(science_angles, [r["moffat_beta"] for r in res_ao], "bo-", label="GLAO")
axes_m[1, 1].set_ylabel("Moffat Beta")
axes_m[1, 1].set_xlabel("Field Angle [arcmin]")

for ax in axes_m.flatten():
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 5. RESULTS TABLE
# ==========================================

results_data = []

for i, angle in enumerate(science_angles):
    results_data.append({
        "Field Angle [arcmin]": angle,
        "AO Mode": "No AO",
        "Strehl": f"{res_no[i]['strehl']:.4f}",
        "Gaussian FWHM [L/D]": f"{res_no[i]['gaussian_fwhm']:.4f}",
        "Gaussian Ellipticity": f"{res_no[i]['gaussian_ell']:.4f}",
        "Moffat FWHM [L/D]": f"{res_no[i]['moffat_fwhm']:.4f}",
        "Moffat Ellipticity": f"{res_no[i]['moffat_ell']:.4f}",
        "Moffat Alpha [L/D]": f"{res_no[i]['moffat_alpha']:.4f}",
        "Moffat Beta": f"{res_no[i]['moffat_beta']:.4f}",
        "EE80 [L/D]": f"{res_no[i]['ee80']:.4f}",
    })
    results_data.append({
        "Field Angle [arcmin]": angle,
        "AO Mode": "GLAO",
        "Strehl": f"{res_ao[i]['strehl']:.4f}",
        "Gaussian FWHM [L/D]": f"{res_ao[i]['gaussian_fwhm']:.4f}",
        "Gaussian Ellipticity": f"{res_ao[i]['gaussian_ell']:.4f}",
        "Moffat FWHM [L/D]": f"{res_ao[i]['moffat_fwhm']:.4f}",
        "Moffat Ellipticity": f"{res_ao[i]['moffat_ell']:.4f}",
        "Moffat Alpha [L/D]": f"{res_ao[i]['moffat_alpha']:.4f}",
        "Moffat Beta": f"{res_ao[i]['moffat_beta']:.4f}",
        "EE80 [L/D]": f"{res_ao[i]['ee80']:.4f}",
    })

df_results = pd.DataFrame(results_data)

df_no_ao = df_results[df_results["AO Mode"] == "No AO"].reset_index(drop=True)
df_glao = df_results[df_results["AO Mode"] == "GLAO"].reset_index(drop=True)

print("\n" + "=" * 100)
print("DIAGNOSTIC RESULTS TABLE - NO AO")
print("=" * 100)
print(df_no_ao.to_string(index=False))

print("\n" + "=" * 100)
print("DIAGNOSTIC RESULTS TABLE - GLAO")
print("=" * 100)
print(df_glao.to_string(index=False))
print("=" * 100)

results_filename = "psf_analysis_results.csv"
results_no_ao_filename = "psf_analysis_results_no_ao.csv"
results_glao_filename = "psf_analysis_results_glao.csv"

df_results.to_csv(results_filename, index=False)
df_no_ao.to_csv(results_no_ao_filename, index=False)
df_glao.to_csv(results_glao_filename, index=False)

print(f"\nResults saved to: {results_filename}")
print(f"Results also saved to: {results_no_ao_filename} and {results_glao_filename}")

# import os
# import copy
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from scipy.ndimage import gaussian_filter
# import pandas as pd
# import beam_trace as bt  # Assumes beam_trace.py is in the same directory

# # ==========================================
# # 1. CORE PHYSICS & ANALYSIS FUNCTIONS
# # ==========================================

# def calculate_marechal_strehl(phase_map_rad, mask):
#     """Maréchal/Mahajan approximation: S ~ exp(-sigma_phi^2)."""
#     if not np.any(mask):
#         return 0.0
#     phase_var = np.var(phase_map_rad[mask])
#     return np.exp(-phase_var)


# def phase_to_opd(phase_rad, wavelength_m):
#     """Convert phase [rad] to OPD [m]."""
#     return phase_rad * wavelength_m / (2.0 * np.pi)


# def opd_to_phase(opd_m, wavelength_m):
#     """Convert OPD [m] to phase [rad]."""
#     return (2.0 * np.pi / wavelength_m) * opd_m


# def apply_dm_correction_opd(opd_map, acts, mask):
#     """
#     Simulates DM spatial filtering on an OPD map [m].

#     Returns the low-spatial-frequency content that the DM is assumed
#     capable of correcting.
#     """
#     if acts == 0:
#         return np.zeros_like(opd_map)

#     avg_opd = np.mean(opd_map[mask])
#     work_opd = np.where(mask, opd_map, avg_opd)

#     sigma = (opd_map.shape[0] / acts) * 0.5
#     low_spatial = gaussian_filter(work_opd, sigma=sigma, mode='reflect')

#     return np.where(mask, low_spatial, 0.0)


# def pad_and_fft_psf(sample, pad_to=2048):
#     """
#     Generates high-resolution raw intensity PSF.
#     """
#     mask = sample["mask"]
#     phase = np.nan_to_num(sample["phase_map_rad"])
#     amp = sample["amplitude"]

#     field = np.where(mask, amp * np.exp(1j * phase), 0.0)

#     if pad_to < field.shape[0]:
#         raise ValueError(f"pad_to={pad_to} must be >= field size {field.shape[0]}")

#     pad_total = pad_to - field.shape[0]
#     pad_before = pad_total // 2
#     pad_after = pad_total - pad_before

#     padded = np.pad(
#         field,
#         ((pad_before, pad_after), (pad_before, pad_after)),
#         mode='constant'
#     )

#     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
#     return np.abs(ef) ** 2


# def compute_encircled_energy(psf, peak_x, peak_y, angular_pixel_scale,
#                              r_max_ld=10.0, n_radii=400):
#     """
#     Compute encircled energy on a regular radial grid in lambda/D.

#     Returns
#     -------
#     ee_radii : ndarray
#         Radius values [lambda/D].
#     ee_curve : ndarray
#         Encircled energy fraction at each radius.
#     ee80 : float
#         Radius containing 80% of the flux [lambda/D].
#     """
#     total_flux = np.sum(psf)
#     if total_flux <= 0:
#         return np.array([]), np.array([]), np.nan

#     yy, xx = np.indices(psf.shape)
#     r_ld = np.sqrt((xx - peak_x)**2 + (yy - peak_y)**2) * angular_pixel_scale

#     ee_radii = np.linspace(0.0, r_max_ld, n_radii)
#     ee_curve = np.empty_like(ee_radii)

#     for i, r in enumerate(ee_radii):
#         ee_curve[i] = np.sum(psf[r_ld <= r]) / total_flux

#     if np.all(np.isnan(ee_curve)) or np.max(ee_curve) < 0.80:
#         ee80 = np.nan
#     else:
#         ee80 = np.interp(0.80, ee_curve, ee_radii)

#     return ee_radii, ee_curve, ee80


# def analyze_psf(psf, perfect_psf, angular_pixel_scale, ao_label="N/A", log_plot=False):
#     """Robust PSF analysis."""
#     total_flux = np.sum(psf)
#     if total_flux <= 0:
#         return {
#             "strehl": np.nan,
#             "ee80": np.nan,
#             "ell": np.nan,
#             "fwhm": np.nan,
#             "psf_crop": psf,
#             "ee_curve": np.array([]),
#             "ee_radii": np.array([]),
#         }

#     perfect_total_flux = np.sum(perfect_psf)
#     strehl_fft = (np.max(psf) / total_flux) / (np.max(perfect_psf) / perfect_total_flux)

#     # Robust peak detection
#     psf_smooth = gaussian_filter(psf, sigma=2)
#     peak_y, peak_x = np.unravel_index(np.argmax(psf_smooth), psf.shape)

#     # Dynamic crop +/- 6 lambda/D
#     lim_pix = max(2, int(6.0 / angular_pixel_scale))
#     y_s, y_e = max(0, peak_y - lim_pix), min(psf.shape[0], peak_y + lim_pix)
#     x_s, x_e = max(0, peak_x - lim_pix), min(psf.shape[1], peak_x + lim_pix)
#     psf_crop = psf[y_s:y_e, x_s:x_e]

#     # Gaussian fit for core shape metrics
#     if np.max(psf_crop) > 0:
#         y_idx, x_idx = np.indices(psf_crop.shape)
#         x_ld = (x_idx - (peak_x - x_s)) * angular_pixel_scale
#         y_ld = (y_idx - (peak_y - y_s)) * angular_pixel_scale
#         fit = bt.fit_2d_gaussian(x_ld, y_ld, psf_crop / np.max(psf_crop))
#         metrics = bt.gaussian_fwhm_and_ellipticity(fit)
#         fwhm = metrics["fwhm_major"]
#         ell = metrics["ellipticity"]
#     else:
#         fwhm = np.nan
#         ell = np.nan

#     ee_radii, ee_curve, ee80 = compute_encircled_energy(
#         psf, peak_x, peak_y, angular_pixel_scale, r_max_ld=10.0, n_radii=400
#     )

#     return {
#         "strehl": strehl_fft,
#         "ee80": ee80,
#         "ell": ell,
#         "fwhm": fwhm,
#         "psf_crop": psf_crop,
#         "ee_curve": ee_curve,
#         "ee_radii": ee_radii,
#     }


# # ==========================================
# # 2. SETUP & DATA LOADING
# # ==========================================

# script_dir = os.path.dirname(os.path.abspath(__file__))
# FITS_PATH = os.path.join(
#     script_dir, "../phasescreens/batch1_test/", "phasescreens_median_dmScaled-1_radialScaled-0.fits"
# )

# with fits.open(FITS_PATH) as hdul:
#     layer_configs = [
#         {"label": "FA",  "z": -2.50,  "hz": 0.2},
#         {"label": "GL3", "z": -0.060, "hz": 1.4},
#         {"label": "GL2", "z": -0.030, "hz": 1.0},
#         {"label": "GL1", "z": -0.001, "hz": 0.7},
#     ]

#     bench = bt.OpticalBench3D()
#     pix_scale = hdul[0].header["PIXSCALE"]

#     for cfg in layer_configs:
#         opd = (hdul[cfg["label"]].data * 500e-9) / (2.0 * np.pi)

#         if cfg["label"] == "FA":
#             bench.add(
#                 bt.RotatingPhaseScreen3D(
#                     point=[26e-3, 0, cfg["z"]],
#                     normal=[0, 0, 1],
#                     opd_map=opd,
#                     map_extent_m=opd.shape[0] * pix_scale,
#                     angular_velocity=2.0 * np.pi * cfg["hz"],
#                     label=cfg["label"],
#                 )
#             )
#         else:
#             bench.add(
#                 bt.RotatingPhaseScreen3D(
#                     point=[34e-3, 0, cfg["z"]],
#                     normal=[0, 0, 1],
#                     opd_map=opd,
#                     map_extent_m=opd.shape[0] * pix_scale,
#                     angular_velocity=2.0 * np.pi * cfg["hz"],
#                     label=cfg["label"],
#                 )
#             )

# # Constants
# WAVELENGTH = 633e-9
# SCIENCE_WAVELENGTH = 2.2e-6
# D_BEAM = 0.013
# NPIX_PUPIL = 256
# PAD_SIZE = 2048

# # Existing lambda/D pixel scaling convention
# ANGULAR_SCALE = 1.0 / (PAD_SIZE / (NPIX_PUPIL / 2.0))

# EXPOSURE_TIME = 2.0
# DT = 0.2
# times = np.arange(0, EXPOSURE_TIME, DT)

# science_angles = np.linspace(0, 10, 5)
# lgs_coords = [(10, 10), (-10, 10), (10, -10), (-10, -10)]

# lgs_beams = [
#     bt.make_converging_beam_from_field_angles(
#         np.deg2rad(x / 60.0),
#         np.deg2rad(y / 60.0),
#         -3.25,
#         [0, 0, 0],
#         D_BEAM,
#         WAVELENGTH,
#         f"L{i}",
#         3,
#         12,
#     )
#     for i, (x, y) in enumerate(lgs_coords)
# ]

# sci_beams = [
#     bt.make_converging_beam_from_field_angles(
#         np.deg2rad(th / 60.0),
#         0.0,
#         -3.25,
#         [0, 0, 0],
#         D_BEAM,
#         SCIENCE_WAVELENGTH,
#         f"S{i}",
#         3,
#         12,
#     )
#     for i, th in enumerate(science_angles)
# ]

# # Perfect baseline at science wavelength
# ref_beam = bt.make_converging_beam_from_field_angles(
#     0.0, 0.0, -3.25, [0, 0, 0], D_BEAM, SCIENCE_WAVELENGTH, "ref", 3, 12
# )
# perf_sample = bt.sample_beam_phase_amplitude_on_pupil_plane(
#     ref_beam, bench, [0, 0, 0], 0.0, NPIX_PUPIL
# )
# perf_sample = copy.deepcopy(perf_sample)
# perf_sample["phase_map_rad"] = np.zeros_like(perf_sample["phase_map_rad"])
# perfect_psf = pad_and_fft_psf(perf_sample, PAD_SIZE)

# # ==========================================
# # 3. GLAO SIMULATION LOOP
# # ==========================================

# accum_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE), dtype=float) for i in range(len(sci_beams))}
# accum_no_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE), dtype=float) for i in range(len(sci_beams))}

# DM_ACTS_ACROSS = 11

# print(f"Exposure: {EXPOSURE_TIME}s | dt: {DT}s")
# print(f"WFS wavelength = {WAVELENGTH * 1e9:.1f} nm")
# print(f"Science wavelength = {SCIENCE_WAVELENGTH * 1e9:.1f} nm")
# print(f"DM correction reconstructed in OPD space with {DM_ACTS_ACROSS} acts across pupil")

# for t in times:
#     lgs_samples = [
#         bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0, 0, 0], t, NPIX_PUPIL)
#         for b in lgs_beams
#     ]

#     lgs_mask = lgs_samples[0]["mask"]

#     lgs_opd_maps = [
#         phase_to_opd(np.nan_to_num(s["phase_map_rad"]), WAVELENGTH)
#         for s in lgs_samples
#     ]

#     gl_opd = np.mean(lgs_opd_maps, axis=0)
#     dm_opd_corr = apply_dm_correction_opd(gl_opd, acts=DM_ACTS_ACROSS, mask=lgs_mask)

#     for i, beam in enumerate(sci_beams):
#         s_samp = bt.sample_beam_phase_amplitude_on_pupil_plane(
#             beam, bench, [0, 0, 0], t, NPIX_PUPIL
#         )

#         accum_no_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)

#         dm_phase_corr_sci = opd_to_phase(dm_opd_corr, SCIENCE_WAVELENGTH)

#         s_samp_corr = copy.deepcopy(s_samp)
#         s_samp_corr["phase_map_rad"] = np.nan_to_num(s_samp_corr["phase_map_rad"]) - dm_phase_corr_sci

#         accum_ao[i] += pad_and_fft_psf(s_samp_corr, PAD_SIZE)

# # ==========================================
# # 4. OUTPUTS & PLOTTING
# # ==========================================

# res_ao = [analyze_psf(accum_ao[i] / len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]
# res_no = [analyze_psf(accum_no_ao[i] / len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]

# USE_LOG = False

# fig_grid, axes_grid = plt.subplots(2, 5, figsize=(18, 8))

# for i in range(5):
#     p_no = res_no[i]["psf_crop"]
#     p_ao = res_ao[i]["psf_crop"]

#     if USE_LOG:
#         img_no = np.log10(np.maximum(p_no / np.max(p_no), 1e-5))
#         img_ao = np.log10(np.maximum(p_ao / np.max(p_ao), 1e-5))
#     else:
#         img_no = p_no
#         img_ao = p_ao

#     extent = [-6, 6, -6, 6]
#     axes_grid[0, i].imshow(img_no, origin="lower", extent=extent, cmap="magma")
#     axes_grid[1, i].imshow(img_ao, origin="lower", extent=extent, cmap="magma")

#     axes_grid[0, i].set_title(f"Uncorr {science_angles[i]}'")
#     axes_grid[1, i].set_title(f"GLAO {science_angles[i]}'")

# for ax in axes_grid.flatten():
#     ax.set_xlabel(r"[$\lambda/D$]")
#     ax.set_ylabel(r"[$\lambda/D$]")

# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# axes[0, 0].plot(science_angles, [r["strehl"] for r in res_no], "ro--", label="No AO")
# axes[0, 0].plot(science_angles, [r["strehl"] for r in res_ao], "bo-", label="GLAO")

# axes[0, 1].plot(science_angles, [r["fwhm"] for r in res_no], "ro--", label="No AO")
# axes[0, 1].plot(science_angles, [r["fwhm"] for r in res_ao], "bo-", label="GLAO")

# axes[1, 0].plot(science_angles, [r["ell"] for r in res_no], "ro--", label="No AO")
# axes[1, 0].plot(science_angles, [r["ell"] for r in res_ao], "bo-", label="GLAO")

# axes[1, 1].plot(res_no[0]["ee_radii"], res_no[0]["ee_curve"], "r--", label="No AO Center")
# axes[1, 1].plot(res_ao[0]["ee_radii"], res_ao[0]["ee_curve"], "b-", label="GLAO Center")
# axes[1, 1].set_xlim(0, 10)

# axes[0, 0].set_ylabel("Strehl")
# axes[0, 1].set_ylabel("FWHM [λ/D]")
# axes[1, 0].set_ylabel("Ellipticity")
# axes[1, 1].set_ylabel("Enc. Energy")

# axes[0, 0].set_xlabel("Field Angle [arcmin]")
# axes[0, 1].set_xlabel("Field Angle [arcmin]")
# axes[1, 0].set_xlabel("Field Angle [arcmin]")
# axes[1, 1].set_xlabel("Radius [λ/D]")

# for ax in axes.flatten():
#     ax.legend()
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # ==========================================
# # 5. RESULTS TABLE
# # ==========================================

# results_data = []

# for i, angle in enumerate(science_angles):
#     results_data.append({
#         "Field Angle [arcmin]": angle,
#         "AO Mode": "No AO",
#         "Strehl": f"{res_no[i]['strehl']:.4f}",
#         "FWHM [L/D]": f"{res_no[i]['fwhm']:.4f}",
#         "Ellipticity": f"{res_no[i]['ell']:.4f}",
#         "EE80 [L/D]": f"{res_no[i]['ee80']:.4f}",
#     })
#     results_data.append({
#         "Field Angle [arcmin]": angle,
#         "AO Mode": "GLAO",
#         "Strehl": f"{res_ao[i]['strehl']:.4f}",
#         "FWHM [L/D]": f"{res_ao[i]['fwhm']:.4f}",
#         "Ellipticity": f"{res_ao[i]['ell']:.4f}",
#         "EE80 [L/D]": f"{res_ao[i]['ee80']:.4f}",
#     })

# df_results = pd.DataFrame(results_data)

# df_no_ao = df_results[df_results["AO Mode"] == "No AO"].reset_index(drop=True)
# df_glao = df_results[df_results["AO Mode"] == "GLAO"].reset_index(drop=True)

# print("\n" + "=" * 80)
# print("DIAGNOSTIC RESULTS TABLE - NO AO")
# print("=" * 80)
# print(df_no_ao.to_string(index=False))

# print("\n" + "=" * 80)
# print("DIAGNOSTIC RESULTS TABLE - GLAO")
# print("=" * 80)
# print(df_glao.to_string(index=False))
# print("=" * 80)

# results_filename = "psf_analysis_results.csv"
# results_no_ao_filename = "psf_analysis_results_no_ao.csv"
# results_glao_filename = "psf_analysis_results_glao.csv"

# df_results.to_csv(results_filename, index=False)
# df_no_ao.to_csv(results_no_ao_filename, index=False)
# df_glao.to_csv(results_glao_filename, index=False)

# print(f"\nResults saved to: {results_filename}")
# print(f"Results also saved to: {results_no_ao_filename} and {results_glao_filename}")

# # import os
# # import copy
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from astropy.io import fits
# # from scipy.ndimage import gaussian_filter
# # import pandas as pd
# # import beam_trace as bt  # Assumes beam_trace.py is in the same directory

# # # ==========================================
# # # 1. CORE PHYSICS & ANALYSIS FUNCTIONS
# # # ==========================================

# # def calculate_marechal_strehl(phase_map_rad, mask):
# #     """Maréchal/Mahajan approximation: S ~ exp(-sigma_phi^2)."""
# #     if not np.any(mask):
# #         return 0.0
# #     phase_var = np.var(phase_map_rad[mask])
# #     return np.exp(-phase_var)


# # def phase_to_opd(phase_rad, wavelength_m):
# #     """
# #     Convert phase [rad] to OPD [m].

# #     phi = 2*pi*OPD/lambda  ->  OPD = phi*lambda/(2*pi)
# #     """
# #     return phase_rad * wavelength_m / (2.0 * np.pi)


# # def opd_to_phase(opd_m, wavelength_m):
# #     """
# #     Convert OPD [m] to phase [rad].

# #     phi = 2*pi*OPD/lambda
# #     """
# #     return (2.0 * np.pi / wavelength_m) * opd_m


# # def apply_dm_correction_opd(opd_map, acts, mask):
# #     """
# #     Simulates DM spatial filtering on an OPD map [m].

# #     This function returns the low-spatial-frequency content that the DM
# #     is assumed capable of correcting. The result is in OPD units [m],
# #     not radians.

# #     Parameters
# #     ----------
# #     opd_map : 2D ndarray
# #         Input OPD map [m].
# #     acts : int
# #         Effective number of actuators across the beam/pupil.
# #     mask : 2D bool ndarray
# #         Pupil mask.

# #     Returns
# #     -------
# #     dm_opd_corr : 2D ndarray
# #         DM correction map in OPD [m].
# #     """
# #     if acts == 0:
# #         return np.zeros_like(opd_map)

# #     avg_opd = np.mean(opd_map[mask])
# #     work_opd = np.where(mask, opd_map, avg_opd)

# #     # Same smoothing prescription as your original script
# #     sigma = (opd_map.shape[0] / acts) * 0.5
# #     low_spatial = gaussian_filter(work_opd, sigma=sigma, mode='reflect')

# #     return np.where(mask, low_spatial, 0.0)


# # def pad_and_fft_psf(sample, pad_to=2048):
# #     """
# #     Generates high-resolution raw intensity PSF.

# #     Expects sample["phase_map_rad"] to already be expressed at the
# #     wavelength of that beam.
# #     """
# #     mask = sample["mask"]
# #     phase = np.nan_to_num(sample["phase_map_rad"])
# #     amp = sample["amplitude"]

# #     field = np.where(mask, amp * np.exp(1j * phase), 0.0)

# #     if pad_to < field.shape[0]:
# #         raise ValueError(f"pad_to={pad_to} must be >= field size {field.shape[0]}")

# #     pad_total = pad_to - field.shape[0]
# #     pad_before = pad_total // 2
# #     pad_after = pad_total - pad_before

# #     padded = np.pad(
# #         field,
# #         ((pad_before, pad_after), (pad_before, pad_after)),
# #         mode='constant'
# #     )

# #     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
# #     return np.abs(ef) ** 2


# # def analyze_psf(psf, perfect_psf, angular_pixel_scale, ao_label="N/A", log_plot=False):
# #     """Robust analysis with optional log-scale visualization."""
# #     total_flux = np.sum(psf)
# #     if total_flux <= 0:
# #         return {
# #             "strehl": np.nan,
# #             "ee80": np.nan,
# #             "ell": np.nan,
# #             "fwhm": np.nan,
# #             "psf_crop": psf,
# #             "ee_curve": np.array([])
# #         }

# #     perfect_total_flux = np.sum(perfect_psf)
# #     strehl_fft = (np.max(psf) / total_flux) / (np.max(perfect_psf) / perfect_total_flux)

# #     # Smoothing for robust sub-pixel peak detection
# #     psf_smooth = gaussian_filter(psf, sigma=2)
# #     peak_y, peak_x = np.unravel_index(np.argmax(psf_smooth), psf.shape)

# #     # Dynamic crop +/- 6 lambda/D
# #     lim_pix = int(6.0 / angular_pixel_scale)
# #     y_s, y_e = max(0, peak_y - lim_pix), min(psf.shape[0], peak_y + lim_pix)
# #     x_s, x_e = max(0, peak_x - lim_pix), min(psf.shape[1], peak_x + lim_pix)
# #     psf_crop = psf[y_s:y_e, x_s:x_e]

# #     # 2D Gaussian Fit for core stats
# #     if np.max(psf_crop) > 0:
# #         y_idx, x_idx = np.indices(psf_crop.shape)
# #         x_ld = (x_idx - (peak_x - x_s)) * angular_pixel_scale
# #         y_ld = (y_idx - (peak_y - y_s)) * angular_pixel_scale
# #         fit = bt.fit_2d_gaussian(x_ld, y_ld, psf_crop / np.max(psf_crop))
# #         metrics = bt.gaussian_fwhm_and_ellipticity(fit)
# #         fwhm = metrics["fwhm_major"]
# #         ell = metrics["ellipticity"]
# #     else:
# #         fwhm = np.nan
# #         ell = np.nan

# #     # Encircled Energy profile calculation
# #     yy, xx = np.indices(psf.shape)
# #     r_pix = np.sqrt((xx - peak_x) ** 2 + (yy - peak_y) ** 2).flatten()
# #     idx = np.argsort(r_pix)

# #     r_sorted = r_pix[idx] * angular_pixel_scale
# #     ee_curve = np.cumsum(psf.flatten()[idx]) / total_flux

# #     ee80_idx = np.searchsorted(ee_curve, 0.80)
# #     ee80_idx = min(ee80_idx, len(r_sorted) - 1)
# #     ee80 = r_sorted[ee80_idx]

# #     # # Encircled Energy profile calculation
# #     # yy, xx = np.indices(psf.shape)
# #     # r_pix = np.sqrt((xx - peak_x) ** 2 + (yy - peak_y) ** 2).flatten()
# #     # idx = np.argsort(r_pix)
# #     # ee_curve = np.cumsum(psf.flatten()[idx]) / total_flux

# #     # ee80_idx = np.searchsorted(ee_curve, 0.80)
# #     # ee80_idx = min(ee80_idx, len(idx) - 1)
# #     # ee80 = (r_pix[idx] * angular_pixel_scale)[ee80_idx]

# #     return {
# #         "strehl": strehl_fft,
# #         "ee80": ee80,
# #         "ell": ell,
# #         "fwhm": fwhm,
# #         "psf_crop": psf_crop,
# #         "ee_curve": ee_curve
# #     }


# # # ==========================================
# # # 2. SETUP & DATA LOADING
# # # ==========================================

# # # FITS file path relative to this script's directory
# # script_dir = os.path.dirname(os.path.abspath(__file__))
# # FITS_PATH = os.path.join(script_dir, "../phasescreens/batch1_test/", "phasescreens_median_dmScaled-1_radialScaled-0.fits")

# # with fits.open(FITS_PATH) as hdul:
# #     # Ensure correct optical order from source (z=-3.25) to pupil (z=0)
# #     layer_configs = [
# #         {"label": "FA",  "z": -2.50,  "hz": 0.2},
# #         {"label": "GL3", "z": -0.060, "hz": 1.4},
# #         {"label": "GL2", "z": -0.030, "hz": 1.0},
# #         {"label": "GL1", "z": -0.001, "hz": 0.7},
# #     ]

# #     bench = bt.OpticalBench3D()
# #     pix_scale = hdul[0].header['PIXSCALE']

# #     for cfg in layer_configs:
# #         # FITS data appears to be in phase [rad] at 500 nm, so convert to OPD [m]
# #         opd = (hdul[cfg["label"]].data * 500e-9) / (2.0 * np.pi)

# #         if cfg["label"] == "FA":
# #             bench.add(
# #                 bt.RotatingPhaseScreen3D(
# #                     point=[26e-3, 0, cfg["z"]],
# #                     normal=[0, 0, 1],
# #                     opd_map=opd,
# #                     map_extent_m=opd.shape[0] * pix_scale,
# #                     angular_velocity=2.0 * np.pi * cfg["hz"],
# #                     label=cfg["label"]
# #                 )
# #             )
# #         else:
# #             bench.add(
# #                 bt.RotatingPhaseScreen3D(
# #                     point=[34e-3, 0, cfg["z"]],
# #                     normal=[0, 0, 1],
# #                     opd_map=opd,
# #                     map_extent_m=opd.shape[0] * pix_scale,
# #                     angular_velocity=2.0 * np.pi * cfg["hz"],
# #                     label=cfg["label"]
# #                 )
# #             )

# # # Constants
# # WAVELENGTH = 633e-9              # LGS / sensing wavelength [m]
# # SCIENCE_WAVELENGTH = 2.2e-6      # science wavelength [m]
# # D_BEAM = 0.013                   # beam diameter [m]
# # NPIX_PUPIL = 256
# # PAD_SIZE = 2048

# # # Note: this is your existing lambda/D pixel scaling convention
# # ANGULAR_SCALE = 1.0 / (PAD_SIZE / (NPIX_PUPIL / 2.0))

# # EXPOSURE_TIME = 2.0
# # DT = 0.2
# # times = np.arange(0, EXPOSURE_TIME, DT)

# # science_angles = np.linspace(0, 10, 5)  # arcmin
# # lgs_coords = [(10, 10), (-10, 10), (10, -10), (-10, -10)]

# # lgs_beams = [
# #     bt.make_converging_beam_from_field_angles(
# #         np.deg2rad(x / 60.0),
# #         np.deg2rad(y / 60.0),
# #         -3.25,
# #         [0, 0, 0],
# #         D_BEAM,
# #         WAVELENGTH,
# #         f"L{i}",
# #         3,
# #         12
# #     )
# #     for i, (x, y) in enumerate(lgs_coords)
# # ]

# # sci_beams = [
# #     bt.make_converging_beam_from_field_angles(
# #         np.deg2rad(th / 60.0),
# #         0.0,
# #         -3.25,
# #         [0, 0, 0],
# #         D_BEAM,
# #         SCIENCE_WAVELENGTH,
# #         f"S{i}",
# #         3,
# #         12
# #     )
# #     for i, th in enumerate(science_angles)
# # ]

# # # Perfect baseline at science wavelength
# # ref_beam = bt.make_converging_beam_from_field_angles(
# #     0.0, 0.0, -3.25, [0, 0, 0], D_BEAM, SCIENCE_WAVELENGTH, "ref", 3, 12
# # )
# # perf_sample = bt.sample_beam_phase_amplitude_on_pupil_plane(
# #     ref_beam, bench, [0, 0, 0], 0.0, NPIX_PUPIL
# # )
# # perf_sample = copy.deepcopy(perf_sample)
# # perf_sample["phase_map_rad"] = np.zeros_like(perf_sample["phase_map_rad"])
# # perfect_psf = pad_and_fft_psf(perf_sample, PAD_SIZE)

# # # ==========================================
# # # 3. GLAO SIMULATION LOOP
# # # ==========================================

# # accum_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE), dtype=float) for i in range(len(sci_beams))}
# # accum_no_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE), dtype=float) for i in range(len(sci_beams))}

# # DM_ACTS_ACROSS = 11

# # print(f"Exposure: {EXPOSURE_TIME}s | dt: {DT}s")
# # print(f"WFS wavelength = {WAVELENGTH*1e9:.1f} nm")
# # print(f"Science wavelength = {SCIENCE_WAVELENGTH*1e9:.1f} nm")
# # print(f"DM correction reconstructed in OPD space with {DM_ACTS_ACROSS} acts across pupil")

# # for t in times:
# #     # 1. Reconstruct GL correction from 4 LGS beams in OPD space
# #     lgs_samples = [
# #         bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0, 0, 0], t, NPIX_PUPIL)
# #         for b in lgs_beams
# #     ]

# #     lgs_mask = lgs_samples[0]["mask"]

# #     # Convert each LGS phase map [rad at WAVELENGTH] -> OPD [m]
# #     lgs_opd_maps = [
# #         phase_to_opd(s["phase_map_rad"], WAVELENGTH)
# #         for s in lgs_samples
# #     ]

# #     # Average ground-layer estimate in OPD
# #     gl_opd = np.mean(lgs_opd_maps, axis=0)

# #     # DM correction remains in OPD units
# #     dm_opd_corr = apply_dm_correction_opd(gl_opd, acts=DM_ACTS_ACROSS, mask=lgs_mask)

# #     # 2. Apply to science beams
# #     for i, beam in enumerate(sci_beams):
# #         s_samp = bt.sample_beam_phase_amplitude_on_pupil_plane(
# #             beam, bench, [0, 0, 0], t, NPIX_PUPIL
# #         )

# #         # Uncorrected science PSF
# #         accum_no_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)

# #         # Convert DM OPD correction into phase at science wavelength
# #         dm_phase_corr_sci = opd_to_phase(dm_opd_corr, SCIENCE_WAVELENGTH)

# #         # Apply science-wavelength phase correction
# #         s_samp_corr = copy.deepcopy(s_samp)
# #         s_samp_corr["phase_map_rad"] = s_samp_corr["phase_map_rad"] - dm_phase_corr_sci

# #         accum_ao[i] += pad_and_fft_psf(s_samp_corr, PAD_SIZE)

# # # ==========================================
# # # 4. OUTPUTS & PLOTTING
# # # ==========================================

# # res_ao = [analyze_psf(accum_ao[i] / len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]
# # res_no = [analyze_psf(accum_no_ao[i] / len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]

# # USE_LOG = False  # Set True for log10(Intensity)

# # fig_grid, axes_grid = plt.subplots(2, 5, figsize=(18, 8))

# # for i in range(5):
# #     p_no = res_no[i]["psf_crop"]
# #     p_ao = res_ao[i]["psf_crop"]

# #     if USE_LOG:
# #         img_no = np.log10(np.maximum(p_no / np.max(p_no), 1e-5))
# #         img_ao = np.log10(np.maximum(p_ao / np.max(p_ao), 1e-5))
# #     else:
# #         img_no = p_no
# #         img_ao = p_ao

# #     extent = [-6, 6, -6, 6]
# #     axes_grid[0, i].imshow(img_no, origin='lower', extent=extent, cmap='magma')
# #     axes_grid[1, i].imshow(img_ao, origin='lower', extent=extent, cmap='magma')

# #     axes_grid[0, i].set_title(f"Uncorr {science_angles[i]}'")
# #     axes_grid[1, i].set_title(f"GLAO {science_angles[i]}'")

# # for ax in axes_grid.flatten():
# #     ax.set_xlabel(r'[$\lambda/D$]')
# #     ax.set_ylabel(r'[$\lambda/D$]')

# # plt.tight_layout()
# # plt.show()

# # # Diagnostic Plots
# # fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # axes[0, 0].plot(science_angles, [r["strehl"] for r in res_no], 'ro--', label='No AO')
# # axes[0, 0].plot(science_angles, [r["strehl"] for r in res_ao], 'bo-', label='GLAO')

# # axes[0, 1].plot(science_angles, [r["fwhm"] for r in res_no], 'ro--', label='No AO')
# # axes[0, 1].plot(science_angles, [r["fwhm"] for r in res_ao], 'bo-', label='GLAO')

# # axes[1, 0].plot(science_angles, [r["ell"] for r in res_no], 'ro--', label='No AO')
# # axes[1, 0].plot(science_angles, [r["ell"] for r in res_ao], 'bo-', label='GLAO')

# # # EE Profile Center Point
# # r_axis = np.linspace(0, 10, 1000)
# # axes[1, 1].plot(res_no[0]["ee_radii"], res_no[0]["ee_curve"], 'r--', label='No AO Center')
# # axes[1, 1].plot(res_ao[0]["ee_radii"], res_ao[0]["ee_curve"], 'b-', label='GLAO Center')
# # axes[1, 1].set_xlim(0, 10)

# # # r_axis = np.linspace(0, 10, 1000)
# # # axes[1, 1].plot(r_axis, res_no[0]["ee_curve"][:1000], 'r--', label='No AO Center')
# # # axes[1, 1].plot(r_axis, res_ao[0]["ee_curve"][:1000], 'b-', label='GLAO Center')

# # axes[0, 0].set_ylabel("Strehl")
# # axes[0, 1].set_ylabel("FWHM [λ/D]")
# # axes[1, 0].set_ylabel("Ellipticity")
# # axes[1, 1].set_ylabel("Enc. Energy")

# # for ax in axes.flatten():
# #     ax.set_xlabel("Field Angle [arcmin]")
# #     ax.legend()
# #     ax.grid(True, alpha=0.3)

# # plt.tight_layout()
# # plt.show()

# # # ==========================================
# # # 5. RESULTS TABLE
# # # ==========================================

# # results_data = []

# # for i, angle in enumerate(science_angles):
# #     results_data.append({
# #         "Field Angle [arcmin]": angle,
# #         "AO Mode": "No AO",
# #         "Strehl": f"{res_no[i]['strehl']:.4f}",
# #         "FWHM [L/D]": f"{res_no[i]['fwhm']:.4f}",
# #         "Ellipticity": f"{res_no[i]['ell']:.4f}",
# #         "EE80 [L/D]": f"{res_no[i]['ee80']:.4f}"
# #     })
# #     results_data.append({
# #         "Field Angle [arcmin]": angle,
# #         "AO Mode": "GLAO",
# #         "Strehl": f"{res_ao[i]['strehl']:.4f}",
# #         "FWHM [L/D]": f"{res_ao[i]['fwhm']:.4f}",
# #         "Ellipticity": f"{res_ao[i]['ell']:.4f}",
# #         "EE80 [L/D]": f"{res_ao[i]['ee80']:.4f}"
# #     })

# # df_results = pd.DataFrame(results_data)

# # df_no_ao = df_results[df_results["AO Mode"] == "No AO"].reset_index(drop=True)
# # df_glao = df_results[df_results["AO Mode"] == "GLAO"].reset_index(drop=True)

# # print("\n" + "=" * 80)
# # print("DIAGNOSTIC RESULTS TABLE - NO AO")
# # print("=" * 80)
# # print(df_no_ao.to_string(index=False))

# # print("\n" + "=" * 80)
# # print("DIAGNOSTIC RESULTS TABLE - GLAO")
# # print("=" * 80)
# # print(df_glao.to_string(index=False))
# # print("=" * 80)

# # # Save results to CSV
# # results_filename = "psf_analysis_results.csv"
# # results_no_ao_filename = "psf_analysis_results_no_ao.csv"
# # results_glao_filename = "psf_analysis_results_glao.csv"

# # df_results.to_csv(results_filename, index=False)
# # df_no_ao.to_csv(results_no_ao_filename, index=False)
# # df_glao.to_csv(results_glao_filename, index=False)

# # print(f"\nResults saved to: {results_filename}")
# # print(f"Results also saved to: {results_no_ao_filename} and {results_glao_filename}")

# # # import os
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from astropy.io import fits
# # # from scipy.ndimage import gaussian_filter
# # # import beam_trace as bt  # Assumes beam_trace.py is in the same directory

# # # # ==========================================
# # # # 1. CORE PHYSICS & ANALYSIS FUNCTIONS
# # # # ==========================================

# # # def calculate_marechal_strehl(phase_map, mask):
# # #     """Maréchal/Mahajan approximation: S ~ exp(-sigma_phi^2)."""
# # #     if not np.any(mask): return 0.0
# # #     phase_var = np.var(phase_map[mask])
# # #     return np.exp(-phase_var)

# # # def apply_dm_correction(phase_map, acts, mask):
# # #     """Simulates high-pass spatial filtering for DM correction with strict masking."""
# # #     if acts == 0: return np.zeros_like(phase_map)
# # #     avg_phase = np.mean(phase_map[mask])
# # #     work_phase = np.where(mask, phase_map, avg_phase)
# # #     sigma = (phase_map.shape[0] / acts) * 0.5
# # #     low_spatial = gaussian_filter(work_phase, sigma=sigma, mode='reflect')
# # #     return np.where(mask, low_spatial, 0)

# # # def remove_tip_tilt(phase_map, mask):
# # #     """Remove tip-tilt (linear plane fit) from phase map."""
# # #     y, x = np.indices(phase_map.shape)
# # #     valid = mask.flatten()
# # #     x_flat = x.flatten()[valid]
# # #     y_flat = y.flatten()[valid]
# # #     phase_flat = phase_map.flatten()[valid]
    
# # #     # Fit plane: phase = a*x + b*y + c
# # #     A = np.column_stack([x_flat, y_flat, np.ones_like(x_flat)])
# # #     coeffs, _, _, _ = np.linalg.lstsq(A, phase_flat, rcond=None)
# # #     a, b, c = coeffs
    
# # #     # Subtract fitted plane
# # #     fitted_plane = a * x + b * y + c
# # #     return phase_map - fitted_plane

# # # def pad_and_fft_psf(sample, pad_to=2048):
# # #     """Generates high-resolution RAW intensity PSF."""
# # #     mask, phase, amp = sample["mask"], np.nan_to_num(sample["phase_map_rad"]), sample["amplitude"]
# # #     field = np.where(mask, amp * np.exp(1j * phase), 0)
# # #     pad_w = (pad_to - field.shape[0]) // 2
# # #     padded = np.pad(field, pad_w, mode='constant')
# # #     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
# # #     return np.abs(ef) ** 2

# # # def analyze_psf(psf, perfect_psf, angular_pixel_scale, ao_label="N/A", log_plot=False, fit_mode="moffat"):
# # #     """Robust analysis with optional log-scale visualization."""
# # #     total_flux = np.sum(psf)
# # #     strehl_fft = (np.max(psf) / total_flux) / (np.max(perfect_psf) / np.sum(perfect_psf))
    
# # #     # Smoothing for robust sub-pixel peak detection
# # #     psf_smooth = gaussian_filter(psf, sigma=2)
# # #     peak_y, peak_x = np.unravel_index(np.argmax(psf_smooth), psf.shape)
    
# # #     # Dynamic crop +/- 6 lambda/D
# # #     lim_pix = int(6.0 / angular_pixel_scale)
# # #     y_s, y_e = max(0, peak_y-lim_pix), min(psf.shape[0], peak_y+lim_pix)
# # #     x_s, x_e = max(0, peak_x-lim_pix), min(psf.shape[1], peak_x+lim_pix)
# # #     psf_crop = psf[y_s:y_e, x_s:x_e]
    
# # #     # 2D Gaussian Fit for core stats
# # #     y_idx, x_idx = np.indices(psf_crop.shape)
# # #     x_ld = (x_idx - (peak_x - x_s)) * angular_pixel_scale
# # #     y_ld = (y_idx - (peak_y - y_s)) * angular_pixel_scale
# # #     psf_crop_norm = psf_crop / np.max(psf_crop)
# # #     fit_mode = fit_mode.lower()
# # #     if fit_mode == "gaussian":
# # #         fit = bt.fit_2d_gaussian(x_ld, y_ld, psf_crop_norm)
# # #         metrics = bt.gaussian_fwhm_and_ellipticity(fit)
# # #     elif fit_mode == "moffat":
# # #         fit = bt.fit_2d_moffat(x_ld, y_ld, psf_crop_norm, fit_region="all")
# # #         metrics = bt.moffat_fwhm_and_ellipticity(fit)
# # #     elif fit_mode in ("moffat_wings", "moffat-wings", "moffatwings"):
# # #         fit = bt.fit_2d_moffat(x_ld, y_ld, psf_crop_norm, fit_region="wings")
# # #         metrics = bt.moffat_fwhm_and_ellipticity(fit)
# # #     else:
# # #         raise ValueError("fit_mode must be 'gaussian', 'moffat', or 'moffat_wings'.")

# # #     # Encircled Energy profile calculation
# # #     yy, xx = np.indices(psf.shape)
# # #     r_pix = np.sqrt((xx - peak_x)**2 + (yy - peak_y)**2).flatten()
# # #     idx = np.argsort(r_pix)
# # #     ee_curve = np.cumsum(psf.flatten()[idx]) / total_flux
# # #     ee80 = (r_pix[idx] * angular_pixel_scale)[np.searchsorted(ee_curve, 0.80)]

# # #     return {"strehl": strehl_fft, "ee80": ee80, "ell": metrics["ellipticity"], 
# # #             "fwhm": metrics["fwhm_major"], "psf_crop": psf_crop, "ee_curve": ee_curve,
# # #             "fit_mode": fit_mode}


# # # def save_gl_slice_plot(gl_phase, gl_corr, output_dir, t_stamp):
# # #     n = gl_phase.shape[0]
# # #     center_idx = n // 2
# # #     pixel_axis = np.arange(n) - center_idx
# # #     pixel_axis = pixel_axis[1:-1]

# # #     gl_phase_y0 = gl_phase[center_idx, 1:-1]
# # #     gl_phase_x0 = gl_phase[1:-1, center_idx]
# # #     gl_corr_y0 = gl_corr[center_idx, 1:-1]
# # #     gl_corr_x0 = gl_corr[1:-1, center_idx]

# # #     fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# # #     axes[0].plot(pixel_axis, gl_phase_y0, label='y = 0 slice (varying x)', color='tab:blue')
# # #     axes[0].plot(pixel_axis, gl_phase_x0, label='x = 0 slice (varying y)', color='tab:orange')
# # #     axes[0].set_title('GL phase slices')
# # #     axes[0].set_xlabel('Pixel offset from center')
# # #     axes[0].set_ylabel('Phase (rad)')
# # #     axes[0].grid(True, alpha=0.3)
# # #     axes[0].legend()

# # #     axes[1].plot(pixel_axis, gl_corr_y0, label='y = 0 slice (varying x)', color='tab:blue')
# # #     axes[1].plot(pixel_axis, gl_corr_x0, label='x = 0 slice (varying y)', color='tab:orange')
# # #     axes[1].set_title('GL correction slices')
# # #     axes[1].set_xlabel('Pixel offset from center')
# # #     axes[1].set_ylabel('Phase (rad)')
# # #     axes[1].grid(True, alpha=0.3)
# # #     axes[1].legend()

# # #     fig.suptitle(f'GL centerline slices at t = {t_stamp:.3f} s')
# # #     fig.tight_layout()

# # #     out_path = os.path.join(output_dir, 'gl_phase_gl_corr_slices.png')
# # #     fig.savefig(out_path, dpi=150, bbox_inches='tight')
# # #     plt.close(fig)
# # #     print(f"Saved plot: {out_path}")

# # # # ==========================================
# # # # 2. SETUP & DATA LOADING
# # # # ==========================================

# # # #FITS_PATH = "/home/bbarrer/mq_glao_testbench_sim/phasescreens/batch1_test/phasescreens_median_dmScaled-1_radialScaled-0.fits" #"phasescreens_median_dmScaled-1_radialScaled-0.fits"
# # # #FITS_PATH = "C:/Users/bmcinnes/OneDrive - Macquarie University/Documents/GitHub/mq_glao_testbench_sim/phasescreens_median_dmScaled-0_radialScaled-0.fits"

# # # # FITS file path relative to this script's directory
# # # script_dir = os.path.dirname(os.path.abspath(__file__))
# # # analysis_plots_dir = os.path.join(script_dir, 'psf_analysis_plots')
# # # os.makedirs(analysis_plots_dir, exist_ok=True)
# # # FITS_PATH = os.path.join(script_dir, "..", "phasescreens_median_dmScaled-1_radialScaled-0.fits")

# # # with fits.open(FITS_PATH) as hdul:
# # #     # Ensure correct optical order from source (z=-3.25) to pupil (z=0)
# # #     #layer_configs = [{"label": "FA", "z": -2.50, "hz": 0.2}, {"label": "GL3", "z": -0.060, "hz": 1.4},
# # #     #                {"label": "GL2", "z": -0.030, "hz": 1.0}, {"label": "GL1", "z": -0.001, "hz": 0.7}]
# # #     layer_configs = [{"label": "FA", "z": -2.50, "hz": 1.0}, {"label": "GL3", "z": -0.060, "hz": 1.0},
# # #                     {"label": "GL2", "z": -0.030, "hz": 1.0}, {"label": "GL1", "z": -0.001, "hz": 1.0}]
 
# # #     bench = bt.OpticalBench3D()
# # #     pix_scale = hdul[0].header['PIXSCALE']
# # #     for cfg in layer_configs:
# # #         opd = (hdul[cfg["label"]].data * 500e-9) / (2*np.pi)
# # #         if cfg["label"] == "FA":
# # #             bench.add(bt.RotatingPhaseScreen3D(point=[25e-3,0,cfg["z"]], normal=[0,0,1], opd_map=opd,
# # #                       map_extent_m=opd.shape[0]*pix_scale, angular_velocity=2*np.pi*cfg["hz"], label=cfg["label"]))
# # #         else:
# # #             bench.add(bt.RotatingPhaseScreen3D(point=[34.5e-3,0,cfg["z"]], normal=[0,0,1], opd_map=opd,
# # #                       map_extent_m=opd.shape[0]*pix_scale, angular_velocity=2*np.pi*cfg["hz"], label=cfg["label"]))

# # # # Constants
# # # WAVELENGTH, SCIENCE_WAVELENGTH, D_BEAM, NPIX_PUPIL, PAD_SIZE = 589e-9, 589e-9, 0.013, 256, 2048
# # # ANGULAR_SCALE = 1.0 / (PAD_SIZE / (NPIX_PUPIL / 2.0))
# # # EXPOSURE_TIME, DT = 1.0, 0.125#
# # # #EXPOSURE_TIME, DT = 2.0, 0.4 # Reduced for quick testing
# # # times = np.arange(0, EXPOSURE_TIME, DT)

# # # science_angles = np.linspace(0, 10, 5)
# # # lgs_coords = [(10,10), (-10,10), (10,-10), (-10,-10)]
# # # lgs_beams = [bt.make_converging_beam_from_field_angles(np.deg2rad(x/60), np.deg2rad(y/60), -3.25, [0,0,0], D_BEAM, WAVELENGTH, f"L", 3, 12) for x,y in lgs_coords]
# # # sci_beams = [bt.make_converging_beam_from_field_angles(np.deg2rad(th/60), 0, -3.25, [0,0,0], D_BEAM, SCIENCE_WAVELENGTH, f"S", 3, 12) for th in science_angles]

# # # # Perfect Baseline (identical beam geometry at science wavelength)
# # # ref_beam = bt.make_converging_beam_from_field_angles(0, 0, -3.25, [0,0,0], D_BEAM, SCIENCE_WAVELENGTH, "ref", 3, 12)
# # # perf_sample = bt.sample_beam_phase_amplitude_on_pupil_plane(ref_beam, bench, [0,0,0], 0.0, NPIX_PUPIL)
# # # perf_sample["phase_map_rad"] *= 0
# # # perfect_psf = pad_and_fft_psf(perf_sample, PAD_SIZE)

# # # # ==========================================
# # # # 3. GLAO SIMULATION LOOP
# # # # ==========================================

# # # accum_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE)) for i in range(len(sci_beams))}
# # # accum_no_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE)) for i in range(len(sci_beams))}

# # # print(f"Exposure: {EXPOSURE_TIME}s | dt: {DT}s")
# # # for t in times:
# # #     # 1. Reconstruct GL correction from 4 corners
# # #     lgs_s = [bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0,0,0], t, NPIX_PUPIL) for b in lgs_beams]
# # #     gl_phase = np.mean([s["phase_map_rad"] for s in lgs_s], axis=0)
    
# # #     #Optional Remove tip-tilt from averaged GL phase
# # #     #gl_phase = remove_tip_tilt(gl_phase, lgs_s[0]["mask"])
    
# # #     #gl_corr = apply_dm_correction(gl_phase, acts=35, mask=lgs_s[0]["mask"])
# # #     gl_corr = apply_dm_correction(gl_phase, acts=11, mask=lgs_s[0]["mask"])
    
# # #     # 2. Apply to Target beams
# # #     for i, beam in enumerate(sci_beams):
# # #         s_samp = bt.sample_beam_phase_amplitude_on_pupil_plane(beam, bench, [0,0,0], t, NPIX_PUPIL)
# # #         accum_no_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)
# # #         s_samp["phase_map_rad"] -= gl_corr
# # #         accum_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)

# # # # A sanity check plot of the GL phase and correction slices at the end of the simulation (can be commented out )
# # # #if len(times) > 0:
# # # #    save_gl_slice_plot(gl_phase, gl_corr, analysis_plots_dir, times[-1])
 

# # # # ==========================================
# # # # 4. OUTPUTS & PLOTTING
# # # # ==========================================

# # # res_ao = [analyze_psf(accum_ao[i]/len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]
# # # res_no = [analyze_psf(accum_no_ao[i]/len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]

# # # # Grid Plot Config
# # # USE_LOG = False # Set to True for log10(Intensity)

# # # fig_grid, axes_grid = plt.subplots(2, 5, figsize=(18, 8))
# # # for i in range(5):
# # #     p_no = res_no[i]["psf_crop"]
# # #     p_ao = res_ao[i]["psf_crop"]
    
# # #     # Apply scaling based on USE_LOG toggle
# # #     img_no = np.log10(np.maximum(p_no / np.max(p_no), 1e-5)) if USE_LOG else p_no
# # #     img_ao = np.log10(np.maximum(p_ao / np.max(p_ao), 1e-5)) if USE_LOG else p_ao
    
# # #     extent = [-6, 6, -6, 6]
# # #     im0 = axes_grid[0,i].imshow(img_no, origin='lower', extent=extent, cmap='magma')
# # #     im1 = axes_grid[1,i].imshow(img_ao, origin='lower', extent=extent, cmap='magma')
    
# # #     axes_grid[0,i].set_title(f"Uncorr {science_angles[i]}'")
# # #     axes_grid[1,i].set_title(f"GLAO {science_angles[i]}'")

# # # # Diagnostic Plots
# # # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# # # axes[0,0].plot(science_angles, [r["strehl"] for r in res_no], 'ro--', label='No AO')
# # # axes[0,0].plot(science_angles, [r["strehl"] for r in res_ao], 'bo-', label='GLAO')
# # # axes[0,1].plot(science_angles, [r["fwhm"] for r in res_no], 'ro--', label='No AO')
# # # axes[0,1].plot(science_angles, [r["fwhm"] for r in res_ao], 'bo-', label='GLAO')
# # # axes[1,0].plot(science_angles, [r["ell"] for r in res_no], 'ro--', label='No AO')
# # # axes[1,0].plot(science_angles, [r["ell"] for r in res_ao], 'bo-', label='GLAO')

# # # # EE Profile Center Point
# # # r_axis = np.linspace(0, 10, 1000) # Grid for EE display
# # # axes[1,1].plot(r_axis, res_no[0]["ee_curve"][:1000], 'r--', label='No AO Center')
# # # axes[1,1].plot(r_axis, res_ao[0]["ee_curve"][:1000], 'b-', label='GLAO Center')

# # # axes[0,0].set_ylabel("Strehl"); axes[0,1].set_ylabel("FWHM [L/D]"); axes[1,0].set_ylabel("Ellipticity"); axes[1,1].set_ylabel("Enc. Energy")
# # # for ax in axes.flatten(): ax.set_xlabel("Field Angle [arcmin]"); ax.legend(); ax.grid(True, alpha=0.3)
# # # fig_grid.tight_layout()
# # # fig.tight_layout()

# # # grid_plot_filename = "psf_grid_plot.png"
# # # diagnostic_plot_filename = "psf_diagnostic_plot.png"
# # # fig_grid.savefig(grid_plot_filename, dpi=150, bbox_inches='tight')
# # # fig.savefig(diagnostic_plot_filename, dpi=150, bbox_inches='tight')
# # # plt.close(fig_grid)
# # # plt.close(fig)

# # # print(f"Saved plot: {grid_plot_filename}")
# # # print(f"Saved plot: {diagnostic_plot_filename}")

# # # # ==========================================
# # # # 5. RESULTS TABLE
# # # # ==========================================

# # # import pandas as pd

# # # results_data = []
# # # for i, angle in enumerate(science_angles):
# # #     results_data.append({
# # #         "Field Angle [arcmin]": angle,
# # #         "AO Mode": "No AO",
# # #         "Strehl": f"{res_no[i]['strehl']:.4f}",
# # #         "FWHM [L/D]": f"{res_no[i]['fwhm']:.4f}",
# # #         "Ellipticity": f"{res_no[i]['ell']:.4f}",
# # #         "EE80 [L/D]": f"{res_no[i]['ee80']:.4f}"
# # #     })
# # #     results_data.append({
# # #         "Field Angle [arcmin]": angle,
# # #         "AO Mode": "GLAO",
# # #         "Strehl": f"{res_ao[i]['strehl']:.4f}",
# # #         "FWHM [L/D]": f"{res_ao[i]['fwhm']:.4f}",
# # #         "Ellipticity": f"{res_ao[i]['ell']:.4f}",
# # #         "EE80 [L/D]": f"{res_ao[i]['ee80']:.4f}"
# # #     })

# # # df_results = pd.DataFrame(results_data)

# # # df_no_ao = df_results[df_results["AO Mode"] == "No AO"].reset_index(drop=True)
# # # df_glao = df_results[df_results["AO Mode"] == "GLAO"].reset_index(drop=True)

# # # print("\n" + "="*80)
# # # print("DIAGNOSTIC RESULTS TABLE - NO AO")
# # # print("="*80)
# # # print(df_no_ao.to_string(index=False))
# # # print("\n" + "="*80)
# # # print("DIAGNOSTIC RESULTS TABLE - GLAO")
# # # print("="*80)
# # # print(df_glao.to_string(index=False))
# # # print("="*80)

# # # # Save results to CSV
# # # results_filename = "psf_analysis_results.csv"
# # # df_results.to_csv(results_filename, index=False)
# # # results_no_ao_filename = "psf_analysis_results_no_ao.csv"
# # # results_glao_filename = "psf_analysis_results_glao.csv"
# # # df_no_ao.to_csv(results_no_ao_filename, index=False)
# # # df_glao.to_csv(results_glao_filename, index=False)
# # # print(f"\nResults saved to: {results_filename}")
# # # print(f"Results also saved to: {results_no_ao_filename} and {results_glao_filename}")


