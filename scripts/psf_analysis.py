import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import beam_trace as bt  # Assumes beam_trace.py is in the same directory

# ==========================================
# 1. CORE PHYSICS & ANALYSIS FUNCTIONS
# ==========================================

def calculate_marechal_strehl(phase_map, mask):
    """Maréchal/Mahajan approximation: S ~ exp(-sigma_phi^2)."""
    if not np.any(mask): return 0.0
    phase_var = np.var(phase_map[mask])
    return np.exp(-phase_var)

def apply_dm_correction(phase_map, acts, mask):
    """Simulates high-pass spatial filtering for DM correction with strict masking."""
    if acts == 0: return np.zeros_like(phase_map)
    avg_phase = np.mean(phase_map[mask])
    work_phase = np.where(mask, phase_map, avg_phase)
    sigma = (phase_map.shape[0] / acts) * 0.5
    low_spatial = gaussian_filter(work_phase, sigma=sigma, mode='reflect')
    return np.where(mask, low_spatial, 0)

def pad_and_fft_psf(sample, pad_to=2048):
    """Generates high-resolution RAW intensity PSF."""
    mask, phase, amp = sample["mask"], np.nan_to_num(sample["phase_map_rad"]), sample["amplitude"]
    field = np.where(mask, amp * np.exp(1j * phase), 0)
    pad_w = (pad_to - field.shape[0]) // 2
    padded = np.pad(field, pad_w, mode='constant')
    ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return np.abs(ef) ** 2

def analyze_psf(psf, perfect_psf, angular_pixel_scale, ao_label="N/A", log_plot=False):
    """Robust analysis with optional log-scale visualization."""
    total_flux = np.sum(psf)
    strehl_fft = (np.max(psf) / total_flux) / (np.max(perfect_psf) / np.sum(perfect_psf))
    
    # Smoothing for robust sub-pixel peak detection
    psf_smooth = gaussian_filter(psf, sigma=2)
    peak_y, peak_x = np.unravel_index(np.argmax(psf_smooth), psf.shape)
    
    # Dynamic crop +/- 6 lambda/D
    lim_pix = int(6.0 / angular_pixel_scale)
    y_s, y_e = max(0, peak_y-lim_pix), min(psf.shape[0], peak_y+lim_pix)
    x_s, x_e = max(0, peak_x-lim_pix), min(psf.shape[1], peak_x+lim_pix)
    psf_crop = psf[y_s:y_e, x_s:x_e]
    
    # 2D Gaussian Fit for core stats
    y_idx, x_idx = np.indices(psf_crop.shape)
    x_ld = (x_idx - (peak_x - x_s)) * angular_pixel_scale
    y_ld = (y_idx - (peak_y - y_s)) * angular_pixel_scale
    fit = bt.fit_2d_gaussian(x_ld, y_ld, psf_crop / np.max(psf_crop))
    metrics = bt.gaussian_fwhm_and_ellipticity(fit)

    # Encircled Energy profile calculation
    yy, xx = np.indices(psf.shape)
    r_pix = np.sqrt((xx - peak_x)**2 + (yy - peak_y)**2).flatten()
    idx = np.argsort(r_pix)
    ee_curve = np.cumsum(psf.flatten()[idx]) / total_flux
    ee80 = (r_pix[idx] * angular_pixel_scale)[np.searchsorted(ee_curve, 0.80)]

    return {"strehl": strehl_fft, "ee80": ee80, "ell": metrics["ellipticity"], 
            "fwhm": metrics["fwhm_major"], "psf_crop": psf_crop, "ee_curve": ee_curve}

# ==========================================
# 2. SETUP & DATA LOADING
# ==========================================

FITS_PATH = "/home/bbarrer/mq_glao_testbench_sim/phasescreens/batch1_test/phasescreens_median_dmScaled-1_radialScaled-0.fits" #"phasescreens_median_dmScaled-1_radialScaled-0.fits"
with fits.open(FITS_PATH) as hdul:
    # Ensure correct optical order from source (z=-3.25) to pupil (z=0)
    layer_configs = [{"label": "FA", "z": -2.50, "hz": 0.2}, {"label": "GL3", "z": -0.096, "hz": 1.4},
                    {"label": "GL2", "z": -0.048, "hz": 1.0}, {"label": "GL1", "z": -0.024, "hz": 0.7}]
    bench = bt.OpticalBench3D()
    pix_scale = hdul[0].header['PIXSCALE']
    for cfg in layer_configs:
        opd = (hdul[cfg["label"]].data * 500e-9) / (2*np.pi)
        bench.add(bt.RotatingPhaseScreen3D(point=[0,0,cfg["z"]], normal=[0,0,1], opd_map=opd, 
                  map_extent_m=opd.shape[0]*pix_scale, angular_velocity=2*np.pi*cfg["hz"], label=cfg["label"]))

# Constants
WAVELENGTH, D_BEAM, NPIX_PUPIL, PAD_SIZE = 589e-9, 0.013, 256, 2048
ANGULAR_SCALE = 1.0 / (PAD_SIZE / (NPIX_PUPIL / 2.0))
EXPOSURE_TIME, DT = 2.0, 0.4 # Reduced for quick testing
times = np.arange(0, EXPOSURE_TIME, DT)

science_angles = np.linspace(0, 10, 5)
lgs_coords = [(10,10), (-10,10), (10,-10), (-10,-10)]
lgs_beams = [bt.make_converging_beam_from_field_angles(np.deg2rad(x/60), np.deg2rad(y/60), -3.25, [0,0,0], D_BEAM, WAVELENGTH, f"L", 3, 12) for x,y in lgs_coords]
sci_beams = [bt.make_converging_beam_from_field_angles(np.deg2rad(th/60), 0, -3.25, [0,0,0], D_BEAM, WAVELENGTH, f"S", 3, 12) for th in science_angles]

# Perfect Baseline (identical beam geometry)
ref_beam = bt.make_converging_beam_from_field_angles(0, 0, -3.25, [0,0,0], D_BEAM, WAVELENGTH, "ref", 3, 12)
perf_sample = bt.sample_beam_phase_amplitude_on_pupil_plane(ref_beam, bench, [0,0,0], 0.0, NPIX_PUPIL)
perf_sample["phase_map_rad"] *= 0
perfect_psf = pad_and_fft_psf(perf_sample, PAD_SIZE)

# ==========================================
# 3. GLAO SIMULATION LOOP
# ==========================================

accum_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE)) for i in range(len(sci_beams))}
accum_no_ao = {i: np.zeros((PAD_SIZE, PAD_SIZE)) for i in range(len(sci_beams))}

print(f"Exposure: {EXPOSURE_TIME}s | dt: {DT}s")
for t in times:
    # 1. Reconstruct GL correction from 4 corners
    lgs_s = [bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0,0,0], t, NPIX_PUPIL) for b in lgs_beams]
    gl_phase = np.mean([s["phase_map_rad"] for s in lgs_s], axis=0)
    gl_corr = apply_dm_correction(gl_phase, acts=35, mask=lgs_s[0]["mask"])
    
    # 2. Apply to Target beams
    for i, beam in enumerate(sci_beams):
        s_samp = bt.sample_beam_phase_amplitude_on_pupil_plane(beam, bench, [0,0,0], t, NPIX_PUPIL)
        accum_no_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)
        s_samp["phase_map_rad"] -= gl_corr
        accum_ao[i] += pad_and_fft_psf(s_samp, PAD_SIZE)

# ==========================================
# 4. OUTPUTS & PLOTTING
# ==========================================

res_ao = [analyze_psf(accum_ao[i]/len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]
res_no = [analyze_psf(accum_no_ao[i]/len(times), perfect_psf, ANGULAR_SCALE) for i in range(len(sci_beams))]

# Grid Plot Config
USE_LOG = False # Set to True for log10(Intensity)

fig_grid, axes_grid = plt.subplots(2, 5, figsize=(18, 8))
for i in range(5):
    p_no = res_no[i]["psf_crop"]
    p_ao = res_ao[i]["psf_crop"]
    
    # Apply scaling based on USE_LOG toggle
    img_no = np.log10(np.maximum(p_no / np.max(p_no), 1e-5)) if USE_LOG else p_no
    img_ao = np.log10(np.maximum(p_ao / np.max(p_ao), 1e-5)) if USE_LOG else p_ao
    
    extent = [-6, 6, -6, 6]
    im0 = axes_grid[0,i].imshow(img_no, origin='lower', extent=extent, cmap='magma')
    im1 = axes_grid[1,i].imshow(img_ao, origin='lower', extent=extent, cmap='magma')
    
    axes_grid[0,i].set_title(f"Uncorr {science_angles[i]}'")
    axes_grid[1,i].set_title(f"GLAO {science_angles[i]}'")

# Diagnostic Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(science_angles, [r["strehl"] for r in res_no], 'ro--', label='No AO')
axes[0,0].plot(science_angles, [r["strehl"] for r in res_ao], 'bo-', label='GLAO')
axes[0,1].plot(science_angles, [r["fwhm"] for r in res_no], 'ro--', label='No AO')
axes[0,1].plot(science_angles, [r["fwhm"] for r in res_ao], 'bo-', label='GLAO')
axes[1,0].plot(science_angles, [r["ell"] for r in res_no], 'ro--', label='No AO')
axes[1,0].plot(science_angles, [r["ell"] for r in res_ao], 'bo-', label='GLAO')

# EE Profile Center Point
r_axis = np.linspace(0, 10, 1000) # Grid for EE display
axes[1,1].plot(r_axis, res_no[0]["ee_curve"][:1000], 'r--', label='No AO Center')
axes[1,1].plot(r_axis, res_ao[0]["ee_curve"][:1000], 'b-', label='GLAO Center')

axes[0,0].set_ylabel("Strehl"); axes[0,1].set_ylabel("FWHM [L/D]"); axes[1,0].set_ylabel("Ellipticity"); axes[1,1].set_ylabel("Enc. Energy")
for ax in axes.flatten(): ax.set_xlabel("Field Angle [arcmin]"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

