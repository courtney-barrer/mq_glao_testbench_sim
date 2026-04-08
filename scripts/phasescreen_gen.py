import os
import numpy as np
import matplotlib.pyplot as plt
import aotools
from astropy.io import fits

script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'Phase Screen Plots')
os.makedirs(plots_dir, exist_ok=True)

# ==============================
# PARAMETERS
# ==============================
N = 4096
D_tel = 8.2                 # telescope diameter (m)
DMActuators_tel = 35        # DM actuators across D_tel

D_test = 0.013              # test bench beam diameter (m)
DMActuators_test = 11       # DM actuators across D_test
D_plate = 0.083             # phase plate useable OD (m)

Aperture_scale = D_tel / D_test
Plate_scale = D_tel / D_plate
pixel_scale = D_plate / N   # metres per pixel

# PSD model options: 'kolmogorov' (default) or 'von_karman'
PSD_MODEL = 'von_karman' #'kolmogorov'#
L0 = 4.1/Plate_scale  # von Karman outer scale (m) scaled to plate size.  4.1m scales to 41.5mm which is x3.2 on beam diameter D_test

# Fried parameters to simulate (m) - Median seeing at 500 nm
r0s_names = ['GL1', 'GL2', 'GL3', 'FA'] 
r0s = np.array([0.279, 0.416, 0.920, 0.244]) 
r0s = r0s / Aperture_scale  # Scale to test bench size

# Taper up in the outer 13 mm (one beam diameter)
R_transition = (83 - 2 * 13) / 83
Scale_edge = 1.5 # Scale at the edge of the plate up for stonger turbulence  

# Convention : "<turb_strength>_<DMscaled>_<radial_scaled>"
#batch_name = 'median_dmScaled-1_radialScaled-0' 
#batch_name_list = [f'median_dmScaled-{dm}_radialScaled-{radial}' for dm, radial in [[0,0],[1,0],[0,1],[1,1]]]
batch_name_list = [f'median_dmScaled-{dm}_radialScaled-{radial}' for dm, radial in [[1,0],[1,1]]]# just run a couple

# itrerate over the batch names and generate a file for each one
# if dm_scaled = 0, then we will not apply the DM scaling to the phase screen
# if radial_scaled = 0, then we will not apply the radial scaling to the phase

# define a 2D mask with radial scaling that can be applied to the phase screen to simulate a larger r0 at the edges of the plate
# make the scaling factor 1 up to some radius and then a linear ramp to the edge of the plate, where the scaling factor is r0_edge/r0_center. The radius of the flat region and the start of the ramp can be defined as a fraction of the plate radius. The outer edge of the ramp is at the corner of the square array (which is sqrt(2) times the half-width).
def make_radial_mask(size=4096, flat_radius_fraction=0.5, start_value=0.0, end_value=1.0):
    half = size / 2.0
    y, x = np.ogrid[-half:half, -half:half]
    r = np.sqrt(x**2 + y**2) / half
    flat_r  = flat_radius_fraction
    outer_r = 1.0
    t = np.clip((r - flat_r) / (outer_r - flat_r), 0.0, 1.0)
    mask = np.where(
        r <= flat_r,start_value,
        np.where(r >= outer_r, end_value, start_value + t * (end_value - start_value))
    ).astype(np.float32)
    return mask


def save_phase_linecuts(phase, pixel_scale, batch_name, screen_name, output_dir):
    # For even-sized arrays, x=0 and y=0 are between two pixels; use the nearest central index.
    n = phase.shape[0]
    center_idx = n // 2
    x_axis_mm = (np.arange(n) - center_idx) * pixel_scale * 1e3

    y0_line = phase[center_idx, :]  # y ~= 0, varying x
    x0_line = phase[:, center_idx]  # x ~= 0, varying y

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis_mm, y0_line, label='y ~= 0 line (phase vs x)', color='tab:blue')
    ax.plot(x_axis_mm, x0_line, label='x ~= 0 line (phase vs y)', color='tab:orange')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Phase (rad @ 500 nm)')
    ax.set_title(f'Phase line cuts - {batch_name} - {screen_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_name = f'phase_linecuts_{batch_name}-{screen_name}.png'
    fig.savefig(os.path.join(output_dir, output_name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_phase_psd_plot(phase, pixel_scale, batch_name, screen_name, output_dir, psd_model, r0, l0=None):
    n = phase.shape[0]

    # Measured 2D PSD from generated phase screen
    ft = np.fft.fft2(phase)
    psd2d = (np.abs(ft) ** 2) * (pixel_scale ** 2) / (n ** 2)

    fx = np.fft.fftfreq(n, d=pixel_scale)
    fy = np.fft.fftfreq(n, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fy)
    fr = np.sqrt(FX**2 + FY**2)

    # Radial average on log-spaced bins
    f_min = 1.0 / (n * pixel_scale)
    f_max = 0.5 / pixel_scale
    edges = np.geomspace(f_min, f_max, 200)
    f_flat = fr.ravel()
    p_flat = psd2d.ravel()
    valid = f_flat > 0
    f_flat = f_flat[valid]
    p_flat = p_flat[valid]

    f_centers = np.sqrt(edges[:-1] * edges[1:])
    p_radial = np.full_like(f_centers, np.nan, dtype=float)
    for i in range(len(f_centers)):
        in_bin = (f_flat >= edges[i]) & (f_flat < edges[i + 1])
        if np.any(in_bin):
            p_radial[i] = np.mean(p_flat[in_bin])
    ok = np.isfinite(p_radial)

    # Simple theoretical reference for shape comparison (scaled to measured first point)
    f_ref = f_centers[ok]
    if len(f_ref) > 0:
        if psd_model == 'von_karman' and l0 is not None:
            f0 = 1.0 / l0
            p_ref = 0.023 * r0**(-5/3) * (f_ref**2 + f0**2)**(-11/6)
        else:
            p_ref = 0.023 * r0**(-5/3) * (f_ref**2 + 1e-20)**(-11/6)
        if np.isfinite(p_radial[ok][0]) and p_ref[0] > 0:
            p_ref *= p_radial[ok][0] / p_ref[0]

    psd2d_shift = np.fft.fftshift(psd2d)
    extent = [fx.min(), fx.max(), fy.min(), fy.max()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(np.log10(np.maximum(psd2d_shift, 1e-30)), origin='lower', extent=extent, cmap='magma')
    axes[0].set_title('2D phase PSD (log10)')
    axes[0].set_xlabel('fx (cycles/m)')
    axes[0].set_ylabel('fy (cycles/m)')
    fig.colorbar(im, ax=axes[0], label='log10(PSD)')

    axes[1].loglog(f_centers[ok], p_radial[ok], label='Measured radial PSD', color='tab:blue')
    if len(f_ref) > 0:
        axes[1].loglog(f_ref, p_ref, '--', label=f'{psd_model} reference', color='tab:orange')

    #add dotted lines for referencce
    beam_diameter_m = 0.013
    beam_freq = 1.0 / beam_diameter_m
    axes[1].axvline(beam_freq, color='tab:red', linestyle=':', linewidth=1.5, label='Beam diameter (1/0.013 m)')
    act_spacing_m = 0.0015
    act_freq = 1.0 / act_spacing_m
    axes[1].axvline(act_freq, color='tab:green', linestyle=':', linewidth=1.5, label='Actuator spacing (1/0.0015 m)')


    axes[1].set_title('Radial phase PSD')
    axes[1].set_xlabel('Spatial frequency (cycles/m)')
    axes[1].set_ylabel('PSD (rad^2 m^2)')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend()

    fig.suptitle(f'Phase PSD - {batch_name} - {screen_name}')
    fig.tight_layout()

    output_name = f'phase_psd_{batch_name}-{screen_name}.png'
    fig.savefig(os.path.join(output_dir, output_name), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================
# BATCH LOOP
# ==============================
for batch_name in batch_name_list:
    print(f"\n=== Generating Phase Screens for Batch: {batch_name} ===")

    # ==============================
    # LOOP OVER r0 VALUES
    # ==============================
    fits_list = []
    seed=10# for repaetable random numbers
    for r0, name in zip(r0s, r0s_names):

         # Apply DM Scaling if specified
        if 'dmScaled-1' in batch_name:
            r0 = r0 * (DMActuators_tel / DMActuators_test)
        
        r0_mm = np.round(r0 * 1e3, decimals=3)
        print(f"\n--- Generating Screen: {name} (r0 = {r0_mm} mm) ---")

        # Generate Kolmogorov phase screen using Fourier Method
        fx = np.fft.fftfreq(N, pixel_scale)
        fy = np.fft.fftfreq(N, pixel_scale)
        FX, FY = np.meshgrid(fx, fy)
        f = np.sqrt(FX**2 + FY**2)

        # Kolmogorov: PSD ~ f^(-11/3), implemented as (f^2)^(-11/6)
        if PSD_MODEL == 'kolmogorov':
            PSD_phi = 0.023 * r0**(-5/3) * (f**2 + 1e-10)**(-11/6)
        elif PSD_MODEL == 'von_karman':
            f0 = 1.0 / L0
            PSD_phi = 0.023 * r0**(-5/3) * (f**2 + f0**2)**(-11/6)
        else:
            raise ValueError(f"Unsupported PSD_MODEL: {PSD_MODEL}. Use 'kolmogorov' or 'von_karman'.")

        df = 1.0 / (N * pixel_scale)
        rng = np.random.default_rng(seed) 
        seed += 1 # Increment seed for next screen
        cn = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
        #cn = (np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))) # Original non-reproducible version
        phase = np.real(np.fft.ifft2(cn * np.sqrt(PSD_phi) * df)) * N**2

        # Remove Tip / Tilt
        xx = np.linspace(-1, 1, N)
        XX, YY = np.meshgrid(xx, xx)
        A = np.column_stack([np.ones(N*N), XX.flatten(), YY.flatten()])
        coeff, *_ = np.linalg.lstsq(A, phase.flatten(), rcond=None)
        plane = (coeff[0] + coeff[1]*XX + coeff[2]*YY)
        phase = phase - plane

       
        if 'radialScaled-1' in batch_name:
            # Create Radial r0 Mask (Tapering)
            #never scale the FA screen as it will not be laterally shifted to change r0
            if 'FA' not in name:
                scale_mask = make_radial_mask(N, R_transition, 1, Scale_edge)
                phase = phase * scale_mask

        # Metadata Reporting
        OPD = (np.max(phase) - np.min(phase)) * 500e-9 / (2*np.pi) * 1e6
        print(f"Max OPD for {name}: {OPD:.2f} um")

        #optional save line cuts for visualization and sanity check
        #save_phase_linecuts(phase, pixel_scale, batch_name, name, plots_dir)
        #print(f"Saved phase line cuts for {name}")

        save_phase_psd_plot(phase, pixel_scale, batch_name, name, plots_dir, PSD_MODEL, r0, l0=L0)
        print(f"Saved phase PSD for {name}")

        #optional visualization of the phase screen
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(phase, cmap='RdBu')
        ax.set_title(f"Phase Screen (radians @ 500 nm) — {name} r0 = {r0_mm} mm")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"phase_screen_{batch_name+"-"+name}.png"), dpi=150)
        plt.close(fig)
        
        # Build individual HDU
        hdu = fits.PrimaryHDU(phase)
        hdu.header['BUNIT'] = ('rad', 'Phase in radians at 500 nm')
        hdu.header['PIXSCALE'] = (pixel_scale, 'Plate pixel scale (m/pixel)')
        hdu.header['r0'] = (r0, 'Fried parameter (m)')
        fits_list.append(hdu)

    # ==============================
    # SAVE MULTI-EXTENSION FITS
    # ==============================

    # 1. Name the Primary HDU (GL1)
    fits_list[0].name = r0s_names[0]

    # 2. Convert the rest of the list into named ImageHDU extensions
    extension_hdus = []
    for hdu, name in zip(fits_list[1:], r0s_names[1:]):
        ext_hdu = fits.ImageHDU(data=hdu.data, header=hdu.header, name=name)
        extension_hdus.append(ext_hdu)

    # 3. Combine into a single HDUList
    # HDU 0 is the named Primary; the rest are named ImageHDU extensions
    combined_hdul = fits.HDUList([fits_list[0]] + extension_hdus)

    output_filename = f"phasescreens_{batch_name}.fits"
    combined_hdul.writeto(output_filename, overwrite=True)

    print(f"\nSuccessfully saved combined file: {output_filename}")
