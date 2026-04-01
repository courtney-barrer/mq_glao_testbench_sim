import os
import numpy as np
import matplotlib.pyplot as plt
import aotools
from astropy.io import fits

script_dir = os.path.dirname(os.path.abspath(__file__))

# Convention : "<turb_strength>_<DMscaled>_<radial_scaled>"
batch_name = 'median_dmScaled-1_radialScaled-0' 
batch_name_list = [f'median_dmScaled-{dm}_radialScaled-{radial}' for dm, radial in [[0,0],[1,0],[0,1],[1,1]]]

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
pixel_scale = D_plate / N   # metres per pixel

# Fried parameters to simulate (m) - Median seeing at 500 nm
r0s = np.array([0.279, 0.416, 0.920, 0.244]) 

r0s = r0s / Aperture_scale  # Scale to test bench size
r0s_names = ['GL1', 'GL2', 'GL3', 'FA']      

# Taper up in the outer 13 mm (one beam diameter)
R_transition = (83 - 2 * 13) / 83
Scale_edge = 1.5 # Scale at the edge of the plate up for stonger turbulence  

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

        PSD_phi = 0.023 * r0**(-5/3) * (f**2 + 1e-10)**(-11/6)
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
            scale_mask = make_radial_mask(N, R_transition, 1, Scale_edge)
            phase = phase * scale_mask

        # Metadata Reporting
        OPD = (np.max(phase) - np.min(phase)) * 500e-9 / (2*np.pi) * 1e6
        print(f"Max OPD for {name}: {OPD:.2f} um")

        #optional visualization of the phase screen
        # fig, ax = plt.subplots(figsize=(6, 5))
        # im = ax.imshow(phase, cmap='RdBu')
        # ax.set_title(f"Phase Screen (radians @ 500 nm) — {name} r0 = {r0_mm} mm")
        # fig.colorbar(im, ax=ax)
        # fig.tight_layout()
        # fig.savefig(os.path.join(script_dir, f"phase_screen_{batch_name+"-"+name}.png"), dpi=150)
        # plt.close(fig)
        
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
