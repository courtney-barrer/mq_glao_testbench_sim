import os
import numpy as np
import matplotlib.pyplot as plt
import aotools
from astropy.io import fits

script_dir = os.path.dirname(os.path.abspath(__file__))

# Convention : "<turb_strength>_<DMscaled>_<radial_scaled>"
batch_name = 'median_dmScaled-1_radialScaled-0' 

# ==============================
# PARAMETERS
# ==============================
N = 4096
D_tel = 8.2                 # telescope diameter (m)
D_test = 0.013              # test bench beam diameter (m)
D_plate = 0.083             # phase plate useable OD (m)

Aperture_scale = D_tel / D_test
pixel_scale = D_plate / N   # metres per pixel

# Fried parameters to simulate (m) - Median seeing at 500 nm
r0s = np.array([0.279, 0.416, 0.920, 0.244]) 
r0s = r0s / Aperture_scale  # Scale to test bench size
r0s_names = ['GL1', 'GL2', 'GL3', 'FA']      

# Taper up in the outer 13 mm (one beam diameter)
R_transition = (83 - 2 * 13) / 83     

def make_radial_mask(size=4096, flat_radius_fraction=0.5, start_value=0.0, end_value=1.0):
    half = size / 2.0
    y, x = np.ogrid[-half:half, -half:half]
    r = np.sqrt(x**2 + y**2) / half
    flat_r  = flat_radius_fraction
    outer_r = 1.0
    t = np.clip((r - flat_r) / (outer_r - flat_r), 0.0, 1.0)
    mask = np.where(
        r <= flat_r,
        start_value,
        np.where(r >= outer_r, end_value, start_value + t * (end_value - start_value))
    ).astype(np.float32)
    return mask

# ==============================
# LOOP OVER r0 VALUES
# ==============================
fits_list = []
for r0, name in zip(r0s, r0s_names):
    r0_mm = np.round(r0 * 1e3, decimals=3)
    print(f"\n--- Generating Screen: {name} (r0 = {r0_mm} mm) ---")

    # Generate Kolmogorov phase screen using Fourier Method
    fx = np.fft.fftfreq(N, pixel_scale)
    fy = np.fft.fftfreq(N, pixel_scale)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)

    PSD_phi = 0.023 * r0**(-5/3) * (f**2 + 1e-10)**(-11/6)
    df = 1.0 / (N * pixel_scale)
    rng = np.random.default_rng(2026)
    cn = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    phase = np.real(np.fft.ifft2(cn * np.sqrt(PSD_phi) * df)) * N**2

    # Remove Tip / Tilt
    xx = np.linspace(-1, 1, N)
    XX, YY = np.meshgrid(xx, xx)
    A = np.column_stack([np.ones(N*N), XX.flatten(), YY.flatten()])
    coeff, *_ = np.linalg.lstsq(A, phase.flatten(), rcond=None)
    plane = (coeff[0] + coeff[1]*XX + coeff[2]*YY)
    phase = phase - plane

    # Create Radial r0 Mask (Tapering)
    r0_edge = r0 / 1.4
    scale_mask = make_radial_mask(N, R_transition, 1, r0 / r0_edge)
    phase = phase * scale_mask

    # Metadata Reporting
    OPD = (np.max(phase) - np.min(phase)) * 500e-9 / (2*np.pi) * 1e6
    print(f"Max OPD for {name}: {OPD:.2f} um")

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

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import aotools
# from scipy import stats
# from numpy.fft import fft2, fftshift
# from astropy.io import fits

# script_dir = os.path.dirname(os.path.abspath(__file__))


# batch_name = 'phasescreens_median_dmScaled-1_radialScaled-0' # convention : "<turb_strength>_<DMscaled>_<radial_scaled>"

# # ==============================
# # PARAMETERS
# # ==============================

# N = 4096
# D_tel = 8.2                 # telescope diameter (m)
# D_test = 0.013              # test bench beam diameter (m)
# D_plate = 0.083             # phase plate useable OD (m)

# Aperture_scale = D_tel / D_test #630.77 for 8.2m/13mm
# #Plate_scale = D_tel/D_plate

# #pixel_scale = D_tel/N       # metres per pixel
# pixel_scale = D_plate/N       # metres per pixel  Small sample warining using AOtools!!!

# #lamda = 500e-9        # wavelength (m)

# r0s = np.array([0.279, 0.416, 0.920, 0.244]) # Fried parameters to simulate (m) Median seeing at 500 nm

# r0s = r0s / Aperture_scale # need to be scaled to the test bench size

# r0s_names = ['GL1', 'GL2', 'GL3', 'FA']      # filename labels for each r0

# #r0s = r0s * (35/11) # scale to actuator count
# R_transition = (83-2*13)/83     #taper up in the outer 13 mm (one beam diameter)  # fraction of radius where r0 transitions from flat to edge (m)

# def make_radial_mask(size=4096, flat_radius_fraction=0.5, start_value=0.0, end_value=1.0):
#     """
#     Create a 2D radial mask of shape (size, size).

#     The mask is:
#       - start_value in the flat central region (r <= flat_radius)
#       - linearly ramping from start_value → end_value between flat_radius and the corner
#       - end_value beyond the corner distance

#     Parameters
#     ----------
#     size : int
#         Width and height of the square array (default 4096).
#     flat_radius_fraction : float
#         Radius of the flat region as a fraction of the half-width (default 0.5).
#     start_value : float
#         Mask value in the flat central region (default 0.0).
#     end_value : float
#         Mask value at and beyond the corners (default 1.0).
#     """
#     half = size / 2.0

#     # Pixel coordinate grids, centred at (0, 0)
#     y, x = np.ogrid[-half:half, -half:half]

#     # Normalised radius: 0 at centre, 1 at half-width of the array
#     r = np.sqrt(x**2 + y**2) / half

#     flat_r  = flat_radius_fraction          # inner edge of the ramp
#     outer_r = 1.0#np.sqrt(2.0)                  # corner of a unit square ≈ 1.414

#     # Build the mask: start_value in the flat centre, linear ramp to end_value at edges
#     t = np.clip((r - flat_r) / (outer_r - flat_r), 0.0, 1.0)  # normalised ramp 0→1
#     mask = np.where(
#         r <= flat_r,
#         start_value,                                     # flat centre
#         np.where(
#             r >= outer_r,
#             end_value,                                   # beyond corners
#             start_value + t * (end_value - start_value) # linear ramp
#         )
#     ).astype(np.float32)

#     return mask

# # ==============================
# # LOOP OVER r0 VALUES
# # ==============================

# fits_list = []
# for r0, name in zip(r0s, r0s_names):

#     r0_mm=np.round(r0*1e3, decimals=3) # convert to mm for printing
#     print(f"\n--- r0 = {r0_mm} mm ({name}) ---")
#     # ==============================
#     # GENERATE BASE PHASE SCREEN
#     # ==============================

#     if False:
        
#         L0 = 50.0          # outer scale (m)
#         l0 = 0.001         # inner scale (m)
#         phase = aotools.turbulence.phasescreen.ft_phase_screen(
#             r0=r0, N=N, delta=pixel_scale, L0=L0, l0=l0, seed=1
#         )
#     else:
        
#         #or use a custom implementation of Kolmogorov Fourier
#         # --- Generate Kolmogorov phase screen ---
#         fx = np.fft.fftfreq(N, pixel_scale)
#         fy = np.fft.fftfreq(N, pixel_scale)
#         FX, FY = np.meshgrid(fx, fy)
#         f = np.sqrt(FX**2 + FY**2)

#         PSD_phi = 0.023 * r0**(-5/3) * (f**2 + 1e-10)**(-11/6)
#         df = 1.0 / (N * pixel_scale)   # frequency bin spacing (m⁻¹)
#         rng = np.random.default_rng(2026)
#         cn = (rng.normal(size=(N,N)) + 1j*rng.normal(size=(N,N)))
#         # φ = N² · ifft2[ sqrt(S(f)) · Δf · cn ]  (corrects for numpy's 1/N² normalisation)
#         phase = np.real(np.fft.ifft2(cn * np.sqrt(PSD_phi) * df)) * N**2

#     # ==============================
#     # REMOVE TIP / TILT
#     # ==============================

#     xx = np.linspace(-1, 1, N)
#     XX, YY = np.meshgrid(xx, xx)

#     A = np.column_stack([
#         np.ones(N*N),
#         XX.flatten(),
#         YY.flatten()
#     ])

#     coeff, *_ = np.linalg.lstsq(A, phase.flatten(), rcond=None)

#     plane = (coeff[0]
#              + coeff[1]*XX
#              + coeff[2]*YY)

#     phase = phase - plane

#     # ==============================
#     # CREATE RADIAL r0 MASK
#     # ==============================
#     #flare up the edges of the phase screen to simulate a larger r0 at the edges of the plate

#     r0_edge   = r0/1.4     #40% increase    # Fried parameter at edge (m)

#     # scaling law: not implemented
#     # phase variance ∝ r0^(-5/3)
#     #scale_mask = (make_radial_mask(N, R_transition, 1, r0/r0_edge))**(5/6)
#     scale_mask = make_radial_mask(N, R_transition, 1, r0/r0_edge)
    
#     phase = phase * scale_mask

#     # ==============================
#     # RESULTS
#     # ==============================

#     #Lexitek can manufacture a range up to 30 um OPD, so we want to make sure we are within that limit.
#     OPD = (np.max(phase) - np.min(phase)) * 500e-9 / (2*np.pi) * 1e6  # convert from radians to metres at 500 nm
#     print(f"The max OPD is: {OPD:.2f} um")

#     # fig, ax = plt.subplots(figsize=(6, 5))
#     # im = ax.imshow(phase, cmap='RdBu')
#     # ax.set_title(f"Phase Screen (radians @ 500 nm) — {name} r0 = {r0_mm} mm")
#     # fig.colorbar(im, ax=ax)
#     # fig.tight_layout()
#     # fig.savefig(os.path.join(script_dir, f"phase_screen_{name}.png"), dpi=150)
#     # plt.close(fig)

#     fname = os.path.join(script_dir, f"phase_screen_{name}")
#     # np.save(f"{fname}.npy", phase)

#     hdu = fits.PrimaryHDU(phase)
#     hdu.header['LAYER'] = f'{name}'
#     hdu.header['BUNIT'] = ('rad', 'Phase in radians at 500 nm')
#     hdu.header['PIXSCALE'] = (pixel_scale, 'Plate pixel scale (m/pixel)')
#     hdu.header['r0'] = (r0, 'Fried parameter (m)')

#     fits_list.append( hdu )


# extension_hdus = []
# for hdu, name in zip(fits_list, r0s_names):
#     # Passing 'name' here explicitly sets the EXTNAME keyword
#     extension_hdu = fits.ImageHDU(data=hdu.data, header=hdu.header, name=name)
#     extension_hdus.append(extension_hdu)


# combined_hdul = fits.HDUList([fits_list[0]] + extension_hdus[1:])

# combined_hdul.writeto(f"phasescreens_{batch_name}.fits", overwrite=True)

# #hdu.writeto(f"{fname}.fits", overwrite=True)
# print(f"Saved {fname}.fits")
