import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import beam_trace as bt 

# ==========================================
# 1. ROBUST PHYSICS HELPERS
# ==========================================

def get_rotating_screen_image(elem, t, npix=256):
    r_max = elem.clear_radius
    u = np.linspace(-r_max, r_max, npix)
    uu, vv = np.meshgrid(u, u)
    uv_grid = np.stack([uu, vv], axis=-1)
    opd, valid = elem.sample_uv(uv_grid.reshape(-1, 2), t=t)
    # Use 0 instead of nan for visualization to ensure the plot renders
    return np.where(valid.reshape(uu.shape), opd.reshape(uu.shape), 0)

def simulate_sci_performance(beam, gl_corr, bench, t, pad_size, npix_pupil):
    samp = bt.sample_beam_phase_amplitude_on_pupil_plane(beam, bench, [0,0,0], t, npix_pupil)
    # FIX: Use nan_to_num to prevent NaN phase from breaking the FFT
    raw_phase = np.nan_to_num(samp["phase_map_rad"])
    residual = np.where(samp["mask"], raw_phase - gl_corr, 0)
    
    field = np.where(samp["mask"], samp["amplitude"] * np.exp(1j * residual), 0)
    padded = np.pad(field, (pad_size - npix_pupil) // 2, mode='constant')
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))))**2
    return residual, psf / np.max(psf) if np.max(psf) > 0 else psf

# ==========================================
# 2. TELEMETRY GENERATION
# ==========================================

def run_glao_telemetry(exposure_s=3.6, dt=0.1):
    WAVELENGTH, D_BEAM, NPIX_PUPIL, PAD_SIZE = 589e-9, 0.013, 256, 1024
    SCI_OFFS = [0.0, 5.0, 10.0] 
    FITS_PATH =  "/home/bbarrer/mq_glao_testbench_sim/phasescreens/batch1_test/phasescreens_median_dmScaled-1_radialScaled-0.fits" #"phasescreens_median_dmScaled-1_radialScaled-0.fits"


    with fits.open(FITS_PATH) as hdul:
        pix_scale = hdul[0].header['PIXSCALE']
        bench = bt.OpticalBench3D()
        layers = [{"lbl":"FA", "z":-2.50, "hz":0.4}, {"lbl":"GL3", "z":-0.096, "hz":1.4}, 
                  {"lbl":"GL2", "z":-0.048, "hz":1.0}, {"lbl":"GL1", "z":-0.024, "hz":0.7}]
        for l in layers:
            opd = (hdul[l["lbl"]].data * 500e-9) / (2*np.pi)
            bench.add(bt.RotatingPhaseScreen3D(point=[0,0,l["z"]], normal=[0,0,1], opd_map=opd, 
                      map_extent_m=opd.shape[0]*pix_scale, angular_velocity=2*np.pi*l["hz"], label=l["lbl"]))

    lgs_coords = [(10,10), (-10,10), (10,-10), (-10,-10)]
    lgs_beams = [bt.make_converging_beam_from_field_angles(np.deg2rad(x/60), np.deg2rad(y/60), -3.25, [0,0,0], D_BEAM, WAVELENGTH, "LGS", 3, 12) for x,y in lgs_coords]
    sci_beams = [bt.make_converging_beam_from_field_angles(np.deg2rad(th/60), 0, -3.25, [0,0,0], D_BEAM, WAVELENGTH, f"Sci", 3, 12) for th in SCI_OFFS]

    telemetry = {"times": np.arange(0, exposure_s, dt), "bench": bench, "frames": [], 
                 "sci_angles": SCI_OFFS, "lgs_beams": lgs_beams, "sci_beams": sci_beams}
    
    for t in telemetry["times"]:
        frame = {"t": t}
        # FIX: Use nanmean so that a single missed ray doesn't kill the average
        lgs_samps = [bt.sample_beam_phase_amplitude_on_pupil_plane(b, bench, [0,0,0], t, NPIX_PUPIL) for b in lgs_beams]
        avg_lgs_phase = np.nanmean([np.nan_to_num(s["phase_map_rad"]) for s in lgs_samps], axis=0)
        
        sigma = (NPIX_PUPIL / 35) * 0.5
        dm_shape = gaussian_filter(avg_lgs_phase, sigma=sigma, mode='reflect')
        dm_shape = np.where(lgs_samps[0]["mask"], dm_shape, 0)
        
        frame["recon"], frame["dm"] = avg_lgs_phase, dm_shape
        frame["sci"] = [simulate_sci_performance(b, dm_shape, bench, t, PAD_SIZE, NPIX_PUPIL) for b in sci_beams]
        telemetry["frames"].append(frame)
    return telemetry

# ==========================================
# 3. FIXED MOVIE FUNCTION
# ==========================================

def make_movie(tel, base_filename="glao_telemetry"):
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(4, 4, width_ratios=[1.3, 1, 1, 1])
    
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_scr = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    ax_pup = [fig.add_subplot(gs[i, 2]) for i in range(3)] 
    ax_psf = [fig.add_subplot(gs[i, 3]) for i in range(3)]

    # Draw the 3D Trace ONCE outside the update loop to keep it persistent
    for b in tel["lgs_beams"]:
        paths, _ = tel["bench"].trace_beam(b, s_end=0.1, t=0.0)
        for p in paths: ax3d.plot(p[:,0], p[:,1], p[:,2], lw=0.8, alpha=0.5, color='orange')
    for elem in tel["bench"].elements:
        e1, e2 = elem.plane_basis()
        rad = elem.clear_radius if isinstance(elem, bt.RotatingPhaseScreen3D) else 0.001
        circ = np.linspace(0, 2*np.pi, 100)
        ring = elem.point[None,:] + rad*(np.cos(circ)[:,None]*e1 + np.sin(circ)[:,None]*e2)
        ax3d.plot(ring[:,0], ring[:,1], ring[:,2], 'k-', lw=1.5)
        ax3d.text(elem.point[0], elem.point[1], elem.point[2], elem.label)
    ax3d.set_box_aspect([1,1,1.5])

    def update(idx):
        f = tel["frames"][idx]
        [ax.clear() for ax in ax_scr + ax_pup + ax_psf]
        
        # Col 2: Screens & Overlays
        for i, elem in enumerate(tel["bench"].elements):
            img = get_rotating_screen_image(elem, f["t"])
            ax_scr[i].imshow(img, cmap='RdBu', origin='lower', extent=[-15, 15, -15, 15])
            ax_scr[i].set_title(elem.label)
            for b in tel["lgs_beams"]:
                inter = tel["bench"].trace_chief_intersections(b, t=f["t"])
                if elem.label in inter:
                    u, v = elem.local_coordinates(inter[elem.label]["point"])
                    ax_scr[i].add_patch(plt.Circle((u*1000, v*1000), 6.5, color='red', fill=False))
            for b in tel["sci_beams"]:
                inter = tel["bench"].trace_chief_intersections(b, t=f["t"])
                if elem.label in inter:
                    u, v = elem.local_coordinates(inter[elem.label]["point"])
                    ax_scr[i].scatter(u*1000, v*1000, marker='x', color='white', s=20)
            ax_scr[i].axis('off')

        # Col 3: Pupil
        # Apply mask to everything in Column 3 for visual consistency
        mask = np.nan_to_num(tel["frames"][0]["dm"]) != 0
        ax_pup[0].imshow(np.where(mask, f["recon"], np.nan), cmap='viridis'); ax_pup[0].set_title("Recon")
        ax_pup[1].imshow(np.where(mask, f["dm"], np.nan), cmap='viridis'); ax_pup[1].set_title("DM Shape")
        ax_pup[2].imshow(np.where(mask, f["sci"][2][0], np.nan), cmap='viridis'); ax_pup[2].set_title("Resid (10')")
        for ax in ax_pup: ax.axis('off')

        # Col 4: PSFs
        lim = int(6 * 8)
        for i, (res, psf) in enumerate(f["sci"]):
            # FIX: Robust centering even if PSF is very broken
            cy, cx = 512, 512 # Default to center
            if np.max(psf) > 0:
                cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
            
            crop = psf[cy-lim:cy+lim, cx-lim:cx+lim]
            # FIX: Use explicit vmin/vmax to force visibility against the background
            ax_psf[i].imshow(np.log10(np.maximum(crop, 1e-4)), cmap='magma', 
                             origin='lower', vmin=-4, vmax=0)
            ax_psf[i].set_title(f"PSF {tel['sci_angles'][i]}'")
            ax_psf[i].axis('off')

        fig.suptitle(f"t = {f['t']:.2f} s")

    ani = FuncAnimation(fig, update, frames=len(tel["frames"]), interval=250)
    
    try:
        ani.save(f"{base_filename}.mp4", writer='ffmpeg')
    except:
        ani.save(f"{base_filename}.gif", writer='pillow')
    plt.close()

tel_data = run_glao_telemetry(exposure_s=3.6, dt=0.1)
make_movie(tel_data)

