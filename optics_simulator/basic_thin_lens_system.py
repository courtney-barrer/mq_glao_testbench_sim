#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 02:52:39 2023

@author: bcourtne

basic 2D thin lens system simulation 
"""

import numpy as np
import matplotlib.pyplot as plt 


class lens:
    def __init__(self,x,f):
        self.f = f
        self.x = x
        
    def get_image(self,_object):
        #returns a new object in the image plane
        do = (self.x - _object.x)
        if do != self.f:
            di = do * self.f / (do - self.f)
        else:
            di = np.inf
        
        M = -di/do
        
        i = obj(x = self.x + di, h = M * _object.h)
        
        return( i )
    
class obj:
    def __init__(self,x,h):
        self.x = x
        self.h = h
        
        
def propagate_object(lens_list, _object, plot=True):
    
    object_list = [ _object ]
    
    if plot:
        plt.figure()
        plt.axhline(0,color='k',lw=2)
        plt.plot( _object.x, _object.h ,'x' ,color='g',label='input object')

    for i,l in  enumerate(lens_list):
        next_object = l.get_image(object_list[-1])
        object_list.append( next_object )
        
        if plot:
            plt.axvline(l.x,linestyle=':',color='k',label=f'lens {i+1}, f={l.f}')
        
    if plot:
        plt.plot( object_list[-1].x, object_list[-1].h ,'x' ,color='b',label='object image')
        plt.legend()
        plt.xlabel( 'x' )
        plt.ylabel( 'y' )
        plt.show()
    return( object_list )
        

def plot_sys(sys, object_list, plot_initial_object=False):
    plt.figure()
    plt.axhline(0,color='k',lw=2)
    if plot_initial_object:
        plt.plot( object_list[0].x, object_list[0].h ,'x' ,color='g',label='input object')
    for i,l in enumerate(sys):
        plt.axvline(l.x,linestyle=':',color='k',label=f'lens {i+1}, f={l.f}')
    plt.plot( object_list[-1].x, object_list[-1].h ,'x' ,color='b',label='object image')
    plt.legend()
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )

    plt.show()



def main():
    """
    Single-LGS 2D thin-lens design tracker.

    Optical sequence
    ----------------
    finite LGS source
    -> input pupil at z=0
    -> Lens 1 / Lens 2 common pupil relay
    -> DM pupil image
    -> Lens 3 forms an off-axis focused LGS image after the DM
    -> LGS pickoff image plane
    -> Lens 4 reimages the pupil onto the SH-WFS lenslet plane
    -> SH-WFS pupil image plane

    Units
    -----
    z, x, focal lengths : mm
    angles             : radians
    """

    # ============================================================
    # Editable design inputs
    # ============================================================

    pupil_diameter_mm = 13.5
    pupil_radius_mm = 0.5 * pupil_diameter_mm

    lgs_angle_arcmin = -10.0
    lgs_source_distance_mm = 3250.0

    # Desired focused LGS image position after the DM.
    # This is where a future pickoff mirror would be centred.
    desired_pickoff_offset_mm = +5.0

    # Common relay: input pupil -> DM pupil image.
    z_lens1_mm = 150.0
    f_lens1_mm = 150.0

    z_lens2_mm = 450.0
    f_lens2_mm = 150.0

    z_dm_mm = 600.0

    # Lens 3 forms the LGS image after the DM.
    z_lens3_mm = 750.0

    # Lens 4 relays the pupil after the LGS pickoff plane to the SH-WFS.
    lens4_after_pickoff_mm = 150.0

    # Desired pupil diameter on the SH-WFS lenslet array.
    target_sh_pupil_diameter_mm = 6.0
    target_sh_pupil_magnification = target_sh_pupil_diameter_mm / pupil_diameter_mm

    # Extra fiducials after SH-WFS for context only.
    downstream_fiducials_after_sh_mm = [150.0, 300.0]

    # Aperture radii for first-order checks.
    lens1_ap_radius_mm = 25.0
    lens2_ap_radius_mm = 25.0
    lens3_ap_radius_mm = 25.0
    lens4_ap_radius_mm = 25.0
    dm_ap_radius_mm = pupil_radius_mm

    # Pickoff is intentionally off-axis and centred on the LGS image.
    pickoff_plane_ap_radius_mm = 2.0

    # Give the SH plane a tiny practical margin over the exact target radius.
    sh_plane_ap_radius_mm = 0.5 * target_sh_pupil_diameter_mm + 0.05

    clearance_tol_mm = 1e-6

    # ============================================================
    # Initial finite-LGS geometry at input pupil
    # ============================================================

    theta_chief = np.deg2rad(lgs_angle_arcmin / 60.0)

    z_source = -lgs_source_distance_mm

    # Source x chosen so chief ray crosses x=0 at z=0.
    x_source = -theta_chief * lgs_source_distance_mm

    # Ray state is [x, theta] at z=0.
    chief0 = np.array([0.0, theta_chief], dtype=float)

    top0 = np.array(
        [
            +pupil_radius_mm,
            (+pupil_radius_mm - x_source) / (0.0 - z_source),
        ],
        dtype=float,
    )

    bottom0 = np.array(
        [
            -pupil_radius_mm,
            (-pupil_radius_mm - x_source) / (0.0 - z_source),
        ],
        dtype=float,
    )

    # ============================================================
    # ABCD helpers
    # ============================================================

    def free_space(d_mm):
        return np.array([[1.0, d_mm], [0.0, 1.0]], dtype=float)

    def thin_lens_matrix(f_mm):
        return np.array([[1.0, 0.0], [-1.0 / f_mm, 1.0]], dtype=float)

    def crossing_distance(ray_a, ray_b):
        xa, ta = ray_a
        xb, tb = ray_b
        denom = ta - tb

        if abs(denom) < 1e-15:
            return np.inf

        return -(xa - xb) / denom

    def pupil_conjugate_distance_after_plane(M):
        """
        Distance after current plane where the input pupil is reimaged.

        M maps the input pupil plane to the current plane.
        After free propagation d:
            B_new = B + dD
        Pupil conjugate condition:
            B_new = 0
        """
        B = M[0, 1]
        D = M[1, 1]

        if abs(D) < 1e-15:
            return np.inf

        return -B / D

    def image_from_object_thin_lens(z_object, z_lens, f):
        s = z_lens - z_object

        if abs(s - f) < 1e-15:
            return np.inf, np.inf

        sp = s * f / (s - f)
        mag = -sp / s

        return z_lens + sp, mag

    def trace_to_plane(layout, z_target):
        """
        Trace chief/top/bottom rays and ABCD matrix to z_target through all
        lenses in layout at or before z_target.
        """
        M = np.eye(2)

        r_chief = chief0.copy()
        r_top = top0.copy()
        r_bottom = bottom0.copy()

        z_prev = 0.0

        for elem in layout:
            z = float(elem["z"])

            if z > z_target + 1e-12:
                break

            dz = z - z_prev

            if dz < -1e-12:
                raise ValueError("Layout z positions must be monotonically increasing.")

            P = free_space(dz)

            M = P @ M
            r_chief = P @ r_chief
            r_top = P @ r_top
            r_bottom = P @ r_bottom

            if elem["type"].lower() == "lens":
                L = thin_lens_matrix(float(elem["f"]))

                M = L @ M
                r_chief = L @ r_chief
                r_top = L @ r_top
                r_bottom = L @ r_bottom

            z_prev = z

        dz = z_target - z_prev

        if dz < -1e-12:
            raise ValueError("z_target is upstream of latest traced element.")

        P = free_space(dz)

        M = P @ M
        r_chief = P @ r_chief
        r_top = P @ r_top
        r_bottom = P @ r_bottom

        return M, r_chief, r_top, r_bottom

    def solve_post_dm_lens_for_lgs_image(
        r_chief_pre,
        r_top_pre,
        r_bottom_pre,
        desired_x_img,
    ):
        """
        Solve Lens 3 focal length so the finite LGS beam focuses after Lens 3
        with chief-ray image height desired_x_img.
        """
        xc, tc = r_chief_pre
        xt, tt = r_top_pre
        xb, tb = r_bottom_pre

        dx = xt - xb
        dt = tt - tb
        X = desired_x_img

        if abs(dx) < 1e-15:
            raise ValueError("Top and bottom ray heights are identical at Lens 3.")

        if abs(X) < 1e-15:
            raise ValueError("desired_pickoff_offset_mm cannot be zero for this solver.")

        # Derived from:
        #   d = -dx / (dt - q dx)
        #   X = xc + d * (tc - q xc)
        # where q = 1/f.
        q = ((X - xc) * dt + dx * tc) / (dx * X)

        if abs(q) < 1e-15:
            raise ValueError("Solved Lens 3 has infinite focal length.")

        f = 1.0 / q

        L = thin_lens_matrix(f)

        rc = L @ r_chief_pre
        rt = L @ r_top_pre
        rb = L @ r_bottom_pre

        d_img = crossing_distance(rt, rb)

        if not np.isfinite(d_img):
            raise ValueError("Solved Lens 3 does not produce a finite image.")

        if d_img <= 0:
            raise ValueError(
                "Solved LGS image is upstream of Lens 3. "
                "Try desired_pickoff_offset_mm with the opposite sign."
            )

        x_check = rc[0] + d_img * rc[1]

        if abs(x_check - X) > 1e-6:
            raise RuntimeError("Internal Lens 3 solve check failed.")

        return f, d_img

    def solve_single_lens_pupil_relay(M_pre_lens, m_target):
        """
        Solve one lens plus downstream free-space distance so the input pupil
        is reimaged with transverse magnification m_target.

        M_pre_lens maps input pupil -> just before lens.

        We solve for q=1/f and d such that:

            M_total = P(d) @ L(q) @ M_pre_lens

        has:
            B_total = 0
            A_total = m_target
        """
        A, B = M_pre_lens[0, 0], M_pre_lens[0, 1]
        C, D = M_pre_lens[1, 0], M_pre_lens[1, 1]

        if abs(B) < 1e-15:
            raise ValueError("Cannot solve pupil relay: B≈0 at Lens 4.")
        if abs(m_target) < 1e-15:
            raise ValueError("Target pupil magnification cannot be zero.")

        q = (B * C + (m_target - A) * D) / (m_target * B)

        if abs(q) < 1e-15:
            raise ValueError("Solved Lens 4 has infinite focal length.")

        f = 1.0 / q

        denom = D - q * B

        if abs(denom) < 1e-15:
            raise ValueError("Solved SH-WFS distance is singular.")

        d = -B / denom

        if d <= 0:
            raise ValueError(
                "Solved SH-WFS pupil image is upstream of Lens 4. "
                "Try moving Lens 4 or changing target magnification sign."
            )

        return f, d

    # ============================================================
    # Common relay up to the DM
    # ============================================================

    z_lgs_image_after_l1, mag_lgs_l1 = image_from_object_thin_lens(
        z_object=z_source,
        z_lens=z_lens1_mm,
        f=f_lens1_mm,
    )

    common_layout = [
        dict(
            label="Input pupil",
            type="Pupil",
            z=0.0,
            f=None,
            ap_radius=pupil_radius_mm,
            ap_center_mm=0.0,
        ),
        dict(
            label="Lens 1: pupil relay A",
            type="Lens",
            z=z_lens1_mm,
            f=f_lens1_mm,
            ap_radius=lens1_ap_radius_mm,
            ap_center_mm=0.0,
        ),
        dict(
            label="Intermediate LGS image",
            type="Fiducial",
            z=z_lgs_image_after_l1,
            f=None,
            ap_radius=2.0,
            ap_center_mm=mag_lgs_l1 * x_source,
        ),
        dict(
            label="Lens 2: pupil relay B",
            type="Lens",
            z=z_lens2_mm,
            f=f_lens2_mm,
            ap_radius=lens2_ap_radius_mm,
            ap_center_mm=0.0,
        ),
        dict(
            label="DM pupil image",
            type="DM",
            z=z_dm_mm,
            f=None,
            ap_radius=dm_ap_radius_mm,
            ap_center_mm=0.0,
        ),
    ]

    # ============================================================
    # Solve Lens 3 for post-DM LGS pickoff image
    # ============================================================

    _, r_chief_l3_pre, r_top_l3_pre, r_bottom_l3_pre = trace_to_plane(
        common_layout,
        z_lens3_mm,
    )

    f_lens3_mm, pickoff_after_lens3_mm = solve_post_dm_lens_for_lgs_image(
        r_chief_pre=r_chief_l3_pre,
        r_top_pre=r_top_l3_pre,
        r_bottom_pre=r_bottom_l3_pre,
        desired_x_img=desired_pickoff_offset_mm,
    )

    z_pickoff_mm = z_lens3_mm + pickoff_after_lens3_mm

    if z_pickoff_mm <= z_dm_mm:
        raise RuntimeError("Solved pickoff plane is not after the DM; check layout.")

    # ============================================================
    # Solve Lens 4 for SH-WFS pupil image
    # ============================================================

    z_lens4_mm = z_pickoff_mm + lens4_after_pickoff_mm

    layout_to_lens4 = common_layout + [
        dict(
            label="Lens 3: post-DM LGS focus",
            type="Lens",
            z=z_lens3_mm,
            f=f_lens3_mm,
            ap_radius=lens3_ap_radius_mm,
            ap_center_mm=0.0,
        ),
        dict(
            label="LGS pickoff image plane",
            type="Pickoff/Fiducial",
            z=z_pickoff_mm,
            f=None,
            ap_radius=pickoff_plane_ap_radius_mm,
            ap_center_mm=desired_pickoff_offset_mm,
        ),
    ]

    layout_to_lens4 = sorted(layout_to_lens4, key=lambda e: float(e["z"]))

    M_pre_lens4, _, _, _ = trace_to_plane(layout_to_lens4, z_lens4_mm)

    f_lens4_mm, sh_after_lens4_mm = solve_single_lens_pupil_relay(
        M_pre_lens=M_pre_lens4,
        m_target=target_sh_pupil_magnification,
    )

    z_sh_mm = z_lens4_mm + sh_after_lens4_mm

    # ============================================================
    # Final layout
    # ============================================================

    layout = layout_to_lens4 + [
        dict(
            label="Lens 4: SH pupil relay",
            type="Lens",
            z=z_lens4_mm,
            f=f_lens4_mm,
            ap_radius=lens4_ap_radius_mm,
            ap_center_mm=0.0,
        ),
        dict(
            label="SH-WFS lenslet pupil plane",
            type="SH-WFS/Pupil",
            z=z_sh_mm,
            f=None,
            ap_radius=sh_plane_ap_radius_mm,
            ap_center_mm=0.0,
        ),
    ]

    for i, dz in enumerate(downstream_fiducials_after_sh_mm):
        layout.append(
            dict(
                label=f"Downstream fiducial {i + 1}",
                type="Fiducial",
                z=z_sh_mm + dz,
                f=None,
                ap_radius=25.0,
                ap_center_mm=0.0,
            )
        )

    layout = sorted(layout, key=lambda e: float(e["z"]))

    # ============================================================
    # Trace final layout
    # ============================================================

    rows = []

    M = np.eye(2)

    r_chief = chief0.copy()
    r_top = top0.copy()
    r_bottom = bottom0.copy()

    z_prev = layout[0]["z"]

    for i, elem in enumerate(layout):
        z = float(elem["z"])

        if i == 0:
            dz = 0.0
        else:
            dz = z - z_prev

        if dz < -1e-12:
            raise ValueError("Layout z positions must be monotonically increasing.")

        P = free_space(dz)

        M = P @ M
        r_chief = P @ r_chief
        r_top = P @ r_top
        r_bottom = P @ r_bottom

        if elem["type"].lower() == "lens":
            L = thin_lens_matrix(float(elem["f"]))

            M = L @ M
            r_chief = L @ r_chief
            r_top = L @ r_top
            r_bottom = L @ r_bottom

        beam_center = 0.5 * (r_top[0] + r_bottom[0])
        beam_radius = 0.5 * abs(r_top[0] - r_bottom[0])

        d_img = crossing_distance(r_top, r_bottom)
        z_img = z + d_img if np.isfinite(d_img) else np.inf
        x_img_chief = r_chief[0] + d_img * r_chief[1] if np.isfinite(d_img) else np.nan

        d_pup = pupil_conjugate_distance_after_plane(M)
        z_pup = z + d_pup if np.isfinite(d_pup) else np.inf

        ap_radius = float(elem.get("ap_radius", np.nan))
        ap_center = float(elem.get("ap_center_mm", 0.0))

        max_abs_ray_height_relative_to_aperture = max(
            abs(r_chief[0] - ap_center),
            abs(r_top[0] - ap_center),
            abs(r_bottom[0] - ap_center),
        )

        clearance = ap_radius - max_abs_ray_height_relative_to_aperture

        rows.append(
            dict(
                step=i,
                label=elem["label"],
                type=elem["type"],
                z_mm=z,
                dz_mm=dz,
                f_mm=elem["f"],
                A=M[0, 0],
                B=M[0, 1],
                C=M[1, 0],
                D=M[1, 1],
                chief_x_mm=r_chief[0],
                chief_theta_mrad=1e3 * r_chief[1],
                top_x_mm=r_top[0],
                top_theta_mrad=1e3 * r_top[1],
                bottom_x_mm=r_bottom[0],
                bottom_theta_mrad=1e3 * r_bottom[1],
                beam_center_mm=beam_center,
                beam_radius_mm=beam_radius,
                image_d_after_mm=d_img,
                image_z_mm=z_img,
                image_chief_x_mm=x_img_chief,
                pupil_d_after_mm=d_pup,
                pupil_z_mm=z_pup,
                ap_radius_mm=ap_radius,
                ap_center_mm=ap_center,
                clearance_mm=clearance,
            )
        )

        z_prev = z

    # ============================================================
    # Print summary
    # ============================================================

    print()
    print("Single-LGS 2D thin-lens system: post-DM pickoff + SH pupil relay")
    print("=" * 98)
    print(f"Input pupil diameter:              {pupil_diameter_mm:.3f} mm")
    print(f"LGS field angle:                   {lgs_angle_arcmin:.3f} arcmin")
    print(f"LGS chief angle:                   {theta_chief * 1e3:.6f} mrad")
    print(f"LGS source z:                      {z_source:.3f} mm")
    print(f"LGS source x:                      {x_source:+.3f} mm")
    print()
    print("Common pupil relay")
    print(f"  Lens 1: z={z_lens1_mm:.3f} mm, f={f_lens1_mm:.3f} mm")
    print(f"  Lens 2: z={z_lens2_mm:.3f} mm, f={f_lens2_mm:.3f} mm")
    print(f"  DM pupil image z:                {z_dm_mm:.3f} mm")
    print(f"  Intermediate LGS image z:        {z_lgs_image_after_l1:.3f} mm")
    print(f"  Intermediate LGS image x:        {mag_lgs_l1 * x_source:+.3f} mm")
    print()
    print("Post-DM LGS pickoff")
    print(f"  desired pickoff image x:         {desired_pickoff_offset_mm:+.3f} mm")
    print(f"  Lens 3 z:                        {z_lens3_mm:.3f} mm")
    print(f"  solved Lens 3 focal length:      {f_lens3_mm:.3f} mm")
    print(f"  pickoff distance after Lens 3:   {pickoff_after_lens3_mm:.3f} mm")
    print(f"  pickoff plane z:                 {z_pickoff_mm:.3f} mm")
    print(f"  pickoff after DM by:             {z_pickoff_mm - z_dm_mm:.3f} mm")
    print()
    print("SH-WFS pupil relay")
    print(f"  target SH pupil diameter:        {target_sh_pupil_diameter_mm:.3f} mm")
    print(f"  target pupil magnification:      {target_sh_pupil_magnification:+.6f}")
    print(f"  Lens 4 z:                        {z_lens4_mm:.3f} mm")
    print(f"  solved Lens 4 focal length:      {f_lens4_mm:.3f} mm")
    print(f"  SH distance after Lens 4:        {sh_after_lens4_mm:.3f} mm")
    print(f"  SH-WFS pupil plane z:            {z_sh_mm:.3f} mm")
    print()

    header = (
        f"{'i':>2s} {'label':32s} {'type':16s} "
        f"{'z':>10s} {'f':>10s} "
        f"{'chief x':>10s} {'radius':>10s} "
        f"{'ap ctr':>10s} {'clear':>10s}"
    )

    print(header)
    print("-" * len(header))

    for r in rows:
        f_txt = "" if r["f_mm"] is None else f"{r['f_mm']:.3f}"

        print(
            f"{r['step']:2d} "
            f"{r['label'][:32]:32s} "
            f"{r['type'][:16]:16s} "
            f"{r['z_mm']:10.3f} "
            f"{f_txt:>10s} "
            f"{r['chief_x_mm']:10.3f} "
            f"{r['beam_radius_mm']:10.3f} "
            f"{r['ap_center_mm']:10.3f} "
            f"{r['clearance_mm']:10.3f}"
        )

    dm_rows = [r for r in rows if r["type"].lower() == "dm"]
    pickoff_rows = [r for r in rows if "pickoff" in r["type"].lower()]
    sh_rows = [r for r in rows if "sh-wfs" in r["type"].lower()]

    if dm_rows:
        dm = dm_rows[0]
        dm_diameter = 2.0 * dm["beam_radius_mm"]

        print()
        print("DM pupil image check")
        print(f"  traced DM pupil diameter:         {dm_diameter:.3f} mm")
        print(f"  target DM pupil diameter:         {pupil_diameter_mm:.3f} mm")
        print(f"  diameter error:                   {dm_diameter - pupil_diameter_mm:+.6f} mm")
        print(f"  DM chief x:                       {dm['chief_x_mm']:+.6f} mm")

    if pickoff_rows:
        p = pickoff_rows[0]

        print()
        print("LGS pickoff image check")
        print(f"  pickoff z:                        {p['z_mm']:.3f} mm")
        print(f"  pickoff chief x:                  {p['chief_x_mm']:+.6f} mm")
        print(f"  pickoff aperture centre:          {p['ap_center_mm']:+.6f} mm")
        print(f"  requested chief x:                {desired_pickoff_offset_mm:+.6f} mm")
        print(f"  geometric beam radius at pickoff: {p['beam_radius_mm']:.6e} mm")
        print(f"  aperture clearance at pickoff:    {p['clearance_mm']:+.6f} mm")
        print(f"  pickoff after DM?                 {p['z_mm'] > z_dm_mm}")

        two_lgs_sep = 2.0 * abs(p["chief_x_mm"])
        print(f"  2D +/-10 arcmin image separation: {two_lgs_sep:.3f} mm")

    if sh_rows:
        s = sh_rows[0]
        sh_diameter = 2.0 * s["beam_radius_mm"]

        print()
        print("SH-WFS pupil image check")
        print(f"  SH-WFS z:                         {s['z_mm']:.3f} mm")
        print(f"  SH-WFS chief x:                   {s['chief_x_mm']:+.6f} mm")
        print(f"  traced SH pupil diameter:         {sh_diameter:.3f} mm")
        print(f"  target SH pupil diameter:         {target_sh_pupil_diameter_mm:.3f} mm")
        print(f"  diameter error:                   {sh_diameter - target_sh_pupil_diameter_mm:+.6f} mm")
        print(f"  aperture clearance at SH plane:   {s['clearance_mm']:+.6f} mm")

    print()
    print("Aperture clearance check")
    for r in rows:
        status = "OK" if r["clearance_mm"] >= -clearance_tol_mm else "CLIPPING"
        print(
            f"  {r['label'][:34]:34s}: "
            f"centre={r['ap_center_mm']:+8.3f} mm | "
            f"clearance={r['clearance_mm']:+10.6f} mm -> {status}"
        )

    # ============================================================
    # Plot ray trace
    # ============================================================

    z_vals = np.array([r["z_mm"] for r in rows])
    chief_x = np.array([r["chief_x_mm"] for r in rows])
    top_x = np.array([r["top_x_mm"] for r in rows])
    bottom_x = np.array([r["bottom_x_mm"] for r in rows])
    radius = np.array([r["beam_radius_mm"] for r in rows])
    centre = np.array([r["beam_center_mm"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)

    ax = axes[0]
    ax.plot(z_vals, chief_x, "o-", label="chief ray")
    ax.plot(z_vals, top_x, "o-", label="top ray")
    ax.plot(z_vals, bottom_x, "o-", label="bottom ray")
    ax.fill_between(
        z_vals,
        bottom_x,
        top_x,
        alpha=0.15,
        label="beam envelope",
    )


    fs = 14
    for r in rows:
        rtype = r["type"].lower()

        if rtype == "lens":
            ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.35)
            ax.text(
                r["z_mm"],
                ax.get_ylim()[1],
                r["label"],
                rotation=90,
                va="top",
                ha="right",
                fontsize=fs,
            )

        elif "pickoff" in rtype:
            ax.axvline(r["z_mm"], color="tab:red", linestyle="-.", alpha=0.85)
            ax.scatter(
                [r["z_mm"]],
                [r["ap_center_mm"]],
                marker="s",
                s=60,
                color="tab:red",
                label="pickoff aperture centre",
                zorder=5,
            )
            ax.text(
                r["z_mm"],
                ax.get_ylim()[1],
                "LGS pickoff plane",
                rotation=90,
                va="top",
                ha="left",
                fontsize=fs,
                color="tab:red",
            )

        elif rtype == "dm":
            ax.axvline(r["z_mm"], color="tab:blue", linestyle="-.", alpha=0.65)
            ax.text(
                r["z_mm"],
                ax.get_ylim()[1],
                "DM pupil image",
                rotation=90,
                va="top",
                ha="left",
                fontsize=fs,
                color="tab:blue",
            )

        elif "sh-wfs" in rtype:
            ax.axvline(r["z_mm"], color="tab:purple", linestyle="-.", alpha=0.85)
            ax.text(
                r["z_mm"],
                ax.get_ylim()[1],
                "SH-WFS pupil",
                rotation=90,
                va="top",
                ha="left",
                fontsize=fs,
                color="tab:purple",
            )

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.35)
    ax.set_ylabel("x [mm]")
    ax.set_title("Single-LGS 2D ray trace: DM, off-axis pickoff, then SH-WFS pupil image")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(z_vals, centre, "o-", label="beam centre")
    ax.plot(z_vals, radius, "o-", label="beam radius")
    ax.axhline(
        pupil_radius_mm,
        color="k",
        linestyle=":",
        alpha=0.5,
        label="input pupil radius",
    )
    ax.axhline(
        0.5 * target_sh_pupil_diameter_mm,
        color="tab:purple",
        linestyle=":",
        alpha=0.6,
        label="target SH pupil radius",
    )

    for r in rows:
        rtype = r["type"].lower()
        if "pickoff" in rtype:
            ax.axvline(r["z_mm"], color="tab:red", linestyle="-.", alpha=0.85)
        elif rtype == "dm":
            ax.axvline(r["z_mm"], color="tab:blue", linestyle="-.", alpha=0.65)
        elif "sh-wfs" in rtype:
            ax.axvline(r["z_mm"], color="tab:purple", linestyle="-.", alpha=0.85)
        elif rtype == "lens":
            ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.25)

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x / radius [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    return rows


if __name__ == "__main__":
    main()

# def main():
#     """
#     Single-LGS 2D thin-lens design tracker.

#     Optical sequence
#     ----------------
#     finite LGS source
#     -> input pupil at z=0
#     -> Lens 1 / Lens 2 common pupil relay
#     -> DM pupil image
#     -> Lens 3 forms an off-axis focused LGS image after the DM
#     -> LGS pickoff image plane
#     -> Lens 4 reimages the pupil onto the SH-WFS lenslet plane
#     -> SH-WFS pupil image plane

#     Units
#     -----
#     z, x, focal lengths : mm
#     angles             : radians
#     """

#     # ============================================================
#     # Editable design inputs
#     # ============================================================

#     pupil_diameter_mm = 13.5
#     pupil_radius_mm = 0.5 * pupil_diameter_mm

#     lgs_angle_arcmin = -10.0
#     lgs_source_distance_mm = 3250.0

#     # Desired focused LGS image position after the DM.
#     # This is the field-image pickoff location.
#     desired_pickoff_offset_mm = +5.0

#     # Common relay: input pupil -> DM pupil image.
#     z_lens1_mm = 150.0
#     f_lens1_mm = 150.0

#     z_lens2_mm = 450.0
#     f_lens2_mm = 150.0

#     z_dm_mm = 600.0

#     # Lens 3 forms the LGS image after the DM.
#     z_lens3_mm = 750.0

#     # Lens 4 relays the pupil after the LGS pickoff plane to the SH-WFS.
#     lens4_after_pickoff_mm = 150.0

#     # Desired pupil diameter on the SH-WFS lenslet array.
#     target_sh_pupil_diameter_mm = 6.0
#     target_sh_pupil_magnification = target_sh_pupil_diameter_mm / pupil_diameter_mm

#     # Extra fiducials after SH-WFS for context only.
#     downstream_fiducials_after_sh_mm = [150.0, 300.0]

#     # Aperture radii for first-order checks.
#     lens1_ap_radius_mm = 25.0
#     lens2_ap_radius_mm = 25.0
#     lens3_ap_radius_mm = 25.0
#     lens4_ap_radius_mm = 25.0
#     dm_ap_radius_mm = pupil_radius_mm
#     pickoff_plane_ap_radius_mm = 2.0
#     sh_plane_ap_radius_mm = 0.5 * target_sh_pupil_diameter_mm

#     # ============================================================
#     # Initial finite-LGS geometry at input pupil
#     # ============================================================

#     theta_chief = np.deg2rad(lgs_angle_arcmin / 60.0)

#     z_source = -lgs_source_distance_mm

#     # Source x chosen so chief ray crosses x=0 at z=0.
#     x_source = -theta_chief * lgs_source_distance_mm

#     # Ray state is [x, theta] at z=0.
#     chief0 = np.array([0.0, theta_chief], dtype=float)

#     top0 = np.array(
#         [
#             +pupil_radius_mm,
#             (+pupil_radius_mm - x_source) / (0.0 - z_source),
#         ],
#         dtype=float,
#     )

#     bottom0 = np.array(
#         [
#             -pupil_radius_mm,
#             (-pupil_radius_mm - x_source) / (0.0 - z_source),
#         ],
#         dtype=float,
#     )

#     # ============================================================
#     # ABCD helpers
#     # ============================================================

#     def free_space(d_mm):
#         return np.array([[1.0, d_mm], [0.0, 1.0]], dtype=float)

#     def thin_lens_matrix(f_mm):
#         return np.array([[1.0, 0.0], [-1.0 / f_mm, 1.0]], dtype=float)

#     def crossing_distance(ray_a, ray_b):
#         xa, ta = ray_a
#         xb, tb = ray_b
#         denom = ta - tb

#         if abs(denom) < 1e-15:
#             return np.inf

#         return -(xa - xb) / denom

#     def pupil_conjugate_distance_after_plane(M):
#         """
#         Distance after current plane where the input pupil is reimaged.

#         M maps the input pupil plane to the current plane.
#         After free propagation d:
#             B_new = B + dD
#         Pupil conjugate condition:
#             B_new = 0
#         """
#         B = M[0, 1]
#         D = M[1, 1]

#         if abs(D) < 1e-15:
#             return np.inf

#         return -B / D

#     def image_from_object_thin_lens(z_object, z_lens, f):
#         s = z_lens - z_object

#         if abs(s - f) < 1e-15:
#             return np.inf, np.inf

#         sp = s * f / (s - f)
#         mag = -sp / s

#         return z_lens + sp, mag

#     def trace_to_plane(layout, z_target):
#         """
#         Trace chief/top/bottom rays and ABCD matrix to z_target through all
#         lenses in layout at or before z_target.
#         """
#         M = np.eye(2)

#         r_chief = chief0.copy()
#         r_top = top0.copy()
#         r_bottom = bottom0.copy()

#         z_prev = 0.0

#         for elem in layout:
#             z = float(elem["z"])

#             if z > z_target + 1e-12:
#                 break

#             dz = z - z_prev

#             if dz < -1e-12:
#                 raise ValueError("Layout z positions must be monotonically increasing.")

#             P = free_space(dz)

#             M = P @ M
#             r_chief = P @ r_chief
#             r_top = P @ r_top
#             r_bottom = P @ r_bottom

#             if elem["type"].lower() == "lens":
#                 L = thin_lens_matrix(float(elem["f"]))

#                 M = L @ M
#                 r_chief = L @ r_chief
#                 r_top = L @ r_top
#                 r_bottom = L @ r_bottom

#             z_prev = z

#         dz = z_target - z_prev

#         if dz < -1e-12:
#             raise ValueError("z_target is upstream of latest traced element.")

#         P = free_space(dz)

#         M = P @ M
#         r_chief = P @ r_chief
#         r_top = P @ r_top
#         r_bottom = P @ r_bottom

#         return M, r_chief, r_top, r_bottom

#     def solve_post_dm_lens_for_lgs_image(
#         r_chief_pre,
#         r_top_pre,
#         r_bottom_pre,
#         desired_x_img,
#     ):
#         """
#         Solve Lens 3 focal length so the finite LGS beam focuses after Lens 3
#         with chief-ray image height desired_x_img.
#         """
#         xc, tc = r_chief_pre
#         xt, tt = r_top_pre
#         xb, tb = r_bottom_pre

#         dx = xt - xb
#         dt = tt - tb
#         X = desired_x_img

#         if abs(dx) < 1e-15:
#             raise ValueError("Top and bottom ray heights are identical at Lens 3.")

#         if abs(X) < 1e-15:
#             raise ValueError("desired_pickoff_offset_mm cannot be zero for this solver.")

#         # Derived from:
#         #   d = -dx / (dt - q dx)
#         #   X = xc + d * (tc - q xc)
#         # where q = 1/f.
#         q = ((X - xc) * dt + dx * tc) / (dx * X)

#         if abs(q) < 1e-15:
#             raise ValueError("Solved Lens 3 has infinite focal length.")

#         f = 1.0 / q

#         L = thin_lens_matrix(f)

#         rc = L @ r_chief_pre
#         rt = L @ r_top_pre
#         rb = L @ r_bottom_pre

#         d_img = crossing_distance(rt, rb)

#         if not np.isfinite(d_img):
#             raise ValueError("Solved Lens 3 does not produce a finite image.")

#         if d_img <= 0:
#             raise ValueError(
#                 "Solved LGS image is upstream of Lens 3. "
#                 "Try desired_pickoff_offset_mm with the opposite sign."
#             )

#         x_check = rc[0] + d_img * rc[1]

#         if abs(x_check - X) > 1e-6:
#             raise RuntimeError("Internal Lens 3 solve check failed.")

#         return f, d_img

#     def solve_single_lens_pupil_relay(M_pre_lens, m_target):
#         """
#         Solve one lens plus downstream free-space distance so the input pupil
#         is reimaged with transverse magnification m_target.

#         M_pre_lens maps input pupil -> just before lens.

#         We solve for q=1/f and d such that:

#             M_total = P(d) @ L(q) @ M_pre_lens

#         has:
#             B_total = 0
#             A_total = m_target
#         """
#         A, B = M_pre_lens[0, 0], M_pre_lens[0, 1]
#         C, D = M_pre_lens[1, 0], M_pre_lens[1, 1]

#         if abs(B) < 1e-15:
#             raise ValueError("Cannot solve pupil relay: B≈0 at Lens 4.")
#         if abs(m_target) < 1e-15:
#             raise ValueError("Target pupil magnification cannot be zero.")

#         q = (B * C + (m_target - A) * D) / (m_target * B)

#         if abs(q) < 1e-15:
#             raise ValueError("Solved Lens 4 has infinite focal length.")

#         f = 1.0 / q

#         denom = D - q * B

#         if abs(denom) < 1e-15:
#             raise ValueError("Solved SH-WFS distance is singular.")

#         d = -B / denom

#         if d <= 0:
#             raise ValueError(
#                 "Solved SH-WFS pupil image is upstream of Lens 4. "
#                 "Try moving Lens 4 or changing target magnification sign."
#             )

#         return f, d

#     # ============================================================
#     # Common relay up to the DM
#     # ============================================================

#     z_lgs_image_after_l1, mag_lgs_l1 = image_from_object_thin_lens(
#         z_object=z_source,
#         z_lens=z_lens1_mm,
#         f=f_lens1_mm,
#     )

#     common_layout = [
#         dict(
#             label="Input pupil",
#             type="Pupil",
#             z=0.0,
#             f=None,
#             ap_radius=pupil_radius_mm,
#         ),
#         dict(
#             label="Lens 1: pupil relay A",
#             type="Lens",
#             z=z_lens1_mm,
#             f=f_lens1_mm,
#             ap_radius=lens1_ap_radius_mm,
#         ),
#         dict(
#             label="Intermediate LGS image",
#             type="Fiducial",
#             z=z_lgs_image_after_l1,
#             f=None,
#             ap_radius=2.0,
#         ),
#         dict(
#             label="Lens 2: pupil relay B",
#             type="Lens",
#             z=z_lens2_mm,
#             f=f_lens2_mm,
#             ap_radius=lens2_ap_radius_mm,
#         ),
#         dict(
#             label="DM pupil image",
#             type="DM",
#             z=z_dm_mm,
#             f=None,
#             ap_radius=dm_ap_radius_mm,
#         ),
#     ]

#     # ============================================================
#     # Solve Lens 3 for post-DM LGS pickoff image
#     # ============================================================

#     _, r_chief_l3_pre, r_top_l3_pre, r_bottom_l3_pre = trace_to_plane(
#         common_layout,
#         z_lens3_mm,
#     )

#     f_lens3_mm, pickoff_after_lens3_mm = solve_post_dm_lens_for_lgs_image(
#         r_chief_pre=r_chief_l3_pre,
#         r_top_pre=r_top_l3_pre,
#         r_bottom_pre=r_bottom_l3_pre,
#         desired_x_img=desired_pickoff_offset_mm,
#     )

#     z_pickoff_mm = z_lens3_mm + pickoff_after_lens3_mm

#     if z_pickoff_mm <= z_dm_mm:
#         raise RuntimeError("Solved pickoff plane is not after the DM; check layout.")

#     # ============================================================
#     # Solve Lens 4 for SH-WFS pupil image
#     # ============================================================

#     z_lens4_mm = z_pickoff_mm + lens4_after_pickoff_mm

#     layout_to_lens4 = common_layout + [
#         dict(
#             label="Lens 3: post-DM LGS focus",
#             type="Lens",
#             z=z_lens3_mm,
#             f=f_lens3_mm,
#             ap_radius=lens3_ap_radius_mm,
#         ),
#         dict(
#             label="LGS pickoff image plane",
#             type="Pickoff/Fiducial",
#             z=z_pickoff_mm,
#             f=None,
#             ap_radius=pickoff_plane_ap_radius_mm,
#         ),
#     ]

#     layout_to_lens4 = sorted(layout_to_lens4, key=lambda e: float(e["z"]))

#     M_pre_lens4, _, _, _ = trace_to_plane(layout_to_lens4, z_lens4_mm)

#     f_lens4_mm, sh_after_lens4_mm = solve_single_lens_pupil_relay(
#         M_pre_lens=M_pre_lens4,
#         m_target=target_sh_pupil_magnification,
#     )

#     z_sh_mm = z_lens4_mm + sh_after_lens4_mm

#     # ============================================================
#     # Final layout
#     # ============================================================

#     layout = layout_to_lens4 + [
#         dict(
#             label="Lens 4: SH pupil relay",
#             type="Lens",
#             z=z_lens4_mm,
#             f=f_lens4_mm,
#             ap_radius=lens4_ap_radius_mm,
#         ),
#         dict(
#             label="SH-WFS lenslet pupil plane",
#             type="SH-WFS/Pupil",
#             z=z_sh_mm,
#             f=None,
#             ap_radius=sh_plane_ap_radius_mm,
#         ),
#     ]

#     for i, dz in enumerate(downstream_fiducials_after_sh_mm):
#         layout.append(
#             dict(
#                 label=f"Downstream fiducial {i + 1}",
#                 type="Fiducial",
#                 z=z_sh_mm + dz,
#                 f=None,
#                 ap_radius=25.0,
#             )
#         )

#     layout = sorted(layout, key=lambda e: float(e["z"]))

#     # ============================================================
#     # Trace final layout
#     # ============================================================

#     rows = []

#     M = np.eye(2)

#     r_chief = chief0.copy()
#     r_top = top0.copy()
#     r_bottom = bottom0.copy()

#     z_prev = layout[0]["z"]

#     for i, elem in enumerate(layout):
#         z = float(elem["z"])

#         if i == 0:
#             dz = 0.0
#         else:
#             dz = z - z_prev

#         if dz < -1e-12:
#             raise ValueError("Layout z positions must be monotonically increasing.")

#         P = free_space(dz)

#         M = P @ M
#         r_chief = P @ r_chief
#         r_top = P @ r_top
#         r_bottom = P @ r_bottom

#         if elem["type"].lower() == "lens":
#             L = thin_lens_matrix(float(elem["f"]))

#             M = L @ M
#             r_chief = L @ r_chief
#             r_top = L @ r_top
#             r_bottom = L @ r_bottom

#         beam_center = 0.5 * (r_top[0] + r_bottom[0])
#         beam_radius = 0.5 * abs(r_top[0] - r_bottom[0])

#         d_img = crossing_distance(r_top, r_bottom)
#         z_img = z + d_img if np.isfinite(d_img) else np.inf
#         x_img_chief = r_chief[0] + d_img * r_chief[1] if np.isfinite(d_img) else np.nan

#         d_pup = pupil_conjugate_distance_after_plane(M)
#         z_pup = z + d_pup if np.isfinite(d_pup) else np.inf

#         ap_radius = float(elem.get("ap_radius", np.nan))

#         max_abs_ray_height = max(
#             abs(r_chief[0]),
#             abs(r_top[0]),
#             abs(r_bottom[0]),
#         )

#         clearance = ap_radius - max_abs_ray_height

#         rows.append(
#             dict(
#                 step=i,
#                 label=elem["label"],
#                 type=elem["type"],
#                 z_mm=z,
#                 dz_mm=dz,
#                 f_mm=elem["f"],
#                 A=M[0, 0],
#                 B=M[0, 1],
#                 C=M[1, 0],
#                 D=M[1, 1],
#                 chief_x_mm=r_chief[0],
#                 chief_theta_mrad=1e3 * r_chief[1],
#                 top_x_mm=r_top[0],
#                 top_theta_mrad=1e3 * r_top[1],
#                 bottom_x_mm=r_bottom[0],
#                 bottom_theta_mrad=1e3 * r_bottom[1],
#                 beam_center_mm=beam_center,
#                 beam_radius_mm=beam_radius,
#                 image_d_after_mm=d_img,
#                 image_z_mm=z_img,
#                 image_chief_x_mm=x_img_chief,
#                 pupil_d_after_mm=d_pup,
#                 pupil_z_mm=z_pup,
#                 ap_radius_mm=ap_radius,
#                 clearance_mm=clearance,
#             )
#         )

#         z_prev = z

#     # ============================================================
#     # Optional original object/image checks
#     # ============================================================

#     lens_list = [
#         lens(x=e["z"], f=e["f"])
#         for e in layout
#         if e["type"].lower() == "lens"
#     ]

#     lgs_object = obj(x=z_source, h=x_source)
#     pupil_object = obj(x=0.0, h=pupil_radius_mm)

#     lgs_images = propagate_object(lens_list, lgs_object, plot=False)
#     pupil_images = propagate_object(lens_list, pupil_object, plot=False)

#     # ============================================================
#     # Print summary
#     # ============================================================

#     print()
#     print("Single-LGS 2D thin-lens system: post-DM pickoff + SH pupil relay")
#     print("=" * 98)
#     print(f"Input pupil diameter:              {pupil_diameter_mm:.3f} mm")
#     print(f"LGS field angle:                   {lgs_angle_arcmin:.3f} arcmin")
#     print(f"LGS chief angle:                   {theta_chief * 1e3:.6f} mrad")
#     print(f"LGS source z:                      {z_source:.3f} mm")
#     print(f"LGS source x:                      {x_source:+.3f} mm")
#     print()
#     print("Common pupil relay")
#     print(f"  Lens 1: z={z_lens1_mm:.3f} mm, f={f_lens1_mm:.3f} mm")
#     print(f"  Lens 2: z={z_lens2_mm:.3f} mm, f={f_lens2_mm:.3f} mm")
#     print(f"  DM pupil image z:                {z_dm_mm:.3f} mm")
#     print(f"  Intermediate LGS image z:        {z_lgs_image_after_l1:.3f} mm")
#     print(f"  Intermediate LGS image x:        {mag_lgs_l1 * x_source:+.3f} mm")
#     print()
#     print("Post-DM LGS pickoff")
#     print(f"  desired pickoff image x:         {desired_pickoff_offset_mm:+.3f} mm")
#     print(f"  Lens 3 z:                        {z_lens3_mm:.3f} mm")
#     print(f"  solved Lens 3 focal length:      {f_lens3_mm:.3f} mm")
#     print(f"  pickoff distance after Lens 3:   {pickoff_after_lens3_mm:.3f} mm")
#     print(f"  pickoff plane z:                 {z_pickoff_mm:.3f} mm")
#     print(f"  pickoff after DM by:             {z_pickoff_mm - z_dm_mm:.3f} mm")
#     print()
#     print("SH-WFS pupil relay")
#     print(f"  target SH pupil diameter:        {target_sh_pupil_diameter_mm:.3f} mm")
#     print(f"  target pupil magnification:      {target_sh_pupil_magnification:+.6f}")
#     print(f"  Lens 4 z:                        {z_lens4_mm:.3f} mm")
#     print(f"  solved Lens 4 focal length:      {f_lens4_mm:.3f} mm")
#     print(f"  SH distance after Lens 4:        {sh_after_lens4_mm:.3f} mm")
#     print(f"  SH-WFS pupil plane z:            {z_sh_mm:.3f} mm")
#     print()

#     header = (
#         f"{'i':>2s} {'label':32s} {'type':16s} "
#         f"{'z':>10s} {'f':>10s} "
#         f"{'chief x':>10s} {'radius':>10s} "
#         f"{'image z':>11s} {'pupil z':>11s} "
#         f"{'clear':>10s}"
#     )

#     print(header)
#     print("-" * len(header))

#     for r in rows:
#         f_txt = "" if r["f_mm"] is None else f"{r['f_mm']:.3f}"
#         img_txt = "inf" if not np.isfinite(r["image_z_mm"]) else f"{r['image_z_mm']:.3f}"
#         pup_txt = "inf" if not np.isfinite(r["pupil_z_mm"]) else f"{r['pupil_z_mm']:.3f}"

#         print(
#             f"{r['step']:2d} "
#             f"{r['label'][:32]:32s} "
#             f"{r['type'][:16]:16s} "
#             f"{r['z_mm']:10.3f} "
#             f"{f_txt:>10s} "
#             f"{r['chief_x_mm']:10.3f} "
#             f"{r['beam_radius_mm']:10.3f} "
#             f"{img_txt:>11s} "
#             f"{pup_txt:>11s} "
#             f"{r['clearance_mm']:10.3f}"
#         )

#     dm_rows = [r for r in rows if r["type"].lower() == "dm"]
#     pickoff_rows = [r for r in rows if "pickoff" in r["type"].lower()]
#     sh_rows = [r for r in rows if "sh-wfs" in r["type"].lower()]

#     if dm_rows:
#         dm = dm_rows[0]
#         dm_diameter = 2.0 * dm["beam_radius_mm"]

#         print()
#         print("DM pupil image check")
#         print(f"  traced DM pupil diameter:         {dm_diameter:.3f} mm")
#         print(f"  target DM pupil diameter:         {pupil_diameter_mm:.3f} mm")
#         print(f"  diameter error:                   {dm_diameter - pupil_diameter_mm:+.6f} mm")
#         print(f"  DM chief x:                       {dm['chief_x_mm']:+.6f} mm")

#     if pickoff_rows:
#         p = pickoff_rows[0]

#         print()
#         print("LGS pickoff image check")
#         print(f"  pickoff z:                        {p['z_mm']:.3f} mm")
#         print(f"  pickoff chief x:                  {p['chief_x_mm']:+.6f} mm")
#         print(f"  requested chief x:                {desired_pickoff_offset_mm:+.6f} mm")
#         print(f"  geometric beam radius at pickoff: {p['beam_radius_mm']:.6e} mm")
#         print(f"  pickoff after DM?                 {p['z_mm'] > z_dm_mm}")

#         two_lgs_sep = 2.0 * abs(p["chief_x_mm"])
#         print(f"  2D +/-10 arcmin image separation: {two_lgs_sep:.3f} mm")

#     if sh_rows:
#         s = sh_rows[0]
#         sh_diameter = 2.0 * s["beam_radius_mm"]

#         print()
#         print("SH-WFS pupil image check")
#         print(f"  SH-WFS z:                         {s['z_mm']:.3f} mm")
#         print(f"  SH-WFS chief x:                   {s['chief_x_mm']:+.6f} mm")
#         print(f"  traced SH pupil diameter:         {sh_diameter:.3f} mm")
#         print(f"  target SH pupil diameter:         {target_sh_pupil_diameter_mm:.3f} mm")
#         print(f"  diameter error:                   {sh_diameter - target_sh_pupil_diameter_mm:+.6f} mm")

#     print()
#     print("Original object/image propagation checks")
#     print(f"  final LGS object/image:           x={lgs_images[-1].x:.3f} mm, h={lgs_images[-1].h:.3f} mm")
#     print(f"  final pupil object/image:         x={pupil_images[-1].x:.3f} mm, h={pupil_images[-1].h:.3f} mm")

#     print()
#     print("Aperture clearance check")
#     for r in rows:
#         status = "OK" if r["clearance_mm"] >= -1e-9 else "CLIPPING"
#         print(
#             f"  {r['label'][:34]:34s}: "
#             f"clearance={r['clearance_mm']:+10.3f} mm -> {status}"
#         )

#     # ============================================================
#     # Plot ray trace
#     # ============================================================

#     z_vals = np.array([r["z_mm"] for r in rows])
#     chief_x = np.array([r["chief_x_mm"] for r in rows])
#     top_x = np.array([r["top_x_mm"] for r in rows])
#     bottom_x = np.array([r["bottom_x_mm"] for r in rows])
#     radius = np.array([r["beam_radius_mm"] for r in rows])
#     centre = np.array([r["beam_center_mm"] for r in rows])

#     fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)

#     ax = axes[0]
#     ax.plot(z_vals, chief_x, "o-", label="chief ray")
#     ax.plot(z_vals, top_x, "o-", label="top ray")
#     ax.plot(z_vals, bottom_x, "o-", label="bottom ray")
#     ax.fill_between(
#         z_vals,
#         bottom_x,
#         top_x,
#         alpha=0.15,
#         label="beam envelope",
#     )

#     for r in rows:
#         rtype = r["type"].lower()

#         if rtype == "lens":
#             ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.35)
#             ax.text(
#                 r["z_mm"],
#                 ax.get_ylim()[1],
#                 r["label"],
#                 rotation=90,
#                 va="top",
#                 ha="right",
#                 fontsize=8,
#             )

#         elif "pickoff" in rtype:
#             ax.axvline(r["z_mm"], color="tab:red", linestyle="-.", alpha=0.85)
#             ax.text(
#                 r["z_mm"],
#                 ax.get_ylim()[1],
#                 "LGS pickoff plane",
#                 rotation=90,
#                 va="top",
#                 ha="left",
#                 fontsize=9,
#                 color="tab:red",
#             )

#         elif rtype == "dm":
#             ax.axvline(r["z_mm"], color="tab:blue", linestyle="-.", alpha=0.65)
#             ax.text(
#                 r["z_mm"],
#                 ax.get_ylim()[1],
#                 "DM pupil image",
#                 rotation=90,
#                 va="top",
#                 ha="left",
#                 fontsize=9,
#                 color="tab:blue",
#             )

#         elif "sh-wfs" in rtype:
#             ax.axvline(r["z_mm"], color="tab:purple", linestyle="-.", alpha=0.85)
#             ax.text(
#                 r["z_mm"],
#                 ax.get_ylim()[1],
#                 "SH-WFS pupil",
#                 rotation=90,
#                 va="top",
#                 ha="left",
#                 fontsize=9,
#                 color="tab:purple",
#             )

#     ax.axhline(0.0, color="k", lw=0.8, alpha=0.35)
#     ax.set_ylabel("x [mm]")
#     ax.set_title("Single-LGS 2D ray trace: DM, LGS pickoff, then SH-WFS pupil image")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best")

#     ax = axes[1]
#     ax.plot(z_vals, centre, "o-", label="beam centre")
#     ax.plot(z_vals, radius, "o-", label="beam radius")
#     ax.axhline(
#         pupil_radius_mm,
#         color="k",
#         linestyle=":",
#         alpha=0.5,
#         label="input pupil radius",
#     )
#     ax.axhline(
#         0.5 * target_sh_pupil_diameter_mm,
#         color="tab:purple",
#         linestyle=":",
#         alpha=0.6,
#         label="target SH pupil radius",
#     )

#     for r in rows:
#         rtype = r["type"].lower()

#         if "pickoff" in rtype:
#             ax.axvline(r["z_mm"], color="tab:red", linestyle="-.", alpha=0.85)
#         elif rtype == "dm":
#             ax.axvline(r["z_mm"], color="tab:blue", linestyle="-.", alpha=0.65)
#         elif "sh-wfs" in rtype:
#             ax.axvline(r["z_mm"], color="tab:purple", linestyle="-.", alpha=0.85)
#         elif rtype == "lens":
#             ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.25)

#     ax.set_xlabel("z [mm]")
#     ax.set_ylabel("x / radius [mm]")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best")

#     plt.tight_layout()
#     plt.show()

#     return rows


# if __name__ == "__main__":
#     main()

# # def main():
# #     """
# #     Single-LGS 2D thin-lens design tracker.

# #     Purpose
# #     -------
# #     First-order 2D design for one finite-distance LGS beam:

# #         finite LGS source
# #         -> input pupil at z=0
# #         -> Lens 1 / Lens 2 pupil relay
# #         -> DM pupil image
# #         -> Lens 3 forms a focused LGS image after the DM
# #         -> LGS pickoff image plane

# #     This version enforces that the LGS pickoff plane is AFTER the DM.

# #     Units
# #     -----
# #     z, x, focal lengths : mm
# #     angles             : radians
# #     """

# #     # ============================================================
# #     # Editable design inputs
# #     # ============================================================

# #     pupil_diameter_mm = 13.5
# #     pupil_radius_mm = 0.5 * pupil_diameter_mm

# #     lgs_angle_arcmin = -10.0
# #     lgs_source_distance_mm = 3250.0

# #     # Desired focused LGS image position after the DM.
# #     # With the default relay sign convention, a -10 arcmin LGS focuses to
# #     # positive x after Lens 3. Try +5, +8, etc.
# #     desired_pickoff_offset_mm = +5.0

# #     # Common pupil relay to DM.
# #     z_lens1_mm = 150.0
# #     f_lens1_mm = 150.0

# #     z_lens2_mm = 450.0
# #     f_lens2_mm = 150.0

# #     z_dm_mm = 600.0

# #     # Lens after the DM. Its focal length is solved automatically so that
# #     # the LGS focuses to desired_pickoff_offset_mm.
# #     z_lens3_mm = 750.0

# #     # Extra planes after the pickoff image for context only.
# #     downstream_fiducials_after_pickoff_mm = [150.0, 300.0]

# #     # Aperture radii for first-order sanity checks.
# #     lens1_ap_radius_mm = 25.0
# #     lens2_ap_radius_mm = 25.0
# #     lens3_ap_radius_mm = 25.0
# #     dm_ap_radius_mm = pupil_radius_mm
# #     pickoff_plane_ap_radius_mm = 2.0

# #     # ============================================================
# #     # Initial finite LGS geometry at input pupil
# #     # ============================================================

# #     theta_chief = np.deg2rad(lgs_angle_arcmin / 60.0)

# #     z_source = -lgs_source_distance_mm

# #     # Source x chosen so chief ray crosses x=0 at z=0.
# #     # theta = (0 - x_source) / (0 - z_source)
# #     x_source = -theta_chief * lgs_source_distance_mm

# #     # Ray state is [x, theta] at z=0.
# #     chief0 = np.array([0.0, theta_chief], dtype=float)

# #     top0 = np.array(
# #         [
# #             +pupil_radius_mm,
# #             (+pupil_radius_mm - x_source) / (0.0 - z_source),
# #         ],
# #         dtype=float,
# #     )

# #     bottom0 = np.array(
# #         [
# #             -pupil_radius_mm,
# #             (-pupil_radius_mm - x_source) / (0.0 - z_source),
# #         ],
# #         dtype=float,
# #     )

# #     # ============================================================
# #     # ABCD helpers
# #     # ============================================================

# #     def free_space(d_mm):
# #         return np.array([[1.0, d_mm], [0.0, 1.0]], dtype=float)

# #     def thin_lens_matrix(f_mm):
# #         return np.array([[1.0, 0.0], [-1.0 / f_mm, 1.0]], dtype=float)

# #     def crossing_distance(ray_a, ray_b):
# #         """
# #         Distance after current plane where two rays cross.
# #         """
# #         xa, ta = ray_a
# #         xb, tb = ray_b
# #         denom = ta - tb

# #         if abs(denom) < 1e-15:
# #             return np.inf

# #         return -(xa - xb) / denom

# #     def pupil_conjugate_distance_after_plane(M):
# #         """
# #         Distance after current plane where the input pupil is reimaged.

# #         If M maps the input pupil to the current plane, then after free
# #         propagation d:

# #             B_new = B + d D

# #         Pupil conjugate condition: B_new = 0.
# #         """
# #         B = M[0, 1]
# #         D = M[1, 1]

# #         if abs(D) < 1e-15:
# #             return np.inf

# #         return -B / D

# #     def image_from_object_thin_lens(z_object, z_lens, f):
# #         """
# #         Thin-lens image location using:
# #             s  = z_lens - z_object
# #             s' = s f / (s - f)
# #             z_image = z_lens + s'
# #             M = -s'/s
# #         """
# #         s = z_lens - z_object

# #         if abs(s - f) < 1e-15:
# #             return np.inf, np.inf

# #         sp = s * f / (s - f)
# #         mag = -sp / s

# #         return z_lens + sp, mag

# #     def trace_to_plane(layout, z_target):
# #         """
# #         Trace chief/top/bottom rays and ABCD matrix to z_target through all
# #         lenses in layout that occur at or before z_target.

# #         Returns
# #         -------
# #         M, r_chief, r_top, r_bottom
# #         """
# #         M = np.eye(2)

# #         r_chief = chief0.copy()
# #         r_top = top0.copy()
# #         r_bottom = bottom0.copy()

# #         z_prev = 0.0

# #         for elem in layout:
# #             z = float(elem["z"])

# #             if z > z_target + 1e-12:
# #                 break

# #             dz = z - z_prev

# #             if dz < -1e-12:
# #                 raise ValueError("Layout z positions must be monotonically increasing.")

# #             P = free_space(dz)

# #             M = P @ M
# #             r_chief = P @ r_chief
# #             r_top = P @ r_top
# #             r_bottom = P @ r_bottom

# #             if elem["type"].lower() == "lens":
# #                 L = thin_lens_matrix(float(elem["f"]))

# #                 M = L @ M
# #                 r_chief = L @ r_chief
# #                 r_top = L @ r_top
# #                 r_bottom = L @ r_bottom

# #             z_prev = z

# #         dz = z_target - z_prev

# #         if dz < -1e-12:
# #             raise ValueError("z_target is upstream of latest traced element.")

# #         P = free_space(dz)

# #         M = P @ M
# #         r_chief = P @ r_chief
# #         r_top = P @ r_top
# #         r_bottom = P @ r_bottom

# #         return M, r_chief, r_top, r_bottom

# #     def solve_post_dm_lens_for_lgs_image(
# #         r_chief_pre,
# #         r_top_pre,
# #         r_bottom_pre,
# #         desired_x_img,
# #     ):
# #         """
# #         Solve Lens 3 focal length so the finite LGS beam focuses after Lens 3
# #         with chief-ray image height desired_x_img.

# #         The lens is assumed to be at the current plane. The input ray states are
# #         before the lens.

# #         Returns
# #         -------
# #         f_mm, d_image_after_lens_mm
# #         """
# #         xc, tc = r_chief_pre
# #         xt, tt = r_top_pre
# #         xb, tb = r_bottom_pre

# #         dx = xt - xb
# #         dt = tt - tb
# #         X = desired_x_img

# #         if abs(dx) < 1e-15:
# #             raise ValueError("Top and bottom ray heights are identical at Lens 3.")

# #         if abs(X) < 1e-15:
# #             raise ValueError("desired_pickoff_offset_mm cannot be zero for this solver.")

# #         # Derived from:
# #         #   d = -dx / (dt - q dx)
# #         #   X = xc + d * (tc - q xc)
# #         # where q = 1/f.
# #         q = ((X - xc) * dt + dx * tc) / (dx * X)

# #         if abs(q) < 1e-15:
# #             raise ValueError("Solved Lens 3 has infinite focal length.")

# #         f = 1.0 / q

# #         # Apply lens and compute crossing.
# #         L = thin_lens_matrix(f)
# #         rc = L @ r_chief_pre
# #         rt = L @ r_top_pre
# #         rb = L @ r_bottom_pre

# #         d_img = crossing_distance(rt, rb)

# #         if not np.isfinite(d_img):
# #             raise ValueError("Solved Lens 3 does not produce a finite image.")

# #         if d_img <= 0:
# #             raise ValueError(
# #                 "Solved LGS image is upstream of Lens 3. "
# #                 "For the default relay, try desired_pickoff_offset_mm with the opposite sign."
# #             )

# #         x_check = rc[0] + d_img * rc[1]

# #         if abs(x_check - X) > 1e-6:
# #             raise RuntimeError("Internal Lens 3 solve check failed.")

# #         return f, d_img

# #     # ============================================================
# #     # Define common relay up to the DM
# #     # ============================================================

# #     z_lgs_image_after_l1, mag_lgs_l1 = image_from_object_thin_lens(
# #         z_object=z_source,
# #         z_lens=z_lens1_mm,
# #         f=f_lens1_mm,
# #     )

# #     common_layout = [
# #         dict(
# #             label="Input pupil",
# #             type="Pupil",
# #             z=0.0,
# #             f=None,
# #             ap_radius=pupil_radius_mm,
# #         ),
# #         dict(
# #             label="Lens 1: pupil relay A",
# #             type="Lens",
# #             z=z_lens1_mm,
# #             f=f_lens1_mm,
# #             ap_radius=lens1_ap_radius_mm,
# #         ),
# #         dict(
# #             label="Intermediate LGS image",
# #             type="Fiducial",
# #             z=z_lgs_image_after_l1,
# #             f=None,
# #             ap_radius=2.0,
# #         ),
# #         dict(
# #             label="Lens 2: pupil relay B",
# #             type="Lens",
# #             z=z_lens2_mm,
# #             f=f_lens2_mm,
# #             ap_radius=lens2_ap_radius_mm,
# #         ),
# #         dict(
# #             label="DM pupil image",
# #             type="DM",
# #             z=z_dm_mm,
# #             f=None,
# #             ap_radius=dm_ap_radius_mm,
# #         ),
# #     ]

# #     # Trace to just before Lens 3 and solve Lens 3.
# #     _, r_chief_l3_pre, r_top_l3_pre, r_bottom_l3_pre = trace_to_plane(
# #         common_layout,
# #         z_lens3_mm,
# #     )

# #     f_lens3_mm, pickoff_after_lens3_mm = solve_post_dm_lens_for_lgs_image(
# #         r_chief_pre=r_chief_l3_pre,
# #         r_top_pre=r_top_l3_pre,
# #         r_bottom_pre=r_bottom_l3_pre,
# #         desired_x_img=desired_pickoff_offset_mm,
# #     )

# #     z_pickoff_mm = z_lens3_mm + pickoff_after_lens3_mm

# #     if z_pickoff_mm <= z_dm_mm:
# #         raise RuntimeError("Solved pickoff plane is not after the DM; check layout.")

# #     # ============================================================
# #     # Final layout
# #     # ============================================================

# #     layout = common_layout + [
# #         dict(
# #             label="Lens 3: post-DM LGS focus",
# #             type="Lens",
# #             z=z_lens3_mm,
# #             f=f_lens3_mm,
# #             ap_radius=lens3_ap_radius_mm,
# #         ),
# #         dict(
# #             label="LGS pickoff image plane",
# #             type="Pickoff/Fiducial",
# #             z=z_pickoff_mm,
# #             f=None,
# #             ap_radius=pickoff_plane_ap_radius_mm,
# #         ),
# #     ]

# #     for i, dz in enumerate(downstream_fiducials_after_pickoff_mm):
# #         layout.append(
# #             dict(
# #                 label=f"Downstream fiducial {i + 1}",
# #                 type="Fiducial",
# #                 z=z_pickoff_mm + dz,
# #                 f=None,
# #                 ap_radius=25.0,
# #             )
# #         )

# #     # Sort by z in case the intermediate LGS image is inserted between lenses.
# #     layout = sorted(layout, key=lambda e: float(e["z"]))

# #     # ============================================================
# #     # Trace final layout
# #     # ============================================================

# #     rows = []

# #     M = np.eye(2)

# #     r_chief = chief0.copy()
# #     r_top = top0.copy()
# #     r_bottom = bottom0.copy()

# #     z_prev = layout[0]["z"]

# #     for i, elem in enumerate(layout):
# #         z = float(elem["z"])

# #         if i == 0:
# #             dz = 0.0
# #         else:
# #             dz = z - z_prev

# #         if dz < -1e-12:
# #             raise ValueError("Layout z positions must be monotonically increasing.")

# #         P = free_space(dz)

# #         M = P @ M
# #         r_chief = P @ r_chief
# #         r_top = P @ r_top
# #         r_bottom = P @ r_bottom

# #         if elem["type"].lower() == "lens":
# #             L = thin_lens_matrix(float(elem["f"]))

# #             M = L @ M
# #             r_chief = L @ r_chief
# #             r_top = L @ r_top
# #             r_bottom = L @ r_bottom

# #         beam_center = 0.5 * (r_top[0] + r_bottom[0])
# #         beam_radius = 0.5 * abs(r_top[0] - r_bottom[0])

# #         d_img = crossing_distance(r_top, r_bottom)
# #         z_img = z + d_img if np.isfinite(d_img) else np.inf
# #         x_img_chief = r_chief[0] + d_img * r_chief[1] if np.isfinite(d_img) else np.nan

# #         d_pup = pupil_conjugate_distance_after_plane(M)
# #         z_pup = z + d_pup if np.isfinite(d_pup) else np.inf

# #         ap_radius = float(elem.get("ap_radius", np.nan))

# #         max_abs_ray_height = max(
# #             abs(r_chief[0]),
# #             abs(r_top[0]),
# #             abs(r_bottom[0]),
# #         )

# #         clearance = ap_radius - max_abs_ray_height

# #         rows.append(
# #             dict(
# #                 step=i,
# #                 label=elem["label"],
# #                 type=elem["type"],
# #                 z_mm=z,
# #                 dz_mm=dz,
# #                 f_mm=elem["f"],
# #                 A=M[0, 0],
# #                 B=M[0, 1],
# #                 C=M[1, 0],
# #                 D=M[1, 1],
# #                 chief_x_mm=r_chief[0],
# #                 chief_theta_mrad=1e3 * r_chief[1],
# #                 top_x_mm=r_top[0],
# #                 top_theta_mrad=1e3 * r_top[1],
# #                 bottom_x_mm=r_bottom[0],
# #                 bottom_theta_mrad=1e3 * r_bottom[1],
# #                 beam_center_mm=beam_center,
# #                 beam_radius_mm=beam_radius,
# #                 image_d_after_mm=d_img,
# #                 image_z_mm=z_img,
# #                 image_chief_x_mm=x_img_chief,
# #                 pupil_d_after_mm=d_pup,
# #                 pupil_z_mm=z_pup,
# #                 ap_radius_mm=ap_radius,
# #                 clearance_mm=clearance,
# #             )
# #         )

# #         z_prev = z

# #     # ============================================================
# #     # Optional original object/image propagation checks
# #     # ============================================================

# #     lens_list = [
# #         lens(x=e["z"], f=e["f"])
# #         for e in layout
# #         if e["type"].lower() == "lens"
# #     ]

# #     lgs_object = obj(x=z_source, h=x_source)
# #     pupil_object = obj(x=0.0, h=pupil_radius_mm)

# #     lgs_images = propagate_object(lens_list, lgs_object, plot=False)
# #     pupil_images = propagate_object(lens_list, pupil_object, plot=False)

# #     # ============================================================
# #     # Print summary
# #     # ============================================================

# #     print()
# #     print("Single-LGS 2D thin-lens system: post-DM LGS pickoff")
# #     print("=" * 92)
# #     print(f"Input pupil diameter:          {pupil_diameter_mm:.3f} mm")
# #     print(f"LGS field angle:               {lgs_angle_arcmin:.3f} arcmin")
# #     print(f"LGS chief angle:               {theta_chief * 1e3:.6f} mrad")
# #     print(f"LGS source z:                  {z_source:.3f} mm")
# #     print(f"LGS source x:                  {x_source:+.3f} mm")
# #     print()
# #     print("Common pupil relay")
# #     print(f"  Lens 1: z={z_lens1_mm:.3f} mm, f={f_lens1_mm:.3f} mm")
# #     print(f"  Lens 2: z={z_lens2_mm:.3f} mm, f={f_lens2_mm:.3f} mm")
# #     print(f"  DM pupil image z:            {z_dm_mm:.3f} mm")
# #     print(f"  Intermediate LGS image z:    {z_lgs_image_after_l1:.3f} mm")
# #     print(f"  Intermediate LGS image x:    {mag_lgs_l1 * x_source:+.3f} mm")
# #     print()
# #     print("Post-DM LGS pickoff")
# #     print(f"  desired pickoff image x:     {desired_pickoff_offset_mm:+.3f} mm")
# #     print(f"  Lens 3 z:                    {z_lens3_mm:.3f} mm")
# #     print(f"  solved Lens 3 focal length:  {f_lens3_mm:.3f} mm")
# #     print(f"  pickoff distance after L3:   {pickoff_after_lens3_mm:.3f} mm")
# #     print(f"  pickoff plane z:             {z_pickoff_mm:.3f} mm")
# #     print(f"  pickoff after DM by:         {z_pickoff_mm - z_dm_mm:.3f} mm")
# #     print()

# #     header = (
# #         f"{'i':>2s} {'label':30s} {'type':16s} "
# #         f"{'z':>10s} {'f':>10s} "
# #         f"{'chief x':>10s} {'radius':>10s} "
# #         f"{'image z':>11s} {'pupil z':>11s} "
# #         f"{'clear':>10s}"
# #     )

# #     print(header)
# #     print("-" * len(header))

# #     for r in rows:
# #         f_txt = "" if r["f_mm"] is None else f"{r['f_mm']:.3f}"
# #         img_txt = "inf" if not np.isfinite(r["image_z_mm"]) else f"{r['image_z_mm']:.3f}"
# #         pup_txt = "inf" if not np.isfinite(r["pupil_z_mm"]) else f"{r['pupil_z_mm']:.3f}"

# #         print(
# #             f"{r['step']:2d} "
# #             f"{r['label'][:30]:30s} "
# #             f"{r['type'][:16]:16s} "
# #             f"{r['z_mm']:10.3f} "
# #             f"{f_txt:>10s} "
# #             f"{r['chief_x_mm']:10.3f} "
# #             f"{r['beam_radius_mm']:10.3f} "
# #             f"{img_txt:>11s} "
# #             f"{pup_txt:>11s} "
# #             f"{r['clearance_mm']:10.3f}"
# #         )

# #     dm_rows = [r for r in rows if r["type"].lower() == "dm"]

# #     if dm_rows:
# #         dm = dm_rows[0]
# #         dm_diameter = 2.0 * dm["beam_radius_mm"]

# #         print()
# #         print("DM pupil image check")
# #         print(f"  traced DM pupil diameter:     {dm_diameter:.3f} mm")
# #         print(f"  target DM pupil diameter:     {pupil_diameter_mm:.3f} mm")
# #         print(f"  diameter error:               {dm_diameter - pupil_diameter_mm:+.6f} mm")
# #         print(f"  DM chief x:                   {dm['chief_x_mm']:+.6f} mm")

# #     pickoff_rows = [r for r in rows if "pickoff" in r["type"].lower()]

# #     if pickoff_rows:
# #         p = pickoff_rows[0]
# #         print()
# #         print("LGS pickoff image check")
# #         print(f"  pickoff z:                    {p['z_mm']:.3f} mm")
# #         print(f"  pickoff chief x:              {p['chief_x_mm']:+.6f} mm")
# #         print(f"  requested chief x:            {desired_pickoff_offset_mm:+.6f} mm")
# #         print(f"  pickoff beam radius:          {p['beam_radius_mm']:.6e} mm")
# #         print(f"  pickoff after DM?             {p['z_mm'] > z_dm_mm}")

# #     print()
# #     print("Original object/image propagation checks")
# #     print(f"  final LGS object/image:       x={lgs_images[-1].x:.3f} mm, h={lgs_images[-1].h:.3f} mm")
# #     print(f"  final pupil object/image:     x={pupil_images[-1].x:.3f} mm, h={pupil_images[-1].h:.3f} mm")

# #     print()
# #     print("Aperture clearance check")
# #     for r in rows:
# #         status = "OK" if r["clearance_mm"] >= -1e-9 else "CLIPPING"
# #         print(
# #             f"  {r['label'][:32]:32s}: "
# #             f"clearance={r['clearance_mm']:+10.3f} mm -> {status}"
# #         )

# #     # ============================================================
# #     # Plot ray trace
# #     # ============================================================

# #     z_vals = np.array([r["z_mm"] for r in rows])
# #     chief_x = np.array([r["chief_x_mm"] for r in rows])
# #     top_x = np.array([r["top_x_mm"] for r in rows])
# #     bottom_x = np.array([r["bottom_x_mm"] for r in rows])
# #     radius = np.array([r["beam_radius_mm"] for r in rows])
# #     centre = np.array([r["beam_center_mm"] for r in rows])

# #     fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)

# #     ax = axes[0]
# #     ax.plot(z_vals, chief_x, "o-", label="chief ray")
# #     ax.plot(z_vals, top_x, "o-", label="top ray")
# #     ax.plot(z_vals, bottom_x, "o-", label="bottom ray")
# #     ax.fill_between(
# #         z_vals,
# #         bottom_x,
# #         top_x,
# #         alpha=0.15,
# #         label="beam envelope",
# #     )

# #     for r in rows:
# #         rtype = r["type"].lower()

# #         if rtype == "lens":
# #             ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.35)
# #             ax.text(
# #                 r["z_mm"],
# #                 ax.get_ylim()[1],
# #                 r["label"],
# #                 rotation=90,
# #                 va="top",
# #                 ha="right",
# #                 fontsize=8,
# #             )

# #         elif "pickoff" in rtype:
# #             ax.axvline(r["z_mm"], color="tab:red", linestyle="-.", alpha=0.85)
# #             ax.text(
# #                 r["z_mm"],
# #                 ax.get_ylim()[1],
# #                 "LGS pickoff plane",
# #                 rotation=90,
# #                 va="top",
# #                 ha="left",
# #                 fontsize=9,
# #                 color="tab:red",
# #             )

# #         elif rtype == "dm":
# #             ax.axvline(r["z_mm"], color="tab:blue", linestyle="-.", alpha=0.65)
# #             ax.text(
# #                 r["z_mm"],
# #                 ax.get_ylim()[1],
# #                 "DM pupil image",
# #                 rotation=90,
# #                 va="top",
# #                 ha="left",
# #                 fontsize=9,
# #                 color="tab:blue",
# #             )

# #     ax.axhline(0.0, color="k", lw=0.8, alpha=0.35)
# #     ax.set_ylabel("x [mm]")
# #     ax.set_title("Single-LGS 2D ray trace: DM first, then post-DM LGS pickoff")
# #     ax.grid(True, alpha=0.3)
# #     ax.legend(loc="best")

# #     ax = axes[1]
# #     ax.plot(z_vals, centre, "o-", label="beam centre")
# #     ax.plot(z_vals, radius, "o-", label="beam radius")
# #     ax.axhline(
# #         pupil_radius_mm,
# #         color="k",
# #         linestyle=":",
# #         alpha=0.5,
# #         label="input pupil radius",
# #     )

# #     for r in rows:
# #         rtype = r["type"].lower()

# #         if "pickoff" in rtype:
# #             ax.axvline(r["z_mm"], color="tab:red", linestyle="-.", alpha=0.85)
# #         elif rtype == "dm":
# #             ax.axvline(r["z_mm"], color="tab:blue", linestyle="-.", alpha=0.65)
# #         elif rtype == "lens":
# #             ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.25)

# #         if np.isfinite(r["image_z_mm"]) and z_vals.min() <= r["image_z_mm"] <= z_vals.max():
# #             ax.axvline(r["image_z_mm"], color="tab:red", linestyle="--", alpha=0.18)

# #         if np.isfinite(r["pupil_z_mm"]) and z_vals.min() <= r["pupil_z_mm"] <= z_vals.max():
# #             ax.axvline(r["pupil_z_mm"], color="tab:blue", linestyle=":", alpha=0.18)

# #     ax.set_xlabel("z [mm]")
# #     ax.set_ylabel("x / radius [mm]")
# #     ax.grid(True, alpha=0.3)
# #     ax.legend(loc="best")

# #     plt.tight_layout()
# #     plt.show()

# #     return rows


# # if __name__ == "__main__":
# #     main()



# # # def main():
# # #     """
# # #     Single-LGS 2D thin-lens tracker.

# # #     Units
# # #     -----
# # #     z / x / focal lengths : mm
# # #     angles               : radians

# # #     Coordinate convention
# # #     ---------------------
# # #     z = optical axis, increasing downstream.
# # #     x = transverse coordinate.
# # #     Input telescope pupil is at z = 0.
# # #     One finite LGS source is upstream at z = -lgs_source_distance_mm.
# # #     """

# # #     # ============================================================
# # #     # Editable design inputs
# # #     # ============================================================

# # #     pupil_diameter_mm = 13.5
# # #     pupil_radius_mm = 0.5 * pupil_diameter_mm

# # #     lgs_angle_arcmin = -10.0
# # #     lgs_source_distance_mm = 3250.0

# # #     theta_chief = np.deg2rad(lgs_angle_arcmin / 60.0)

# # #     z_source = -lgs_source_distance_mm

# # #     # Choose source x so that chief ray crosses x=0 at z=0.
# # #     # theta = (0 - x_source) / (0 - z_source)
# # #     # therefore x_source = -theta * lgs_source_distance.
# # #     x_source = -theta_chief * lgs_source_distance_mm

# # #     # Initial ray states at input pupil z=0: [x, theta].
# # #     chief0 = np.array([0.0, theta_chief], dtype=float)

# # #     top0 = np.array(
# # #         [
# # #             +pupil_radius_mm,
# # #             (+pupil_radius_mm - x_source) / (0.0 - z_source),
# # #         ],
# # #         dtype=float,
# # #     )

# # #     bottom0 = np.array(
# # #         [
# # #             -pupil_radius_mm,
# # #             (-pupil_radius_mm - x_source) / (0.0 - z_source),
# # #         ],
# # #         dtype=float,
# # #     )

# # #     # ------------------------------------------------------------
# # #     # First-order optical layout
# # #     # ------------------------------------------------------------
# # #     # f=None means a fiducial plane, not a powered optic.
# # #     #
# # #     # Default layout:
# # #     #   input pupil
# # #     #   -> Lens 1
# # #     #   -> LGS image fiducial
# # #     #   -> Lens 2
# # #     #   -> DM pupil image
# # #     #   -> Lens 3
# # #     #   -> downstream optimisation planes

# # #     layout = [
# # #         dict(label="Input pupil", type="Pupil", z=0.0, f=None, ap_radius=pupil_radius_mm),

# # #         dict(label="Lens 1", type="Lens", z=150.0, f=150.0, ap_radius=25.0),

# # #         # For the default finite LGS source, Lens 1 forms the LGS image near ~307 mm.
# # #         dict(label="LGS image fiducial", type="Fiducial", z=307.0, f=None, ap_radius=5.0),

# # #         dict(label="Lens 2", type="Lens", z=450.0, f=150.0, ap_radius=25.0),

# # #         # Lens 1 + Lens 2 form a 1:1 pupil relay from z=0 to z=600 mm.
# # #         dict(label="DM pupil image", type="DM", z=600.0, f=None, ap_radius=pupil_radius_mm),

# # #         # One lens after the DM to start downstream optimisation.
# # #         dict(label="Lens 3 after DM", type="Lens", z=750.0, f=150.0, ap_radius=25.0),

# # #         dict(label="Downstream fiducial 1", type="Fiducial", z=900.0, f=None, ap_radius=20.0),
# # #         dict(label="Downstream fiducial 2", type="Fiducial", z=1050.0, f=None, ap_radius=20.0),
# # #     ]

# # #     target_dm_pupil_diameter_mm = pupil_diameter_mm

# # #     # ============================================================
# # #     # ABCD helpers
# # #     # ============================================================

# # #     def free_space(d_mm):
# # #         return np.array(
# # #             [
# # #                 [1.0, d_mm],
# # #                 [0.0, 1.0],
# # #             ],
# # #             dtype=float,
# # #         )

# # #     def thin_lens(f_mm):
# # #         return np.array(
# # #             [
# # #                 [1.0, 0.0],
# # #                 [-1.0 / f_mm, 1.0],
# # #             ],
# # #             dtype=float,
# # #         )

# # #     def crossing_distance(ray_a, ray_b):
# # #         """
# # #         Distance after the current plane where two rays cross.

# # #         ray = [x, theta]
# # #         x_a + d theta_a = x_b + d theta_b
# # #         """
# # #         xa, ta = ray_a
# # #         xb, tb = ray_b

# # #         denom = ta - tb

# # #         if abs(denom) < 1e-15:
# # #             return np.inf

# # #         return -(xa - xb) / denom

# # #     def pupil_conjugate_distance_after_plane(M):
# # #         """
# # #         Distance after current plane where the input pupil is reimaged.

# # #         M maps the input pupil plane to the current plane:

# # #             [x]     [A B] [x0]
# # #             [t]  =  [C D] [t0]

# # #         After free propagation by d:

# # #             B_new = B + d D

# # #         Pupil conjugate condition:

# # #             B_new = 0
# # #             d = -B/D
# # #         """
# # #         B = M[0, 1]
# # #         D = M[1, 1]

# # #         if abs(D) < 1e-15:
# # #             return np.inf

# # #         return -B / D

# # #     def image_from_object_thin_lens(z_object, z_lens, f):
# # #         """
# # #         Thin-lens image location using the sign convention:
# # #             s  = z_lens - z_object
# # #             s' = s f / (s - f)
# # #             z_image = z_lens + s'
# # #         """
# # #         s = z_lens - z_object

# # #         if abs(s - f) < 1e-15:
# # #             return np.inf, np.inf

# # #         sp = s * f / (s - f)
# # #         M = -sp / s

# # #         return z_lens + sp, M

# # #     # ============================================================
# # #     # Trace chief/top/bottom rays
# # #     # ============================================================

# # #     rows = []

# # #     M = np.eye(2)
# # #     z_prev = layout[0]["z"]

# # #     r_chief = chief0.copy()
# # #     r_top = top0.copy()
# # #     r_bottom = bottom0.copy()

# # #     for i, elem in enumerate(layout):
# # #         z = float(elem["z"])

# # #         if i == 0:
# # #             dz = 0.0
# # #         else:
# # #             dz = z - z_prev

# # #         if dz < -1e-12:
# # #             raise ValueError("Layout z positions must be monotonically increasing.")

# # #         # Free-space propagation to this plane.
# # #         P = free_space(dz)

# # #         M = P @ M
# # #         r_chief = P @ r_chief
# # #         r_top = P @ r_top
# # #         r_bottom = P @ r_bottom

# # #         # Apply powered optic if this is a lens.
# # #         if elem["type"].lower() == "lens":
# # #             L = thin_lens(float(elem["f"]))

# # #             M = L @ M
# # #             r_chief = L @ r_chief
# # #             r_top = L @ r_top
# # #             r_bottom = L @ r_bottom

# # #         beam_center = 0.5 * (r_top[0] + r_bottom[0])
# # #         beam_radius = 0.5 * abs(r_top[0] - r_bottom[0])

# # #         # Image estimate from actual top/bottom ray crossing after this plane.
# # #         d_img = crossing_distance(r_top, r_bottom)
# # #         z_img = z + d_img if np.isfinite(d_img) else np.inf
# # #         x_img_chief = r_chief[0] + d_img * r_chief[1] if np.isfinite(d_img) else np.nan

# # #         # Pupil conjugate estimate from ABCD B=0.
# # #         d_pup = pupil_conjugate_distance_after_plane(M)
# # #         z_pup = z + d_pup if np.isfinite(d_pup) else np.inf

# # #         ap_radius = float(elem.get("ap_radius", np.nan))

# # #         max_abs_ray_height = max(
# # #             abs(r_chief[0]),
# # #             abs(r_top[0]),
# # #             abs(r_bottom[0]),
# # #         )

# # #         clearance = ap_radius - max_abs_ray_height

# # #         rows.append(
# # #             dict(
# # #                 step=i,
# # #                 label=elem["label"],
# # #                 type=elem["type"],
# # #                 z_mm=z,
# # #                 dz_mm=dz,
# # #                 f_mm=elem["f"],
# # #                 A=M[0, 0],
# # #                 B=M[0, 1],
# # #                 C=M[1, 0],
# # #                 D=M[1, 1],
# # #                 chief_x_mm=r_chief[0],
# # #                 chief_theta_mrad=1e3 * r_chief[1],
# # #                 top_x_mm=r_top[0],
# # #                 top_theta_mrad=1e3 * r_top[1],
# # #                 bottom_x_mm=r_bottom[0],
# # #                 bottom_theta_mrad=1e3 * r_bottom[1],
# # #                 beam_center_mm=beam_center,
# # #                 beam_radius_mm=beam_radius,
# # #                 image_d_after_mm=d_img,
# # #                 image_z_mm=z_img,
# # #                 image_chief_x_mm=x_img_chief,
# # #                 pupil_d_after_mm=d_pup,
# # #                 pupil_z_mm=z_pup,
# # #                 ap_radius_mm=ap_radius,
# # #                 clearance_mm=clearance,
# # #             )
# # #         )

# # #         z_prev = z

# # #     # ============================================================
# # #     # Thin-lens object checks using original lens/obj classes
# # #     # ============================================================

# # #     lens_list = [
# # #         lens(x=e["z"], f=e["f"])
# # #         for e in layout
# # #         if e["type"].lower() == "lens"
# # #     ]

# # #     lgs_object = obj(x=z_source, h=x_source)
# # #     pupil_object = obj(x=0.0, h=pupil_radius_mm)

# # #     lgs_images = propagate_object(lens_list, lgs_object, plot=False)
# # #     pupil_images = propagate_object(lens_list, pupil_object, plot=False)

# # #     # Direct prediction for Lens 1 only.
# # #     z_lgs_img_l1, M_lgs_l1 = image_from_object_thin_lens(
# # #         z_object=z_source,
# # #         z_lens=layout[1]["z"],
# # #         f=layout[1]["f"],
# # #     )

# # #     # ============================================================
# # #     # Print summary
# # #     # ============================================================

# # #     print()
# # #     print("Single-LGS 2D thin-lens system")
# # #     print("=" * 80)
# # #     print(f"Input pupil diameter:       {pupil_diameter_mm:.3f} mm")
# # #     print(f"Input pupil radius:         {pupil_radius_mm:.3f} mm")
# # #     print(f"LGS field angle:            {lgs_angle_arcmin:.3f} arcmin")
# # #     print(f"LGS chief angle:            {theta_chief * 1e3:.6f} mrad")
# # #     print(f"LGS source z:               {z_source:.3f} mm")
# # #     print(f"LGS source x:               {x_source:.3f} mm")
# # #     print()
# # #     print("Lens-1 direct LGS image prediction")
# # #     print(f"  z image after Lens 1:      {z_lgs_img_l1:.3f} mm")
# # #     print(f"  image magnification:       {M_lgs_l1:.6f}")
# # #     print()

# # #     header = (
# # #         f"{'i':>2s} {'label':26s} {'type':10s} "
# # #         f"{'z':>9s} {'f':>9s} "
# # #         f"{'chief x':>10s} {'radius':>10s} "
# # #         f"{'image z':>11s} {'pupil z':>11s} "
# # #         f"{'clear':>10s}"
# # #     )

# # #     print(header)
# # #     print("-" * len(header))

# # #     for r in rows:
# # #         f_txt = "" if r["f_mm"] is None else f"{r['f_mm']:.3f}"
# # #         img_txt = "inf" if not np.isfinite(r["image_z_mm"]) else f"{r['image_z_mm']:.3f}"
# # #         pup_txt = "inf" if not np.isfinite(r["pupil_z_mm"]) else f"{r['pupil_z_mm']:.3f}"

# # #         print(
# # #             f"{r['step']:2d} "
# # #             f"{r['label'][:26]:26s} "
# # #             f"{r['type'][:10]:10s} "
# # #             f"{r['z_mm']:9.3f} "
# # #             f"{f_txt:>9s} "
# # #             f"{r['chief_x_mm']:10.3f} "
# # #             f"{r['beam_radius_mm']:10.3f} "
# # #             f"{img_txt:>11s} "
# # #             f"{pup_txt:>11s} "
# # #             f"{r['clearance_mm']:10.3f}"
# # #         )

# # #     print()
# # #     print("Original object/image propagation checks")
# # #     print(f"  Final LGS image/object:     x={lgs_images[-1].x:.3f} mm, h={lgs_images[-1].h:.3f} mm")
# # #     print(f"  Final pupil image/object:   x={pupil_images[-1].x:.3f} mm, h={pupil_images[-1].h:.3f} mm")

# # #     dm_rows = [r for r in rows if "DM" in r["label"]]

# # #     if dm_rows:
# # #         dm = dm_rows[0]
# # #         dm_diameter = 2.0 * dm["beam_radius_mm"]

# # #         print()
# # #         print("DM pupil image check")
# # #         print(f"  traced DM pupil diameter:   {dm_diameter:.3f} mm")
# # #         print(f"  target DM pupil diameter:   {target_dm_pupil_diameter_mm:.3f} mm")
# # #         print(f"  diameter error:             {dm_diameter - target_dm_pupil_diameter_mm:+.3f} mm")

# # #     print()
# # #     print("Aperture clearance check")
# # #     for r in rows:
# # #         status = "OK" if r["clearance_mm"] >= -1e-9 else "CLIPPING"
# # #         print(
# # #             f"  {r['label'][:28]:28s}: "
# # #             f"clearance={r['clearance_mm']:+9.3f} mm -> {status}"
# # #         )

# # #     # ============================================================
# # #     # Plot ray trace
# # #     # ============================================================

# # #     z_vals = np.array([r["z_mm"] for r in rows])
# # #     chief_x = np.array([r["chief_x_mm"] for r in rows])
# # #     top_x = np.array([r["top_x_mm"] for r in rows])
# # #     bottom_x = np.array([r["bottom_x_mm"] for r in rows])
# # #     radius = np.array([r["beam_radius_mm"] for r in rows])
# # #     centre = np.array([r["beam_center_mm"] for r in rows])

# # #     fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# # #     ax = axes[0]
# # #     ax.plot(z_vals, chief_x, "o-", label="chief ray")
# # #     ax.plot(z_vals, top_x, "o-", label="top ray")
# # #     ax.plot(z_vals, bottom_x, "o-", label="bottom ray")
# # #     ax.fill_between(
# # #         z_vals,
# # #         bottom_x,
# # #         top_x,
# # #         alpha=0.15,
# # #         label="beam envelope",
# # #     )

# # #     for r in rows:
# # #         if r["type"].lower() == "lens":
# # #             ax.axvline(r["z_mm"], color="k", linestyle="--", alpha=0.35)
# # #             ax.text(
# # #                 r["z_mm"],
# # #                 ax.get_ylim()[1],
# # #                 r["label"],
# # #                 rotation=90,
# # #                 va="top",
# # #                 ha="right",
# # #                 fontsize=8,
# # #             )

# # #     ax.set_ylabel("x [mm]")
# # #     ax.set_title("Single-LGS 2D ray trace")
# # #     ax.grid(True, alpha=0.3)
# # #     ax.legend(loc="best")

# # #     ax = axes[1]
# # #     ax.plot(z_vals, centre, "o-", label="beam centre")
# # #     ax.plot(z_vals, radius, "o-", label="beam radius")
# # #     ax.axhline(pupil_radius_mm, color="k", linestyle=":", alpha=0.5, label="input pupil radius")

# # #     for r in rows:
# # #         if np.isfinite(r["image_z_mm"]) and z_vals.min() <= r["image_z_mm"] <= z_vals.max():
# # #             ax.axvline(r["image_z_mm"], color="tab:red", linestyle="--", alpha=0.25)
# # #         if np.isfinite(r["pupil_z_mm"]) and z_vals.min() <= r["pupil_z_mm"] <= z_vals.max():
# # #             ax.axvline(r["pupil_z_mm"], color="tab:blue", linestyle=":", alpha=0.25)

# # #     ax.set_xlabel("z [mm]")
# # #     ax.set_ylabel("x / radius [mm]")
# # #     ax.grid(True, alpha=0.3)
# # #     ax.legend(loc="best")

# # #     plt.tight_layout()
# # #     plt.show()

# # #     return rows


# # # if __name__ == "__main__":
# # #     main()


# # # #%% sanity check following results from https://www.youtube.com/watch?v=aHHa0cK_3as
# # # if __name__=="__main__":
# # #     obj1= obj(x=-50, h=4e-3 ) 
# # #     l1 = lens(x= 0,        f= 30 )
# # #     l2 = lens(x= 10,        f= 20)

# # #     sys=[l1,l2]
# # #     test0 = propagate_object(sys, obj1, plot=True)

# # # #image formed at 25.3cm 


# # # #%% simulating Baldr optical design described in system described in baldr_calc_8

# # # o0 = obj(x=0,h=6) # pupil 

# # # #lenses
# # # l1 = lens(x= 2110, f = 254.016)
# # # l2 = lens(x= 2110 + 254.016 + 30.747, f= 30.747)
# # # l3 = lens(x= 2110 + 254.016 + 30.747 + 1200, f= 204.996 )

# # # #combined system
# # # sys = [l1,l2,l3]

# # # #propagate object through each lens
# # # objs = propagate_object(sys, o0, plot=True)

# # # for i,o in enumerate( objs[1:] ) :
# # #     print( f'object image position after lens {i+1} = {round(o.x)}mm')
    

# # # #%% # constraint 1 - check l1 virtual image DM edge (4mm) is at x=0 and h= 12mm/2 = 6mm
# # # # ++++++++++ relative distances between things (mm)
# # # x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
# # # x2 = 1000  # DM to lens 1 

# # # # ++++++++++ lens focal lens (mm) 
# # # f1 = 1500
# # # f2 = 254
# # # f3 = 30
# # # f4 = 200

# # # # ++++++++++ define lens 
# # # l1 = lens(x= x1 + x2,        f= f1 )

# # # DM_rad = 3.6 #mm
# # # oDM = obj(x=x1, h=DM_rad )  # DM

# # # # check l1 virtual image DM edge (4mm) is at x=0 and h= 12mm/2 = 6mm
# # # test1 = propagate_object([l1], oDM, plot=True)

# # # #%%  constraint 2 - lens 2 to FPM : star needs to be imaged on FPM 
# # # # ++++++++++ relative distances between things (mm)
# # # x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
# # # x2 = 1000  # DM to lens 1 
# # # x3 = 20  # lens 1 to OAP (lens 2) 

# # # # ++++++++++ lens focal lens (mm) 
# # # f1 = 1500
# # # f2 = 254
# # # f3 = 30
# # # f4 = 200

# # # # ++++++++++ define lens 
# # # l1 = lens(x= x1 + x2,        f= f1 )
# # # l2 = lens(x= l1.x + x3,      f= f2 )

# # # Sta_rad = 2 #mm
# # # oSta = obj(x=-1e20, h=Sta_rad )  # star

# # # # lens 2 to FPM (Constraint: star needs to be imaged on FPM )
# # # test2 = propagate_object([l1,l2], oSta, plot=True)
# # # print( f'l2.x  + x4 = {test2[-1].x} (Constraint: star needs to be imaged on FPM )\ntherefore x4 = {test2[-1].x-l2.x}')

# # # #%% # test 3 , given pupil edge image height propagated after lens3 set x6,x7 such that it matches our desired # pixels
# # # # we need to set x6,x7 such that lens 4 images h1 to h2 (constrained by det 
# # # # pitch and how many pixels we want to image across)

# # # # ++++++++++ relative distances between things (mm)
# # # x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
# # # x2 = 1000  # DM to lens 1 
# # # x3 = 20  # lens 1 to OAP (lens 2) 
# # # x4 = 216.794  # lens 2 to FPM (Constraint: star needs to be imaged on FPM )
# # # x5 = 30  # FPM to lens 3

# # # # ++++++++++ lens focal lens (mm) 
# # # f1 = 1500
# # # f2 = 254
# # # f3 = 30
# # # f4 = 200

# # # # ++++++++++ define lens 
# # # l1 = lens(x= x1 + x2,        f= f1 )
# # # l2 = lens(x= l1.x + x3,      f= f2 )
# # # l3 = lens(x= l2.x + x4 + x5, f= f3 )
# # # l4 = lens(x= np.nan,     f= f4 )
# # # Pup_rad = 2 #mm
# # # oPup = obj(x=0, h=Pup_rad ) #pupil
# # # test3 = propagate_object([l1,l2,l3], oPup, plot=True)

# # # # Detector 640x512 pixels with 15um pitch (image across 12 pixels = 180um, h = 90um)
# # # desired_h2 = -np.sign(test3[-1].h) * 90e-3 # mm ()



# # # def get_x6_x7(objs_, l3, l4, desired_h2):
    
# # #     """
# # #        obj_.x
# # #     l3  |         l4 (lens)
# # #     |   | h       |
# # #  ---|---.---------|-------| DET
# # #     |             |
# # #      x'    x_o      x_i
     
# # #     |-  - x6- -  -|- x7  -|
    
# # #     M=-x_i/x_o,  
# # #     x_i = (x_o * f) / (x_o-f)
    
# # #     wolfram solves x_o = f(M-1)/M
# # #     """
    
# # #     M = desired_h2 / objs_[-1].h
    
# # #     x_o = l4.f * (M-1)/M
    
# # #     x7 = -M * x_o

# # #     x6 =  (objs_[-1].x  - l3.x) + x_o
    
# # #     return( (x6,x7) )

# # # x6, x7 = get_x6_x7( test3, l3,l4, desired_h2 )
# # # print( f'x6={x6}, x7={x7} calculated such that pupil edge images to h2={desired_h2}   ')


# # #
# # #%% Lets see the complete system
# # # define pupil at x=0 !

# # # DM  BMC 492-1.5 aperture = 6.90mm, pitch (300um) , 24 actuators across pupil ,=> r = 3.6mm
# # # Detector 640x512 pixels with 15um pitch (image across 12 pixels = 180um, h =90um)

# # Sta_rad = 2 #mm
# # Pup_rad = 2 #mm
# # DM_rad = 3.6 #mm


# # # ++++++++++ relative distances between things (mm)
# # x1 = 2000  # pupil to DM (Constraint: lens 1 needs to image DM (virtually))
# # x2 = 1000  # DM to lens 1 
# # x3 = 20  # lens 1 to OAP (lens 2) 
# # x4 = 216.794  # lens 2 to FPM (Constraint: star needs to be imaged on FPM )
# # x5 = 30  # FPM to lens 3
# # x6 = 784.686 #1614.05 #1.200  # lens 3 to lens 4
# # x7 = 265.915 #250  # lens 4 to detector (Constraint: needs to image pupil )

# # x_s = [x1,x2,x3,x4,x5,x6,x7] #relative positions
# # z_s = np.cumsum( x_s ) #absolute positions 

# # # ++++++++++ lens focal lens (mm) 
# # f1 = 1500 #mm
# # f2 = 254
# # f3 = 30
# # f4 = 200

# # # ++++++++++ define lens 
# # l1 = lens(x= x1 + x2,        f= f1 )
# # l2 = lens(x= l1.x + x3,      f= f2 )
# # l3 = lens(x= l2.x + x4 + x5, f= f3 )
# # l4 = lens(x= l3.x + x6 ,     f= f4 )

# # # ++++++++++ define of objects to study (propagate)
# # oPup = obj(x=0, h=Pup_rad  ) #pupil
# # oSta = obj(x=-1e15, h=Sta_rad  )   # star
# # oDM = obj(x=x1, h=DM_rad )  # DM
# # oFPM = obj(x=l2.x + x4, h=0 ) #FPM

# # sys = [l1, l2, l3, l4]

# # # test
# # pup_ims = propagate_object(sys, oPup, plot=True)


# # print( pup_ims[-1].h ) #check it matches expectations 

