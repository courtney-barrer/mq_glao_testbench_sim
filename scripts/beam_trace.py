import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


"""
beam_trace.py

Geometric ray-tracing + rotating phase-screen module.

Main example:
- no relay optics
- 4 tilted collimated LGS beams + 1 slightly off-axis NGS
- all beams are 13 mm diameter
- beams originate from z = -3.25 m and converge to the same pupil-plane point at z = 0
- 3 ground-layer screens near the pupil plane
- 1 free-atmosphere screen higher upstream
- long-exposure PSFs evaluated at 5 field points
- per-screen phase contribution plots

This remains geometric / collimated:
- no Fresnel propagation
- no screen-slope refraction
- phase screens contribute OPD patches across the beam diameter


"""


"""
Main example: GLAO atmospheric test-bench simulation
====================================================

This example simulates a simplified geometric model of the atmospheric section
of a ground-layer adaptive optics (GLAO) test bench.

What is simulated
-----------------
The model contains:
- 4 laser guide star (LGS) beams
- 1 natural guide star (NGS) beam
- 3 rotating phase screens representing near-pupil / ground-layer turbulence
- 1 rotating phase screen representing a higher free-atmosphere layer

All beams are treated as collimated circular pupils of fixed diameter, and the
chief rays are tilted so that they converge to the same pupil-plane point.

Geometry
--------
The pupil plane is defined at:
    z = 0 m

The source plane is defined at:
    z = -3.25 m

The 4 LGS beams are placed on a ring such that they simulate a full field of
view of 20 arcmin. This corresponds to a half-angle of:

    theta_LGS = 10 arcmin = 5.81/2 mrad ≈ 2.905 mrad

Each LGS beam is a perfectly collimated beam of diameter:
    D_beam = 13 mm

and is tilted so that its chief ray intersects the common pupil-plane point.

The NGS beam is also modeled as a collimated 13 mm beam, but is placed at a
small off-axis position near the field centre.

Phase screens
-------------
All phase screens have diameter:
    D_screen = 83 mm

The screen planes are normal to the z-axis and located at:

Free atmosphere:
    z = -2.50 m

Ground layer:
    z = -96 mm
    z = -48 mm
    z = -24 mm

Each screen is represented by a 2D OPD map generated from a von Kármán-like
random phase realization, and is rotated in time at a prescribed angular
velocity.

How phase is applied
--------------------
This model is geometric rather than Fresnel-based.

For each beam and each phase screen:
- the chief-ray intersection with the screen is found
- a circular OPD patch, with the same diameter as the beam, is sampled from the
  rotating screen OPD map
- that OPD patch is added to the beam phase

The pupil-plane phase of a beam is therefore modeled as the sum of the sampled
screen OPD patches projected onto a common beam-sized pupil grid.

Assumptions and approximations
------------------------------
This is an intentionally simplified bench model. In particular:

1. Geometric propagation only
   Rays are traced geometrically. There is no Fresnel diffraction propagation
   between optical planes.

2. Collimated beams
   Each beam is represented as a collimated circular pupil with fixed diameter.
   The beam footprint does not expand or contract.

3. No refraction from screen slopes
   The phase screens add OPD only. Local gradients in the OPD map do not
   refract or steer rays.

4. Ideal pupil transport
   The sampled OPD patch is carried directly into the pupil-plane phase model.
   Magnification, distortion, and rotation of the pupil by relay optics are not
   modeled in this main example.

5. Monochromatic model
   The simulation is performed at a single wavelength:
       lambda = 589 nm

6. Uniform pupil amplitude
   The pupil amplitude is taken to be 1 inside the circular beam and 0 outside.

PSF calculation
---------------
For a given beam, the complex pupil field is constructed as:

    E(x, y) = A(x, y) * exp(i * phi(x, y))

where:
- A(x, y) is the circular pupil amplitude mask
- phi(x, y) is the accumulated phase from the sampled OPD maps

The PSF is then computed as the Fraunhofer intensity:

    PSF = | FFT(E) |^2

and normalized to unit peak.

Long-exposure PSF
-----------------
The long-exposure PSF is formed by averaging instantaneous PSFs over a discrete
set of times spanning the exposure duration.

In the current main example:
- the total simulated exposure is 3 s
- instantaneous PSFs are computed at discrete times separated by dt

Reference field points
----------------------
The long-exposure PSF is evaluated at 5 field points:
- 4 corners of the 20 arcmin field
- 1 central field point

These reference beams are generated with the same converging-collimated beam
model used for the LGS/NGS geometry.

Definition of FWHM
------------------
The PSF width is measured by fitting a 2D rotated Gaussian model to the cropped
PSF image:

    G(x, y) = offset + amp * exp(-0.5 * [ (x'/sigma_x)^2 + (y'/sigma_y)^2 ])

where (x', y') are coordinates rotated by an angle theta.

From the fitted Gaussian widths:
    FWHM_x = 2 * sqrt(2 * ln 2) * sigma_x
    FWHM_y = 2 * sqrt(2 * ln 2) * sigma_y

These are reported as:
- FWHM_major = max(FWHM_x, FWHM_y)
- FWHM_minor = min(FWHM_x, FWHM_y)

and are expressed in units of lambda / D.

Definition of ellipticity
-------------------------
The PSF ellipticity is defined from the fitted Gaussian widths as:

    ellipticity = 1 - FWHM_minor / FWHM_major

so that:
- ellipticity = 0 corresponds to a circular Gaussian
- larger ellipticity indicates a more elongated PSF

Contour shown on PSF plots
--------------------------
The white contour overlaid on each PSF plot is the half-maximum contour of the
fitted 2D Gaussian model. It is the ellipse corresponding to:

    G(x, y) = 0.5 * peak

and therefore traces the fitted FWHM shape and orientation.

Purpose of this example
-----------------------
This example is intended as a lightweight validation tool for:
- checking the geometric overlap of LGS beams on different phase screens
- assessing whether the chosen phase-screen layout plausibly mimics a GLAO bench
- visualizing how near-pupil and higher-altitude layers affect the pupil phase
- comparing long-exposure PSF quality across field points

It is not intended to be a high-fidelity physical propagation model.
"""


# ============================================================
# Helpers
# ============================================================

def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(n, ref))
    e2 = normalize(np.cross(n, e1))
    return e1, e2


def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def bilinear_sample(grid: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
    ny, nx = grid.shape

    x0 = np.floor(x_pix).astype(int)
    y0 = np.floor(y_pix).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, nx - 1)
    x1 = np.clip(x1, 0, nx - 1)
    y0 = np.clip(y0, 0, ny - 1)
    y1 = np.clip(y1, 0, ny - 1)

    Ia = grid[y0, x0]
    Ib = grid[y0, x1]
    Ic = grid[y1, x0]
    Id = grid[y1, x1]

    wa = (x1 - x_pix) * (y1 - y_pix)
    wb = (x_pix - x0) * (y1 - y_pix)
    wc = (x1 - x_pix) * (y_pix - y0)
    wd = (x_pix - x0) * (y_pix - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    psf = np.abs(ef) ** 2
    if np.max(psf) > 0:
        psf = psf / np.max(psf)

    ny, nx = field.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    return {"psf": psf, "fx": fx, "fy": fy}


def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
    x_ld = psf_pack["fx"] * pupil_diameter_m
    y_ld = psf_pack["fy"] * pupil_diameter_m
    return x_ld, y_ld


def crop_psf_to_lambda_over_d(
    psf: np.ndarray,
    x_ld: np.ndarray,
    y_ld: np.ndarray,
    half_width_ld: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
    iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
    if len(ix) == 0 or len(iy) == 0:
        return psf, x_ld, y_ld
    return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]


# ============================================================
# 2D Gaussian fitting for PSF
# ============================================================

def gaussian2d_rotated(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    ct = np.cos(theta)
    st = np.sin(theta)

    xp = ct * (x - x0) + st * (y - y0)
    yp = -st * (x - x0) + ct * (y - y0)

    g = offset + amp * np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))
    return g.ravel()


def second_moment_initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    z = np.asarray(z, dtype=float)
    z = z - np.nanmin(z)
    if np.nansum(z) <= 0:
        return None

    z = z / np.nansum(z)
    x0 = np.nansum(x * z)
    y0 = np.nansum(y * z)

    xx = np.nansum((x - x0) ** 2 * z)
    yy = np.nansum((y - y0) ** 2 * z)
    xy = np.nansum((x - x0) * (y - y0) * z)

    cov = np.array([[xx, xy], [xy, yy]])
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)

    sigma_minor = np.sqrt(evals[0])
    sigma_major = np.sqrt(evals[1])
    vec_major = evecs[:, 1]
    theta = np.arctan2(vec_major[1], vec_major[0])

    amp = np.nanmax(z)
    offset = 0.0
    return amp, x0, y0, sigma_major, sigma_minor, theta, offset


def fit_2d_gaussian(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, Any]:
    z = np.asarray(z, dtype=float)
    z_fit = z - np.nanmin(z)
    if np.nanmax(z_fit) > 0:
        z_fit = z_fit / np.nanmax(z_fit)

    guess = second_moment_initial_guess(x, y, z_fit)
    if guess is None:
        return {"success": False}

    if not SCIPY_AVAILABLE:
        amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
            "offset": offset,
            "method": "moments",
        }

    p0 = guess
    lower = [0.0, np.nanmin(x), np.nanmin(y), 1e-6, 1e-6, -np.pi, -0.5]
    upper = [2.0, np.nanmax(x), np.nanmax(y), 10.0, 10.0, np.pi, 1.0]

    try:
        popt, _ = curve_fit(
            gaussian2d_rotated,
            (x.ravel(), y.ravel()),
            z_fit.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        amp, x0, y0, sigma_x, sigma_y, theta, offset = popt
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
            "offset": offset,
            "method": "curve_fit",
        }
    except Exception:
        amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
        return {
            "success": True,
            "amp": amp,
            "x0": x0,
            "y0": y0,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
            "offset": offset,
            "method": "moments_fallback",
        }


def gaussian_fwhm_and_ellipticity(fit: Dict[str, Any]) -> Dict[str, float]:
    if not fit.get("success", False):
        return {
            "fwhm_major": np.nan,
            "fwhm_minor": np.nan,
            "ellipticity": np.nan,
        }

    sigma1 = max(fit["sigma_x"], fit["sigma_y"])
    sigma2 = min(fit["sigma_x"], fit["sigma_y"])
    factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    fwhm_major = factor * sigma1
    fwhm_minor = factor * sigma2
    ellipticity = 1.0 - fwhm_minor / fwhm_major if fwhm_major > 0 else np.nan

    return {
        "fwhm_major": float(fwhm_major),
        "fwhm_minor": float(fwhm_minor),
        "ellipticity": float(ellipticity),
    }


def gaussian_halfmax_contour(fit: Dict[str, Any], npts: int = 300):
    if not fit.get("success", False):
        return None, None

    sigma_x = fit["sigma_x"]
    sigma_y = fit["sigma_y"]
    theta = fit["theta"]
    x0 = fit["x0"]
    y0 = fit["y0"]

    t = np.linspace(0.0, 2.0 * np.pi, npts)
    a = np.sqrt(2.0 * np.log(2.0)) * sigma_x
    b = np.sqrt(2.0 * np.log(2.0)) * sigma_y

    xp = a * np.cos(t)
    yp = b * np.sin(t)

    ct = np.cos(theta)
    st = np.sin(theta)

    xc = x0 + ct * xp - st * yp
    yc = y0 + st * xp + ct * yp
    return xc, yc


# ============================================================
# Ray / beam
# ============================================================

@dataclass
class Ray3D:
    r: np.ndarray
    d: np.ndarray
    alive: bool = True
    opd: float = 0.0

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float).reshape(3)
        self.d = normalize(self.d)

    def copy(self):
        return Ray3D(self.r.copy(), self.d.copy(), self.alive, self.opd)


@dataclass
class Beam3D:
    rays: List[Ray3D]
    label: str = "beam"
    wavelength: float = 589e-9
    diameter: float = 13e-3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def chief_ray(self) -> Ray3D:
        return self.rays[0]

    @classmethod
    def collimated_circular(
        cls,
        radius: float = 6.5e-3,
        nrings: int = 3,
        nphi: int = 12,
        origin=(0, 0, 0),
        direction=(0, 0, 1),
        wavelength: float = 589e-9,
        label: str = "collimated",
    ):
        origin = np.asarray(origin, dtype=float)
        direction = normalize(direction)

        rays = [Ray3D(origin.copy(), direction.copy())]
        for ir in range(1, nrings + 1):
            rr = radius * ir / nrings
            n_this = max(8, nphi * ir)
            for k in range(n_this):
                phi = 2 * np.pi * k / n_this
                pos = origin + np.array([rr * np.cos(phi), rr * np.sin(phi), 0.0])
                rays.append(Ray3D(pos, direction.copy()))

        return cls(rays=rays, label=label, wavelength=wavelength, diameter=2 * radius)

    @classmethod
    def converging_collimated_from_point_to_pupil(
        cls,
        source_position: Tuple[float, float, float],
        pupil_point: Tuple[float, float, float],
        radius: float = 6.5e-3,
        nrings: int = 3,
        nphi: int = 12,
        wavelength: float = 589e-9,
        label: str = "beam",
    ):
        source_position = np.asarray(source_position, dtype=float)
        pupil_point = np.asarray(pupil_point, dtype=float)
        direction = normalize(pupil_point - source_position)
        return cls.collimated_circular(
            radius=radius,
            nrings=nrings,
            nphi=nphi,
            origin=source_position,
            direction=direction,
            wavelength=wavelength,
            label=label,
        )

    def copy(self) -> "Beam3D":
        return copy.deepcopy(self)


# ============================================================
# Optical elements
# ============================================================

@dataclass
class OpticalElement3D:
    point: np.ndarray
    normal: np.ndarray
    label: str = "element"

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=float).reshape(3)
        self.normal = normalize(self.normal)
        self._e1, self._e2 = orthonormal_basis_from_normal(self.normal)

    def intersect_parameter(self, ray: Ray3D, eps: float = 1e-12):
        denom = np.dot(ray.d, self.normal)
        if abs(denom) < eps:
            return None
        s = np.dot(self.point - ray.r, self.normal) / denom
        if s < 0:
            return None
        return s

    def intersect_point(self, ray: Ray3D):
        s = self.intersect_parameter(ray)
        if s is None:
            return None, None
        return ray.r + s * ray.d, s

    def plane_basis(self):
        return self._e1, self._e2

    def local_coordinates(self, p):
        dp = p - self.point
        return np.dot(dp, self._e1), np.dot(dp, self._e2)

    def apply(self, ray: Ray3D):
        return ray


@dataclass
class RotatingPhaseScreen3D(OpticalElement3D):
    opd_map: np.ndarray = None
    map_extent_m: float = 0.10
    clear_radius: float = 41.5e-3
    angular_velocity: float = 0.0
    rotation_angle0: float = 0.0
    label: str = "phase screen"

    def __post_init__(self):
        super().__post_init__()
        if self.opd_map is None:
            raise ValueError("opd_map must be provided.")
        self.opd_map = np.asarray(self.opd_map, dtype=float)
        if self.opd_map.ndim != 2:
            raise ValueError("opd_map must be a 2D array.")

    @property
    def map_ny(self) -> int:
        return self.opd_map.shape[0]

    @property
    def map_nx(self) -> int:
        return self.opd_map.shape[1]

    def current_rotation_angle(self, t: float) -> float:
        return self.rotation_angle0 + self.angular_velocity * t

    def uv_to_pixel(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = uv[..., 0]
        v = uv[..., 1]
        x_pix = (u / self.map_extent_m + 0.5) * (self.map_nx - 1)
        y_pix = (v / self.map_extent_m + 0.5) * (self.map_ny - 1)
        return x_pix, y_pix

    def contains_uv(self, uv: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(np.asarray(uv) ** 2, axis=-1))
        return r <= self.clear_radius

    def sample_uv(self, uv: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=float)
        rot = rotation_matrix_2d(-self.current_rotation_angle(t))
        uv_rot = np.einsum("ij,...j->...i", rot, uv)
        inside = self.contains_uv(uv_rot)

        x_pix, y_pix = self.uv_to_pixel(uv_rot)
        valid = inside & (x_pix >= 0) & (x_pix <= self.map_nx - 1) & (y_pix >= 0) & (y_pix <= self.map_ny - 1)

        opd = np.zeros_like(x_pix, dtype=float)
        if np.any(valid):
            opd[valid] = bilinear_sample(self.opd_map, x_pix[valid], y_pix[valid])
        return opd, valid

    def apply(self, ray: Ray3D, t: float = 0.0):
        p, _ = self.intersect_point(ray)
        if p is None:
            ray.alive = False
            return ray

        u, v = self.local_coordinates(p)
        opd, valid = self.sample_uv(np.array([[u, v]]), t=t)
        if not bool(valid[0]):
            ray.alive = False
            return ray

        ray.opd += float(opd[0])
        ray.r = p
        return ray


# ============================================================
# Bench
# ============================================================

@dataclass
class OpticalBench3D:
    elements: List[OpticalElement3D] = field(default_factory=list)

    def add(self, element: OpticalElement3D):
        self.elements.append(element)

    def trace_beam(self, beam: Beam3D, s_end: float = 0.2, n_line_samples: int = 60, t: float = 0.0):
        all_paths = []
        traced_rays = []

        for ray0 in beam.rays:
            ray = ray0.copy()
            path = [ray.r.copy()]

            for elem in self.elements:
                if not ray.alive:
                    break
                p, s = elem.intersect_point(ray)
                if p is None:
                    continue

                seg_s = np.linspace(0.0, s, n_line_samples)
                for ss in seg_s[1:]:
                    path.append(ray.r + ss * ray.d)

                if isinstance(elem, RotatingPhaseScreen3D):
                    ray = elem.apply(ray, t=t)
                else:
                    ray = elem.apply(ray)
                path.append(ray.r.copy())

            if ray.alive:
                seg_s = np.linspace(0.0, s_end, n_line_samples)
                for ss in seg_s[1:]:
                    path.append(ray.r + ss * ray.d)

            all_paths.append(np.array(path))
            traced_rays.append(ray)

        return all_paths, traced_rays

    def trace_chief_intersections(self, beam: Beam3D, t: float = 0.0) -> Dict[str, Dict[str, Any]]:
        ray = beam.chief_ray.copy()
        out = {}

        for elem in self.elements:
            if not ray.alive:
                break
            p, _ = elem.intersect_point(ray)
            if p is None:
                continue

            out[elem.label] = {
                "point": p.copy(),
                "direction_in": ray.d.copy(),
                "element": elem,
            }

            if isinstance(elem, RotatingPhaseScreen3D):
                ray = elem.apply(ray, t=t)
            else:
                ray = elem.apply(ray)

            out[elem.label]["direction_out"] = ray.d.copy()
            out[elem.label]["ray_after"] = ray.copy()

        out["final_ray"] = ray
        return out

    def plot_3d(self, beams: List[Beam3D], pupil_point: np.ndarray, s_end: float = 0.2, figsize=(10, 8), title: str = "3D optical bench", t: float = 0.0):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        for beam in beams:
            paths, _ = self.trace_beam(beam, s_end=s_end, t=t)
            for path in paths:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.0)

        for elem in self.elements:
            e1, e2 = elem.plane_basis()
            R = elem.clear_radius if isinstance(elem, RotatingPhaseScreen3D) else 1e-3
            tt = np.linspace(0, 2 * np.pi, 200)
            ring = elem.point[None, :] + R * (
                np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :]
            )
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], lw=2.0)
            ax.text(elem.point[0], elem.point[1], elem.point[2], elem.label)

        ax.scatter([pupil_point[0]], [pupil_point[1]], [pupil_point[2]], marker="x", s=80)
        ax.text(pupil_point[0], pupil_point[1], pupil_point[2], " pupil plane")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1.5])
        plt.tight_layout()
        return fig


# ============================================================
# Plane sampling / decomposition
# ============================================================

def sample_screen_patch_for_beam(
    screen: RotatingPhaseScreen3D,
    center_point: np.ndarray,
    beam_diameter: float,
    t: float,
    npix: int = 192,
) -> Dict[str, Any]:
    r = 0.5 * beam_diameter
    x = np.linspace(-r, r, npix)
    y = np.linspace(-r, r, npix)
    xx, yy = np.meshgrid(x, y)
    mask = (xx ** 2 + yy ** 2) <= r ** 2

    center_uv = screen.local_coordinates(center_point)
    uv = np.stack([center_uv[0] + xx, center_uv[1] + yy], axis=-1)

    opd, valid = screen.sample_uv(uv.reshape(-1, 2), t=t)
    opd = opd.reshape(xx.shape)
    valid = valid.reshape(xx.shape)

    opd = np.where(mask & valid, opd, 0.0)
    return {
        "xx": xx,
        "yy": yy,
        "mask": mask,
        "opd_map_m": np.where(mask, opd, np.nan),
    }


def sample_beam_phase_amplitude_on_pupil_plane(
    beam: Beam3D,
    bench: OpticalBench3D,
    pupil_point: np.ndarray,
    t: float,
    npix: int = 192,
    diameter: Optional[float] = None,
    screen_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    diameter = beam.diameter if diameter is None else diameter
    r = 0.5 * diameter
    x = np.linspace(-r, r, npix)
    y = np.linspace(-r, r, npix)
    xx, yy = np.meshgrid(x, y)
    mask = (xx ** 2 + yy ** 2) <= r ** 2

    chief_info = bench.trace_chief_intersections(beam, t=t)
    total_opd = np.zeros_like(xx, dtype=float)

    for elem in bench.elements:
        if not isinstance(elem, RotatingPhaseScreen3D):
            continue
        if screen_labels is not None and elem.label not in screen_labels:
            continue
        if elem.label not in chief_info:
            continue

        center_point = chief_info[elem.label]["point"]
        patch = sample_screen_patch_for_beam(
            screen=elem,
            center_point=center_point,
            beam_diameter=diameter,
            t=t,
            npix=npix,
        )
        total_opd += np.nan_to_num(patch["opd_map_m"], nan=0.0)

    total_opd = np.where(mask, total_opd, np.nan)
    phase = 2.0 * np.pi * total_opd / beam.wavelength
    dx = xx[0, 1] - xx[0, 0]
    amp = mask.astype(float)

    return {
        "xx": xx,
        "yy": yy,
        "mask": mask,
        "amplitude": amp,
        "opd_map_m": total_opd,
        "phase_map_rad": phase,
        "dx": dx,
    }


def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    mask = sample["mask"]
    phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
    amp = np.where(mask, sample["amplitude"], 0.0)
    field = amp * np.exp(1j * phase)
    out = fft_psf_from_pupil_field(field, sample["dx"])
    return {**sample, **out}


# ============================================================
# Example builders
# ============================================================

def make_von_karman_opd_map(
    n: int = 512,
    extent_m: float = 0.12,
    r0: float = 0.03,
    L0: float = 10.0,
    rms_opd_m: float = 150e-9,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dx = extent_m / n
    fx = np.fft.fftfreq(n, d=dx)
    fy = np.fft.fftfreq(n, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX ** 2 + FY ** 2)
    f0 = 1.0 / L0
    psd = (f ** 2 + f0 ** 2) ** (-11.0 / 6.0)
    psd[0, 0] = 0.0
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    screen = np.fft.ifft2(noise * np.sqrt(psd)).real
    screen -= np.mean(screen)
    if np.std(screen) > 0:
        screen *= (0.1 / r0) ** (5.0 / 6.0)
    if rms_opd_m is not None and np.std(screen) > 0:
        screen *= rms_opd_m / np.std(screen)
    return screen


def make_converging_beam_from_field_angles(
    theta_x_rad: float,
    theta_y_rad: float,
    source_plane_z: float,
    pupil_point: np.ndarray,
    beam_diameter: float,
    wavelength: float,
    label: str,
    nrings: int,
    nphi: int,
) -> Beam3D:
    dz = source_plane_z - pupil_point[2]
    x0 = -dz * np.tan(theta_x_rad)
    y0 = -dz * np.tan(theta_y_rad)
    source_pos = np.array([x0, y0, source_plane_z], dtype=float)
    return Beam3D.converging_collimated_from_point_to_pupil(
        source_position=source_pos,
        pupil_point=pupil_point,
        radius=0.5 * beam_diameter,
        nrings=nrings,
        nphi=nphi,
        wavelength=wavelength,
        label=label,
    )


def build_main_example(
    trace_nrings: int = 1,
    trace_nphi: int = 4,
    analysis_nrings: int = 3,
    analysis_nphi: int = 12,
) -> Tuple[OpticalBench3D, List[Beam3D], List[Beam3D], Beam3D, Dict[str, Any]]:
    wavelength = 589e-9
    beam_diameter = 13e-3
    pupil_point = np.array([0.0, 0.0, 0.0])

    source_plane_z = -3.25
    lgs_half_angle_rad = 5.81e-3 / 2.0
    lgs_ring_radius = abs(source_plane_z) * np.tan(lgs_half_angle_rad)

    bench = OpticalBench3D()

    screen_diameter = 83e-3
    screen_radius = 0.5 * screen_diameter
    screen_extent = 0.12

    screen_specs = [
        ("FA PS", -2.50, 180e-9, 0.4, 21, 0.03),
        ("GL PS 3", -0.096, 130e-9, 1.4, 13, 0.05),
        ("GL PS 2", -0.048, 110e-9, 1.0, 12, 0.05),
        ("GL PS 1", -0.024, 90e-9, 0.7, 11, 0.05),
    ]

    for label, z, rms, hz, seed, r0 in screen_specs:
        screen_map = make_von_karman_opd_map(
            n=512,
            extent_m=screen_extent,
            r0=r0,
            L0=10.0,
            rms_opd_m=rms,
            seed=seed,
        )
        bench.add(
            RotatingPhaseScreen3D(
                point=[0.0, 0.0, z],
                normal=[0.0, 0.0, 1.0],
                opd_map=screen_map,
                map_extent_m=screen_extent,
                clear_radius=screen_radius,
                angular_velocity=2 * np.pi * hz,
                label=label,
            )
        )

    lgs_az = np.deg2rad([45.0, 135.0, 225.0, 315.0])

    analysis_lgs_beams = []
    trace_lgs_beams = []

    for i, az in enumerate(lgs_az):
        source_pos = np.array([
            lgs_ring_radius * np.cos(az),
            lgs_ring_radius * np.sin(az),
            source_plane_z,
        ])
        analysis_lgs_beams.append(
            Beam3D.converging_collimated_from_point_to_pupil(
                source_position=source_pos,
                pupil_point=pupil_point,
                radius=0.5 * beam_diameter,
                nrings=analysis_nrings,
                nphi=analysis_nphi,
                wavelength=wavelength,
                label=f"lgs_{i+1}",
            )
        )
        trace_lgs_beams.append(
            Beam3D.converging_collimated_from_point_to_pupil(
                source_position=source_pos,
                pupil_point=pupil_point,
                radius=0.5 * beam_diameter,
                nrings=trace_nrings,
                nphi=trace_nphi,
                wavelength=wavelength,
                label=f"lgs_{i+1}",
            )
        )

    ngs_source_pos = np.array([
        0.22 * lgs_ring_radius,
        -0.14 * lgs_ring_radius,
        source_plane_z,
    ])

    ngs_beam = Beam3D.converging_collimated_from_point_to_pupil(
        source_position=ngs_source_pos,
        pupil_point=pupil_point,
        radius=0.5 * beam_diameter,
        nrings=analysis_nrings,
        nphi=analysis_nphi,
        wavelength=wavelength,
        label="ngs",
    )

    meta = {
        "pupil_point": pupil_point,
        "beam_diameter": beam_diameter,
        "wavelength": wavelength,
        "source_plane_z": source_plane_z,
        "lgs_half_angle_rad": lgs_half_angle_rad,
        "field_points_arcmin": [
            (+10.0, +10.0, "corner ++"),
            (-10.0, +10.0, "corner -+"),
            (+10.0, -10.0, "corner +-"),
            (-10.0, -10.0, "corner --"),
            (0.0, 0.0, "center"),
        ],
        "screen_labels": ["FA PS", "GL PS 1", "GL PS 2", "GL PS 3"],
        "trace_nrings": trace_nrings,
        "trace_nphi": trace_nphi,
        "analysis_nrings": analysis_nrings,
        "analysis_nphi": analysis_nphi,
    }
    return bench, analysis_lgs_beams, trace_lgs_beams, ngs_beam, meta


# ============================================================
# Example plotting routines
# ============================================================

def plot_field_map(ax, meta: Dict[str, Any]):
    pts = meta["field_points_arcmin"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    labels = [p[2] for p in pts]

    ax.scatter(xs, ys, s=80)
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, f"  {label}", va="center", ha="left")

    lim = max(max(np.abs(xs)), max(np.abs(ys))) + 2.0
    ax.axhline(0.0, color="0.7", lw=1)
    ax.axvline(0.0, color="0.7", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("field x [arcmin]")
    ax.set_ylabel("field y [arcmin]")
    ax.set_title("PSF sample locations")
    ax.grid(True, alpha=0.3)


def plot_field_long_exposure_psfs(
    bench: OpticalBench3D,
    meta: Dict[str, Any],
    exposure_s: float = 3.0,
    dt_s: float = 0.5,
    npix: int = 1024,
    half_width_ld: float = 5.0,
):
    times = np.arange(0.0, exposure_s, dt_s)
    pupil_point = meta["pupil_point"]
    beam_diameter = meta["beam_diameter"]
    wavelength = meta["wavelength"]
    source_plane_z = meta["source_plane_z"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    plot_field_map(axes[0], meta)

    for i, (fx_arcmin, fy_arcmin, label) in enumerate(meta["field_points_arcmin"], start=1):
        beam = make_converging_beam_from_field_angles(
            theta_x_rad=np.deg2rad(fx_arcmin / 60.0),
            theta_y_rad=np.deg2rad(fy_arcmin / 60.0),
            source_plane_z=source_plane_z,
            pupil_point=pupil_point,
            beam_diameter=beam_diameter,
            wavelength=wavelength,
            label=label,
            nrings=meta["analysis_nrings"],
            nphi=meta["analysis_nphi"],
        )

        stack = []
        first_pack = None
        for t in times:
            sample = sample_beam_phase_amplitude_on_pupil_plane(
                beam=beam,
                bench=bench,
                pupil_point=pupil_point,
                t=float(t),
                npix=npix,
                diameter=beam_diameter,
            )
            psf_pack = psf_from_plane_sample(sample)
            stack.append(psf_pack["psf"])
            if first_pack is None:
                first_pack = psf_pack

        long_psf = np.mean(np.array(stack), axis=0)
        if np.max(long_psf) > 0:
            long_psf = long_psf / np.max(long_psf)
        pack = dict(first_pack)
        pack["psf"] = long_psf

        x_ld, y_ld = psf_coords_lambda_over_d(pack, beam_diameter)
        psf_crop, x_crop, y_crop = crop_psf_to_lambda_over_d(pack["psf"], x_ld, y_ld, half_width_ld=half_width_ld)
        XX, YY = np.meshgrid(x_crop, y_crop)

        fit = fit_2d_gaussian(XX, YY, psf_crop)
        metrics = gaussian_fwhm_and_ellipticity(fit)
        xc, yc = gaussian_halfmax_contour(fit)

        ax = axes[i]
        im = ax.imshow(
            np.log10(np.maximum(psf_crop, 1e-10)),
            origin="lower",
            extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()],
        )
        if xc is not None:
            ax.plot(xc, yc, color="white", lw=1.2)

        ax.set_title(label)
        ax.set_xlabel(r"$\lambda/D$")
        ax.set_ylabel(r"$\lambda/D$")
        ax.set_aspect("equal")
        ax.text(
            0.04, 0.93,
            (
                f"FWHMmaj={metrics['fwhm_major']:.2f} $\\lambda/D$\n"
                f"FWHMmin={metrics['fwhm_minor']:.2f} $\\lambda/D$\n"
                f"ell={metrics['ellipticity']:.3f}"
            ),
            transform=ax.transAxes,
            color="white",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
        )
        plt.colorbar(im, ax=ax, label="log10 PSF")

    fig.suptitle(f"Long-exposure PSFs ({exposure_s:.1f} s, dt={dt_s:.2f} s)", y=0.98)
    fig.tight_layout()
    return fig


def plot_phase_screen_contributions(
    bench: OpticalBench3D,
    beams: List[Beam3D],
    meta: Dict[str, Any],
    t: float = 0.0,
    npix: int = 180,
):
    pupil_point = meta["pupil_point"]
    beam_diameter = meta["beam_diameter"]
    screen_labels = meta["screen_labels"]

    nrows = len(beams)
    ncols = len(screen_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)

    vmax = 0.0
    data_cache = {}
    for beam in beams:
        for screen_label in screen_labels:
            sample = sample_beam_phase_amplitude_on_pupil_plane(
                beam=beam,
                bench=bench,
                pupil_point=pupil_point,
                t=t,
                npix=npix,
                diameter=beam_diameter,
                screen_labels=[screen_label],
            )
            phase = sample["phase_map_rad"]
            data_cache[(beam.label, screen_label)] = sample
            vmax = max(vmax, np.nanmax(np.abs(np.nan_to_num(phase, nan=0.0))))

    if vmax <= 0:
        vmax = 1.0

    for i, beam in enumerate(beams):
        for j, screen_label in enumerate(screen_labels):
            sample = data_cache[(beam.label, screen_label)]
            xx = sample["xx"] * 1e3
            yy = sample["yy"] * 1e3
            ax = axes[i, j]
            im = ax.imshow(
                sample["phase_map_rad"],
                origin="lower",
                extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                vmin=-vmax,
                vmax=vmax,
            )
            if i == 0:
                ax.set_title(screen_label)
            if j == 0:
                ax.set_ylabel(f"{beam.label}\ny [mm]")
            else:
                ax.set_ylabel("y [mm]")
            ax.set_xlabel("x [mm]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="phase [rad]")

    fig.suptitle(f"Per-screen phase contributions at t = {t:.2f} s", y=0.995)
    fig.tight_layout()
    return fig


# ============================================================
# Main example
# ============================================================

if __name__ == "__main__":
    # Sparse LGS beams for 3D display, denser beams for calculations
    bench, analysis_lgs_beams, trace_lgs_beams, ngs_beam, meta = build_main_example(
        trace_nrings=1,
        trace_nphi=4,
        analysis_nrings=3,
        analysis_nphi=12,
    )

    # 3D plot shows only the 4 LGS beams
    fig1 = bench.plot_3d(
        beams=trace_lgs_beams,
        pupil_point=meta["pupil_point"],
        s_end=0.15,
        title=(
            "Converging 4-LGS geometry with rotating phase screens\n"
            f"3D trace uses sparse beams: nrings={meta['trace_nrings']}, nphi={meta['trace_nphi']}"
        ),
        t=0.0,
    )

    # Long-exposure PSFs reduced to 3 s
    fig2 = plot_field_long_exposure_psfs(
        bench=bench,
        meta=meta,
        exposure_s=3.0,
        dt_s=0.5,
        npix=1024,
        half_width_ld=5.0,
    )

    # Per-screen phase contribution plots use only the 4 LGS beams
    fig3 = plot_phase_screen_contributions(
        bench=bench,
        beams=analysis_lgs_beams,
        meta=meta,
        t=0.0,
        npix=180,
    )

    plt.show()

# import copy
# from dataclasses import dataclass, field
# from typing import List, Optional, Tuple, Dict, Any

# import matplotlib.pyplot as plt
# import numpy as np

# try:
#     from scipy.optimize import curve_fit
#     SCIPY_AVAILABLE = True
# except Exception:
#     SCIPY_AVAILABLE = False


# """
# beam_trace.py

# Geometric ray-tracing + rotating phase-screen module.

# Main example:
# - no relay optics
# - 4 tilted collimated LGS beams + 1 slightly off-axis NGS
# - all beams are 13 mm diameter
# - beams originate from z = -3.25 m and converge to the same pupil-plane point at z = 0
# - 3 ground-layer screens near the pupil plane
# - 1 free-atmosphere screen higher upstream
# - long-exposure PSFs evaluated at 5 field points
# - per-screen phase contribution plots

# This remains geometric / collimated:
# - no Fresnel propagation
# - no screen-slope refraction
# - phase screens contribute OPD patches across the beam diameter
# """


# # ============================================================
# # Helpers
# # ============================================================

# def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
#     v = np.asarray(v, dtype=float)
#     n = np.linalg.norm(v)
#     if n < eps:
#         raise ValueError("Cannot normalize near-zero vector.")
#     return v / n


# def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     n = normalize(normal)
#     ref = np.array([1.0, 0.0, 0.0])
#     if abs(np.dot(ref, n)) > 0.9:
#         ref = np.array([0.0, 1.0, 0.0])
#     e1 = normalize(np.cross(n, ref))
#     e2 = normalize(np.cross(n, e1))
#     return e1, e2


# def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
#     c = np.cos(angle_rad)
#     s = np.sin(angle_rad)
#     return np.array([[c, -s], [s, c]])


# def bilinear_sample(grid: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
#     ny, nx = grid.shape

#     x0 = np.floor(x_pix).astype(int)
#     y0 = np.floor(y_pix).astype(int)
#     x1 = x0 + 1
#     y1 = y0 + 1

#     x0 = np.clip(x0, 0, nx - 1)
#     x1 = np.clip(x1, 0, nx - 1)
#     y0 = np.clip(y0, 0, ny - 1)
#     y1 = np.clip(y1, 0, ny - 1)

#     Ia = grid[y0, x0]
#     Ib = grid[y0, x1]
#     Ic = grid[y1, x0]
#     Id = grid[y1, x1]

#     wa = (x1 - x_pix) * (y1 - y_pix)
#     wb = (x_pix - x0) * (y1 - y_pix)
#     wc = (x1 - x_pix) * (y_pix - y0)
#     wd = (x_pix - x0) * (y_pix - y0)

#     return wa * Ia + wb * Ib + wc * Ic + wd * Id


# def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
#     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
#     psf = np.abs(ef) ** 2
#     if np.max(psf) > 0:
#         psf = psf / np.max(psf)

#     ny, nx = field.shape
#     fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
#     fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
#     return {"psf": psf, "fx": fx, "fy": fy}


# def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
#     x_ld = psf_pack["fx"] * pupil_diameter_m
#     y_ld = psf_pack["fy"] * pupil_diameter_m
#     return x_ld, y_ld


# def crop_psf_to_lambda_over_d(
#     psf: np.ndarray,
#     x_ld: np.ndarray,
#     y_ld: np.ndarray,
#     half_width_ld: float = 5.0,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
#     iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
#     if len(ix) == 0 or len(iy) == 0:
#         return psf, x_ld, y_ld
#     return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]


# # ============================================================
# # 2D Gaussian fitting for PSF
# # ============================================================

# def gaussian2d_rotated(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
#     x, y = coords
#     ct = np.cos(theta)
#     st = np.sin(theta)

#     xp = ct * (x - x0) + st * (y - y0)
#     yp = -st * (x - x0) + ct * (y - y0)

#     g = offset + amp * np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))
#     return g.ravel()


# def second_moment_initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray):
#     z = np.asarray(z, dtype=float)
#     z = z - np.nanmin(z)
#     if np.nansum(z) <= 0:
#         return None

#     z = z / np.nansum(z)
#     x0 = np.nansum(x * z)
#     y0 = np.nansum(y * z)

#     xx = np.nansum((x - x0) ** 2 * z)
#     yy = np.nansum((y - y0) ** 2 * z)
#     xy = np.nansum((x - x0) * (y - y0) * z)

#     cov = np.array([[xx, xy], [xy, yy]])
#     evals, evecs = np.linalg.eigh(cov)
#     evals = np.clip(evals, 1e-12, None)

#     sigma_minor = np.sqrt(evals[0])
#     sigma_major = np.sqrt(evals[1])
#     vec_major = evecs[:, 1]
#     theta = np.arctan2(vec_major[1], vec_major[0])

#     amp = np.nanmax(z)
#     offset = 0.0
#     return amp, x0, y0, sigma_major, sigma_minor, theta, offset


# def fit_2d_gaussian(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, Any]:
#     z = np.asarray(z, dtype=float)
#     z_fit = z - np.nanmin(z)
#     if np.nanmax(z_fit) > 0:
#         z_fit = z_fit / np.nanmax(z_fit)

#     guess = second_moment_initial_guess(x, y, z_fit)
#     if guess is None:
#         return {"success": False}

#     if not SCIPY_AVAILABLE:
#         amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
#         return {
#             "success": True,
#             "amp": amp,
#             "x0": x0,
#             "y0": y0,
#             "sigma_x": sigma_x,
#             "sigma_y": sigma_y,
#             "theta": theta,
#             "offset": offset,
#             "method": "moments",
#         }

#     p0 = guess
#     lower = [0.0, np.nanmin(x), np.nanmin(y), 1e-6, 1e-6, -np.pi, -0.5]
#     upper = [2.0, np.nanmax(x), np.nanmax(y), 10.0, 10.0, np.pi, 1.0]

#     try:
#         popt, _ = curve_fit(
#             gaussian2d_rotated,
#             (x.ravel(), y.ravel()),
#             z_fit.ravel(),
#             p0=p0,
#             bounds=(lower, upper),
#             maxfev=20000,
#         )
#         amp, x0, y0, sigma_x, sigma_y, theta, offset = popt
#         return {
#             "success": True,
#             "amp": amp,
#             "x0": x0,
#             "y0": y0,
#             "sigma_x": sigma_x,
#             "sigma_y": sigma_y,
#             "theta": theta,
#             "offset": offset,
#             "method": "curve_fit",
#         }
#     except Exception:
#         amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
#         return {
#             "success": True,
#             "amp": amp,
#             "x0": x0,
#             "y0": y0,
#             "sigma_x": sigma_x,
#             "sigma_y": sigma_y,
#             "theta": theta,
#             "offset": offset,
#             "method": "moments_fallback",
#         }


# def gaussian_fwhm_and_ellipticity(fit: Dict[str, Any]) -> Dict[str, float]:
#     if not fit.get("success", False):
#         return {
#             "fwhm_major": np.nan,
#             "fwhm_minor": np.nan,
#             "ellipticity": np.nan,
#         }

#     sigma1 = max(fit["sigma_x"], fit["sigma_y"])
#     sigma2 = min(fit["sigma_x"], fit["sigma_y"])
#     factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

#     fwhm_major = factor * sigma1
#     fwhm_minor = factor * sigma2
#     ellipticity = 1.0 - fwhm_minor / fwhm_major if fwhm_major > 0 else np.nan

#     return {
#         "fwhm_major": float(fwhm_major),
#         "fwhm_minor": float(fwhm_minor),
#         "ellipticity": float(ellipticity),
#     }


# def gaussian_halfmax_contour(fit: Dict[str, Any], npts: int = 300):
#     if not fit.get("success", False):
#         return None, None

#     sigma_x = fit["sigma_x"]
#     sigma_y = fit["sigma_y"]
#     theta = fit["theta"]
#     x0 = fit["x0"]
#     y0 = fit["y0"]

#     t = np.linspace(0.0, 2.0 * np.pi, npts)
#     a = np.sqrt(2.0 * np.log(2.0)) * sigma_x
#     b = np.sqrt(2.0 * np.log(2.0)) * sigma_y

#     xp = a * np.cos(t)
#     yp = b * np.sin(t)

#     ct = np.cos(theta)
#     st = np.sin(theta)

#     xc = x0 + ct * xp - st * yp
#     yc = y0 + st * xp + ct * yp
#     return xc, yc


# # ============================================================
# # Ray / beam
# # ============================================================

# @dataclass
# class Ray3D:
#     r: np.ndarray
#     d: np.ndarray
#     alive: bool = True
#     opd: float = 0.0

#     def __post_init__(self):
#         self.r = np.asarray(self.r, dtype=float).reshape(3)
#         self.d = normalize(self.d)

#     def copy(self):
#         return Ray3D(self.r.copy(), self.d.copy(), self.alive, self.opd)


# @dataclass
# class Beam3D:
#     rays: List[Ray3D]
#     label: str = "beam"
#     wavelength: float = 589e-9
#     diameter: float = 13e-3
#     metadata: Dict[str, Any] = field(default_factory=dict)

#     @property
#     def radius(self) -> float:
#         return 0.5 * self.diameter

#     @property
#     def chief_ray(self) -> Ray3D:
#         return self.rays[0]

#     @classmethod
#     def collimated_circular(
#         cls,
#         radius: float = 6.5e-3,
#         nrings: int = 3,
#         nphi: int = 12,
#         origin=(0, 0, 0),
#         direction=(0, 0, 1),
#         wavelength: float = 589e-9,
#         label: str = "collimated",
#     ):
#         origin = np.asarray(origin, dtype=float)
#         direction = normalize(direction)

#         rays = [Ray3D(origin.copy(), direction.copy())]
#         for ir in range(1, nrings + 1):
#             rr = radius * ir / nrings
#             n_this = max(8, nphi * ir)
#             for k in range(n_this):
#                 phi = 2 * np.pi * k / n_this
#                 pos = origin + np.array([rr * np.cos(phi), rr * np.sin(phi), 0.0])
#                 rays.append(Ray3D(pos, direction.copy()))

#         return cls(rays=rays, label=label, wavelength=wavelength, diameter=2 * radius)

#     @classmethod
#     def converging_collimated_from_point_to_pupil(
#         cls,
#         source_position: Tuple[float, float, float],
#         pupil_point: Tuple[float, float, float],
#         radius: float = 6.5e-3,
#         nrings: int = 3,
#         nphi: int = 12,
#         wavelength: float = 589e-9,
#         label: str = "beam",
#     ):
#         source_position = np.asarray(source_position, dtype=float)
#         pupil_point = np.asarray(pupil_point, dtype=float)
#         direction = normalize(pupil_point - source_position)
#         return cls.collimated_circular(
#             radius=radius,
#             nrings=nrings,
#             nphi=nphi,
#             origin=source_position,
#             direction=direction,
#             wavelength=wavelength,
#             label=label,
#         )

#     def copy(self) -> "Beam3D":
#         return copy.deepcopy(self)


# # ============================================================
# # Optical elements
# # ============================================================

# @dataclass
# class OpticalElement3D:
#     point: np.ndarray
#     normal: np.ndarray
#     label: str = "element"

#     def __post_init__(self):
#         self.point = np.asarray(self.point, dtype=float).reshape(3)
#         self.normal = normalize(self.normal)
#         self._e1, self._e2 = orthonormal_basis_from_normal(self.normal)

#     def intersect_parameter(self, ray: Ray3D, eps: float = 1e-12):
#         denom = np.dot(ray.d, self.normal)
#         if abs(denom) < eps:
#             return None
#         s = np.dot(self.point - ray.r, self.normal) / denom
#         if s < 0:
#             return None
#         return s

#     def intersect_point(self, ray: Ray3D):
#         s = self.intersect_parameter(ray)
#         if s is None:
#             return None, None
#         return ray.r + s * ray.d, s

#     def plane_basis(self):
#         return self._e1, self._e2

#     def local_coordinates(self, p):
#         dp = p - self.point
#         return np.dot(dp, self._e1), np.dot(dp, self._e2)

#     def apply(self, ray: Ray3D):
#         return ray


# @dataclass
# class RotatingPhaseScreen3D(OpticalElement3D):
#     opd_map: np.ndarray = None
#     map_extent_m: float = 0.10
#     clear_radius: float = 41.5e-3
#     angular_velocity: float = 0.0
#     rotation_angle0: float = 0.0
#     label: str = "phase screen"

#     def __post_init__(self):
#         super().__post_init__()
#         if self.opd_map is None:
#             raise ValueError("opd_map must be provided.")
#         self.opd_map = np.asarray(self.opd_map, dtype=float)
#         if self.opd_map.ndim != 2:
#             raise ValueError("opd_map must be a 2D array.")

#     @property
#     def map_ny(self) -> int:
#         return self.opd_map.shape[0]

#     @property
#     def map_nx(self) -> int:
#         return self.opd_map.shape[1]

#     def current_rotation_angle(self, t: float) -> float:
#         return self.rotation_angle0 + self.angular_velocity * t

#     def uv_to_pixel(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         u = uv[..., 0]
#         v = uv[..., 1]
#         x_pix = (u / self.map_extent_m + 0.5) * (self.map_nx - 1)
#         y_pix = (v / self.map_extent_m + 0.5) * (self.map_ny - 1)
#         return x_pix, y_pix

#     def contains_uv(self, uv: np.ndarray) -> np.ndarray:
#         r = np.sqrt(np.sum(np.asarray(uv) ** 2, axis=-1))
#         return r <= self.clear_radius

#     def sample_uv(self, uv: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
#         uv = np.asarray(uv, dtype=float)
#         rot = rotation_matrix_2d(-self.current_rotation_angle(t))
#         uv_rot = np.einsum("ij,...j->...i", rot, uv)
#         inside = self.contains_uv(uv_rot)

#         x_pix, y_pix = self.uv_to_pixel(uv_rot)
#         valid = inside & (x_pix >= 0) & (x_pix <= self.map_nx - 1) & (y_pix >= 0) & (y_pix <= self.map_ny - 1)

#         opd = np.zeros_like(x_pix, dtype=float)
#         if np.any(valid):
#             opd[valid] = bilinear_sample(self.opd_map, x_pix[valid], y_pix[valid])
#         return opd, valid

#     def apply(self, ray: Ray3D, t: float = 0.0):
#         p, _ = self.intersect_point(ray)
#         if p is None:
#             ray.alive = False
#             return ray

#         u, v = self.local_coordinates(p)
#         opd, valid = self.sample_uv(np.array([[u, v]]), t=t)
#         if not bool(valid[0]):
#             ray.alive = False
#             return ray

#         ray.opd += float(opd[0])
#         ray.r = p
#         return ray


# # ============================================================
# # Bench
# # ============================================================

# @dataclass
# class OpticalBench3D:
#     elements: List[OpticalElement3D] = field(default_factory=list)

#     def add(self, element: OpticalElement3D):
#         self.elements.append(element)

#     def trace_beam(self, beam: Beam3D, s_end: float = 0.2, n_line_samples: int = 60, t: float = 0.0):
#         all_paths = []
#         traced_rays = []

#         for ray0 in beam.rays:
#             ray = ray0.copy()
#             path = [ray.r.copy()]

#             for elem in self.elements:
#                 if not ray.alive:
#                     break
#                 p, s = elem.intersect_point(ray)
#                 if p is None:
#                     continue

#                 seg_s = np.linspace(0.0, s, n_line_samples)
#                 for ss in seg_s[1:]:
#                     path.append(ray.r + ss * ray.d)

#                 if isinstance(elem, RotatingPhaseScreen3D):
#                     ray = elem.apply(ray, t=t)
#                 else:
#                     ray = elem.apply(ray)
#                 path.append(ray.r.copy())

#             if ray.alive:
#                 seg_s = np.linspace(0.0, s_end, n_line_samples)
#                 for ss in seg_s[1:]:
#                     path.append(ray.r + ss * ray.d)

#             all_paths.append(np.array(path))
#             traced_rays.append(ray)

#         return all_paths, traced_rays

#     def trace_chief_intersections(self, beam: Beam3D, t: float = 0.0) -> Dict[str, Dict[str, Any]]:
#         ray = beam.chief_ray.copy()
#         out = {}

#         for elem in self.elements:
#             if not ray.alive:
#                 break
#             p, _ = elem.intersect_point(ray)
#             if p is None:
#                 continue

#             out[elem.label] = {
#                 "point": p.copy(),
#                 "direction_in": ray.d.copy(),
#                 "element": elem,
#             }

#             if isinstance(elem, RotatingPhaseScreen3D):
#                 ray = elem.apply(ray, t=t)
#             else:
#                 ray = elem.apply(ray)

#             out[elem.label]["direction_out"] = ray.d.copy()
#             out[elem.label]["ray_after"] = ray.copy()

#         out["final_ray"] = ray
#         return out

#     def plot_3d(self, beams: List[Beam3D], pupil_point: np.ndarray, s_end: float = 0.2, figsize=(10, 8), title: str = "3D optical bench", t: float = 0.0):
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111, projection="3d")

#         for beam in beams:
#             paths, _ = self.trace_beam(beam, s_end=s_end, t=t)
#             for path in paths:
#                 ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.0)

#         for elem in self.elements:
#             e1, e2 = elem.plane_basis()
#             R = elem.clear_radius if isinstance(elem, RotatingPhaseScreen3D) else 1e-3
#             tt = np.linspace(0, 2 * np.pi, 200)
#             ring = elem.point[None, :] + R * (
#                 np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :]
#             )
#             ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], lw=2.0)
#             ax.text(elem.point[0], elem.point[1], elem.point[2], elem.label)

#         ax.scatter([pupil_point[0]], [pupil_point[1]], [pupil_point[2]], marker="x", s=80)
#         ax.text(pupil_point[0], pupil_point[1], pupil_point[2], " pupil plane")

#         ax.set_xlabel("x [m]")
#         ax.set_ylabel("y [m]")
#         ax.set_zlabel("z [m]")
#         ax.set_title(title)
#         ax.set_box_aspect([1, 1, 1.5])
#         plt.tight_layout()
#         return fig


# # ============================================================
# # Plane sampling / decomposition
# # ============================================================

# def sample_screen_patch_for_beam(
#     screen: RotatingPhaseScreen3D,
#     center_point: np.ndarray,
#     beam_diameter: float,
#     t: float,
#     npix: int = 192,
# ) -> Dict[str, Any]:
#     r = 0.5 * beam_diameter
#     x = np.linspace(-r, r, npix)
#     y = np.linspace(-r, r, npix)
#     xx, yy = np.meshgrid(x, y)
#     mask = (xx ** 2 + yy ** 2) <= r ** 2

#     center_uv = screen.local_coordinates(center_point)
#     uv = np.stack([center_uv[0] + xx, center_uv[1] + yy], axis=-1)

#     opd, valid = screen.sample_uv(uv.reshape(-1, 2), t=t)
#     opd = opd.reshape(xx.shape)
#     valid = valid.reshape(xx.shape)

#     opd = np.where(mask & valid, opd, 0.0)
#     return {
#         "xx": xx,
#         "yy": yy,
#         "mask": mask,
#         "opd_map_m": np.where(mask, opd, np.nan),
#     }


# def sample_beam_phase_amplitude_on_pupil_plane(
#     beam: Beam3D,
#     bench: OpticalBench3D,
#     pupil_point: np.ndarray,
#     t: float,
#     npix: int = 192,
#     diameter: Optional[float] = None,
#     screen_labels: Optional[List[str]] = None,
# ) -> Dict[str, Any]:
#     diameter = beam.diameter if diameter is None else diameter
#     r = 0.5 * diameter
#     x = np.linspace(-r, r, npix)
#     y = np.linspace(-r, r, npix)
#     xx, yy = np.meshgrid(x, y)
#     mask = (xx ** 2 + yy ** 2) <= r ** 2

#     chief_info = bench.trace_chief_intersections(beam, t=t)
#     total_opd = np.zeros_like(xx, dtype=float)

#     for elem in bench.elements:
#         if not isinstance(elem, RotatingPhaseScreen3D):
#             continue
#         if screen_labels is not None and elem.label not in screen_labels:
#             continue
#         if elem.label not in chief_info:
#             continue

#         center_point = chief_info[elem.label]["point"]
#         patch = sample_screen_patch_for_beam(
#             screen=elem,
#             center_point=center_point,
#             beam_diameter=diameter,
#             t=t,
#             npix=npix,
#         )
#         total_opd += np.nan_to_num(patch["opd_map_m"], nan=0.0)

#     total_opd = np.where(mask, total_opd, np.nan)
#     phase = 2.0 * np.pi * total_opd / beam.wavelength
#     dx = xx[0, 1] - xx[0, 0]
#     amp = mask.astype(float)

#     return {
#         "xx": xx,
#         "yy": yy,
#         "mask": mask,
#         "amplitude": amp,
#         "opd_map_m": total_opd,
#         "phase_map_rad": phase,
#         "dx": dx,
#     }


# def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
#     mask = sample["mask"]
#     phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
#     amp = np.where(mask, sample["amplitude"], 0.0)
#     field = amp * np.exp(1j * phase)
#     out = fft_psf_from_pupil_field(field, sample["dx"])
#     return {**sample, **out}


# # ============================================================
# # Example builders
# # ============================================================

# def make_von_karman_opd_map(
#     n: int = 512,
#     extent_m: float = 0.12,
#     r0: float = 0.03,
#     L0: float = 10.0,
#     rms_opd_m: float = 150e-9,
#     seed: Optional[int] = None,
# ) -> np.ndarray:
#     rng = np.random.default_rng(seed)
#     dx = extent_m / n
#     fx = np.fft.fftfreq(n, d=dx)
#     fy = np.fft.fftfreq(n, d=dx)
#     FX, FY = np.meshgrid(fx, fy)
#     f = np.sqrt(FX ** 2 + FY ** 2)
#     f0 = 1.0 / L0
#     psd = (f ** 2 + f0 ** 2) ** (-11.0 / 6.0)
#     psd[0, 0] = 0.0
#     noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
#     screen = np.fft.ifft2(noise * np.sqrt(psd)).real
#     screen -= np.mean(screen)
#     if np.std(screen) > 0:
#         screen *= (0.1 / r0) ** (5.0 / 6.0)
#     if rms_opd_m is not None and np.std(screen) > 0:
#         screen *= rms_opd_m / np.std(screen)
#     return screen


# def make_converging_beam_from_field_angles(
#     theta_x_rad: float,
#     theta_y_rad: float,
#     source_plane_z: float,
#     pupil_point: np.ndarray,
#     beam_diameter: float,
#     wavelength: float,
#     label: str,
#     nrings: int,
#     nphi: int,
# ) -> Beam3D:
#     dz = source_plane_z - pupil_point[2]
#     x0 = -dz * np.tan(theta_x_rad)
#     y0 = -dz * np.tan(theta_y_rad)
#     source_pos = np.array([x0, y0, source_plane_z], dtype=float)
#     return Beam3D.converging_collimated_from_point_to_pupil(
#         source_position=source_pos,
#         pupil_point=pupil_point,
#         radius=0.5 * beam_diameter,
#         nrings=nrings,
#         nphi=nphi,
#         wavelength=wavelength,
#         label=label,
#     )


# def build_main_example(
#     trace_nrings: int = 1,
#     trace_nphi: int = 4,
#     analysis_nrings: int = 3,
#     analysis_nphi: int = 12,
# ) -> Tuple[OpticalBench3D, List[Beam3D], List[Beam3D], Dict[str, Any]]:
#     wavelength = 589e-9
#     beam_diameter = 13e-3
#     pupil_point = np.array([0.0, 0.0, 0.0])

#     source_plane_z = -3.25
#     lgs_half_angle_rad = 5.81e-3 / 2.0
#     lgs_ring_radius = abs(source_plane_z) * np.tan(lgs_half_angle_rad)

#     bench = OpticalBench3D()

#     screen_diameter = 83e-3
#     screen_radius = 0.5 * screen_diameter
#     screen_extent = 0.12

#     screen_specs = [
#         ("FA PS", -2.50, 180e-9, 0.4, 21, 0.03),
#         ("GL PS 3", -0.096, 130e-9, 1.4, 13, 0.05),
#         ("GL PS 2", -0.048, 110e-9, 1.0, 12, 0.05),
#         ("GL PS 1", -0.024, 90e-9, 0.7, 11, 0.05),
#     ]

#     for label, z, rms, hz, seed, r0 in screen_specs:
#         screen_map = make_von_karman_opd_map(
#             n=512,
#             extent_m=screen_extent,
#             r0=r0,
#             L0=10.0,
#             rms_opd_m=rms,
#             seed=seed,
#         )
#         bench.add(
#             RotatingPhaseScreen3D(
#                 point=[0.0, 0.0, z],
#                 normal=[0.0, 0.0, 1.0],
#                 opd_map=screen_map,
#                 map_extent_m=screen_extent,
#                 clear_radius=screen_radius,
#                 angular_velocity=2 * np.pi * hz,
#                 label=label,
#             )
#         )

#     lgs_az = np.deg2rad([45.0, 135.0, 225.0, 315.0])

#     analysis_beams = []
#     trace_beams = []

#     for i, az in enumerate(lgs_az):
#         source_pos = np.array([
#             lgs_ring_radius * np.cos(az),
#             lgs_ring_radius * np.sin(az),
#             source_plane_z,
#         ])
#         analysis_beams.append(
#             Beam3D.converging_collimated_from_point_to_pupil(
#                 source_position=source_pos,
#                 pupil_point=pupil_point,
#                 radius=0.5 * beam_diameter,
#                 nrings=analysis_nrings,
#                 nphi=analysis_nphi,
#                 wavelength=wavelength,
#                 label=f"lgs_{i+1}",
#             )
#         )
#         trace_beams.append(
#             Beam3D.converging_collimated_from_point_to_pupil(
#                 source_position=source_pos,
#                 pupil_point=pupil_point,
#                 radius=0.5 * beam_diameter,
#                 nrings=trace_nrings,
#                 nphi=trace_nphi,
#                 wavelength=wavelength,
#                 label=f"lgs_{i+1}",
#             )
#         )

#     ngs_source_pos = np.array([
#         0.22 * lgs_ring_radius,
#         -0.14 * lgs_ring_radius,
#         source_plane_z,
#     ])

#     analysis_beams.append(
#         Beam3D.converging_collimated_from_point_to_pupil(
#             source_position=ngs_source_pos,
#             pupil_point=pupil_point,
#             radius=0.5 * beam_diameter,
#             nrings=analysis_nrings,
#             nphi=analysis_nphi,
#             wavelength=wavelength,
#             label="ngs",
#         )
#     )
#     trace_beams.append(
#         Beam3D.converging_collimated_from_point_to_pupil(
#             source_position=ngs_source_pos,
#             pupil_point=pupil_point,
#             radius=0.5 * beam_diameter,
#             nrings=trace_nrings,
#             nphi=trace_nphi,
#             wavelength=wavelength,
#             label="ngs",
#         )
#     )

#     meta = {
#         "pupil_point": pupil_point,
#         "beam_diameter": beam_diameter,
#         "wavelength": wavelength,
#         "source_plane_z": source_plane_z,
#         "lgs_half_angle_rad": lgs_half_angle_rad,
#         "field_points_arcmin": [
#             (+10.0, +10.0, "corner ++"),
#             (-10.0, +10.0, "corner -+"),
#             (+10.0, -10.0, "corner +-"),
#             (-10.0, -10.0, "corner --"),
#             (0.0, 0.0, "center"),
#         ],
#         "screen_labels": ["FA PS", "GL PS 1", "GL PS 2", "GL PS 3"],
#         "trace_nrings": trace_nrings,
#         "trace_nphi": trace_nphi,
#         "analysis_nrings": analysis_nrings,
#         "analysis_nphi": analysis_nphi,
#     }
#     return bench, analysis_beams, trace_beams, meta


# # ============================================================
# # Example plotting routines
# # ============================================================

# def plot_field_map(ax, meta: Dict[str, Any]):
#     pts = meta["field_points_arcmin"]
#     xs = [p[0] for p in pts]
#     ys = [p[1] for p in pts]
#     labels = [p[2] for p in pts]

#     ax.scatter(xs, ys, s=80)
#     for x, y, label in zip(xs, ys, labels):
#         ax.text(x, y, f"  {label}", va="center", ha="left")

#     lim = max(max(np.abs(xs)), max(np.abs(ys))) + 2.0
#     ax.axhline(0.0, color="0.7", lw=1)
#     ax.axvline(0.0, color="0.7", lw=1)
#     ax.set_xlim(-lim, lim)
#     ax.set_ylim(-lim, lim)
#     ax.set_aspect("equal")
#     ax.set_xlabel("field x [arcmin]")
#     ax.set_ylabel("field y [arcmin]")
#     ax.set_title("PSF sample locations")
#     ax.grid(True, alpha=0.3)


# def plot_field_long_exposure_psfs(
#     bench: OpticalBench3D,
#     meta: Dict[str, Any],
#     exposure_s: float = 30.0,
#     dt_s: float = 0.5,
#     npix: int = 1024,
#     half_width_ld: float = 5.0,
# ):
#     times = np.arange(0.0, exposure_s, dt_s)
#     pupil_point = meta["pupil_point"]
#     beam_diameter = meta["beam_diameter"]
#     wavelength = meta["wavelength"]
#     source_plane_z = meta["source_plane_z"]

#     fig, axes = plt.subplots(2, 3, figsize=(14, 9))
#     axes = axes.ravel()

#     plot_field_map(axes[0], meta)

#     for i, (fx_arcmin, fy_arcmin, label) in enumerate(meta["field_points_arcmin"], start=1):
#         beam = make_converging_beam_from_field_angles(
#             theta_x_rad=np.deg2rad(fx_arcmin / 60.0),
#             theta_y_rad=np.deg2rad(fy_arcmin / 60.0),
#             source_plane_z=source_plane_z,
#             pupil_point=pupil_point,
#             beam_diameter=beam_diameter,
#             wavelength=wavelength,
#             label=label,
#             nrings=meta["analysis_nrings"],
#             nphi=meta["analysis_nphi"],
#         )

#         stack = []
#         first_pack = None
#         for t in times:
#             sample = sample_beam_phase_amplitude_on_pupil_plane(
#                 beam=beam,
#                 bench=bench,
#                 pupil_point=pupil_point,
#                 t=float(t),
#                 npix=npix,
#                 diameter=beam_diameter,
#             )
#             psf_pack = psf_from_plane_sample(sample)
#             stack.append(psf_pack["psf"])
#             if first_pack is None:
#                 first_pack = psf_pack

#         long_psf = np.mean(np.array(stack), axis=0)
#         if np.max(long_psf) > 0:
#             long_psf = long_psf / np.max(long_psf)
#         pack = dict(first_pack)
#         pack["psf"] = long_psf

#         x_ld, y_ld = psf_coords_lambda_over_d(pack, beam_diameter)
#         psf_crop, x_crop, y_crop = crop_psf_to_lambda_over_d(pack["psf"], x_ld, y_ld, half_width_ld=half_width_ld)
#         XX, YY = np.meshgrid(x_crop, y_crop)

#         fit = fit_2d_gaussian(XX, YY, psf_crop)
#         metrics = gaussian_fwhm_and_ellipticity(fit)
#         xc, yc = gaussian_halfmax_contour(fit)

#         ax = axes[i]
#         im = ax.imshow(
#             np.log10(np.maximum(psf_crop, 1e-10)),
#             origin="lower",
#             extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()],
#         )
#         if xc is not None:
#             ax.plot(xc, yc, color="white", lw=1.2)

#         ax.set_title(label)
#         ax.set_xlabel(r"$\lambda/D$")
#         ax.set_ylabel(r"$\lambda/D$")
#         ax.set_aspect("equal")
#         ax.text(
#             0.04, 0.93,
#             (
#                 f"FWHMmaj={metrics['fwhm_major']:.2f} $\\lambda/D$\n"
#                 f"FWHMmin={metrics['fwhm_minor']:.2f} $\\lambda/D$\n"
#                 f"ell={metrics['ellipticity']:.3f}"
#             ),
#             transform=ax.transAxes,
#             color="white",
#             ha="left",
#             va="top",
#             fontsize=10,
#             bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
#         )
#         plt.colorbar(im, ax=ax, label="log10 PSF")

#     fig.suptitle(f"Long-exposure PSFs ({exposure_s:.1f} s, dt={dt_s:.2f} s)", y=0.98)
#     fig.tight_layout()
#     return fig


# def plot_phase_screen_contributions(
#     bench: OpticalBench3D,
#     beams: List[Beam3D],
#     meta: Dict[str, Any],
#     t: float = 0.0,
#     npix: int = 180,
# ):
#     pupil_point = meta["pupil_point"]
#     beam_diameter = meta["beam_diameter"]
#     screen_labels = meta["screen_labels"]

#     nrows = len(beams)
#     ncols = len(screen_labels)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)

#     vmax = 0.0
#     data_cache = {}
#     for beam in beams:
#         for screen_label in screen_labels:
#             sample = sample_beam_phase_amplitude_on_pupil_plane(
#                 beam=beam,
#                 bench=bench,
#                 pupil_point=pupil_point,
#                 t=t,
#                 npix=npix,
#                 diameter=beam_diameter,
#                 screen_labels=[screen_label],
#             )
#             phase = sample["phase_map_rad"]
#             data_cache[(beam.label, screen_label)] = sample
#             vmax = max(vmax, np.nanmax(np.abs(np.nan_to_num(phase, nan=0.0))))

#     if vmax <= 0:
#         vmax = 1.0

#     for i, beam in enumerate(beams):
#         for j, screen_label in enumerate(screen_labels):
#             sample = data_cache[(beam.label, screen_label)]
#             xx = sample["xx"] * 1e3
#             yy = sample["yy"] * 1e3
#             ax = axes[i, j]
#             im = ax.imshow(
#                 sample["phase_map_rad"],
#                 origin="lower",
#                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
#                 vmin=-vmax,
#                 vmax=vmax,
#             )
#             if i == 0:
#                 ax.set_title(screen_label)
#             if j == 0:
#                 ax.set_ylabel(f"{beam.label}\ny [mm]")
#             else:
#                 ax.set_ylabel("y [mm]")
#             ax.set_xlabel("x [mm]")
#             ax.set_aspect("equal")
#             plt.colorbar(im, ax=ax, label="phase [rad]")

#     fig.suptitle(f"Per-screen phase contributions at t = {t:.2f} s", y=0.995)
#     fig.tight_layout()
#     return fig


# # ============================================================
# # Main example
# # ============================================================

# if __name__ == "__main__":
#     # Fewer rays for the 3D trace, denser beams for PSF/phase calculations
#     bench, analysis_beams, trace_beams, meta = build_main_example(
#         trace_nrings=1,
#         trace_nphi=4,
#         analysis_nrings=3,
#         analysis_nphi=12,
#     )

#     fig1 = bench.plot_3d(
#         beams=trace_beams,
#         pupil_point=meta["pupil_point"],
#         s_end=0.15,
#         title=(
#             "Converging 4-LGS + off-axis NGS geometry with rotating phase screens\n"
#             f"3D trace uses sparse beams: nrings={meta['trace_nrings']}, nphi={meta['trace_nphi']}"
#         ),
#         t=0.0,
#     )

#     fig2 = plot_field_long_exposure_psfs(
#         bench=bench,
#         meta=meta,
#         exposure_s=30.0,
#         dt_s=0.5,
#         npix=1024,
#         half_width_ld=5.0,
#     )

#     fig3 = plot_phase_screen_contributions(
#         bench=bench,
#         beams=analysis_beams,
#         meta=meta,
#         t=0.0,
#         npix=180,
#     )

#     plt.show()

# # import copy
# # from dataclasses import dataclass, field
# # from typing import List, Optional, Tuple, Dict, Any

# # import matplotlib.pyplot as plt
# # import numpy as np

# # try:
# #     from scipy.optimize import curve_fit
# #     SCIPY_AVAILABLE = True
# # except Exception:
# #     SCIPY_AVAILABLE = False


# # """
# # beam_trace.py

# # Geometric ray-tracing + rotating phase-screen module.

# # Main example:
# # - no relay optics
# # - 4 tilted collimated LGS beams + 1 slightly off-axis NGS
# # - all beams are 13 mm diameter
# # - beams originate from z = -3.25 m and converge to the same pupil-plane point at z = 0
# # - 3 ground-layer screens near the pupil plane
# # - 1 free-atmosphere screen higher upstream
# # - long-exposure PSFs evaluated at 5 field points
# # - per-screen phase contribution plots
# # """


# # # ============================================================
# # # Helpers
# # # ============================================================

# # def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
# #     v = np.asarray(v, dtype=float)
# #     n = np.linalg.norm(v)
# #     if n < eps:
# #         raise ValueError("Cannot normalize near-zero vector.")
# #     return v / n


# # def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# #     n = normalize(normal)
# #     ref = np.array([1.0, 0.0, 0.0])
# #     if abs(np.dot(ref, n)) > 0.9:
# #         ref = np.array([0.0, 1.0, 0.0])
# #     e1 = normalize(np.cross(n, ref))
# #     e2 = normalize(np.cross(n, e1))
# #     return e1, e2


# # def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
# #     c = np.cos(angle_rad)
# #     s = np.sin(angle_rad)
# #     return np.array([[c, -s], [s, c]])


# # def bilinear_sample(grid: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
# #     ny, nx = grid.shape

# #     x0 = np.floor(x_pix).astype(int)
# #     y0 = np.floor(y_pix).astype(int)
# #     x1 = x0 + 1
# #     y1 = y0 + 1

# #     x0 = np.clip(x0, 0, nx - 1)
# #     x1 = np.clip(x1, 0, nx - 1)
# #     y0 = np.clip(y0, 0, ny - 1)
# #     y1 = np.clip(y1, 0, ny - 1)

# #     Ia = grid[y0, x0]
# #     Ib = grid[y0, x1]
# #     Ic = grid[y1, x0]
# #     Id = grid[y1, x1]

# #     wa = (x1 - x_pix) * (y1 - y_pix)
# #     wb = (x_pix - x0) * (y1 - y_pix)
# #     wc = (x1 - x_pix) * (y_pix - y0)
# #     wd = (x_pix - x0) * (y_pix - y0)

# #     return wa * Ia + wb * Ib + wc * Ic + wd * Id


# # def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
# #     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
# #     psf = np.abs(ef) ** 2
# #     if np.max(psf) > 0:
# #         psf = psf / np.max(psf)

# #     ny, nx = field.shape
# #     fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
# #     fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
# #     return {"psf": psf, "fx": fx, "fy": fy}


# # def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
# #     x_ld = psf_pack["fx"] * pupil_diameter_m
# #     y_ld = psf_pack["fy"] * pupil_diameter_m
# #     return x_ld, y_ld


# # def crop_psf_to_lambda_over_d(
# #     psf: np.ndarray,
# #     x_ld: np.ndarray,
# #     y_ld: np.ndarray,
# #     half_width_ld: float = 5.0,
# # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #     ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
# #     iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
# #     if len(ix) == 0 or len(iy) == 0:
# #         return psf, x_ld, y_ld
# #     return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]


# # # ============================================================
# # # 2D Gaussian fitting for PSF
# # # ============================================================

# # def gaussian2d_rotated(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
# #     x, y = coords
# #     ct = np.cos(theta)
# #     st = np.sin(theta)

# #     xp = ct * (x - x0) + st * (y - y0)
# #     yp = -st * (x - x0) + ct * (y - y0)

# #     g = offset + amp * np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))
# #     return g.ravel()


# # def second_moment_initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray):
# #     z = np.asarray(z, dtype=float)
# #     z = z - np.nanmin(z)
# #     if np.nansum(z) <= 0:
# #         return None

# #     z = z / np.nansum(z)
# #     x0 = np.nansum(x * z)
# #     y0 = np.nansum(y * z)

# #     xx = np.nansum((x - x0) ** 2 * z)
# #     yy = np.nansum((y - y0) ** 2 * z)
# #     xy = np.nansum((x - x0) * (y - y0) * z)

# #     cov = np.array([[xx, xy], [xy, yy]])
# #     evals, evecs = np.linalg.eigh(cov)
# #     evals = np.clip(evals, 1e-12, None)

# #     sigma_minor = np.sqrt(evals[0])
# #     sigma_major = np.sqrt(evals[1])
# #     vec_major = evecs[:, 1]
# #     theta = np.arctan2(vec_major[1], vec_major[0])

# #     amp = np.nanmax(z)
# #     offset = 0.0
# #     return amp, x0, y0, sigma_major, sigma_minor, theta, offset


# # def fit_2d_gaussian(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, Any]:
# #     z = np.asarray(z, dtype=float)
# #     z_fit = z - np.nanmin(z)
# #     if np.nanmax(z_fit) > 0:
# #         z_fit = z_fit / np.nanmax(z_fit)

# #     guess = second_moment_initial_guess(x, y, z_fit)
# #     if guess is None:
# #         return {"success": False}

# #     if not SCIPY_AVAILABLE:
# #         amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
# #         return {
# #             "success": True,
# #             "amp": amp,
# #             "x0": x0,
# #             "y0": y0,
# #             "sigma_x": sigma_x,
# #             "sigma_y": sigma_y,
# #             "theta": theta,
# #             "offset": offset,
# #             "method": "moments",
# #         }

# #     p0 = guess
# #     lower = [0.0, np.nanmin(x), np.nanmin(y), 1e-6, 1e-6, -np.pi, -0.5]
# #     upper = [2.0, np.nanmax(x), np.nanmax(y), 10.0, 10.0, np.pi, 1.0]

# #     try:
# #         popt, _ = curve_fit(
# #             gaussian2d_rotated,
# #             (x.ravel(), y.ravel()),
# #             z_fit.ravel(),
# #             p0=p0,
# #             bounds=(lower, upper),
# #             maxfev=20000,
# #         )
# #         amp, x0, y0, sigma_x, sigma_y, theta, offset = popt
# #         return {
# #             "success": True,
# #             "amp": amp,
# #             "x0": x0,
# #             "y0": y0,
# #             "sigma_x": sigma_x,
# #             "sigma_y": sigma_y,
# #             "theta": theta,
# #             "offset": offset,
# #             "method": "curve_fit",
# #         }
# #     except Exception:
# #         amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
# #         return {
# #             "success": True,
# #             "amp": amp,
# #             "x0": x0,
# #             "y0": y0,
# #             "sigma_x": sigma_x,
# #             "sigma_y": sigma_y,
# #             "theta": theta,
# #             "offset": offset,
# #             "method": "moments_fallback",
# #         }


# # def gaussian_fwhm_and_ellipticity(fit: Dict[str, Any]) -> Dict[str, float]:
# #     if not fit.get("success", False):
# #         return {
# #             "fwhm_major": np.nan,
# #             "fwhm_minor": np.nan,
# #             "ellipticity": np.nan,
# #         }

# #     sigma1 = max(fit["sigma_x"], fit["sigma_y"])
# #     sigma2 = min(fit["sigma_x"], fit["sigma_y"])
# #     factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

# #     fwhm_major = factor * sigma1
# #     fwhm_minor = factor * sigma2
# #     ellipticity = 1.0 - fwhm_minor / fwhm_major if fwhm_major > 0 else np.nan

# #     return {
# #         "fwhm_major": float(fwhm_major),
# #         "fwhm_minor": float(fwhm_minor),
# #         "ellipticity": float(ellipticity),
# #     }


# # def gaussian_halfmax_contour(fit: Dict[str, Any], npts: int = 300):
# #     if not fit.get("success", False):
# #         return None, None

# #     sigma_x = fit["sigma_x"]
# #     sigma_y = fit["sigma_y"]
# #     theta = fit["theta"]
# #     x0 = fit["x0"]
# #     y0 = fit["y0"]

# #     t = np.linspace(0.0, 2.0 * np.pi, npts)
# #     a = np.sqrt(2.0 * np.log(2.0)) * sigma_x
# #     b = np.sqrt(2.0 * np.log(2.0)) * sigma_y

# #     xp = a * np.cos(t)
# #     yp = b * np.sin(t)

# #     ct = np.cos(theta)
# #     st = np.sin(theta)

# #     xc = x0 + ct * xp - st * yp
# #     yc = y0 + st * xp + ct * yp
# #     return xc, yc


# # # ============================================================
# # # Ray / beam
# # # ============================================================

# # @dataclass
# # class Ray3D:
# #     r: np.ndarray
# #     d: np.ndarray
# #     alive: bool = True
# #     opd: float = 0.0

# #     def __post_init__(self):
# #         self.r = np.asarray(self.r, dtype=float).reshape(3)
# #         self.d = normalize(self.d)

# #     def copy(self):
# #         return Ray3D(self.r.copy(), self.d.copy(), self.alive, self.opd)


# # @dataclass
# # class Beam3D:
# #     rays: List[Ray3D]
# #     label: str = "beam"
# #     wavelength: float = 589e-9
# #     diameter: float = 13e-3
# #     metadata: Dict[str, Any] = field(default_factory=dict)

# #     @property
# #     def radius(self) -> float:
# #         return 0.5 * self.diameter

# #     @property
# #     def chief_ray(self) -> Ray3D:
# #         return self.rays[0]

# #     @classmethod
# #     def collimated_circular(
# #         cls,
# #         radius: float = 6.5e-3,
# #         nrings: int = 3,
# #         nphi: int = 12,
# #         origin=(0, 0, 0),
# #         direction=(0, 0, 1),
# #         wavelength: float = 589e-9,
# #         label: str = "collimated",
# #     ):
# #         origin = np.asarray(origin, dtype=float)
# #         direction = normalize(direction)

# #         rays = [Ray3D(origin.copy(), direction.copy())]
# #         for ir in range(1, nrings + 1):
# #             rr = radius * ir / nrings
# #             n_this = max(8, nphi * ir)
# #             for k in range(n_this):
# #                 phi = 2 * np.pi * k / n_this
# #                 pos = origin + np.array([rr * np.cos(phi), rr * np.sin(phi), 0.0])
# #                 rays.append(Ray3D(pos, direction.copy()))

# #         return cls(rays=rays, label=label, wavelength=wavelength, diameter=2 * radius)

# #     @classmethod
# #     def converging_collimated_from_point_to_pupil(
# #         cls,
# #         source_position: Tuple[float, float, float],
# #         pupil_point: Tuple[float, float, float],
# #         radius: float = 6.5e-3,
# #         nrings: int = 3,
# #         nphi: int = 12,
# #         wavelength: float = 589e-9,
# #         label: str = "beam",
# #     ):
# #         source_position = np.asarray(source_position, dtype=float)
# #         pupil_point = np.asarray(pupil_point, dtype=float)
# #         direction = normalize(pupil_point - source_position)
# #         return cls.collimated_circular(
# #             radius=radius,
# #             nrings=nrings,
# #             nphi=nphi,
# #             origin=source_position,
# #             direction=direction,
# #             wavelength=wavelength,
# #             label=label,
# #         )

# #     def copy(self) -> "Beam3D":
# #         return copy.deepcopy(self)


# # # ============================================================
# # # Optical elements
# # # ============================================================

# # @dataclass
# # class OpticalElement3D:
# #     point: np.ndarray
# #     normal: np.ndarray
# #     label: str = "element"

# #     def __post_init__(self):
# #         self.point = np.asarray(self.point, dtype=float).reshape(3)
# #         self.normal = normalize(self.normal)
# #         self._e1, self._e2 = orthonormal_basis_from_normal(self.normal)

# #     def intersect_parameter(self, ray: Ray3D, eps: float = 1e-12):
# #         denom = np.dot(ray.d, self.normal)
# #         if abs(denom) < eps:
# #             return None
# #         s = np.dot(self.point - ray.r, self.normal) / denom
# #         if s < 0:
# #             return None
# #         return s

# #     def intersect_point(self, ray: Ray3D):
# #         s = self.intersect_parameter(ray)
# #         if s is None:
# #             return None, None
# #         return ray.r + s * ray.d, s

# #     def plane_basis(self):
# #         return self._e1, self._e2

# #     def local_coordinates(self, p):
# #         dp = p - self.point
# #         return np.dot(dp, self._e1), np.dot(dp, self._e2)

# #     def apply(self, ray: Ray3D):
# #         return ray


# # @dataclass
# # class RotatingPhaseScreen3D(OpticalElement3D):
# #     opd_map: np.ndarray = None
# #     map_extent_m: float = 0.10
# #     clear_radius: float = 41.5e-3
# #     angular_velocity: float = 0.0
# #     rotation_angle0: float = 0.0
# #     label: str = "phase screen"

# #     def __post_init__(self):
# #         super().__post_init__()
# #         if self.opd_map is None:
# #             raise ValueError("opd_map must be provided.")
# #         self.opd_map = np.asarray(self.opd_map, dtype=float)
# #         if self.opd_map.ndim != 2:
# #             raise ValueError("opd_map must be a 2D array.")

# #     @property
# #     def map_ny(self) -> int:
# #         return self.opd_map.shape[0]

# #     @property
# #     def map_nx(self) -> int:
# #         return self.opd_map.shape[1]

# #     def current_rotation_angle(self, t: float) -> float:
# #         return self.rotation_angle0 + self.angular_velocity * t

# #     def uv_to_pixel(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# #         u = uv[..., 0]
# #         v = uv[..., 1]
# #         x_pix = (u / self.map_extent_m + 0.5) * (self.map_nx - 1)
# #         y_pix = (v / self.map_extent_m + 0.5) * (self.map_ny - 1)
# #         return x_pix, y_pix

# #     def contains_uv(self, uv: np.ndarray) -> np.ndarray:
# #         r = np.sqrt(np.sum(np.asarray(uv) ** 2, axis=-1))
# #         return r <= self.clear_radius

# #     def sample_uv(self, uv: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
# #         uv = np.asarray(uv, dtype=float)
# #         rot = rotation_matrix_2d(-self.current_rotation_angle(t))
# #         uv_rot = np.einsum("ij,...j->...i", rot, uv)
# #         inside = self.contains_uv(uv_rot)

# #         x_pix, y_pix = self.uv_to_pixel(uv_rot)
# #         valid = inside & (x_pix >= 0) & (x_pix <= self.map_nx - 1) & (y_pix >= 0) & (y_pix <= self.map_ny - 1)

# #         opd = np.zeros_like(x_pix, dtype=float)
# #         if np.any(valid):
# #             opd[valid] = bilinear_sample(self.opd_map, x_pix[valid], y_pix[valid])
# #         return opd, valid

# #     def apply(self, ray: Ray3D, t: float = 0.0):
# #         p, _ = self.intersect_point(ray)
# #         if p is None:
# #             ray.alive = False
# #             return ray

# #         u, v = self.local_coordinates(p)
# #         opd, valid = self.sample_uv(np.array([[u, v]]), t=t)
# #         if not bool(valid[0]):
# #             ray.alive = False
# #             return ray

# #         ray.opd += float(opd[0])
# #         ray.r = p
# #         return ray


# # # ============================================================
# # # Bench
# # # ============================================================

# # @dataclass
# # class OpticalBench3D:
# #     elements: List[OpticalElement3D] = field(default_factory=list)

# #     def add(self, element: OpticalElement3D):
# #         self.elements.append(element)

# #     def trace_beam(self, beam: Beam3D, s_end: float = 0.2, n_line_samples: int = 60, t: float = 0.0):
# #         all_paths = []
# #         traced_rays = []

# #         for ray0 in beam.rays:
# #             ray = ray0.copy()
# #             path = [ray.r.copy()]

# #             for elem in self.elements:
# #                 if not ray.alive:
# #                     break
# #                 p, s = elem.intersect_point(ray)
# #                 if p is None:
# #                     continue

# #                 seg_s = np.linspace(0.0, s, n_line_samples)
# #                 for ss in seg_s[1:]:
# #                     path.append(ray.r + ss * ray.d)

# #                 if isinstance(elem, RotatingPhaseScreen3D):
# #                     ray = elem.apply(ray, t=t)
# #                 else:
# #                     ray = elem.apply(ray)
# #                 path.append(ray.r.copy())

# #             if ray.alive:
# #                 seg_s = np.linspace(0.0, s_end, n_line_samples)
# #                 for ss in seg_s[1:]:
# #                     path.append(ray.r + ss * ray.d)

# #             all_paths.append(np.array(path))
# #             traced_rays.append(ray)

# #         return all_paths, traced_rays

# #     def trace_chief_intersections(self, beam: Beam3D, t: float = 0.0) -> Dict[str, Dict[str, Any]]:
# #         ray = beam.chief_ray.copy()
# #         out = {}

# #         for elem in self.elements:
# #             if not ray.alive:
# #                 break
# #             p, _ = elem.intersect_point(ray)
# #             if p is None:
# #                 continue

# #             out[elem.label] = {
# #                 "point": p.copy(),
# #                 "direction_in": ray.d.copy(),
# #                 "element": elem,
# #             }

# #             if isinstance(elem, RotatingPhaseScreen3D):
# #                 ray = elem.apply(ray, t=t)
# #             else:
# #                 ray = elem.apply(ray)

# #             out[elem.label]["direction_out"] = ray.d.copy()
# #             out[elem.label]["ray_after"] = ray.copy()

# #         out["final_ray"] = ray
# #         return out

# #     def plot_3d(self, beams: List[Beam3D], pupil_point: np.ndarray, s_end: float = 0.2, figsize=(10, 8), title: str = "3D optical bench", t: float = 0.0):
# #         fig = plt.figure(figsize=figsize)
# #         ax = fig.add_subplot(111, projection="3d")

# #         for beam in beams:
# #             paths, _ = self.trace_beam(beam, s_end=s_end, t=t)
# #             for path in paths:
# #                 ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.0)

# #         for elem in self.elements:
# #             e1, e2 = elem.plane_basis()
# #             R = elem.clear_radius if isinstance(elem, RotatingPhaseScreen3D) else 1e-3
# #             tt = np.linspace(0, 2 * np.pi, 200)
# #             ring = elem.point[None, :] + R * (
# #                 np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :]
# #             )
# #             ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], lw=2.0)
# #             ax.text(elem.point[0], elem.point[1], elem.point[2], elem.label)

# #         ax.scatter([pupil_point[0]], [pupil_point[1]], [pupil_point[2]], marker="x", s=80)
# #         ax.text(pupil_point[0], pupil_point[1], pupil_point[2], " pupil plane")

# #         ax.set_xlabel("x [m]")
# #         ax.set_ylabel("y [m]")
# #         ax.set_zlabel("z [m]")
# #         ax.set_title(title)
# #         ax.set_box_aspect([1, 1, 1.5])
# #         plt.tight_layout()
# #         return fig


# # # ============================================================
# # # Plane sampling / decomposition
# # # ============================================================

# # def sample_screen_patch_for_beam(
# #     screen: RotatingPhaseScreen3D,
# #     center_point: np.ndarray,
# #     beam_diameter: float,
# #     t: float,
# #     npix: int = 192,
# # ) -> Dict[str, Any]:
# #     r = 0.5 * beam_diameter
# #     x = np.linspace(-r, r, npix)
# #     y = np.linspace(-r, r, npix)
# #     xx, yy = np.meshgrid(x, y)
# #     mask = (xx ** 2 + yy ** 2) <= r ** 2

# #     center_uv = screen.local_coordinates(center_point)
# #     uv = np.stack([center_uv[0] + xx, center_uv[1] + yy], axis=-1)

# #     opd, valid = screen.sample_uv(uv.reshape(-1, 2), t=t)
# #     opd = opd.reshape(xx.shape)
# #     valid = valid.reshape(xx.shape)

# #     opd = np.where(mask & valid, opd, 0.0)
# #     return {
# #         "xx": xx,
# #         "yy": yy,
# #         "mask": mask,
# #         "opd_map_m": np.where(mask, opd, np.nan),
# #     }


# # def sample_beam_phase_amplitude_on_pupil_plane(
# #     beam: Beam3D,
# #     bench: OpticalBench3D,
# #     pupil_point: np.ndarray,
# #     t: float,
# #     npix: int = 192,
# #     diameter: Optional[float] = None,
# #     screen_labels: Optional[List[str]] = None,
# # ) -> Dict[str, Any]:
# #     diameter = beam.diameter if diameter is None else diameter
# #     r = 0.5 * diameter
# #     x = np.linspace(-r, r, npix)
# #     y = np.linspace(-r, r, npix)
# #     xx, yy = np.meshgrid(x, y)
# #     mask = (xx ** 2 + yy ** 2) <= r ** 2

# #     chief_info = bench.trace_chief_intersections(beam, t=t)
# #     total_opd = np.zeros_like(xx, dtype=float)

# #     for elem in bench.elements:
# #         if not isinstance(elem, RotatingPhaseScreen3D):
# #             continue
# #         if screen_labels is not None and elem.label not in screen_labels:
# #             continue
# #         if elem.label not in chief_info:
# #             continue

# #         center_point = chief_info[elem.label]["point"]
# #         patch = sample_screen_patch_for_beam(
# #             screen=elem,
# #             center_point=center_point,
# #             beam_diameter=diameter,
# #             t=t,
# #             npix=npix,
# #         )
# #         total_opd += np.nan_to_num(patch["opd_map_m"], nan=0.0)

# #     total_opd = np.where(mask, total_opd, np.nan)
# #     phase = 2.0 * np.pi * total_opd / beam.wavelength
# #     dx = xx[0, 1] - xx[0, 0]
# #     amp = mask.astype(float)

# #     return {
# #         "xx": xx,
# #         "yy": yy,
# #         "mask": mask,
# #         "amplitude": amp,
# #         "opd_map_m": total_opd,
# #         "phase_map_rad": phase,
# #         "dx": dx,
# #     }


# # def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
# #     mask = sample["mask"]
# #     phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
# #     amp = np.where(mask, sample["amplitude"], 0.0)
# #     field = amp * np.exp(1j * phase)
# #     out = fft_psf_from_pupil_field(field, sample["dx"])
# #     return {**sample, **out}


# # # ============================================================
# # # Example builders
# # # ============================================================

# # def make_von_karman_opd_map(
# #     n: int = 512,
# #     extent_m: float = 0.12,
# #     r0: float = 0.03,
# #     L0: float = 10.0,
# #     rms_opd_m: float = 150e-9,
# #     seed: Optional[int] = None,
# # ) -> np.ndarray:
# #     rng = np.random.default_rng(seed)
# #     dx = extent_m / n
# #     fx = np.fft.fftfreq(n, d=dx)
# #     fy = np.fft.fftfreq(n, d=dx)
# #     FX, FY = np.meshgrid(fx, fy)
# #     f = np.sqrt(FX ** 2 + FY ** 2)
# #     f0 = 1.0 / L0
# #     psd = (f ** 2 + f0 ** 2) ** (-11.0 / 6.0)
# #     psd[0, 0] = 0.0
# #     noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
# #     screen = np.fft.ifft2(noise * np.sqrt(psd)).real
# #     screen -= np.mean(screen)
# #     if np.std(screen) > 0:
# #         screen *= (0.1 / r0) ** (5.0 / 6.0)
# #     if rms_opd_m is not None and np.std(screen) > 0:
# #         screen *= rms_opd_m / np.std(screen)
# #     return screen


# # def build_main_example() -> Tuple[OpticalBench3D, List[Beam3D], Dict[str, Any]]:
# #     wavelength = 589e-9
# #     beam_diameter = 13e-3
# #     beam_radius = 0.5 * beam_diameter
# #     pupil_point = np.array([0.0, 0.0, 0.0])

# #     source_plane_z = -3.25
# #     lgs_half_angle_rad = 5.81e-3 / 2.0
# #     lgs_ring_radius = abs(source_plane_z) * np.tan(lgs_half_angle_rad)

# #     bench = OpticalBench3D()

# #     screen_diameter = 83e-3
# #     screen_radius = 0.5 * screen_diameter
# #     screen_extent = 0.12

# #     screen_specs = [
# #         ("FA PS", -2.50, 180e-9, 0.4, 21, 0.03),
# #         ("GL PS 3", -0.096, 130e-9, 1.4, 13, 0.05),
# #         ("GL PS 2", -0.048, 110e-9, 1.0, 12, 0.05),
# #         ("GL PS 1", -0.024, 90e-9, 0.7, 11, 0.05),
# #     ]

# #     for label, z, rms, hz, seed, r0 in screen_specs:
# #         screen_map = make_von_karman_opd_map(
# #             n=512,
# #             extent_m=screen_extent,
# #             r0=r0,
# #             L0=10.0,
# #             rms_opd_m=rms,
# #             seed=seed,
# #         )
# #         bench.add(
# #             RotatingPhaseScreen3D(
# #                 point=[0.0, 0.0, z],
# #                 normal=[0.0, 0.0, 1.0],
# #                 opd_map=screen_map,
# #                 map_extent_m=screen_extent,
# #                 clear_radius=screen_radius,
# #                 angular_velocity=2 * np.pi * hz,
# #                 label=label,
# #             )
# #         )

# #     lgs_az = np.deg2rad([45.0, 135.0, 225.0, 315.0])
# #     beams = []

# #     for i, az in enumerate(lgs_az):
# #         source_pos = np.array([
# #             lgs_ring_radius * np.cos(az),
# #             lgs_ring_radius * np.sin(az),
# #             source_plane_z,
# #         ])
# #         beams.append(
# #             Beam3D.converging_collimated_from_point_to_pupil(
# #                 source_position=source_pos,
# #                 pupil_point=pupil_point,
# #                 radius=beam_radius,
# #                 nrings=3,
# #                 nphi=12,
# #                 wavelength=wavelength,
# #                 label=f"lgs_{i+1}",
# #             )
# #         )

# #     # near-centre, mildly randomish off-axis NGS
# #     ngs_source_pos = np.array([
# #         0.22 * lgs_ring_radius,
# #         -0.14 * lgs_ring_radius,
# #         source_plane_z,
# #     ])
# #     beams.append(
# #         Beam3D.converging_collimated_from_point_to_pupil(
# #             source_position=ngs_source_pos,
# #             pupil_point=pupil_point,
# #             radius=beam_radius,
# #             nrings=3,
# #             nphi=12,
# #             wavelength=wavelength,
# #             label="ngs",
# #         )
# #     )

# #     meta = {
# #         "pupil_point": pupil_point,
# #         "beam_diameter": beam_diameter,
# #         "wavelength": wavelength,
# #         "source_plane_z": source_plane_z,
# #         "lgs_half_angle_rad": lgs_half_angle_rad,
# #         "field_points_arcmin": [
# #             (+10.0, +10.0, "corner ++"),
# #             (-10.0, +10.0, "corner -+"),
# #             (+10.0, -10.0, "corner +-"),
# #             (-10.0, -10.0, "corner --"),
# #             (0.0, 0.0, "center"),
# #         ],
# #         "screen_labels": ["FA PS", "GL PS 1", "GL PS 2", "GL PS 3"],
# #     }
# #     return bench, beams, meta


# # def make_converging_beam_from_field_angles(
# #     theta_x_rad: float,
# #     theta_y_rad: float,
# #     source_plane_z: float,
# #     pupil_point: np.ndarray,
# #     beam_diameter: float,
# #     wavelength: float,
# #     label: str,
# # ) -> Beam3D:
# #     dz = source_plane_z - pupil_point[2]
# #     x0 = -dz * np.tan(theta_x_rad)
# #     y0 = -dz * np.tan(theta_y_rad)
# #     source_pos = np.array([x0, y0, source_plane_z], dtype=float)
# #     return Beam3D.converging_collimated_from_point_to_pupil(
# #         source_position=source_pos,
# #         pupil_point=pupil_point,
# #         radius=0.5 * beam_diameter,
# #         nrings=3,
# #         nphi=12,
# #         wavelength=wavelength,
# #         label=label,
# #     )


# # # ============================================================
# # # Example plotting routines
# # # ============================================================

# # def plot_field_long_exposure_psfs(
# #     bench: OpticalBench3D,
# #     meta: Dict[str, Any],
# #     exposure_s: float = 30.0,
# #     dt_s: float = 0.5,
# #     npix: int = 1024,
# #     half_width_ld: float = 5.0,
# # ):
# #     times = np.arange(0.0, exposure_s, dt_s)
# #     pupil_point = meta["pupil_point"]
# #     beam_diameter = meta["beam_diameter"]
# #     wavelength = meta["wavelength"]
# #     source_plane_z = meta["source_plane_z"]

# #     fig, axes = plt.subplots(2, 3, figsize=(14, 9))
# #     axes = axes.ravel()

# #     for i, (fx_arcmin, fy_arcmin, label) in enumerate(meta["field_points_arcmin"]):
# #         beam = make_converging_beam_from_field_angles(
# #             theta_x_rad=np.deg2rad(fx_arcmin / 60.0),
# #             theta_y_rad=np.deg2rad(fy_arcmin / 60.0),
# #             source_plane_z=source_plane_z,
# #             pupil_point=pupil_point,
# #             beam_diameter=beam_diameter,
# #             wavelength=wavelength,
# #             label=label,
# #         )

# #         stack = []
# #         first_pack = None
# #         for t in times:
# #             sample = sample_beam_phase_amplitude_on_pupil_plane(
# #                 beam=beam,
# #                 bench=bench,
# #                 pupil_point=pupil_point,
# #                 t=float(t),
# #                 npix=npix,
# #                 diameter=beam_diameter,
# #             )
# #             psf_pack = psf_from_plane_sample(sample)
# #             stack.append(psf_pack["psf"])
# #             if first_pack is None:
# #                 first_pack = psf_pack

# #         long_psf = np.mean(np.array(stack), axis=0)
# #         if np.max(long_psf) > 0:
# #             long_psf = long_psf / np.max(long_psf)
# #         pack = dict(first_pack)
# #         pack["psf"] = long_psf

# #         x_ld, y_ld = psf_coords_lambda_over_d(pack, beam_diameter)
# #         psf_crop, x_crop, y_crop = crop_psf_to_lambda_over_d(pack["psf"], x_ld, y_ld, half_width_ld=half_width_ld)
# #         XX, YY = np.meshgrid(x_crop, y_crop)

# #         fit = fit_2d_gaussian(XX, YY, psf_crop)
# #         metrics = gaussian_fwhm_and_ellipticity(fit)
# #         xc, yc = gaussian_halfmax_contour(fit)

# #         ax = axes[i]
# #         im = ax.imshow(
# #             np.log10(np.maximum(psf_crop, 1e-10)),
# #             origin="lower",
# #             extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()],
# #         )
# #         if xc is not None:
# #             ax.plot(xc, yc, color="white", lw=1.2)

# #         ax.set_title(label)
# #         ax.set_xlabel(r"$\lambda/D$")
# #         ax.set_ylabel(r"$\lambda/D$")
# #         ax.set_aspect("equal")
# #         ax.text(
# #             0.04, 0.93,
# #             (
# #                 f"FWHMmaj={metrics['fwhm_major']:.2f} $\\lambda/D$\n"
# #                 f"FWHMmin={metrics['fwhm_minor']:.2f} $\\lambda/D$\n"
# #                 f"ell={metrics['ellipticity']:.3f}"
# #             ),
# #             transform=ax.transAxes,
# #             color="white",
# #             ha="left",
# #             va="top",
# #             fontsize=10,
# #             bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
# #         )
# #         plt.colorbar(im, ax=ax, label="log10 PSF")

# #     if len(axes) > len(meta["field_points_arcmin"]):
# #         axes[-1].axis("off")

# #     fig.suptitle(f"Long-exposure PSFs ({exposure_s:.1f} s, dt={dt_s:.2f} s)", y=0.98)
# #     fig.tight_layout()
# #     return fig


# # def plot_phase_screen_contributions(
# #     bench: OpticalBench3D,
# #     beams: List[Beam3D],
# #     meta: Dict[str, Any],
# #     t: float = 0.0,
# #     npix: int = 180,
# # ):
# #     pupil_point = meta["pupil_point"]
# #     beam_diameter = meta["beam_diameter"]
# #     screen_labels = meta["screen_labels"]

# #     nrows = len(beams)
# #     ncols = len(screen_labels)
# #     fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)

# #     vmax = 0.0
# #     data_cache = {}
# #     for beam in beams:
# #         for screen_label in screen_labels:
# #             sample = sample_beam_phase_amplitude_on_pupil_plane(
# #                 beam=beam,
# #                 bench=bench,
# #                 pupil_point=pupil_point,
# #                 t=t,
# #                 npix=npix,
# #                 diameter=beam_diameter,
# #                 screen_labels=[screen_label],
# #             )
# #             phase = sample["phase_map_rad"]
# #             data_cache[(beam.label, screen_label)] = sample
# #             vmax = max(vmax, np.nanmax(np.abs(np.nan_to_num(phase, nan=0.0))))

# #     if vmax <= 0:
# #         vmax = 1.0

# #     for i, beam in enumerate(beams):
# #         for j, screen_label in enumerate(screen_labels):
# #             sample = data_cache[(beam.label, screen_label)]
# #             xx = sample["xx"] * 1e3
# #             yy = sample["yy"] * 1e3
# #             ax = axes[i, j]
# #             im = ax.imshow(
# #                 sample["phase_map_rad"],
# #                 origin="lower",
# #                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
# #                 vmin=-vmax,
# #                 vmax=vmax,
# #             )
# #             if i == 0:
# #                 ax.set_title(screen_label)
# #             if j == 0:
# #                 ax.set_ylabel(f"{beam.label}\ny [mm]")
# #             else:
# #                 ax.set_ylabel("y [mm]")
# #             ax.set_xlabel("x [mm]")
# #             ax.set_aspect("equal")
# #             plt.colorbar(im, ax=ax, label="phase [rad]")

# #     fig.suptitle(f"Per-screen phase contributions at t = {t:.2f} s", y=0.995)
# #     fig.tight_layout()
# #     return fig


# # # ============================================================
# # # Main example
# # # ============================================================

# # if __name__ == "__main__":
# #     bench, beams, meta = build_main_example()

# #     fig1 = bench.plot_3d(
# #         beams=beams,
# #         pupil_point=meta["pupil_point"],
# #         s_end=0.15,
# #         title="Converging 4-LGS + off-axis NGS geometry with rotating phase screens",
# #         t=0.0,
# #     )

# #     fig2 = plot_field_long_exposure_psfs(
# #         bench=bench,
# #         meta=meta,
# #         exposure_s=30.0,
# #         dt_s=0.5,
# #         npix=1024,
# #         half_width_ld=5.0,
# #     )

# #     fig3 = plot_phase_screen_contributions(
# #         bench=bench,
# #         beams=beams,
# #         meta=meta,
# #         t=0.0,
# #         npix=180,
# #     )

# #     plt.show()

# # # import copy
# # # from dataclasses import dataclass, field
# # # from typing import List, Optional, Tuple, Dict, Any

# # # import matplotlib.pyplot as plt
# # # import numpy as np

# # # try:
# # #     from scipy.optimize import curve_fit
# # #     SCIPY_AVAILABLE = True
# # # except Exception:
# # #     SCIPY_AVAILABLE = False


# # # """
# # # Minimal ray-tracing + rotating phase-screen module.

# # # Key updates in this version
# # # ---------------------------
# # # 1. Phase screens now apply a spatially varying OPD patch across the beam diameter.
# # # 2. Beam phase on the analysis plane is constructed by summing the sampled OPD
# # #    contribution from each phase screen in beam-local coordinates.
# # # 3. Long-exposure PSFs are better sampled.
# # # 4. PSF size/shape is measured from a 2D Gaussian fit, with the fitted half-max
# # #    contour overplotted on each PSF panel.

# # # Still assumed
# # # -------------
# # # - geometric ray tracing
# # # - collimated beams
# # # - no Fresnel propagation
# # # - no slope/refraction from phase-screen gradients
# # # - relay optics transport the pupil coordinates ideally
# # # """


# # # # ============================================================
# # # # Helpers
# # # # ============================================================

# # # def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
# # #     v = np.asarray(v, dtype=float)
# # #     n = np.linalg.norm(v)
# # #     if n < eps:
# # #         raise ValueError("Cannot normalize near-zero vector.")
# # #     return v / n


# # # def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# # #     n = normalize(normal)
# # #     ref = np.array([1.0, 0.0, 0.0])
# # #     if abs(np.dot(ref, n)) > 0.9:
# # #         ref = np.array([0.0, 1.0, 0.0])
# # #     e1 = normalize(np.cross(n, ref))
# # #     e2 = normalize(np.cross(n, e1))
# # #     return e1, e2


# # # def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
# # #     c = np.cos(angle_rad)
# # #     s = np.sin(angle_rad)
# # #     return np.array([[c, -s], [s, c]])


# # # def bilinear_sample(grid: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
# # #     ny, nx = grid.shape

# # #     x0 = np.floor(x_pix).astype(int)
# # #     y0 = np.floor(y_pix).astype(int)
# # #     x1 = x0 + 1
# # #     y1 = y0 + 1

# # #     x0 = np.clip(x0, 0, nx - 1)
# # #     x1 = np.clip(x1, 0, nx - 1)
# # #     y0 = np.clip(y0, 0, ny - 1)
# # #     y1 = np.clip(y1, 0, ny - 1)

# # #     Ia = grid[y0, x0]
# # #     Ib = grid[y0, x1]
# # #     Ic = grid[y1, x0]
# # #     Id = grid[y1, x1]

# # #     wa = (x1 - x_pix) * (y1 - y_pix)
# # #     wb = (x_pix - x0) * (y1 - y_pix)
# # #     wc = (x1 - x_pix) * (y_pix - y0)
# # #     wd = (x_pix - x0) * (y_pix - y0)

# # #     return wa * Ia + wb * Ib + wc * Ic + wd * Id


# # # def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
# # #     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
# # #     psf = np.abs(ef) ** 2
# # #     if np.max(psf) > 0:
# # #         psf = psf / np.max(psf)

# # #     ny, nx = field.shape
# # #     fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
# # #     fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
# # #     return {"psf": psf, "fx": fx, "fy": fy}


# # # def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
# # #     x_ld = psf_pack["fx"] * pupil_diameter_m
# # #     y_ld = psf_pack["fy"] * pupil_diameter_m
# # #     return x_ld, y_ld


# # # def crop_psf_to_lambda_over_d(
# # #     psf: np.ndarray,
# # #     x_ld: np.ndarray,
# # #     y_ld: np.ndarray,
# # #     half_width_ld: float = 5.0,
# # # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #     ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
# # #     iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
# # #     if len(ix) == 0 or len(iy) == 0:
# # #         return psf, x_ld, y_ld
# # #     return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]


# # # # ============================================================
# # # # 2D Gaussian fitting for PSF
# # # # ============================================================

# # # def gaussian2d_rotated(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
# # #     x, y = coords
# # #     ct = np.cos(theta)
# # #     st = np.sin(theta)

# # #     xp = ct * (x - x0) + st * (y - y0)
# # #     yp = -st * (x - x0) + ct * (y - y0)

# # #     g = offset + amp * np.exp(
# # #         -0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2)
# # #     )
# # #     return g.ravel()


# # # def second_moment_initial_guess(x: np.ndarray, y: np.ndarray, z: np.ndarray):
# # #     z = np.asarray(z, dtype=float)
# # #     z = z - np.nanmin(z)
# # #     if np.nansum(z) <= 0:
# # #         return None

# # #     z = z / np.nansum(z)
# # #     x0 = np.nansum(x * z)
# # #     y0 = np.nansum(y * z)

# # #     xx = np.nansum((x - x0) ** 2 * z)
# # #     yy = np.nansum((y - y0) ** 2 * z)
# # #     xy = np.nansum((x - x0) * (y - y0) * z)

# # #     cov = np.array([[xx, xy], [xy, yy]])
# # #     evals, evecs = np.linalg.eigh(cov)
# # #     evals = np.clip(evals, 1e-12, None)

# # #     sigma_minor = np.sqrt(evals[0])
# # #     sigma_major = np.sqrt(evals[1])
# # #     vec_major = evecs[:, 1]
# # #     theta = np.arctan2(vec_major[1], vec_major[0])

# # #     amp = np.nanmax(z)
# # #     offset = 0.0
# # #     return amp, x0, y0, sigma_major, sigma_minor, theta, offset


# # # def fit_2d_gaussian(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, Any]:
# # #     z = np.asarray(z, dtype=float)
# # #     z_fit = z - np.nanmin(z)
# # #     if np.nanmax(z_fit) > 0:
# # #         z_fit = z_fit / np.nanmax(z_fit)

# # #     guess = second_moment_initial_guess(x, y, z_fit)
# # #     if guess is None:
# # #         return {"success": False}

# # #     if not SCIPY_AVAILABLE:
# # #         amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
# # #         return {
# # #             "success": True,
# # #             "amp": amp,
# # #             "x0": x0,
# # #             "y0": y0,
# # #             "sigma_x": sigma_x,
# # #             "sigma_y": sigma_y,
# # #             "theta": theta,
# # #             "offset": offset,
# # #             "method": "moments",
# # #         }

# # #     p0 = guess
# # #     lower = [0.0, np.nanmin(x), np.nanmin(y), 1e-6, 1e-6, -np.pi, -0.5]
# # #     upper = [2.0, np.nanmax(x), np.nanmax(y), 10.0, 10.0, np.pi, 1.0]

# # #     try:
# # #         popt, _ = curve_fit(
# # #             gaussian2d_rotated,
# # #             (x.ravel(), y.ravel()),
# # #             z_fit.ravel(),
# # #             p0=p0,
# # #             bounds=(lower, upper),
# # #             maxfev=20000,
# # #         )
# # #         amp, x0, y0, sigma_x, sigma_y, theta, offset = popt
# # #         return {
# # #             "success": True,
# # #             "amp": amp,
# # #             "x0": x0,
# # #             "y0": y0,
# # #             "sigma_x": sigma_x,
# # #             "sigma_y": sigma_y,
# # #             "theta": theta,
# # #             "offset": offset,
# # #             "method": "curve_fit",
# # #         }
# # #     except Exception:
# # #         amp, x0, y0, sigma_x, sigma_y, theta, offset = guess
# # #         return {
# # #             "success": True,
# # #             "amp": amp,
# # #             "x0": x0,
# # #             "y0": y0,
# # #             "sigma_x": sigma_x,
# # #             "sigma_y": sigma_y,
# # #             "theta": theta,
# # #             "offset": offset,
# # #             "method": "moments_fallback",
# # #         }


# # # def gaussian_fwhm_and_ellipticity(fit: Dict[str, Any]) -> Dict[str, float]:
# # #     if not fit.get("success", False):
# # #         return {
# # #             "fwhm_major": np.nan,
# # #             "fwhm_minor": np.nan,
# # #             "ellipticity": np.nan,
# # #         }

# # #     sigma1 = max(fit["sigma_x"], fit["sigma_y"])
# # #     sigma2 = min(fit["sigma_x"], fit["sigma_y"])
# # #     factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

# # #     fwhm_major = factor * sigma1
# # #     fwhm_minor = factor * sigma2
# # #     ellipticity = 1.0 - fwhm_minor / fwhm_major if fwhm_major > 0 else np.nan

# # #     return {
# # #         "fwhm_major": float(fwhm_major),
# # #         "fwhm_minor": float(fwhm_minor),
# # #         "ellipticity": float(ellipticity),
# # #     }


# # # def gaussian_halfmax_contour(x: np.ndarray, y: np.ndarray, fit: Dict[str, Any], npts: int = 300):
# # #     if not fit.get("success", False):
# # #         return None, None

# # #     sigma_x = fit["sigma_x"]
# # #     sigma_y = fit["sigma_y"]
# # #     theta = fit["theta"]
# # #     x0 = fit["x0"]
# # #     y0 = fit["y0"]

# # #     t = np.linspace(0.0, 2.0 * np.pi, npts)
# # #     a = np.sqrt(2.0 * np.log(2.0)) * sigma_x
# # #     b = np.sqrt(2.0 * np.log(2.0)) * sigma_y

# # #     xp = a * np.cos(t)
# # #     yp = b * np.sin(t)

# # #     ct = np.cos(theta)
# # #     st = np.sin(theta)

# # #     xc = x0 + ct * xp - st * yp
# # #     yc = y0 + st * xp + ct * yp
# # #     return xc, yc


# # # # ============================================================
# # # # Ray / beam
# # # # ============================================================

# # # @dataclass
# # # class Ray3D:
# # #     r: np.ndarray
# # #     d: np.ndarray
# # #     alive: bool = True
# # #     opd: float = 0.0

# # #     def __post_init__(self):
# # #         self.r = np.asarray(self.r, dtype=float).reshape(3)
# # #         self.d = normalize(self.d)

# # #     def copy(self):
# # #         return Ray3D(self.r.copy(), self.d.copy(), self.alive, self.opd)


# # # @dataclass
# # # class Beam3D:
# # #     rays: List[Ray3D]
# # #     label: str = "beam"
# # #     wavelength: float = 532e-9
# # #     diameter: float = 10e-3
# # #     metadata: Dict[str, Any] = field(default_factory=dict)

# # #     @property
# # #     def radius(self) -> float:
# # #         return 0.5 * self.diameter

# # #     @property
# # #     def chief_ray(self) -> Ray3D:
# # #         return self.rays[0]

# # #     @classmethod
# # #     def collimated_circular(
# # #         cls,
# # #         radius: float = 5e-3,
# # #         nrings: int = 3,
# # #         nphi: int = 12,
# # #         origin=(0, 0, 0),
# # #         direction=(0, 0, 1),
# # #         wavelength: float = 532e-9,
# # #         label: str = "collimated",
# # #     ):
# # #         origin = np.asarray(origin, dtype=float)
# # #         direction = normalize(direction)

# # #         rays = [Ray3D(origin.copy(), direction.copy())]
# # #         for ir in range(1, nrings + 1):
# # #             rr = radius * ir / nrings
# # #             n_this = max(8, nphi * ir)
# # #             for k in range(n_this):
# # #                 phi = 2 * np.pi * k / n_this
# # #                 pos = origin + np.array([rr * np.cos(phi), rr * np.sin(phi), 0.0])
# # #                 rays.append(Ray3D(pos, direction.copy()))

# # #         return cls(rays=rays, label=label, wavelength=wavelength, diameter=2 * radius)

# # #     @classmethod
# # #     def field_beam(
# # #         cls,
# # #         radius: float = 5e-3,
# # #         nrings: int = 3,
# # #         nphi: int = 12,
# # #         origin=(0, 0, 0),
# # #         field_angle_x: float = 0.0,
# # #         field_angle_y: float = 0.0,
# # #         wavelength: float = 532e-9,
# # #         label: str = "field beam",
# # #     ):
# # #         d = normalize(np.array([field_angle_x, field_angle_y, 1.0]))
# # #         return cls.collimated_circular(
# # #             radius=radius,
# # #             nrings=nrings,
# # #             nphi=nphi,
# # #             origin=origin,
# # #             direction=d,
# # #             wavelength=wavelength,
# # #             label=label,
# # #         )

# # #     def copy(self) -> "Beam3D":
# # #         return copy.deepcopy(self)


# # # # ============================================================
# # # # Optical elements
# # # # ============================================================

# # # @dataclass
# # # class OpticalElement3D:
# # #     point: np.ndarray
# # #     normal: np.ndarray
# # #     label: str = "element"

# # #     def __post_init__(self):
# # #         self.point = np.asarray(self.point, dtype=float).reshape(3)
# # #         self.normal = normalize(self.normal)
# # #         self._e1, self._e2 = orthonormal_basis_from_normal(self.normal)

# # #     def intersect_parameter(self, ray: Ray3D, eps: float = 1e-12):
# # #         denom = np.dot(ray.d, self.normal)
# # #         if abs(denom) < eps:
# # #             return None
# # #         s = np.dot(self.point - ray.r, self.normal) / denom
# # #         if s < 0:
# # #             return None
# # #         return s

# # #     def intersect_point(self, ray: Ray3D):
# # #         s = self.intersect_parameter(ray)
# # #         if s is None:
# # #             return None, None
# # #         return ray.r + s * ray.d, s

# # #     def plane_basis(self):
# # #         return self._e1, self._e2

# # #     def local_coordinates(self, p):
# # #         dp = p - self.point
# # #         return np.dot(dp, self._e1), np.dot(dp, self._e2)

# # #     def apply(self, ray: Ray3D):
# # #         return ray


# # # @dataclass
# # # class Lens3D(OpticalElement3D):
# # #     f: float = 0.1
# # #     aperture_radius: float = 10e-3
# # #     label: str = "lens"

# # #     def apply(self, ray: Ray3D):
# # #         p, _ = self.intersect_point(ray)
# # #         if p is None:
# # #             ray.alive = False
# # #             return ray

# # #         u, v = self.local_coordinates(p)
# # #         if np.hypot(u, v) > self.aperture_radius:
# # #             ray.alive = False
# # #             return ray

# # #         e1, e2 = self.plane_basis()
# # #         n = self.normal

# # #         du = np.dot(ray.d, e1)
# # #         dv = np.dot(ray.d, e2)
# # #         dn = np.dot(ray.d, n)
# # #         if abs(dn) < 1e-12:
# # #             ray.alive = False
# # #             return ray

# # #         du_out = du - u / self.f
# # #         dv_out = dv - v / self.f
# # #         dn_out = dn

# # #         d_out = du_out * e1 + dv_out * e2 + dn_out * n
# # #         ray.r = p
# # #         ray.d = normalize(d_out)
# # #         return ray


# # # @dataclass
# # # class Mirror3D(OpticalElement3D):
# # #     aperture_radius: float = 10e-3
# # #     label: str = "mirror"

# # #     def apply(self, ray: Ray3D):
# # #         p, _ = self.intersect_point(ray)
# # #         if p is None:
# # #             ray.alive = False
# # #             return ray

# # #         u, v = self.local_coordinates(p)
# # #         if np.hypot(u, v) > self.aperture_radius:
# # #             ray.alive = False
# # #             return ray

# # #         d_out = ray.d - 2 * np.dot(ray.d, self.normal) * self.normal
# # #         ray.r = p
# # #         ray.d = normalize(d_out)
# # #         return ray


# # # @dataclass
# # # class RotatingPhaseScreen3D(OpticalElement3D):
# # #     opd_map: np.ndarray = None
# # #     map_extent_m: float = 0.10
# # #     clear_radius: float = 20e-3
# # #     angular_velocity: float = 0.0
# # #     rotation_angle0: float = 0.0
# # #     label: str = "phase screen"

# # #     def __post_init__(self):
# # #         super().__post_init__()
# # #         if self.opd_map is None:
# # #             raise ValueError("opd_map must be provided.")
# # #         self.opd_map = np.asarray(self.opd_map, dtype=float)
# # #         if self.opd_map.ndim != 2:
# # #             raise ValueError("opd_map must be a 2D array.")

# # #     @property
# # #     def map_ny(self) -> int:
# # #         return self.opd_map.shape[0]

# # #     @property
# # #     def map_nx(self) -> int:
# # #         return self.opd_map.shape[1]

# # #     def current_rotation_angle(self, t: float) -> float:
# # #         return self.rotation_angle0 + self.angular_velocity * t

# # #     def uv_to_pixel(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# # #         u = uv[..., 0]
# # #         v = uv[..., 1]
# # #         x_pix = (u / self.map_extent_m + 0.5) * (self.map_nx - 1)
# # #         y_pix = (v / self.map_extent_m + 0.5) * (self.map_ny - 1)
# # #         return x_pix, y_pix

# # #     def contains_uv(self, uv: np.ndarray) -> np.ndarray:
# # #         r = np.sqrt(np.sum(np.asarray(uv) ** 2, axis=-1))
# # #         return r <= self.clear_radius

# # #     def sample_uv(self, uv: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
# # #         uv = np.asarray(uv, dtype=float)
# # #         rot = rotation_matrix_2d(-self.current_rotation_angle(t))
# # #         uv_rot = np.einsum("ij,...j->...i", rot, uv)
# # #         inside = self.contains_uv(uv_rot)

# # #         x_pix, y_pix = self.uv_to_pixel(uv_rot)
# # #         valid = inside & (x_pix >= 0) & (x_pix <= self.map_nx - 1) & (y_pix >= 0) & (y_pix <= self.map_ny - 1)

# # #         opd = np.zeros_like(x_pix, dtype=float)
# # #         if np.any(valid):
# # #             opd[valid] = bilinear_sample(self.opd_map, x_pix[valid], y_pix[valid])
# # #         return opd, valid

# # #     def apply(self, ray: Ray3D, t: float = 0.0):
# # #         p, _ = self.intersect_point(ray)
# # #         if p is None:
# # #             ray.alive = False
# # #             return ray

# # #         u, v = self.local_coordinates(p)
# # #         opd, valid = self.sample_uv(np.array([[u, v]]), t=t)
# # #         if not bool(valid[0]):
# # #             ray.alive = False
# # #             return ray

# # #         ray.opd += float(opd[0])
# # #         ray.r = p
# # #         return ray


# # # # ============================================================
# # # # Bench
# # # # ============================================================

# # # @dataclass
# # # class OpticalBench3D:
# # #     elements: List[OpticalElement3D] = field(default_factory=list)

# # #     def add(self, element: OpticalElement3D):
# # #         self.elements.append(element)

# # #     def trace_beam(self, beam: Beam3D, s_end: float = 1.0, n_line_samples: int = 60, t: float = 0.0):
# # #         all_paths = []
# # #         traced_rays = []

# # #         for ray0 in beam.rays:
# # #             ray = ray0.copy()
# # #             path = [ray.r.copy()]

# # #             for elem in self.elements:
# # #                 if not ray.alive:
# # #                     break
# # #                 p, s = elem.intersect_point(ray)
# # #                 if p is None:
# # #                     continue

# # #                 seg_s = np.linspace(0.0, s, n_line_samples)
# # #                 for ss in seg_s[1:]:
# # #                     path.append(ray.r + ss * ray.d)

# # #                 if isinstance(elem, RotatingPhaseScreen3D):
# # #                     ray = elem.apply(ray, t=t)
# # #                 else:
# # #                     ray = elem.apply(ray)
# # #                 path.append(ray.r.copy())

# # #             if ray.alive:
# # #                 seg_s = np.linspace(0.0, s_end, n_line_samples)
# # #                 for ss in seg_s[1:]:
# # #                     path.append(ray.r + ss * ray.d)

# # #             all_paths.append(np.array(path))
# # #             traced_rays.append(ray)

# # #         return all_paths, traced_rays

# # #     def trace_chief_intersections(self, beam: Beam3D, t: float = 0.0) -> Dict[str, Dict[str, Any]]:
# # #         ray = beam.chief_ray.copy()
# # #         out = {}

# # #         for elem in self.elements:
# # #             if not ray.alive:
# # #                 break
# # #             p, s = elem.intersect_point(ray)
# # #             if p is None:
# # #                 continue

# # #             out[elem.label] = {
# # #                 "point": p.copy(),
# # #                 "direction_in": ray.d.copy(),
# # #                 "element": elem,
# # #             }

# # #             if isinstance(elem, RotatingPhaseScreen3D):
# # #                 ray = elem.apply(ray, t=t)
# # #             else:
# # #                 ray = elem.apply(ray)

# # #             out[elem.label]["direction_out"] = ray.d.copy()
# # #             out[elem.label]["ray_after"] = ray.copy()

# # #         out["final_ray"] = ray
# # #         return out

# # #     def _draw_element_3d(self, ax, elem, npts: int = 100):
# # #         e1, e2 = elem.plane_basis()
# # #         if isinstance(elem, Lens3D):
# # #             R = elem.aperture_radius
# # #             text = f"{elem.label}\nf={elem.f:.3f} m"
# # #         elif isinstance(elem, Mirror3D):
# # #             R = elem.aperture_radius
# # #             text = elem.label
# # #         elif isinstance(elem, RotatingPhaseScreen3D):
# # #             R = elem.clear_radius
# # #             text = elem.label
# # #         else:
# # #             R = 1e-3
# # #             text = elem.label

# # #         tt = np.linspace(0, 2 * np.pi, npts)
# # #         ring = elem.point[None, :] + R * (
# # #             np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :]
# # #         )
# # #         ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], lw=2)
# # #         ax.text(*elem.point, text)

# # #     @staticmethod
# # #     def _set_axes_equal(ax):
# # #         x_limits = ax.get_xlim3d()
# # #         y_limits = ax.get_ylim3d()
# # #         z_limits = ax.get_zlim3d()
# # #         x_range = abs(x_limits[1] - x_limits[0])
# # #         y_range = abs(y_limits[1] - y_limits[0])
# # #         z_range = abs(z_limits[1] - z_limits[0])
# # #         x_mid = np.mean(x_limits)
# # #         y_mid = np.mean(y_limits)
# # #         z_mid = np.mean(z_limits)
# # #         plot_radius = 0.5 * max([x_range, y_range, z_range])
# # #         ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
# # #         ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
# # #         ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

# # #     def plot_3d(self, beams: List[Beam3D], s_end: float = 0.5, figsize=(10, 8), title: str = "3D optical bench", t: float = 0.0):
# # #         fig = plt.figure(figsize=figsize)
# # #         ax = fig.add_subplot(111, projection="3d")

# # #         for beam in beams:
# # #             paths, _ = self.trace_beam(beam, s_end=s_end, t=t)
# # #             for path in paths:
# # #                 ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.2)

# # #         for elem in self.elements:
# # #             self._draw_element_3d(ax, elem)

# # #         ax.set_xlabel("x [m]")
# # #         ax.set_ylabel("y [m]")
# # #         ax.set_zlabel("z [m]")
# # #         ax.set_title(title)
# # #         self._set_axes_equal(ax)
# # #         plt.tight_layout()
# # #         return fig


# # # # ============================================================
# # # # Plane sampling / decomposition
# # # # ============================================================

# # # def sample_screen_patch_for_beam(
# # #     screen: RotatingPhaseScreen3D,
# # #     center_point: np.ndarray,
# # #     beam_diameter: float,
# # #     t: float,
# # #     npix: int = 192,
# # # ) -> Dict[str, Any]:
# # #     r = 0.5 * beam_diameter
# # #     x = np.linspace(-r, r, npix)
# # #     y = np.linspace(-r, r, npix)
# # #     xx, yy = np.meshgrid(x, y)
# # #     mask = (xx ** 2 + yy ** 2) <= r ** 2

# # #     center_uv = screen.local_coordinates(center_point)
# # #     uv = np.stack([center_uv[0] + xx, center_uv[1] + yy], axis=-1)

# # #     opd, valid = screen.sample_uv(uv.reshape(-1, 2), t=t)
# # #     opd = opd.reshape(xx.shape)
# # #     valid = valid.reshape(xx.shape)

# # #     opd = np.where(mask & valid, opd, 0.0)
# # #     return {
# # #         "xx": xx,
# # #         "yy": yy,
# # #         "mask": mask,
# # #         "opd_map_m": np.where(mask, opd, np.nan),
# # #     }


# # # def sample_beam_phase_amplitude_on_plane(
# # #     beam: Beam3D,
# # #     bench: OpticalBench3D,
# # #     plane_point: np.ndarray,
# # #     plane_normal: np.ndarray,
# # #     t: float,
# # #     npix: int = 192,
# # #     diameter: Optional[float] = None,
# # #     screen_names: Optional[List[str]] = None,
# # # ) -> Dict[str, Any]:
# # #     diameter = beam.diameter if diameter is None else diameter
# # #     r = 0.5 * diameter
# # #     x = np.linspace(-r, r, npix)
# # #     y = np.linspace(-r, r, npix)
# # #     xx, yy = np.meshgrid(x, y)
# # #     mask = (xx ** 2 + yy ** 2) <= r ** 2

# # #     chief_info = bench.trace_chief_intersections(beam, t=t)
# # #     total_opd = np.zeros_like(xx, dtype=float)

# # #     for elem in bench.elements:
# # #         if not isinstance(elem, RotatingPhaseScreen3D):
# # #             continue
# # #         if screen_names is not None and elem.label not in screen_names:
# # #             continue
# # #         if elem.label not in chief_info:
# # #             continue

# # #         center_point = chief_info[elem.label]["point"]
# # #         patch = sample_screen_patch_for_beam(
# # #             screen=elem,
# # #             center_point=center_point,
# # #             beam_diameter=diameter,
# # #             t=t,
# # #             npix=npix,
# # #         )
# # #         total_opd += np.nan_to_num(patch["opd_map_m"], nan=0.0)

# # #     total_opd = np.where(mask, total_opd, np.nan)
# # #     phase = 2.0 * np.pi * total_opd / beam.wavelength
# # #     dx = xx[0, 1] - xx[0, 0]
# # #     amp = mask.astype(float)

# # #     return {
# # #         "xx": xx,
# # #         "yy": yy,
# # #         "mask": mask,
# # #         "amplitude": amp,
# # #         "opd_map_m": total_opd,
# # #         "phase_map_rad": phase,
# # #         "dx": dx,
# # #     }


# # # def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
# # #     mask = sample["mask"]
# # #     phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
# # #     amp = np.where(mask, sample["amplitude"], 0.0)
# # #     field = amp * np.exp(1j * phase)
# # #     out = fft_psf_from_pupil_field(field, sample["dx"])
# # #     return {**sample, **out}


# # # # ============================================================
# # # # Example builders
# # # # ============================================================

# # # def make_von_karman_opd_map(
# # #     n: int = 512,
# # #     extent_m: float = 0.12,
# # #     r0: float = 0.03,
# # #     L0: float = 10.0,
# # #     rms_opd_m: float = 150e-9,
# # #     seed: Optional[int] = None,
# # # ) -> np.ndarray:
# # #     rng = np.random.default_rng(seed)
# # #     dx = extent_m / n
# # #     fx = np.fft.fftfreq(n, d=dx)
# # #     fy = np.fft.fftfreq(n, d=dx)
# # #     FX, FY = np.meshgrid(fx, fy)
# # #     f = np.sqrt(FX ** 2 + FY ** 2)
# # #     f0 = 1.0 / L0
# # #     psd = (f ** 2 + f0 ** 2) ** (-11.0 / 6.0)
# # #     psd[0, 0] = 0.0
# # #     noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
# # #     screen = np.fft.ifft2(noise * np.sqrt(psd)).real
# # #     screen -= np.mean(screen)
# # #     if np.std(screen) > 0:
# # #         screen *= (0.1 / r0) ** (5.0 / 6.0)
# # #     if rms_opd_m is not None and np.std(screen) > 0:
# # #         screen *= rms_opd_m / np.std(screen)
# # #     return screen


# # # def build_example_bench() -> Tuple[OpticalBench3D, List[Beam3D], Dict[str, Any]]:
# # #     beam_diameter = 13e-3
# # #     beam_radius = beam_diameter / 2
# # #     wavelength = 532e-9
# # #     screen_clear_radius = 83e-3 / 2

# # #     z_source = 0.00
# # #     z_fa = 0.14
# # #     z_gl1 = 0.48
# # #     z_gl2 = 0.53
# # #     z_gl3 = 0.58
# # #     z_fold = 0.78

# # #     x_relay_l1 = 0.18
# # #     f_relay1 = 0.08
# # #     f_relay2 = 1.70 * f_relay1
# # #     x_relay_l2 = x_relay_l1 + f_relay1 + f_relay2
# # #     x_pupil = x_relay_l2 + f_relay2

# # #     bench = OpticalBench3D()

# # #     fa_map = make_von_karman_opd_map(n=512, extent_m=0.12, r0=0.03, L0=10.0, rms_opd_m=180e-9, seed=21)
# # #     bench.add(
# # #         RotatingPhaseScreen3D(
# # #             point=[0.0, 0.0, z_fa],
# # #             normal=[0.0, 0.0, 1.0],
# # #             opd_map=fa_map,
# # #             map_extent_m=0.12,
# # #             clear_radius=screen_clear_radius,
# # #             angular_velocity=2 * np.pi * 0.4,
# # #             label="FA PS",
# # #         )
# # #     )

# # #     gl_specs = [
# # #         (z_gl1, 90e-9, 0.7, 11, "GL PS 1"),
# # #         (z_gl2, 110e-9, 1.0, 12, "GL PS 2"),
# # #         (z_gl3, 130e-9, 1.4, 13, "GL PS 3"),
# # #     ]
# # #     for z, rms, hz, seed, label in gl_specs:
# # #         gl_map = make_von_karman_opd_map(n=512, extent_m=0.12, r0=0.05, L0=10.0, rms_opd_m=rms, seed=seed)
# # #         bench.add(
# # #             RotatingPhaseScreen3D(
# # #                 point=[0.0, 0.0, z],
# # #                 normal=[0.0, 0.0, 1.0],
# # #                 opd_map=gl_map,
# # #                 map_extent_m=0.12,
# # #                 clear_radius=screen_clear_radius,
# # #                 angular_velocity=2 * np.pi * hz,
# # #                 label=label,
# # #             )
# # #         )

# # #     d_in = normalize([0.0, 0.0, 1.0])
# # #     d_out = normalize([1.0, 0.0, 0.0])
# # #     mirror_normal = normalize(d_out - d_in)
# # #     bench.add(Mirror3D(point=[0.0, 0.0, z_fold], normal=mirror_normal, aperture_radius=20e-3, label="Fold M1"))
# # #     bench.add(Lens3D(point=[x_relay_l1, 0.0, z_fold], normal=[1.0, 0.0, 0.0], f=f_relay1, aperture_radius=18e-3, label="Relay L1"))
# # #     bench.add(Lens3D(point=[x_relay_l2, 0.0, z_fold], normal=[1.0, 0.0, 0.0], f=f_relay2, aperture_radius=20e-3, label="Relay L2"))

# # #     arcmin_to_rad = np.pi / (180.0 * 60.0)
# # #     theta_10 = 10.0 * arcmin_to_rad
# # #     beams = [
# # #         Beam3D.field_beam(radius=beam_radius, nrings=3, nphi=12, origin=[0.0, 0.0, z_source], field_angle_x=+theta_10, field_angle_y=+theta_10, wavelength=wavelength, label="field +x +y"),
# # #         Beam3D.field_beam(radius=beam_radius, nrings=3, nphi=12, origin=[0.0, 0.0, z_source], field_angle_x=-theta_10, field_angle_y=+theta_10, wavelength=wavelength, label="field -x +y"),
# # #         Beam3D.field_beam(radius=beam_radius, nrings=3, nphi=12, origin=[0.0, 0.0, z_source], field_angle_x=+theta_10, field_angle_y=-theta_10, wavelength=wavelength, label="field +x -y"),
# # #         Beam3D.field_beam(radius=beam_radius, nrings=3, nphi=12, origin=[0.0, 0.0, z_source], field_angle_x=-theta_10, field_angle_y=-theta_10, wavelength=wavelength, label="field -x -y"),
# # #         Beam3D.field_beam(radius=beam_radius, nrings=3, nphi=12, origin=[0.0, 0.0, z_source], field_angle_x=0.0, field_angle_y=0.0, wavelength=wavelength, label="field center"),
# # #     ]

# # #     meta = {
# # #         "analysis_plane_point": np.array([x_pupil, 0.0, z_fold]),
# # #         "analysis_plane_normal": np.array([1.0, 0.0, 0.0]),
# # #         "beam_diameter": beam_diameter,
# # #         "wavelength": wavelength,
# # #         "field_points": [
# # #             (+10.0, +10.0, "corner ++"),
# # #             (-10.0, +10.0, "corner -+"),
# # #             (+10.0, -10.0, "corner +-"),
# # #             (-10.0, -10.0, "corner --"),
# # #             (0.0, 0.0, "center"),
# # #         ],
# # #         "screen_labels": ["FA PS", "GL PS 1", "GL PS 2", "GL PS 3"],
# # #     }
# # #     return bench, beams, meta


# # # # ============================================================
# # # # Example plotting routines
# # # # ============================================================

# # # def plot_field_long_exposure_psfs(
# # #     bench: OpticalBench3D,
# # #     meta: Dict[str, Any],
# # #     exposure_s: float = 30.0,
# # #     dt_s: float = 0.5,
# # #     npix: int = 1024,
# # #     half_width_ld: float = 5.0,
# # # ):
# # #     times = np.arange(0.0, exposure_s, dt_s)
# # #     plane_point = meta["analysis_plane_point"]
# # #     plane_normal = meta["analysis_plane_normal"]
# # #     beam_diameter = meta["beam_diameter"]
# # #     wavelength = meta["wavelength"]

# # #     fig, axes = plt.subplots(2, 3, figsize=(14, 9))
# # #     axes = axes.ravel()

# # #     for i, (fx_arcmin, fy_arcmin, label) in enumerate(meta["field_points"]):
# # #         beam = Beam3D.field_beam(
# # #             radius=0.5 * beam_diameter,
# # #             nrings=3,
# # #             nphi=12,
# # #             origin=[0.0, 0.0, 0.0],
# # #             field_angle_x=np.deg2rad(fx_arcmin / 60.0),
# # #             field_angle_y=np.deg2rad(fy_arcmin / 60.0),
# # #             wavelength=wavelength,
# # #             label=label,
# # #         )

# # #         stack = []
# # #         first_pack = None
# # #         for t in times:
# # #             sample = sample_beam_phase_amplitude_on_plane(
# # #                 beam=beam,
# # #                 bench=bench,
# # #                 plane_point=plane_point,
# # #                 plane_normal=plane_normal,
# # #                 t=float(t),
# # #                 npix=npix,
# # #                 diameter=beam_diameter,
# # #             )
# # #             psf_pack = psf_from_plane_sample(sample)
# # #             stack.append(psf_pack["psf"])
# # #             if first_pack is None:
# # #                 first_pack = psf_pack

# # #         long_psf = np.mean(np.array(stack), axis=0)
# # #         if np.max(long_psf) > 0:
# # #             long_psf = long_psf / np.max(long_psf)
# # #         pack = dict(first_pack)
# # #         pack["psf"] = long_psf

# # #         x_ld, y_ld = psf_coords_lambda_over_d(pack, beam_diameter)
# # #         psf_crop, x_crop, y_crop = crop_psf_to_lambda_over_d(pack["psf"], x_ld, y_ld, half_width_ld=half_width_ld)
# # #         XX, YY = np.meshgrid(x_crop, y_crop)

# # #         fit = fit_2d_gaussian(XX, YY, psf_crop)
# # #         metrics = gaussian_fwhm_and_ellipticity(fit)
# # #         xc, yc = gaussian_halfmax_contour(XX, YY, fit)

# # #         ax = axes[i]
# # #         im = ax.imshow(
# # #             np.log10(np.maximum(psf_crop, 1e-10)),
# # #             origin="lower",
# # #             extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()],
# # #         )
# # #         if xc is not None:
# # #             ax.plot(xc, yc, color="white", lw=1.2)

# # #         ax.set_title(label)
# # #         ax.set_xlabel(r"$\lambda/D$")
# # #         ax.set_ylabel(r"$\lambda/D$")
# # #         ax.set_aspect("equal")
# # #         ax.text(
# # #             0.04, 0.93,
# # #             (
# # #                 f"FWHMmaj={metrics['fwhm_major']:.2f} $\\lambda/D$\n"
# # #                 f"FWHMmin={metrics['fwhm_minor']:.2f} $\\lambda/D$\n"
# # #                 f"ell={metrics['ellipticity']:.3f}"
# # #             ),
# # #             transform=ax.transAxes,
# # #             color="white",
# # #             ha="left",
# # #             va="top",
# # #             fontsize=10,
# # #             bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
# # #         )
# # #         plt.colorbar(im, ax=ax, label="log10 PSF")

# # #     if len(axes) > len(meta["field_points"]):
# # #         axes[-1].axis("off")

# # #     fig.suptitle(f"Long-exposure PSFs ({exposure_s:.1f} s, dt={dt_s:.2f} s)", y=0.98)
# # #     fig.tight_layout()
# # #     return fig


# # # def plot_phase_screen_contributions(
# # #     bench: OpticalBench3D,
# # #     beams: List[Beam3D],
# # #     meta: Dict[str, Any],
# # #     t: float = 0.0,
# # #     npix: int = 180,
# # # ):
# # #     plane_point = meta["analysis_plane_point"]
# # #     plane_normal = meta["analysis_plane_normal"]
# # #     beam_diameter = meta["beam_diameter"]
# # #     screen_labels = meta["screen_labels"]

# # #     nrows = len(beams)
# # #     ncols = len(screen_labels)
# # #     fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)

# # #     vmax = 0.0
# # #     data_cache = {}
# # #     for beam in beams:
# # #         for screen_label in screen_labels:
# # #             sample = sample_beam_phase_amplitude_on_plane(
# # #                 beam=beam,
# # #                 bench=bench,
# # #                 plane_point=plane_point,
# # #                 plane_normal=plane_normal,
# # #                 t=t,
# # #                 npix=npix,
# # #                 diameter=beam_diameter,
# # #                 screen_names=[screen_label],
# # #             )
# # #             phase = sample["phase_map_rad"]
# # #             data_cache[(beam.label, screen_label)] = sample
# # #             vmax = max(vmax, np.nanmax(np.abs(np.nan_to_num(phase, nan=0.0))))

# # #     if vmax <= 0:
# # #         vmax = 1.0

# # #     for i, beam in enumerate(beams):
# # #         for j, screen_label in enumerate(screen_labels):
# # #             sample = data_cache[(beam.label, screen_label)]
# # #             xx = sample["xx"] * 1e3
# # #             yy = sample["yy"] * 1e3
# # #             ax = axes[i, j]
# # #             im = ax.imshow(
# # #                 sample["phase_map_rad"],
# # #                 origin="lower",
# # #                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
# # #                 vmin=-vmax,
# # #                 vmax=vmax,
# # #             )
# # #             if i == 0:
# # #                 ax.set_title(screen_label)
# # #             if j == 0:
# # #                 ax.set_ylabel(f"{beam.label}\ny [mm]")
# # #             else:
# # #                 ax.set_ylabel("y [mm]")
# # #             ax.set_xlabel("x [mm]")
# # #             ax.set_aspect("equal")
# # #             plt.colorbar(im, ax=ax, label="phase [rad]")

# # #     fig.suptitle(f"Per-screen phase contributions at t = {t:.2f} s", y=0.995)
# # #     fig.tight_layout()
# # #     return fig


# # # # ============================================================
# # # # Main example
# # # # ============================================================

# # # if __name__ == "__main__":
# # #     bench, beams, meta = build_example_bench()

# # #     fig1 = bench.plot_3d(beams, s_end=0.25, title="Instrument layout with ray tracing", t=0.0)

# # #     fig2 = plot_field_long_exposure_psfs(
# # #         bench=bench,
# # #         meta=meta,
# # #         exposure_s=30.0,
# # #         dt_s=0.5,
# # #         npix=1024,
# # #         half_width_ld=5.0,
# # #     )

# # #     fig3 = plot_phase_screen_contributions(
# # #         bench=bench,
# # #         beams=beams,
# # #         meta=meta,
# # #         t=0.0,
# # #         npix=180,
# # #     )

# # #     plt.show()


# # # # import copy
# # # # from dataclasses import dataclass, field
# # # # from typing import List, Optional, Tuple, Dict, Any

# # # # import matplotlib.pyplot as plt
# # # # import numpy as np


# # # # """
# # # # Minimal ray-tracing + rotating phase-screen module.

# # # # Goals
# # # # -----
# # # # - Keep a simple ray/instrument structure.
# # # # - Add a beam model with wavelength, diameter, and OPD accumulation.
# # # # - Add rotating phase screens with bilinear sampling.
# # # # - Stay geometric / collimated for now.
# # # # - Provide diagnostics for:
# # # #     * 3D instrument layout with ray tracing
# # # #     * sampled phase and amplitude on planes
# # # #     * long-exposure PSFs at multiple field points
# # # #     * per-screen phase contributions for each beam

# # # # Notes
# # # # -----
# # # # This is not a Fresnel propagator. The beam is represented as a collimated bundle
# # # # with an associated sampled phase map on analysis planes. Optical elements change
# # # # ray directions geometrically. Phase screens add OPD but do not yet refract rays
# # # # via local slopes.
# # # # """


# # # # # ============================================================
# # # # # Helpers
# # # # # ============================================================

# # # # def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
# # # #     v = np.asarray(v, dtype=float)
# # # #     n = np.linalg.norm(v)
# # # #     if n < eps:
# # # #         raise ValueError("Cannot normalize near-zero vector.")
# # # #     return v / n


# # # # def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# # # #     n = normalize(normal)
# # # #     ref = np.array([1.0, 0.0, 0.0])
# # # #     if abs(np.dot(ref, n)) > 0.9:
# # # #         ref = np.array([0.0, 1.0, 0.0])
# # # #     e1 = normalize(np.cross(n, ref))
# # # #     e2 = normalize(np.cross(n, e1))
# # # #     return e1, e2


# # # # def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
# # # #     c = np.cos(angle_rad)
# # # #     s = np.sin(angle_rad)
# # # #     return np.array([[c, -s], [s, c]])


# # # # def bilinear_sample(grid: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
# # # #     ny, nx = grid.shape

# # # #     x0 = np.floor(x_pix).astype(int)
# # # #     y0 = np.floor(y_pix).astype(int)
# # # #     x1 = x0 + 1
# # # #     y1 = y0 + 1

# # # #     x0 = np.clip(x0, 0, nx - 1)
# # # #     x1 = np.clip(x1, 0, nx - 1)
# # # #     y0 = np.clip(y0, 0, ny - 1)
# # # #     y1 = np.clip(y1, 0, ny - 1)

# # # #     Ia = grid[y0, x0]
# # # #     Ib = grid[y0, x1]
# # # #     Ic = grid[y1, x0]
# # # #     Id = grid[y1, x1]

# # # #     wa = (x1 - x_pix) * (y1 - y_pix)
# # # #     wb = (x_pix - x0) * (y1 - y_pix)
# # # #     wc = (x1 - x_pix) * (y_pix - y0)
# # # #     wd = (x_pix - x0) * (y_pix - y0)

# # # #     return wa * Ia + wb * Ib + wc * Ic + wd * Id


# # # # def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
# # # #     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
# # # #     psf = np.abs(ef) ** 2
# # # #     if np.max(psf) > 0:
# # # #         psf = psf / np.max(psf)

# # # #     ny, nx = field.shape
# # # #     fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
# # # #     fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
# # # #     return {"psf": psf, "fx": fx, "fy": fy}


# # # # def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
# # # #     x_ld = psf_pack["fx"] * pupil_diameter_m
# # # #     y_ld = psf_pack["fy"] * pupil_diameter_m
# # # #     return x_ld, y_ld


# # # # def crop_psf_to_lambda_over_d(
# # # #     psf: np.ndarray,
# # # #     x_ld: np.ndarray,
# # # #     y_ld: np.ndarray,
# # # #     half_width_ld: float = 5.0,
# # # # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # # #     ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
# # # #     iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
# # # #     if len(ix) == 0 or len(iy) == 0:
# # # #         return psf, x_ld, y_ld
# # # #     return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]


# # # # def measure_fwhm_1d(x: np.ndarray, y: np.ndarray) -> float:
# # # #     if len(x) < 3:
# # # #         return np.nan
# # # #     y = np.asarray(y, dtype=float)
# # # #     x = np.asarray(x, dtype=float)
# # # #     if np.nanmax(y) <= 0:
# # # #         return np.nan

# # # #     y = y / np.nanmax(y)
# # # #     peak_idx = int(np.nanargmax(y))
# # # #     half = 0.5

# # # #     left = np.nan
# # # #     for i in range(peak_idx, 0, -1):
# # # #         if y[i] >= half and y[i - 1] < half:
# # # #             x1, x2 = x[i - 1], x[i]
# # # #             y1, y2 = y[i - 1], y[i]
# # # #             left = x1 + (half - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[i]
# # # #             break

# # # #     right = np.nan
# # # #     for i in range(peak_idx, len(y) - 1):
# # # #         if y[i] >= half and y[i + 1] < half:
# # # #             x1, x2 = x[i], x[i + 1]
# # # #             y1, y2 = y[i], y[i + 1]
# # # #             right = x1 + (half - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[i]
# # # #             break

# # # #     if np.isfinite(left) and np.isfinite(right):
# # # #         return float(right - left)
# # # #     return np.nan


# # # # def psf_fwhm_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> float:
# # # #     x_ld, _ = psf_coords_lambda_over_d(psf_pack, pupil_diameter_m)
# # # #     psf = psf_pack["psf"]
# # # #     cy = psf.shape[0] // 2
# # # #     return measure_fwhm_1d(x_ld, psf[cy, :])


# # # # def psf_ellipticity_from_moments(psf: np.ndarray) -> float:
# # # #     if np.nanmax(psf) <= 0:
# # # #         return np.nan
# # # #     psf = psf / np.nansum(psf)
# # # #     y, x = np.indices(psf.shape)

# # # #     x0 = np.nansum(x * psf)
# # # #     y0 = np.nansum(y * psf)

# # # #     xx = np.nansum((x - x0) ** 2 * psf)
# # # #     yy = np.nansum((y - y0) ** 2 * psf)
# # # #     xy = np.nansum((x - x0) * (y - y0) * psf)

# # # #     cov = np.array([[xx, xy], [xy, yy]])
# # # #     evals = np.linalg.eigvalsh(cov)
# # # #     evals = np.clip(evals, 0.0, None)
# # # #     if evals[-1] <= 0:
# # # #         return np.nan
# # # #     major = np.sqrt(evals[-1])
# # # #     minor = np.sqrt(evals[0])
# # # #     if major <= 0:
# # # #         return np.nan
# # # #     return float(1.0 - minor / major)


# # # # # ============================================================
# # # # # Ray / beam
# # # # # ============================================================

# # # # @dataclass
# # # # class Ray3D:
# # # #     r: np.ndarray
# # # #     d: np.ndarray
# # # #     alive: bool = True
# # # #     opd: float = 0.0

# # # #     def __post_init__(self):
# # # #         self.r = np.asarray(self.r, dtype=float).reshape(3)
# # # #         self.d = normalize(self.d)

# # # #     def copy(self):
# # # #         return Ray3D(self.r.copy(), self.d.copy(), self.alive, self.opd)


# # # # @dataclass
# # # # class Beam3D:
# # # #     rays: List[Ray3D]
# # # #     label: str = "beam"
# # # #     wavelength: float = 532e-9
# # # #     diameter: float = 10e-3
# # # #     metadata: Dict[str, Any] = field(default_factory=dict)

# # # #     @property
# # # #     def radius(self) -> float:
# # # #         return 0.5 * self.diameter

# # # #     @property
# # # #     def chief_ray(self) -> Ray3D:
# # # #         return self.rays[0]

# # # #     @classmethod
# # # #     def collimated_circular(
# # # #         cls,
# # # #         radius: float = 5e-3,
# # # #         nrings: int = 2,
# # # #         nphi: int = 8,
# # # #         origin=(0, 0, 0),
# # # #         direction=(0, 0, 1),
# # # #         wavelength: float = 532e-9,
# # # #         label: str = "collimated",
# # # #     ):
# # # #         origin = np.asarray(origin, dtype=float)
# # # #         direction = normalize(direction)

# # # #         rays = [Ray3D(origin.copy(), direction.copy())]
# # # #         for ir in range(1, nrings + 1):
# # # #             rr = radius * ir / nrings
# # # #             n_this = max(6, nphi * ir)
# # # #             for k in range(n_this):
# # # #                 phi = 2 * np.pi * k / n_this
# # # #                 pos = origin + np.array([rr * np.cos(phi), rr * np.sin(phi), 0.0])
# # # #                 rays.append(Ray3D(pos, direction.copy()))

# # # #         return cls(rays=rays, label=label, wavelength=wavelength, diameter=2 * radius)

# # # #     @classmethod
# # # #     def field_beam(
# # # #         cls,
# # # #         radius: float = 5e-3,
# # # #         nrings: int = 2,
# # # #         nphi: int = 8,
# # # #         origin=(0, 0, 0),
# # # #         field_angle_x: float = 0.0,
# # # #         field_angle_y: float = 0.0,
# # # #         wavelength: float = 532e-9,
# # # #         label: str = "field beam",
# # # #     ):
# # # #         d = normalize(np.array([field_angle_x, field_angle_y, 1.0]))
# # # #         return cls.collimated_circular(
# # # #             radius=radius,
# # # #             nrings=nrings,
# # # #             nphi=nphi,
# # # #             origin=origin,
# # # #             direction=d,
# # # #             wavelength=wavelength,
# # # #             label=label,
# # # #         )

# # # #     def copy(self) -> "Beam3D":
# # # #         return copy.deepcopy(self)

# # # #     def propagated_to_plane(self, plane_point: np.ndarray, plane_normal: np.ndarray) -> Tuple[float, np.ndarray]:
# # # #         plane_point = np.asarray(plane_point, dtype=float)
# # # #         plane_normal = normalize(np.asarray(plane_normal, dtype=float))
# # # #         ray = self.chief_ray
# # # #         denom = np.dot(ray.d, plane_normal)
# # # #         if abs(denom) < 1e-12:
# # # #             raise ValueError(f"Beam '{self.label}' is parallel to the plane.")
# # # #         s = np.dot(plane_point - ray.r, plane_normal) / denom
# # # #         if s < 0:
# # # #             raise ValueError(f"Plane lies behind beam '{self.label}'.")
# # # #         return s, ray.r + s * ray.d


# # # # # ============================================================
# # # # # Optical elements
# # # # # ============================================================

# # # # @dataclass
# # # # class OpticalElement3D:
# # # #     point: np.ndarray
# # # #     normal: np.ndarray
# # # #     label: str = "element"

# # # #     def __post_init__(self):
# # # #         self.point = np.asarray(self.point, dtype=float).reshape(3)
# # # #         self.normal = normalize(self.normal)
# # # #         self._e1, self._e2 = orthonormal_basis_from_normal(self.normal)

# # # #     def intersect_parameter(self, ray: Ray3D, eps: float = 1e-12):
# # # #         denom = np.dot(ray.d, self.normal)
# # # #         if abs(denom) < eps:
# # # #             return None
# # # #         s = np.dot(self.point - ray.r, self.normal) / denom
# # # #         if s < 0:
# # # #             return None
# # # #         return s

# # # #     def intersect_point(self, ray: Ray3D):
# # # #         s = self.intersect_parameter(ray)
# # # #         if s is None:
# # # #             return None, None
# # # #         return ray.r + s * ray.d, s

# # # #     def plane_basis(self):
# # # #         return self._e1, self._e2

# # # #     def local_coordinates(self, p):
# # # #         dp = p - self.point
# # # #         return np.dot(dp, self._e1), np.dot(dp, self._e2)

# # # #     def apply(self, ray: Ray3D):
# # # #         return ray


# # # # @dataclass
# # # # class Lens3D(OpticalElement3D):
# # # #     f: float = 0.1
# # # #     aperture_radius: float = 10e-3
# # # #     label: str = "lens"

# # # #     def apply(self, ray: Ray3D):
# # # #         p, _ = self.intersect_point(ray)
# # # #         if p is None:
# # # #             ray.alive = False
# # # #             return ray

# # # #         u, v = self.local_coordinates(p)
# # # #         if np.hypot(u, v) > self.aperture_radius:
# # # #             ray.alive = False
# # # #             return ray

# # # #         e1, e2 = self.plane_basis()
# # # #         n = self.normal

# # # #         du = np.dot(ray.d, e1)
# # # #         dv = np.dot(ray.d, e2)
# # # #         dn = np.dot(ray.d, n)
# # # #         if abs(dn) < 1e-12:
# # # #             ray.alive = False
# # # #             return ray

# # # #         du_out = du - u / self.f
# # # #         dv_out = dv - v / self.f
# # # #         dn_out = dn

# # # #         d_out = du_out * e1 + dv_out * e2 + dn_out * n
# # # #         ray.r = p
# # # #         ray.d = normalize(d_out)
# # # #         return ray


# # # # @dataclass
# # # # class Mirror3D(OpticalElement3D):
# # # #     aperture_radius: float = 10e-3
# # # #     label: str = "mirror"

# # # #     def apply(self, ray: Ray3D):
# # # #         p, _ = self.intersect_point(ray)
# # # #         if p is None:
# # # #             ray.alive = False
# # # #             return ray

# # # #         u, v = self.local_coordinates(p)
# # # #         if np.hypot(u, v) > self.aperture_radius:
# # # #             ray.alive = False
# # # #             return ray

# # # #         d_out = ray.d - 2 * np.dot(ray.d, self.normal) * self.normal
# # # #         ray.r = p
# # # #         ray.d = normalize(d_out)
# # # #         return ray


# # # # @dataclass
# # # # class RotatingPhaseScreen3D(OpticalElement3D):
# # # #     opd_map: np.ndarray = None
# # # #     map_extent_m: float = 0.10
# # # #     clear_radius: float = 20e-3
# # # #     angular_velocity: float = 0.0
# # # #     rotation_angle0: float = 0.0
# # # #     label: str = "phase screen"

# # # #     def __post_init__(self):
# # # #         super().__post_init__()
# # # #         if self.opd_map is None:
# # # #             raise ValueError("opd_map must be provided.")
# # # #         self.opd_map = np.asarray(self.opd_map, dtype=float)
# # # #         if self.opd_map.ndim != 2:
# # # #             raise ValueError("opd_map must be a 2D array.")

# # # #     @property
# # # #     def map_ny(self) -> int:
# # # #         return self.opd_map.shape[0]

# # # #     @property
# # # #     def map_nx(self) -> int:
# # # #         return self.opd_map.shape[1]

# # # #     def current_rotation_angle(self, t: float) -> float:
# # # #         return self.rotation_angle0 + self.angular_velocity * t

# # # #     def uv_to_pixel(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# # # #         u = uv[..., 0]
# # # #         v = uv[..., 1]
# # # #         x_pix = (u / self.map_extent_m + 0.5) * (self.map_nx - 1)
# # # #         y_pix = (v / self.map_extent_m + 0.5) * (self.map_ny - 1)
# # # #         return x_pix, y_pix

# # # #     def contains_uv(self, uv: np.ndarray) -> np.ndarray:
# # # #         r = np.sqrt(np.sum(np.asarray(uv) ** 2, axis=-1))
# # # #         return r <= self.clear_radius

# # # #     def sample_uv(self, uv: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
# # # #         uv = np.asarray(uv, dtype=float)
# # # #         rot = rotation_matrix_2d(-self.current_rotation_angle(t))
# # # #         uv_rot = np.einsum("ij,...j->...i", rot, uv)
# # # #         inside = self.contains_uv(uv_rot)

# # # #         x_pix, y_pix = self.uv_to_pixel(uv_rot)
# # # #         valid = inside & (x_pix >= 0) & (x_pix <= self.map_nx - 1) & (y_pix >= 0) & (y_pix <= self.map_ny - 1)

# # # #         opd = np.zeros_like(x_pix, dtype=float)
# # # #         if np.any(valid):
# # # #             opd[valid] = bilinear_sample(self.opd_map, x_pix[valid], y_pix[valid])
# # # #         return opd, valid

# # # #     def apply(self, ray: Ray3D, t: float = 0.0):
# # # #         p, _ = self.intersect_point(ray)
# # # #         if p is None:
# # # #             ray.alive = False
# # # #             return ray

# # # #         u, v = self.local_coordinates(p)
# # # #         opd, valid = self.sample_uv(np.array([[u, v]]), t=t)
# # # #         if not bool(valid[0]):
# # # #             ray.alive = False
# # # #             return ray

# # # #         ray.opd += float(opd[0])
# # # #         ray.r = p
# # # #         return ray


# # # # # ============================================================
# # # # # Bench
# # # # # ============================================================

# # # # @dataclass
# # # # class OpticalBench3D:
# # # #     elements: List[OpticalElement3D] = field(default_factory=list)

# # # #     def add(self, element: OpticalElement3D):
# # # #         self.elements.append(element)

# # # #     def trace_beam(self, beam: Beam3D, s_end: float = 1.0, n_line_samples: int = 60, t: float = 0.0):
# # # #         all_paths = []
# # # #         traced_rays = []

# # # #         for ray0 in beam.rays:
# # # #             ray = ray0.copy()
# # # #             path = [ray.r.copy()]

# # # #             for elem in self.elements:
# # # #                 if not ray.alive:
# # # #                     break
# # # #                 p, s = elem.intersect_point(ray)
# # # #                 if p is None:
# # # #                     continue

# # # #                 seg_s = np.linspace(0.0, s, n_line_samples)
# # # #                 for ss in seg_s[1:]:
# # # #                     path.append(ray.r + ss * ray.d)

# # # #                 if isinstance(elem, RotatingPhaseScreen3D):
# # # #                     ray = elem.apply(ray, t=t)
# # # #                 else:
# # # #                     ray = elem.apply(ray)
# # # #                 path.append(ray.r.copy())

# # # #             if ray.alive:
# # # #                 seg_s = np.linspace(0.0, s_end, n_line_samples)
# # # #                 for ss in seg_s[1:]:
# # # #                     path.append(ray.r + ss * ray.d)

# # # #             all_paths.append(np.array(path))
# # # #             traced_rays.append(ray)

# # # #         return all_paths, traced_rays

# # # #     def _draw_element_3d(self, ax, elem, npts: int = 100):
# # # #         e1, e2 = elem.plane_basis()
# # # #         if isinstance(elem, Lens3D):
# # # #             R = elem.aperture_radius
# # # #             text = f"{elem.label}\nf={elem.f:.3f} m"
# # # #         elif isinstance(elem, Mirror3D):
# # # #             R = elem.aperture_radius
# # # #             text = elem.label
# # # #         elif isinstance(elem, RotatingPhaseScreen3D):
# # # #             R = elem.clear_radius
# # # #             text = elem.label
# # # #         else:
# # # #             R = 1e-3
# # # #             text = elem.label

# # # #         tt = np.linspace(0, 2 * np.pi, npts)
# # # #         ring = elem.point[None, :] + R * (
# # # #             np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :]
# # # #         )
# # # #         ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], lw=2)
# # # #         ax.text(*elem.point, text)

# # # #     @staticmethod
# # # #     def _set_axes_equal(ax):
# # # #         x_limits = ax.get_xlim3d()
# # # #         y_limits = ax.get_ylim3d()
# # # #         z_limits = ax.get_zlim3d()
# # # #         x_range = abs(x_limits[1] - x_limits[0])
# # # #         y_range = abs(y_limits[1] - y_limits[0])
# # # #         z_range = abs(z_limits[1] - z_limits[0])
# # # #         x_mid = np.mean(x_limits)
# # # #         y_mid = np.mean(y_limits)
# # # #         z_mid = np.mean(z_limits)
# # # #         plot_radius = 0.5 * max([x_range, y_range, z_range])
# # # #         ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
# # # #         ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
# # # #         ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

# # # #     def plot_3d(self, beams: List[Beam3D], s_end: float = 0.5, figsize=(10, 8), title: str = "3D optical bench", t: float = 0.0):
# # # #         fig = plt.figure(figsize=figsize)
# # # #         ax = fig.add_subplot(111, projection="3d")

# # # #         for beam in beams:
# # # #             paths, _ = self.trace_beam(beam, s_end=s_end, t=t)
# # # #             for path in paths:
# # # #                 ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.2)

# # # #         for elem in self.elements:
# # # #             self._draw_element_3d(ax, elem)

# # # #         ax.set_xlabel("x [m]")
# # # #         ax.set_ylabel("y [m]")
# # # #         ax.set_zlabel("z [m]")
# # # #         ax.set_title(title)
# # # #         self._set_axes_equal(ax)
# # # #         plt.tight_layout()
# # # #         return fig


# # # # # ============================================================
# # # # # Plane sampling / decomposition
# # # # # ============================================================

# # # # def sample_beam_phase_amplitude_on_plane(
# # # #     beam: Beam3D,
# # # #     bench: OpticalBench3D,
# # # #     plane_point: np.ndarray,
# # # #     plane_normal: np.ndarray,
# # # #     t: float,
# # # #     npix: int = 192,
# # # #     diameter: Optional[float] = None,
# # # #     screen_names: Optional[List[str]] = None,
# # # # ) -> Dict[str, Any]:
# # # #     plane_point = np.asarray(plane_point, dtype=float)
# # # #     plane_normal = normalize(plane_normal)
# # # #     diameter = beam.diameter if diameter is None else diameter

# # # #     e1, e2 = orthonormal_basis_from_normal(plane_normal)
# # # #     r = 0.5 * diameter
# # # #     x = np.linspace(-r, r, npix)
# # # #     y = np.linspace(-r, r, npix)
# # # #     xx, yy = np.meshgrid(x, y)
# # # #     mask = (xx ** 2 + yy ** 2) <= r ** 2

# # # #     chief = beam.rays[0]
# # # #     pts_plane = (
# # # #         plane_point[None, None, :]
# # # #         + xx[:, :, None] * e1[None, None, :]
# # # #         + yy[:, :, None] * e2[None, None, :]
# # # #     )

# # # #     total_opd = np.zeros_like(xx, dtype=float)

# # # #     for elem in bench.elements:
# # # #         if not isinstance(elem, RotatingPhaseScreen3D):
# # # #             continue
# # # #         if screen_names is not None and elem.label not in screen_names:
# # # #             continue

# # # #         denom = np.dot(chief.d, elem.normal)
# # # #         if abs(denom) < 1e-12:
# # # #             continue

# # # #         rel = elem.point[None, None, :] - pts_plane
# # # #         s = np.sum(rel * elem.normal[None, None, :], axis=2) / denom
# # # #         pts_screen = pts_plane + s[:, :, None] * chief.d[None, None, :]

# # # #         e1s, e2s = elem.plane_basis()
# # # #         rels = pts_screen - elem.point[None, None, :]
# # # #         u = np.sum(rels * e1s[None, None, :], axis=2)
# # # #         v = np.sum(rels * e2s[None, None, :], axis=2)
# # # #         uv = np.stack([u, v], axis=-1)

# # # #         opd, valid = elem.sample_uv(uv.reshape(-1, 2), t=t)
# # # #         opd = opd.reshape(xx.shape)
# # # #         valid = valid.reshape(xx.shape)
# # # #         total_opd += np.where(valid, opd, 0.0)

# # # #     total_opd = np.where(mask, total_opd, np.nan)
# # # #     phase = 2.0 * np.pi * total_opd / beam.wavelength
# # # #     dx = xx[0, 1] - xx[0, 0]
# # # #     amp = mask.astype(float)

# # # #     return {
# # # #         "xx": xx,
# # # #         "yy": yy,
# # # #         "mask": mask,
# # # #         "amplitude": amp,
# # # #         "opd_map_m": total_opd,
# # # #         "phase_map_rad": phase,
# # # #         "dx": dx,
# # # #     }


# # # # def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
# # # #     mask = sample["mask"]
# # # #     phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
# # # #     amp = np.where(mask, sample["amplitude"], 0.0)
# # # #     field = amp * np.exp(1j * phase)
# # # #     out = fft_psf_from_pupil_field(field, sample["dx"])
# # # #     return {**sample, **out}


# # # # # ============================================================
# # # # # Example builders
# # # # # ============================================================

# # # # def make_von_karman_opd_map(
# # # #     n: int = 512,
# # # #     extent_m: float = 0.12,
# # # #     r0: float = 0.03,
# # # #     L0: float = 10.0,
# # # #     rms_opd_m: float = 150e-9,
# # # #     seed: Optional[int] = None,
# # # # ) -> np.ndarray:
# # # #     rng = np.random.default_rng(seed)
# # # #     dx = extent_m / n
# # # #     fx = np.fft.fftfreq(n, d=dx)
# # # #     fy = np.fft.fftfreq(n, d=dx)
# # # #     FX, FY = np.meshgrid(fx, fy)
# # # #     f = np.sqrt(FX ** 2 + FY ** 2)
# # # #     f0 = 1.0 / L0
# # # #     psd = (f ** 2 + f0 ** 2) ** (-11.0 / 6.0)
# # # #     psd[0, 0] = 0.0
# # # #     noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
# # # #     screen = np.fft.ifft2(noise * np.sqrt(psd)).real
# # # #     screen -= np.mean(screen)
# # # #     if np.std(screen) > 0:
# # # #         screen *= (0.1 / r0) ** (5.0 / 6.0)
# # # #     if rms_opd_m is not None and np.std(screen) > 0:
# # # #         screen *= rms_opd_m / np.std(screen)
# # # #     return screen


# # # # def build_example_bench() -> Tuple[OpticalBench3D, List[Beam3D], Dict[str, Any]]:
# # # #     beam_diameter = 13e-3
# # # #     beam_radius = beam_diameter / 2
# # # #     wavelength = 532e-9
# # # #     screen_clear_radius = 83e-3 / 2

# # # #     z_source = 0.00
# # # #     z_fa = 0.14
# # # #     z_gl1 = 0.48
# # # #     z_gl2 = 0.53
# # # #     z_gl3 = 0.58
# # # #     z_fold = 0.78

# # # #     x_relay_l1 = 0.18
# # # #     f_relay1 = 0.08
# # # #     f_relay2 = 1.70 * f_relay1
# # # #     x_relay_l2 = x_relay_l1 + f_relay1 + f_relay2
# # # #     x_pupil = x_relay_l2 + f_relay2

# # # #     bench = OpticalBench3D()

# # # #     fa_map = make_von_karman_opd_map(n=512, extent_m=0.12, r0=0.03, L0=10.0, rms_opd_m=180e-9, seed=21)
# # # #     bench.add(
# # # #         RotatingPhaseScreen3D(
# # # #             point=[0.0, 0.0, z_fa],
# # # #             normal=[0.0, 0.0, 1.0],
# # # #             opd_map=fa_map,
# # # #             map_extent_m=0.12,
# # # #             clear_radius=screen_clear_radius,
# # # #             angular_velocity=2 * np.pi * 0.4,
# # # #             label="FA PS",
# # # #         )
# # # #     )

# # # #     gl_specs = [
# # # #         (z_gl1, 90e-9, 0.7, 11, "GL PS 1"),
# # # #         (z_gl2, 110e-9, 1.0, 12, "GL PS 2"),
# # # #         (z_gl3, 130e-9, 1.4, 13, "GL PS 3"),
# # # #     ]
# # # #     for z, rms, hz, seed, label in gl_specs:
# # # #         gl_map = make_von_karman_opd_map(n=512, extent_m=0.12, r0=0.05, L0=10.0, rms_opd_m=rms, seed=seed)
# # # #         bench.add(
# # # #             RotatingPhaseScreen3D(
# # # #                 point=[0.0, 0.0, z],
# # # #                 normal=[0.0, 0.0, 1.0],
# # # #                 opd_map=gl_map,
# # # #                 map_extent_m=0.12,
# # # #                 clear_radius=screen_clear_radius,
# # # #                 angular_velocity=2 * np.pi * hz,
# # # #                 label=label,
# # # #             )
# # # #         )

# # # #     d_in = normalize([0.0, 0.0, 1.0])
# # # #     d_out = normalize([1.0, 0.0, 0.0])
# # # #     mirror_normal = normalize(d_out - d_in)
# # # #     bench.add(Mirror3D(point=[0.0, 0.0, z_fold], normal=mirror_normal, aperture_radius=20e-3, label="Fold M1"))
# # # #     bench.add(Lens3D(point=[x_relay_l1, 0.0, z_fold], normal=[1.0, 0.0, 0.0], f=f_relay1, aperture_radius=18e-3, label="Relay L1"))
# # # #     bench.add(Lens3D(point=[x_relay_l2, 0.0, z_fold], normal=[1.0, 0.0, 0.0], f=f_relay2, aperture_radius=20e-3, label="Relay L2"))

# # # #     arcmin_to_rad = np.pi / (180.0 * 60.0)
# # # #     theta_10 = 10.0 * arcmin_to_rad
# # # #     beams = [
# # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=+theta_10, field_angle_y=+theta_10, wavelength=wavelength, label="field +x +y"),
# # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=-theta_10, field_angle_y=+theta_10, wavelength=wavelength, label="field -x +y"),
# # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=+theta_10, field_angle_y=-theta_10, wavelength=wavelength, label="field +x -y"),
# # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=-theta_10, field_angle_y=-theta_10, wavelength=wavelength, label="field -x -y"),
# # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=0.0, field_angle_y=0.0, wavelength=wavelength, label="field center"),
# # # #     ]

# # # #     meta = {
# # # #         "analysis_plane_point": np.array([x_pupil, 0.0, z_fold]),
# # # #         "analysis_plane_normal": np.array([1.0, 0.0, 0.0]),
# # # #         "beam_diameter": beam_diameter,
# # # #         "wavelength": wavelength,
# # # #         "field_points": [
# # # #             (+10.0, +10.0, "corner ++"),
# # # #             (-10.0, +10.0, "corner -+"),
# # # #             (+10.0, -10.0, "corner +-"),
# # # #             (-10.0, -10.0, "corner --"),
# # # #             (0.0, 0.0, "center"),
# # # #         ],
# # # #         "screen_labels": ["FA PS", "GL PS 1", "GL PS 2", "GL PS 3"],
# # # #     }
# # # #     return bench, beams, meta


# # # # # ============================================================
# # # # # Example plotting routines
# # # # # ============================================================

# # # # def plot_field_long_exposure_psfs(
# # # #     bench: OpticalBench3D,
# # # #     meta: Dict[str, Any],
# # # #     exposure_s: float = 30.0,
# # # #     dt_s: float = 0.5,
# # # #     npix: int = 512,
# # # #     half_width_ld: float = 5.0,
# # # # ):
# # # #     times = np.arange(0.0, exposure_s, dt_s)
# # # #     plane_point = meta["analysis_plane_point"]
# # # #     plane_normal = meta["analysis_plane_normal"]
# # # #     beam_diameter = meta["beam_diameter"]
# # # #     wavelength = meta["wavelength"]

# # # #     fig, axes = plt.subplots(2, 3, figsize=(14, 9))
# # # #     axes = axes.ravel()

# # # #     for i, (fx_arcmin, fy_arcmin, label) in enumerate(meta["field_points"]):
# # # #         beam = Beam3D.field_beam(
# # # #             radius=0.5 * beam_diameter,
# # # #             nrings=2,
# # # #             nphi=10,
# # # #             origin=[0.0, 0.0, 0.0],
# # # #             field_angle_x=np.deg2rad(fx_arcmin / 60.0),
# # # #             field_angle_y=np.deg2rad(fy_arcmin / 60.0),
# # # #             wavelength=wavelength,
# # # #             label=label,
# # # #         )

# # # #         stack = []
# # # #         first_pack = None
# # # #         for t in times:
# # # #             sample = sample_beam_phase_amplitude_on_plane(
# # # #                 beam=beam,
# # # #                 bench=bench,
# # # #                 plane_point=plane_point,
# # # #                 plane_normal=plane_normal,
# # # #                 t=float(t),
# # # #                 npix=npix,
# # # #                 diameter=beam_diameter,
# # # #             )
# # # #             psf_pack = psf_from_plane_sample(sample)
# # # #             stack.append(psf_pack["psf"])
# # # #             if first_pack is None:
# # # #                 first_pack = psf_pack

# # # #         long_psf = np.mean(np.array(stack), axis=0)
# # # #         if np.max(long_psf) > 0:
# # # #             long_psf = long_psf / np.max(long_psf)
# # # #         pack = dict(first_pack)
# # # #         pack["psf"] = long_psf

# # # #         x_ld, y_ld = psf_coords_lambda_over_d(pack, beam_diameter)
# # # #         psf_crop, x_crop, y_crop = crop_psf_to_lambda_over_d(pack["psf"], x_ld, y_ld, half_width_ld=half_width_ld)
# # # #         fwhm_ld = psf_fwhm_lambda_over_d(pack, beam_diameter)
# # # #         ell = psf_ellipticity_from_moments(psf_crop)

# # # #         ax = axes[i]
# # # #         im = ax.imshow(
# # # #             np.log10(np.maximum(psf_crop, 1e-8)),
# # # #             origin="lower",
# # # #             extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()],
# # # #         )
# # # #         ax.set_title(label)
# # # #         ax.set_xlabel(r"$\lambda/D$")
# # # #         ax.set_ylabel(r"$\lambda/D$")
# # # #         ax.set_aspect("equal")
# # # #         ax.text(
# # # #             0.04, 0.93,
# # # #             f"FWHM={fwhm_ld:.2f} $\\lambda/D$\nell={ell:.3f}",
# # # #             transform=ax.transAxes,
# # # #             color="white",
# # # #             ha="left",
# # # #             va="top",
# # # #             fontsize=10,
# # # #             bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
# # # #         )
# # # #         plt.colorbar(im, ax=ax, label="log10 PSF")

# # # #     if len(axes) > len(meta["field_points"]):
# # # #         axes[-1].axis("off")

# # # #     fig.suptitle(f"Long-exposure PSFs ({exposure_s:.1f} s, dt={dt_s:.2f} s)", y=0.98)
# # # #     fig.tight_layout()
# # # #     return fig


# # # # def plot_phase_screen_contributions(
# # # #     bench: OpticalBench3D,
# # # #     beams: List[Beam3D],
# # # #     meta: Dict[str, Any],
# # # #     t: float = 0.0,
# # # #     npix: int = 160,
# # # # ):
# # # #     plane_point = meta["analysis_plane_point"]
# # # #     plane_normal = meta["analysis_plane_normal"]
# # # #     beam_diameter = meta["beam_diameter"]
# # # #     screen_labels = meta["screen_labels"]

# # # #     nrows = len(beams)
# # # #     ncols = len(screen_labels)
# # # #     fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)

# # # #     vmax = 0.0
# # # #     data_cache = {}
# # # #     for beam in beams:
# # # #         for screen_label in screen_labels:
# # # #             sample = sample_beam_phase_amplitude_on_plane(
# # # #                 beam=beam,
# # # #                 bench=bench,
# # # #                 plane_point=plane_point,
# # # #                 plane_normal=plane_normal,
# # # #                 t=t,
# # # #                 npix=npix,
# # # #                 diameter=beam_diameter,
# # # #                 screen_names=[screen_label],
# # # #             )
# # # #             phase = sample["phase_map_rad"]
# # # #             data_cache[(beam.label, screen_label)] = sample
# # # #             vmax = max(vmax, np.nanmax(np.abs(phase)))

# # # #     if vmax <= 0:
# # # #         vmax = 1.0

# # # #     for i, beam in enumerate(beams):
# # # #         for j, screen_label in enumerate(screen_labels):
# # # #             sample = data_cache[(beam.label, screen_label)]
# # # #             xx = sample["xx"] * 1e3
# # # #             yy = sample["yy"] * 1e3
# # # #             ax = axes[i, j]
# # # #             im = ax.imshow(
# # # #                 sample["phase_map_rad"],
# # # #                 origin="lower",
# # # #                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
# # # #                 vmin=-vmax,
# # # #                 vmax=vmax,
# # # #             )
# # # #             if i == 0:
# # # #                 ax.set_title(screen_label)
# # # #             if j == 0:
# # # #                 ax.set_ylabel(f"{beam.label}\ny [mm]")
# # # #             else:
# # # #                 ax.set_ylabel("y [mm]")
# # # #             ax.set_xlabel("x [mm]")
# # # #             ax.set_aspect("equal")
# # # #             plt.colorbar(im, ax=ax, label="phase [rad]")

# # # #     fig.suptitle(f"Per-screen phase contributions at t = {t:.2f} s", y=0.995)
# # # #     fig.tight_layout()
# # # #     return fig


# # # # # ============================================================
# # # # # Main example
# # # # # ============================================================

# # # # if __name__ == "__main__":
# # # #     bench, beams, meta = build_example_bench()

# # # #     fig1 = bench.plot_3d(beams, s_end=0.25, title="Instrument layout with ray tracing", t=0.0)

# # # #     fig2 = plot_field_long_exposure_psfs(
# # # #         bench=bench,
# # # #         meta=meta,
# # # #         exposure_s=30.0,
# # # #         dt_s=0.5,
# # # #         npix=512,
# # # #         half_width_ld=5.0,
# # # #     )

# # # #     fig3 = plot_phase_screen_contributions(
# # # #         bench=bench,
# # # #         beams=beams,
# # # #         meta=meta,
# # # #         t=0.0,
# # # #         npix=160,
# # # #     )

# # # #     plt.show()

# # # # # import copy
# # # # # from dataclasses import dataclass, field
# # # # # from typing import List, Optional, Tuple, Dict, Any

# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np


# # # # # """
# # # # # Minimal ray-tracing + rotating phase-screen module.

# # # # # Goals
# # # # # -----
# # # # # - Keep the ray/instrument structure from the original ray-tracing script.
# # # # # - Add a beam model that carries wavelength, diameter, OPD accumulation,
# # # # #   and can be sampled on arbitrary planes.
# # # # # - Add rotating phase screens with bilinear sampling.
# # # # # - Stay geometric / collimated for now.
# # # # # - Provide diagnostics for:
# # # # #     * 3D instrument layout with ray tracing
# # # # #     * sampled phase and amplitude on planes
# # # # #     * long-exposure PSFs at multiple field points
# # # # #     * per-screen phase contributions for each beam

# # # # # Notes
# # # # # -----
# # # # # This is not a Fresnel propagator. The beam is represented as a collimated bundle
# # # # # with an associated sampled phase map on analysis planes. Optical elements change
# # # # # ray directions geometrically. Phase screens add OPD but do not yet refract rays
# # # # # via local slopes.
# # # # # """


# # # # # # ============================================================
# # # # # # Helpers
# # # # # # ============================================================


# # # # # def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
# # # # #     v = np.asarray(v, dtype=float)
# # # # #     n = np.linalg.norm(v)
# # # # #     if n < eps:
# # # # #         raise ValueError("Cannot normalize near-zero vector.")
# # # # #     return v / n



# # # # # def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# # # # #     n = normalize(normal)
# # # # #     ref = np.array([1.0, 0.0, 0.0])
# # # # #     if abs(np.dot(ref, n)) > 0.9:
# # # # #         ref = np.array([0.0, 1.0, 0.0])
# # # # #     e1 = normalize(np.cross(n, ref))
# # # # #     e2 = normalize(np.cross(n, e1))
# # # # #     return e1, e2



# # # # # def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
# # # # #     c = np.cos(angle_rad)
# # # # #     s = np.sin(angle_rad)
# # # # #     return np.array([[c, -s], [s, c]])



# # # # # def bilinear_sample(grid: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
# # # # #     ny, nx = grid.shape

# # # # #     x0 = np.floor(x_pix).astype(int)
# # # # #     y0 = np.floor(y_pix).astype(int)
# # # # #     x1 = x0 + 1
# # # # #     y1 = y0 + 1

# # # # #     x0 = np.clip(x0, 0, nx - 1)
# # # # #     x1 = np.clip(x1, 0, nx - 1)
# # # # #     y0 = np.clip(y0, 0, ny - 1)
# # # # #     y1 = np.clip(y1, 0, ny - 1)

# # # # #     Ia = grid[y0, x0]
# # # # #     Ib = grid[y0, x1]
# # # # #     Ic = grid[y1, x0]
# # # # #     Id = grid[y1, x1]

# # # # #     wa = (x1 - x_pix) * (y1 - y_pix)
# # # # #     wb = (x_pix - x0) * (y1 - y_pix)
# # # # #     wc = (x1 - x_pix) * (y_pix - y0)
# # # # #     wd = (x_pix - x0) * (y_pix - y0)

# # # # #     return wa * Ia + wb * Ib + wc * Ic + wd * Id



# # # # # def fft_psf_from_pupil_field(field: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
# # # # #     ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
# # # # #     psf = np.abs(ef) ** 2
# # # # #     if np.max(psf) > 0:
# # # # #         psf = psf / np.max(psf)

# # # # #     ny, nx = field.shape
# # # # #     fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
# # # # #     fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
# # # # #     return {"psf": psf, "fx": fx, "fy": fy}



# # # # # def psf_coords_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> Tuple[np.ndarray, np.ndarray]:
# # # # #     x_ld = psf_pack["fx"] * pupil_diameter_m
# # # # #     y_ld = psf_pack["fy"] * pupil_diameter_m
# # # # #     return x_ld, y_ld



# # # # # def crop_psf_to_lambda_over_d(
# # # # #     psf: np.ndarray,
# # # # #     x_ld: np.ndarray,
# # # # #     y_ld: np.ndarray,
# # # # #     half_width_ld: float = 5.0,
# # # # # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # # # #     ix = np.where((x_ld >= -half_width_ld) & (x_ld <= half_width_ld))[0]
# # # # #     iy = np.where((y_ld >= -half_width_ld) & (y_ld <= half_width_ld))[0]
# # # # #     if len(ix) == 0 or len(iy) == 0:
# # # # #         return psf, x_ld, y_ld
# # # # #     return psf[np.ix_(iy, ix)], x_ld[ix], y_ld[iy]



# # # # # def measure_fwhm_1d(x: np.ndarray, y: np.ndarray) -> float:
# # # # #     if len(x) < 3:
# # # # #         return np.nan
# # # # #     y = np.asarray(y, dtype=float)
# # # # #     x = np.asarray(x, dtype=float)
# # # # #     if np.nanmax(y) <= 0:
# # # # #         return np.nan

# # # # #     y = y / np.nanmax(y)
# # # # #     peak_idx = int(np.nanargmax(y))
# # # # #     half = 0.5

# # # # #     left = np.nan
# # # # #     for i in range(peak_idx, 0, -1):
# # # # #         if y[i] >= half and y[i - 1] < half:
# # # # #             x1, x2 = x[i - 1], x[i]
# # # # #             y1, y2 = y[i - 1], y[i]
# # # # #             left = x1 + (half - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[i]
# # # # #             break

# # # # #     right = np.nan
# # # # #     for i in range(peak_idx, len(y) - 1):
# # # # #         if y[i] >= half and y[i + 1] < half:
# # # # #             x1, x2 = x[i], x[i + 1]
# # # # #             y1, y2 = y[i], y[i + 1]
# # # # #             right = x1 + (half - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[i]
# # # # #             break

# # # # #     if np.isfinite(left) and np.isfinite(right):
# # # # #         return float(right - left)
# # # # #     return np.nan



# # # # # def psf_fwhm_lambda_over_d(psf_pack: Dict[str, np.ndarray], pupil_diameter_m: float) -> float:
# # # # #     x_ld, _ = psf_coords_lambda_over_d(psf_pack, pupil_diameter_m)
# # # # #     psf = psf_pack["psf"]
# # # # #     cy = psf.shape[0] // 2
# # # # #     return measure_fwhm_1d(x_ld, psf[cy, :])



# # # # # def psf_ellipticity_from_moments(psf: np.ndarray) -> float:
# # # # #     if np.nanmax(psf) <= 0:
# # # # #         return np.nan
# # # # #     psf = psf / np.nansum(psf)
# # # # #     ny, nx = psf.shape
# # # # #     y, x = np.indices(psf.shape)

# # # # #     x0 = np.nansum(x * psf)
# # # # #     y0 = np.nansum(y * psf)

# # # # #     xx = np.nansum((x - x0) ** 2 * psf)
# # # # #     yy = np.nansum((y - y0) ** 2 * psf)
# # # # #     xy = np.nansum((x - x0) * (y - y0) * psf)

# # # # #     cov = np.array([[xx, xy], [xy, yy]])
# # # # #     evals = np.linalg.eigvalsh(cov)
# # # # #     evals = np.clip(evals, 0.0, None)
# # # # #     if evals[-1] <= 0:
# # # # #         return np.nan
# # # # #     major = np.sqrt(evals[-1])
# # # # #     minor = np.sqrt(evals[0])
# # # # #     if major <= 0:
# # # # #         return np.nan
# # # # #     return float(1.0 - minor / major)


# # # # # # ============================================================
# # # # # # Ray / beam
# # # # # # ============================================================

# # # # # @dataclass
# # # # # class Ray3D:
# # # # #     r: np.ndarray
# # # # #     d: np.ndarray
# # # # #     alive: bool = True
# # # # #     opd: float = 0.0

# # # # #     def __post_init__(self):
# # # # #         self.r = np.asarray(self.r, dtype=float).reshape(3)
# # # # #         self.d = normalize(self.d)

# # # # #     def copy(self):
# # # # #         return Ray3D(self.r.copy(), self.d.copy(), self.alive, self.opd)


# # # # # @dataclass
# # # # # class Beam3D:
# # # # #     rays: List[Ray3D]
# # # # #     label: str = "beam"
# # # # #     wavelength: float = 532e-9
# # # # #     diameter: float = 10e-3
# # # # #     metadata: Dict[str, Any] = field(default_factory=dict)

# # # # #     @property
# # # # #     def radius(self) -> float:
# # # # #         return 0.5 * self.diameter

# # # # #     @property
# # # # #     def chief_ray(self) -> Ray3D:
# # # # #         return self.rays[0]

# # # # #     @classmethod
# # # # #     def collimated_circular(
# # # # #         cls,
# # # # #         radius: float = 5e-3,
# # # # #         nrings: int = 2,
# # # # #         nphi: int = 8,
# # # # #         origin=(0, 0, 0),
# # # # #         direction=(0, 0, 1),
# # # # #         wavelength: float = 532e-9,
# # # # #         label: str = "collimated",
# # # # #     ):
# # # # #         origin = np.asarray(origin, dtype=float)
# # # # #         direction = normalize(direction)

# # # # #         rays = [Ray3D(origin.copy(), direction.copy())]
# # # # #         for ir in range(1, nrings + 1):
# # # # #             rr = radius * ir / nrings
# # # # #             n_this = max(6, nphi * ir)
# # # # #             for k in range(n_this):
# # # # #                 phi = 2 * np.pi * k / n_this
# # # # #                 pos = origin + np.array([rr * np.cos(phi), rr * np.sin(phi), 0.0])
# # # # #                 rays.append(Ray3D(pos, direction.copy()))

# # # # #         return cls(rays=rays, label=label, wavelength=wavelength, diameter=2 * radius)

# # # # #     @classmethod
# # # # #     def field_beam(
# # # # #         cls,
# # # # #         radius: float = 5e-3,
# # # # #         nrings: int = 2,
# # # # #         nphi: int = 8,
# # # # #         origin=(0, 0, 0),
# # # # #         field_angle_x: float = 0.0,
# # # # #         field_angle_y: float = 0.0,
# # # # #         wavelength: float = 532e-9,
# # # # #         label: str = "field beam",
# # # # #     ):
# # # # #         d = normalize(np.array([field_angle_x, field_angle_y, 1.0]))
# # # # #         return cls.collimated_circular(
# # # # #             radius=radius,
# # # # #             nrings=nrings,
# # # # #             nphi=nphi,
# # # # #             origin=origin,
# # # # #             direction=d,
# # # # #             wavelength=wavelength,
# # # # #             label=label,
# # # # #         )

# # # # #     def copy(self) -> "Beam3D":
# # # # #         return copy.deepcopy(self)

# # # # #     def propagated_to_plane(self, plane_point: np.ndarray, plane_normal: np.ndarray) -> Tuple[float, np.ndarray]:
# # # # #         return self.chief_ray.copy().r, self.chief_ray.copy().d  # placeholder for API symmetry


# # # # # # ============================================================
# # # # # # Optical elements
# # # # # # ============================================================

# # # # # @dataclass
# # # # # class OpticalElement3D:
# # # # #     point: np.ndarray
# # # # #     normal: np.ndarray
# # # # #     label: str = "element"

# # # # #     def __post_init__(self):
# # # # #         self.point = np.asarray(self.point, dtype=float).reshape(3)
# # # # #         self.normal = normalize(self.normal)

# # # # #     def intersect_parameter(self, ray: Ray3D, eps: float = 1e-12):
# # # # #         denom = np.dot(ray.d, self.normal)
# # # # #         if abs(denom) < eps:
# # # # #             return None
# # # # #         s = np.dot(self.point - ray.r, self.normal) / denom
# # # # #         if s < 0:
# # # # #             return None
# # # # #         return s

# # # # #     def intersect_point(self, ray: Ray3D):
# # # # #         s = self.intersect_parameter(ray)
# # # # #         if s is None:
# # # # #             return None, None
# # # # #         return ray.r + s * ray.d, s

# # # # #     def plane_basis(self):
# # # # #         return orthonormal_basis_from_normal(self.normal)

# # # # #     def local_coordinates(self, p):
# # # # #         e1, e2 = self.plane_basis()
# # # # #         dp = p - self.point
# # # # #         return np.dot(dp, e1), np.dot(dp, e2)

# # # # #     def apply(self, ray: Ray3D):
# # # # #         return ray


# # # # # @dataclass
# # # # # class Lens3D(OpticalElement3D):
# # # # #     f: float = 0.1
# # # # #     aperture_radius: float = 10e-3
# # # # #     label: str = "lens"

# # # # #     def apply(self, ray: Ray3D):
# # # # #         p, _ = self.intersect_point(ray)
# # # # #         if p is None:
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         u, v = self.local_coordinates(p)
# # # # #         if np.hypot(u, v) > self.aperture_radius:
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         e1, e2 = self.plane_basis()
# # # # #         n = self.normal

# # # # #         du = np.dot(ray.d, e1)
# # # # #         dv = np.dot(ray.d, e2)
# # # # #         dn = np.dot(ray.d, n)
# # # # #         if abs(dn) < 1e-12:
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         du_out = du - u / self.f
# # # # #         dv_out = dv - v / self.f
# # # # #         dn_out = dn

# # # # #         d_out = du_out * e1 + dv_out * e2 + dn_out * n
# # # # #         ray.r = p
# # # # #         ray.d = normalize(d_out)
# # # # #         return ray


# # # # # @dataclass
# # # # # class Mirror3D(OpticalElement3D):
# # # # #     aperture_radius: float = 10e-3
# # # # #     label: str = "mirror"

# # # # #     def apply(self, ray: Ray3D):
# # # # #         p, _ = self.intersect_point(ray)
# # # # #         if p is None:
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         u, v = self.local_coordinates(p)
# # # # #         if np.hypot(u, v) > self.aperture_radius:
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         d_out = ray.d - 2 * np.dot(ray.d, self.normal) * self.normal
# # # # #         ray.r = p
# # # # #         ray.d = normalize(d_out)
# # # # #         return ray


# # # # # @dataclass
# # # # # class RotatingPhaseScreen3D(OpticalElement3D):
# # # # #     opd_map: np.ndarray = None
# # # # #     map_extent_m: float = 0.10
# # # # #     clear_radius: float = 20e-3
# # # # #     angular_velocity: float = 0.0
# # # # #     rotation_angle0: float = 0.0
# # # # #     label: str = "phase screen"

# # # # #     def __post_init__(self):
# # # # #         super().__post_init__()
# # # # #         if self.opd_map is None:
# # # # #             raise ValueError("opd_map must be provided.")
# # # # #         self.opd_map = np.asarray(self.opd_map, dtype=float)
# # # # #         if self.opd_map.ndim != 2:
# # # # #             raise ValueError("opd_map must be a 2D array.")

# # # # #     @property
# # # # #     def map_ny(self) -> int:
# # # # #         return self.opd_map.shape[0]

# # # # #     @property
# # # # #     def map_nx(self) -> int:
# # # # #         return self.opd_map.shape[1]

# # # # #     def current_rotation_angle(self, t: float) -> float:
# # # # #         return self.rotation_angle0 + self.angular_velocity * t

# # # # #     def uv_to_pixel(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# # # # #         u = uv[..., 0]
# # # # #         v = uv[..., 1]
# # # # #         x_pix = (u / self.map_extent_m + 0.5) * (self.map_nx - 1)
# # # # #         y_pix = (v / self.map_extent_m + 0.5) * (self.map_ny - 1)
# # # # #         return x_pix, y_pix

# # # # #     def contains_uv(self, uv: np.ndarray) -> np.ndarray:
# # # # #         r = np.sqrt(np.sum(np.asarray(uv) ** 2, axis=-1))
# # # # #         return r <= self.clear_radius

# # # # #     def sample_uv(self, uv: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
# # # # #         uv = np.asarray(uv, dtype=float)
# # # # #         rot = rotation_matrix_2d(-self.current_rotation_angle(t))
# # # # #         uv_rot = np.einsum("ij,...j->...i", rot, uv)
# # # # #         inside = self.contains_uv(uv_rot)

# # # # #         x_pix, y_pix = self.uv_to_pixel(uv_rot)
# # # # #         valid = inside & (x_pix >= 0) & (x_pix <= self.map_nx - 1) & (y_pix >= 0) & (y_pix <= self.map_ny - 1)

# # # # #         opd = np.zeros_like(x_pix, dtype=float)
# # # # #         if np.any(valid):
# # # # #             opd[valid] = bilinear_sample(self.opd_map, x_pix[valid], y_pix[valid])
# # # # #         return opd, valid

# # # # #     def apply(self, ray: Ray3D, t: float = 0.0):
# # # # #         p, _ = self.intersect_point(ray)
# # # # #         if p is None:
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         u, v = self.local_coordinates(p)
# # # # #         opd, valid = self.sample_uv(np.array([[u, v]]), t=t)
# # # # #         if not bool(valid[0]):
# # # # #             ray.alive = False
# # # # #             return ray

# # # # #         ray.opd += float(opd[0])
# # # # #         ray.r = p
# # # # #         return ray


# # # # # # ============================================================
# # # # # # Bench
# # # # # # ============================================================

# # # # # @dataclass
# # # # # class OpticalBench3D:
# # # # #     elements: List[OpticalElement3D] = field(default_factory=list)

# # # # #     def add(self, element: OpticalElement3D):
# # # # #         self.elements.append(element)

# # # # #     def trace_beam(self, beam: Beam3D, s_end: float = 1.0, n_line_samples: int = 60, t: float = 0.0):
# # # # #         all_paths = []
# # # # #         traced_rays = []

# # # # #         for ray0 in beam.rays:
# # # # #             ray = ray0.copy()
# # # # #             path = [ray.r.copy()]

# # # # #             for elem in self.elements:
# # # # #                 if not ray.alive:
# # # # #                     break
# # # # #                 p, s = elem.intersect_point(ray)
# # # # #                 if p is None:
# # # # #                     continue

# # # # #                 seg_s = np.linspace(0.0, s, n_line_samples)
# # # # #                 for ss in seg_s[1:]:
# # # # #                     path.append(ray.r + ss * ray.d)

# # # # #                 if isinstance(elem, RotatingPhaseScreen3D):
# # # # #                     ray = elem.apply(ray, t=t)
# # # # #                 else:
# # # # #                     ray = elem.apply(ray)
# # # # #                 path.append(ray.r.copy())

# # # # #             if ray.alive:
# # # # #                 seg_s = np.linspace(0.0, s_end, n_line_samples)
# # # # #                 for ss in seg_s[1:]:
# # # # #                     path.append(ray.r + ss * ray.d)

# # # # #             all_paths.append(np.array(path))
# # # # #             traced_rays.append(ray)

# # # # #         return all_paths, traced_rays

# # # # #     def _draw_element_3d(self, ax, elem, npts: int = 100):
# # # # #         e1, e2 = elem.plane_basis()
# # # # #         if isinstance(elem, Lens3D):
# # # # #             R = elem.aperture_radius
# # # # #             text = f"{elem.label}\nf={elem.f:.3f} m"
# # # # #         elif isinstance(elem, Mirror3D):
# # # # #             R = elem.aperture_radius
# # # # #             text = elem.label
# # # # #         elif isinstance(elem, RotatingPhaseScreen3D):
# # # # #             R = elem.clear_radius
# # # # #             text = elem.label
# # # # #         else:
# # # # #             R = 1e-3
# # # # #             text = elem.label

# # # # #         t = np.linspace(0, 2 * np.pi, npts)
# # # # #         ring = elem.point[None, :] + R * (
# # # # #             np.cos(t)[:, None] * e1[None, :] + np.sin(t)[:, None] * e2[None, :]
# # # # #         )
# # # # #         ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], lw=2)
# # # # #         ax.text(*elem.point, text)

# # # # #     @staticmethod
# # # # #     def _set_axes_equal(ax):
# # # # #         x_limits = ax.get_xlim3d()
# # # # #         y_limits = ax.get_ylim3d()
# # # # #         z_limits = ax.get_zlim3d()
# # # # #         x_range = abs(x_limits[1] - x_limits[0])
# # # # #         y_range = abs(y_limits[1] - y_limits[0])
# # # # #         z_range = abs(z_limits[1] - z_limits[0])
# # # # #         x_mid = np.mean(x_limits)
# # # # #         y_mid = np.mean(y_limits)
# # # # #         z_mid = np.mean(z_limits)
# # # # #         plot_radius = 0.5 * max([x_range, y_range, z_range])
# # # # #         ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
# # # # #         ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
# # # # #         ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

# # # # #     def plot_3d(self, beams: List[Beam3D], s_end: float = 0.5, figsize=(10, 8), title: str = "3D optical bench", t: float = 0.0):
# # # # #         fig = plt.figure(figsize=figsize)
# # # # #         ax = fig.add_subplot(111, projection="3d")

# # # # #         for beam in beams:
# # # # #             paths, _ = self.trace_beam(beam, s_end=s_end, t=t)
# # # # #             for path in paths:
# # # # #                 ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=1.2)

# # # # #         for elem in self.elements:
# # # # #             self._draw_element_3d(ax, elem)

# # # # #         ax.set_xlabel("x [m]")
# # # # #         ax.set_ylabel("y [m]")
# # # # #         ax.set_zlabel("z [m]")
# # # # #         ax.set_title(title)
# # # # #         self._set_axes_equal(ax)
# # # # #         plt.tight_layout()
# # # # #         return fig


# # # # # # ============================================================
# # # # # # Plane sampling / decomposition
# # # # # # ============================================================


# # # # # def sample_beam_phase_amplitude_on_plane(
# # # # #     beam: Beam3D,
# # # # #     bench: OpticalBench3D,
# # # # #     plane_point: np.ndarray,
# # # # #     plane_normal: np.ndarray,
# # # # #     t: float,
# # # # #     npix: int = 192,
# # # # #     diameter: Optional[float] = None,
# # # # #     screen_names: Optional[List[str]] = None,
# # # # # ) -> Dict[str, Any]:
# # # # #     plane_point = np.asarray(plane_point, dtype=float)
# # # # #     plane_normal = normalize(plane_normal)
# # # # #     diameter = beam.diameter if diameter is None else diameter

# # # # #     e1, e2 = orthonormal_basis_from_normal(plane_normal)
# # # # #     r = 0.5 * diameter
# # # # #     x = np.linspace(-r, r, npix)
# # # # #     y = np.linspace(-r, r, npix)
# # # # #     xx, yy = np.meshgrid(x, y)
# # # # #     mask = (xx ** 2 + yy ** 2) <= r ** 2

# # # # #     chief = beam.rays[0]
# # # # #     pts_plane = (
# # # # #         plane_point[None, None, :]
# # # # #         + xx[:, :, None] * e1[None, None, :]
# # # # #         + yy[:, :, None] * e2[None, None, :]
# # # # #     )

# # # # #     total_opd = np.zeros_like(xx, dtype=float)

# # # # #     for elem in bench.elements:
# # # # #         if not isinstance(elem, RotatingPhaseScreen3D):
# # # # #             continue
# # # # #         if screen_names is not None and elem.label not in screen_names:
# # # # #             continue

# # # # #         denom = np.dot(chief.d, elem.normal)
# # # # #         if abs(denom) < 1e-12:
# # # # #             continue

# # # # #         rel = elem.point[None, None, :] - pts_plane
# # # # #         s = np.sum(rel * elem.normal[None, None, :], axis=2) / denom
# # # # #         pts_screen = pts_plane + s[:, :, None] * chief.d[None, None, :]

# # # # #         rels = pts_screen - elem.point[None, None, :]
# # # # #         u = np.sum(rels * elem._e1[None, None, :], axis=2)
# # # # #         v = np.sum(rels * elem._e2[None, None, :], axis=2)
# # # # #         uv = np.stack([u, v], axis=-1)

# # # # #         opd, valid = elem.sample_uv(uv.reshape(-1, 2), t=t)
# # # # #         opd = opd.reshape(xx.shape)
# # # # #         valid = valid.reshape(xx.shape)
# # # # #         total_opd += np.where(valid, opd, 0.0)

# # # # #     total_opd = np.where(mask, total_opd, np.nan)
# # # # #     phase = 2.0 * np.pi * total_opd / beam.wavelength
# # # # #     dx = xx[0, 1] - xx[0, 0]
# # # # #     amp = mask.astype(float)

# # # # #     return {
# # # # #         "xx": xx,
# # # # #         "yy": yy,
# # # # #         "mask": mask,
# # # # #         "amplitude": amp,
# # # # #         "opd_map_m": total_opd,
# # # # #         "phase_map_rad": phase,
# # # # #         "dx": dx,
# # # # #     }



# # # # # def psf_from_plane_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
# # # # #     mask = sample["mask"]
# # # # #     phase = np.where(mask, np.nan_to_num(sample["phase_map_rad"], nan=0.0), 0.0)
# # # # #     amp = np.where(mask, sample["amplitude"], 0.0)
# # # # #     field = amp * np.exp(1j * phase)
# # # # #     out = fft_psf_from_pupil_field(field, sample["dx"])
# # # # #     return {**sample, **out}


# # # # # # ============================================================
# # # # # # Example builders
# # # # # # ============================================================


# # # # # def make_von_karman_opd_map(
# # # # #     n: int = 512,
# # # # #     extent_m: float = 0.12,
# # # # #     r0: float = 0.03,
# # # # #     L0: float = 10.0,
# # # # #     rms_opd_m: float = 150e-9,
# # # # #     seed: Optional[int] = None,
# # # # # ) -> np.ndarray:
# # # # #     rng = np.random.default_rng(seed)
# # # # #     dx = extent_m / n
# # # # #     fx = np.fft.fftfreq(n, d=dx)
# # # # #     fy = np.fft.fftfreq(n, d=dx)
# # # # #     FX, FY = np.meshgrid(fx, fy)
# # # # #     f = np.sqrt(FX ** 2 + FY ** 2)
# # # # #     f0 = 1.0 / L0
# # # # #     psd = (f ** 2 + f0 ** 2) ** (-11.0 / 6.0)
# # # # #     psd[0, 0] = 0.0
# # # # #     noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
# # # # #     screen = np.fft.ifft2(noise * np.sqrt(psd)).real
# # # # #     screen -= np.mean(screen)
# # # # #     if np.std(screen) > 0:
# # # # #         screen *= (0.1 / r0) ** (5.0 / 6.0)
# # # # #     if rms_opd_m is not None and np.std(screen) > 0:
# # # # #         screen *= rms_opd_m / np.std(screen)
# # # # #     return screen



# # # # # def build_example_bench() -> Tuple[OpticalBench3D, List[Beam3D], Dict[str, Any]]:
# # # # #     beam_diameter = 13e-3
# # # # #     beam_radius = beam_diameter / 2
# # # # #     wavelength = 532e-9
# # # # #     screen_clear_radius = 83e-3 / 2

# # # # #     z_source = 0.00
# # # # #     z_fa = 0.14
# # # # #     z_gl1 = 0.48
# # # # #     z_gl2 = 0.53
# # # # #     z_gl3 = 0.58
# # # # #     z_fold = 0.78

# # # # #     x_relay_l1 = 0.18
# # # # #     f_relay1 = 0.08
# # # # #     f_relay2 = 1.70 * f_relay1
# # # # #     x_relay_l2 = x_relay_l1 + f_relay1 + f_relay2
# # # # #     x_pupil = x_relay_l2 + f_relay2

# # # # #     bench = OpticalBench3D()

# # # # #     # free atmosphere screen
# # # # #     fa_map = make_von_karman_opd_map(n=512, extent_m=0.12, r0=0.03, L0=10.0, rms_opd_m=180e-9, seed=21)
# # # # #     bench.add(
# # # # #         RotatingPhaseScreen3D(
# # # # #             point=[0.0, 0.0, z_fa],
# # # # #             normal=[0.0, 0.0, 1.0],
# # # # #             opd_map=fa_map,
# # # # #             map_extent_m=0.12,
# # # # #             clear_radius=screen_clear_radius,
# # # # #             angular_velocity=2 * np.pi * 0.4,
# # # # #             label="FA PS",
# # # # #         )
# # # # #     )

# # # # #     gl_specs = [
# # # # #         (z_gl1, 90e-9, 0.7, 11, "GL PS 1"),
# # # # #         (z_gl2, 110e-9, 1.0, 12, "GL PS 2"),
# # # # #         (z_gl3, 130e-9, 1.4, 13, "GL PS 3"),
# # # # #     ]
# # # # #     for z, rms, hz, seed, label in gl_specs:
# # # # #         gl_map = make_von_karman_opd_map(n=512, extent_m=0.12, r0=0.05, L0=10.0, rms_opd_m=rms, seed=seed)
# # # # #         bench.add(
# # # # #             RotatingPhaseScreen3D(
# # # # #                 point=[0.0, 0.0, z],
# # # # #                 normal=[0.0, 0.0, 1.0],
# # # # #                 opd_map=gl_map,
# # # # #                 map_extent_m=0.12,
# # # # #                 clear_radius=screen_clear_radius,
# # # # #                 angular_velocity=2 * np.pi * hz,
# # # # #                 label=label,
# # # # #             )
# # # # #         )

# # # # #     # fold + relay
# # # # #     d_in = normalize([0.0, 0.0, 1.0])
# # # # #     d_out = normalize([1.0, 0.0, 0.0])
# # # # #     mirror_normal = normalize(d_out - d_in)
# # # # #     bench.add(Mirror3D(point=[0.0, 0.0, z_fold], normal=mirror_normal, aperture_radius=20e-3, label="Fold M1"))
# # # # #     bench.add(Lens3D(point=[x_relay_l1, 0.0, z_fold], normal=[1.0, 0.0, 0.0], f=f_relay1, aperture_radius=18e-3, label="Relay L1"))
# # # # #     bench.add(Lens3D(point=[x_relay_l2, 0.0, z_fold], normal=[1.0, 0.0, 0.0], f=f_relay2, aperture_radius=20e-3, label="Relay L2"))

# # # # #     arcmin_to_rad = np.pi / (180.0 * 60.0)
# # # # #     theta_10 = 10.0 * arcmin_to_rad
# # # # #     beams = [
# # # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=+theta_10, field_angle_y=+theta_10, wavelength=wavelength, label="field +x +y"),
# # # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=-theta_10, field_angle_y=+theta_10, wavelength=wavelength, label="field -x +y"),
# # # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=+theta_10, field_angle_y=-theta_10, wavelength=wavelength, label="field +x -y"),
# # # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=-theta_10, field_angle_y=-theta_10, wavelength=wavelength, label="field -x -y"),
# # # # #         Beam3D.field_beam(radius=beam_radius, nrings=2, nphi=10, origin=[0.0, 0.0, z_source], field_angle_x=0.0, field_angle_y=0.0, wavelength=wavelength, label="field center"),
# # # # #     ]

# # # # #     meta = {
# # # # #         "analysis_plane_point": np.array([x_pupil, 0.0, z_fold]),
# # # # #         "analysis_plane_normal": np.array([1.0, 0.0, 0.0]),
# # # # #         "beam_diameter": beam_diameter,
# # # # #         "wavelength": wavelength,
# # # # #         "field_points": [
# # # # #             (+10.0, +10.0, "corner ++"),
# # # # #             (-10.0, +10.0, "corner -+"),
# # # # #             (+10.0, -10.0, "corner +-"),
# # # # #             (-10.0, -10.0, "corner --"),
# # # # #             (0.0, 0.0, "center"),
# # # # #         ],
# # # # #         "screen_labels": ["FA PS", "GL PS 1", "GL PS 2", "GL PS 3"],
# # # # #     }
# # # # #     return bench, beams, meta


# # # # # # ============================================================
# # # # # # Example plotting routines
# # # # # # ============================================================


# # # # # def plot_field_long_exposure_psfs(
# # # # #     bench: OpticalBench3D,
# # # # #     meta: Dict[str, Any],
# # # # #     exposure_s: float = 30.0,
# # # # #     dt_s: float = 0.5,
# # # # #     npix: int = 512,
# # # # #     half_width_ld: float = 5.0,
# # # # # ):
# # # # #     times = np.arange(0.0, exposure_s, dt_s)
# # # # #     plane_point = meta["analysis_plane_point"]
# # # # #     plane_normal = meta["analysis_plane_normal"]
# # # # #     beam_diameter = meta["beam_diameter"]
# # # # #     wavelength = meta["wavelength"]

# # # # #     fig, axes = plt.subplots(2, 3, figsize=(14, 9))
# # # # #     axes = axes.ravel()

# # # # #     for i, (fx_arcmin, fy_arcmin, label) in enumerate(meta["field_points"]):
# # # # #         beam = Beam3D.field_beam(
# # # # #             radius=0.5 * beam_diameter,
# # # # #             nrings=2,
# # # # #             nphi=10,
# # # # #             origin=[0.0, 0.0, 0.0],
# # # # #             field_angle_x=np.deg2rad(fx_arcmin / 60.0),
# # # # #             field_angle_y=np.deg2rad(fy_arcmin / 60.0),
# # # # #             wavelength=wavelength,
# # # # #             label=label,
# # # # #         )

# # # # #         stack = []
# # # # #         first_pack = None
# # # # #         for t in times:
# # # # #             sample = sample_beam_phase_amplitude_on_plane(
# # # # #                 beam=beam,
# # # # #                 bench=bench,
# # # # #                 plane_point=plane_point,
# # # # #                 plane_normal=plane_normal,
# # # # #                 t=float(t),
# # # # #                 npix=npix,
# # # # #                 diameter=beam_diameter,
# # # # #             )
# # # # #             psf_pack = psf_from_plane_sample(sample)
# # # # #             stack.append(psf_pack["psf"])
# # # # #             if first_pack is None:
# # # # #                 first_pack = psf_pack

# # # # #         long_psf = np.mean(np.array(stack), axis=0)
# # # # #         if np.max(long_psf) > 0:
# # # # #             long_psf = long_psf / np.max(long_psf)
# # # # #         pack = dict(first_pack)
# # # # #         pack["psf"] = long_psf

# # # # #         x_ld, y_ld = psf_coords_lambda_over_d(pack, beam_diameter)
# # # # #         psf_crop, x_crop, y_crop = crop_psf_to_lambda_over_d(pack["psf"], x_ld, y_ld, half_width_ld=half_width_ld)
# # # # #         fwhm_ld = psf_fwhm_lambda_over_d(pack, beam_diameter)
# # # # #         ell = psf_ellipticity_from_moments(psf_crop)

# # # # #         ax = axes[i]
# # # # #         im = ax.imshow(
# # # # #             np.log10(np.maximum(psf_crop, 1e-8)),
# # # # #             origin="lower",
# # # # #             extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()],
# # # # #         )
# # # # #         ax.set_title(label)
# # # # #         ax.set_xlabel(r"$\lambda/D$")
# # # # #         ax.set_ylabel(r"$\lambda/D$")
# # # # #         ax.set_aspect("equal")
# # # # #         ax.text(
# # # # #             0.04, 0.93,
# # # # #             f"FWHM={fwhm_ld:.2f} $\\lambda/D$\nell={ell:.3f}",
# # # # #             transform=ax.transAxes,
# # # # #             color="white",
# # # # #             ha="left",
# # # # #             va="top",
# # # # #             fontsize=10,
# # # # #             bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
# # # # #         )
# # # # #         plt.colorbar(im, ax=ax, label="log10 PSF")

# # # # #     if len(axes) > len(meta["field_points"]):
# # # # #         axes[-1].axis("off")

# # # # #     fig.suptitle(f"Long-exposure PSFs ({exposure_s:.1f} s, dt={dt_s:.2f} s)", y=0.98)
# # # # #     fig.tight_layout()
# # # # #     return fig



# # # # # def plot_phase_screen_contributions(
# # # # #     bench: OpticalBench3D,
# # # # #     beams: List[Beam3D],
# # # # #     meta: Dict[str, Any],
# # # # #     t: float = 0.0,
# # # # #     npix: int = 160,
# # # # # ):
# # # # #     plane_point = meta["analysis_plane_point"]
# # # # #     plane_normal = meta["analysis_plane_normal"]
# # # # #     beam_diameter = meta["beam_diameter"]
# # # # #     screen_labels = meta["screen_labels"]

# # # # #     nrows = len(beams)
# # # # #     ncols = len(screen_labels)
# # # # #     fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)

# # # # #     vmax = 0.0
# # # # #     data_cache = {}
# # # # #     for beam in beams:
# # # # #         for screen_label in screen_labels:
# # # # #             sample = sample_beam_phase_amplitude_on_plane(
# # # # #                 beam=beam,
# # # # #                 bench=bench,
# # # # #                 plane_point=plane_point,
# # # # #                 plane_normal=plane_normal,
# # # # #                 t=t,
# # # # #                 npix=npix,
# # # # #                 diameter=beam_diameter,
# # # # #                 screen_names=[screen_label],
# # # # #             )
# # # # #             phase = sample["phase_map_rad"]
# # # # #             data_cache[(beam.label, screen_label)] = sample
# # # # #             vmax = max(vmax, np.nanmax(np.abs(phase)))

# # # # #     if vmax <= 0:
# # # # #         vmax = 1.0

# # # # #     for i, beam in enumerate(beams):
# # # # #         for j, screen_label in enumerate(screen_labels):
# # # # #             sample = data_cache[(beam.label, screen_label)]
# # # # #             xx = sample["xx"] * 1e3
# # # # #             yy = sample["yy"] * 1e3
# # # # #             ax = axes[i, j]
# # # # #             im = ax.imshow(
# # # # #                 sample["phase_map_rad"],
# # # # #                 origin="lower",
# # # # #                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
# # # # #                 vmin=-vmax,
# # # # #                 vmax=vmax,
# # # # #             )
# # # # #             if i == 0:
# # # # #                 ax.set_title(screen_label)
# # # # #             if j == 0:
# # # # #                 ax.set_ylabel(f"{beam.label}\ny [mm]")
# # # # #             else:
# # # # #                 ax.set_ylabel("y [mm]")
# # # # #             ax.set_xlabel("x [mm]")
# # # # #             ax.set_aspect("equal")
# # # # #             plt.colorbar(im, ax=ax, label="phase [rad]")

# # # # #     fig.suptitle(f"Per-screen phase contributions at t = {t:.2f} s", y=0.995)
# # # # #     fig.tight_layout()
# # # # #     return fig


# # # # # # ============================================================
# # # # # # Main example
# # # # # # ============================================================

# # # # # if __name__ == "__main__":
# # # # #     bench, beams, meta = build_example_bench()

# # # # #     # 3D instrument with ray tracing
# # # # #     fig1 = bench.plot_3d(beams, s_end=0.25, title="Instrument layout with ray tracing", t=0.0)

# # # # #     # long-exposure PSF maps at 5 field points
# # # # #     fig2 = plot_field_long_exposure_psfs(
# # # # #         bench=bench,
# # # # #         meta=meta,
# # # # #         exposure_s=30.0,
# # # # #         dt_s=0.5,
# # # # #         npix=512,
# # # # #         half_width_ld=5.0,
# # # # #     )

# # # # #     # decomposed per-screen contributions for each beam
# # # # #     fig3 = plot_phase_screen_contributions(
# # # # #         bench=bench,
# # # # #         beams=beams,
# # # # #         meta=meta,
# # # # #         t=0.0,
# # # # #         npix=160,
# # # # #     )

# # # # #     plt.show()
