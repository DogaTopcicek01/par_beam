
import math
import numpy as np
import torch
from typing import List, Dict, Tuple

# -----------------------
# Default constants
# -----------------------
LEN_PIX = 511    # number of samples used for the 2*Δp interval
LEN_DET = 101   # detector sub-samples per detector (uniform sensitivity)
EPS = 1e-12

# -----------------------
# Helper dataclasses for parameters
# -----------------------
class SinoInfo:
    def __init__(self, Nt:int, Ntheta:int, delta_t:float, theta0:float=0.0, detector_offset:float=0.0):
        self.Nt = Nt
        self.Ntheta = Ntheta
        self.delta_t = delta_t
        self.theta0 = theta0
        # t0: center of detector 0, shifted by detector_offset
        self.t0 = - (delta_t/2.0) * (Nt - 1) + detector_offset

class ImgInfo:
    def __init__(self, Nx:int, Ny:int, lfov:float):
        assert Nx == Ny, "lab assumes Nx == Ny"
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        self.lfov = lfov
        self.delta_p = lfov / Nx   # pixel width Δp
        # coordinates of pixel centers (r0x, r0y) left-bottom
        self.r0x = - (self.delta_p/2.0) * (Nx - 1)
        self.r0y = - (self.delta_p/2.0) * (Ny - 1)

# -----------------------
# Pixel profile computation
# -----------------------
def pix_prof_comp(sino: SinoInfo, img: ImgInfo, device='cpu') -> torch.Tensor:
    """
    Compute pixel profiles for each view.
    Returns: profiles tensor of shape (Ntheta, LEN_PIX) (torch.float32)
    Implementation follows the lab's Table 1 and description.
    """
    Ntheta = sino.Ntheta
    delta_p = img.delta_p
    profiles = torch.zeros((Ntheta, LEN_PIX), dtype=torch.float32, device=device)

    # δ grid from -Δp to +Δp (2*Δp interval) sampled LEN_PIX points:
    delta_vals = torch.linspace(-delta_p, delta_p, LEN_PIX, device=device)  # δ

    for v in range(Ntheta):
        theta = sino.theta0 + v * (math.pi / Ntheta)
        # Determine Lmax, δ1, δ2 depending on theta region (table 1)
        # We'll map theta to [0, pi)
        t = theta % math.pi

        # compute parameters
        if 0 <= t < math.pi/4:
            Lmax = delta_p / math.cos(t)
            d2 = delta_p / math.sqrt(2)
            d1 = delta_p/math.sqrt(2) * abs(math.sin(math.pi/4 - t))
        elif math.pi/4 <= t < math.pi/2:
            Lmax = delta_p / math.sin(t)
            d2 = delta_p / math.sqrt(2)
            d1 = (delta_p/math.sqrt(2)) * abs(math.cos(math.pi/4 - t))
        elif math.pi/2 <= t < 3*math.pi/4:
            Lmax = delta_p / math.sin(t)
            d2 = delta_p / math.sqrt(2)
            d1 = (delta_p/math.sqrt(2)) * abs(math.cos(t - math.pi/4))
        else:
            # [3π/4, π)
            Lmax = delta_p / abs(math.cos(t))
            d2 = delta_p / math.sqrt(2)
            d1 = (delta_p/math.sqrt(2)) * abs(math.sin(math.pi/4 - t))

        # simpler trapezoidal shape: support is [-δ2, +δ2], plateau when |δ| <= δ1
        # triangle edges between δ1 and δ2.
        profile = torch.zeros_like(delta_vals)
        absd = torch.abs(delta_vals)

        # inside plateau
        plateau_mask = absd <= d1 + 1e-14
        profile[plateau_mask] = Lmax

        # rising/falling trapezoid between d1 and d2
        trapezoid_mask = (absd > d1) & (absd <= d2 + 1e-14)
        profile[trapezoid_mask] = Lmax * (1.0 - (absd[trapezoid_mask] - d1) / (d2 - d1 + 1e-15))

        # outside support remains zero
        profiles[v] = profile

    return profiles  # shape (Ntheta, LEN_PIX)

# -----------------------
# compute pixel center coordinates
# -----------------------
def pixel_center_coords(img: ImgInfo, j:int) -> Tuple[float, float]:
    """
    j: pixel linear index (raster order left->right, bottom->top)
    returns (rx, ry)
    """
    jx = j % img.Nx
    jy = j // img.Nx
    rx = img.r0x + jx * img.delta_p
    ry = img.r0y + jy * img.delta_p
    return rx, ry

# -----------------------
# A_col_comp: compute sparse column for pixel j
# -----------------------
def A_col_comp(im_row:int, im_col:int, sino: SinoInfo, img: ImgInfo, profiles: torch.Tensor) -> Dict:
    """
    Compute A_{*,j} for pixel located at image row/col (im_row, im_col).
    Returns a dict with keys: 'indices' (list of sinogram indices i), 'vals' (list of floats)
    This follows the pseudocode in the lab. Uses LEN_DET uniform sensitivity.
    """
    Nx = img.Nx
    jx, jy = im_col, im_row  # note: user might use row/col ordering, keep consistent
    rx = img.r0x + jx * img.delta_p
    ry = img.r0y + jy * img.delta_p

    Nt = sino.Nt
    Ntheta = sino.Ntheta
    delta_t = sino.delta_t
    t0 = sino.t0

    vals = []
    inds = []

    for v in range(Ntheta):
        theta = sino.theta0 + v * (math.pi / Ntheta)
        # precompute sin/cos
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # find tmin/tmax for this pixel profile center (eq 7-8)
        t_center = ry * cos_t - rx * sin_t  # this is tj in lab notation
        tmin = t_center - img.delta_p
        tmax = t_center + img.delta_p

        # compute detector indices possibly intersecting
        dtmin = int(math.floor((tmin - sino.t0) / delta_t))
        dtmax = int(math.ceil((tmax - sino.t0) / delta_t))

        dmin = max(dtmin, 0)
        dmax = min(dtmax, Nt - 1)

        ind_base = v * Nt
        for d in range(dmin, dmax + 1):
            Aval = 0.0
            # sum over detector sub-elements
            for h in range(LEN_DET):
                th = t0 - delta_t/2.0 + d*delta_t + (h * delta_t / LEN_DET)

                # compute profile index lh using eq(12)
                # lh = ceil(([th - (rjy cos θv − rjx sin θv) + ∆p] / (2∆p) * LEN_PIX))
                delta_val = th - t_center  # δ
                # translate δ=0 to center of profiles grid by adding Δp and scaling
                kf = ( (delta_val + img.delta_p) / (2.0 * img.delta_p) ) * LEN_PIX
                lh = int(math.ceil(kf)) - 1  # convert to 0-based index (lab ceil remapped)
                if 0 <= lh < LEN_PIX:
                    Aval += float(profiles[v, lh])
            Aval = Aval / LEN_DET  # uniform sensitivity φ(h) = 1/LEN_DET
            if Aval > 0:
                i = ind_base + d
                vals.append(Aval)
                inds.append(i)

    return {'indices': np.array(inds, dtype=np.int64), 'vals': np.array(vals, dtype=np.float32)}

# -----------------------
# Build sparse A as list of columns (memory friendly)
# -----------------------
def build_A_all_columns(sino: SinoInfo, img: ImgInfo, profiles: torch.Tensor, device='cpu') -> List[Dict]:
    """
    Assemble A as a list of columns where each entry is the dict returned by A_col_comp.
    This can be memory/time heavy; use for moderate-sized problems or sampling.
    """
    cols = []
    for jy in range(img.Ny):
        for jx in range(img.Nx):
            col = A_col_comp(jy, jx, sino, img, profiles)
            cols.append(col)
    return cols

# -----------------------
# Forward projection using sparse columns: Ax
# -----------------------
def forward_project_from_columns(x: torch.Tensor, A_cols: List[Dict], M:int) -> torch.Tensor:
    """
    x: tensor of shape (N,) (flattened image raster)
    A_cols: list of length N of dicts {'indices', 'vals'}
    M: total number of sinogram measurements = Ntheta * Nt
    Returns y = A x as torch tensor of shape (M,)
    """
    device = x.device
    y = torch.zeros((M,), dtype=torch.float32, device=device)
    N = x.numel()
    x_np = x.detach().cpu().numpy()
    for j in range(N):
        col = A_cols[j]
        if col['vals'].size == 0:
            continue
        vals = torch.from_numpy(col['vals']).to(device)
        idx = torch.from_numpy(col['indices']).to(device)
        y[idx] += vals * x[j]

    return y

# -----------------------
# compute initial error sinogram e = y - A x
# -----------------------
def compute_error_sinogram(y: torch.Tensor, x: torch.Tensor, A_cols: List[Dict], M:int) -> torch.Tensor:
    Ax = forward_project_from_columns(x, A_cols, M)
    e = y - Ax
    return e

import torch, numpy as np

# -----------------------
# Compute surrogate weights and prior terms
# -----------------------
def compute_prior_terms(x2d, p, q, sigma_x, b_s_r):
    """
    Compute q-GGMRF surrogate weights for all pixel pairs.
    Returns:
      b_sum: sum_k 2*btilde_{j,k}
      prior_lin: sum_k 2*btilde_{j,k}*(x_j - x_k)
    """
    Ny, Nx = x2d.shape
    b_sum = torch.zeros_like(x2d)
    prior_linear = torch.zeros_like(x2d)
    neighbors = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

    for jy in range(Ny):
        for jx in range(Nx):
            s_val = x2d[jy,jx]
            b_local = 0.0
            prior_lin_local = 0.0
            for dy,dx in neighbors:
                ky, kx = jy+dy, jx+dx
                if 0 <= ky < Ny and 0 <= kx < Nx:
                    r_val = x2d[ky,kx]
                    diff = s_val - r_val
                    absdiff = torch.abs(diff)
                    if absdiff == 0:
                        btilde = b_s_r * (p/(sigma_x**q))*(sigma_x**(q-p))
                    else:
                        term = (absdiff/sigma_x)**(q-p)
                        btilde = b_s_r * ( (p*(absdiff**(p-2))) / (2*(sigma_x**p)) ) * \
                                  ( term * (q/p + term) / ((1+term)**2 + 1e-16) )
                    b_local += 2.0*btilde
                    prior_lin_local += 2.0*btilde * diff
            b_sum[jy,jx] = b_local
            prior_linear[jy,jx] = prior_lin_local
    return b_sum, prior_linear


# -----------------------
# ICD Reconstruction with Proper q-GGMRF Prior
# -----------------------
def icd_reconstruction_qggmrf(y, sino, img, A_cols,
                              initial_x=None,
                              R_diag=None,
                              max_iters=10,
                              stop_thresh=1e-4,
                              p=1.2, q=2.0,
                              sigma_x=1.0, b_s_r=1.0,
                              device='cpu'):
    """
    ICD MBIR reconstruction with q-GGMRF prior.
    """

    M = sino.Nt * sino.Ntheta
    N = img.N

    # initialization
    if initial_x is None:
        x = torch.zeros((N,), dtype=torch.float32, device=device)
    else:
        x = initial_x.clone().to(device)

    if R_diag is None:
        R_diag = torch.ones((M,), dtype=torch.float32, device=device)

    # precompute ||A_{*,j}||^2_R
    A_norm2_R = torch.zeros((N,), dtype=torch.float32, device=device)
    for j in range(N):
        col = A_cols[j]
        if col['vals'].size == 0:
            continue
        vals = torch.from_numpy(col['vals']).to(device)
        idx = torch.from_numpy(col['indices']).to(device)
        A_norm2_R[j] = torch.sum(vals * vals * R_diag[idx])

    # initial residual
    e = y - forward_project_from_columns(x, A_cols, M)

    cost_history = []

    for it in range(max_iters):
        total_update = 0.0

        # recompute prior terms every iteration
        x2d = x.reshape(img.Ny, img.Nx)
        b_sum, prior_lin = compute_prior_terms(x2d, p, q, sigma_x, b_s_r)
        b_sum_flat = b_sum.flatten()
        prior_lin_flat = prior_lin.flatten()

        order = np.random.permutation(N)
        for j in order:
            col = A_cols[j]
            if col['vals'].size == 0:
                continue
            idx = torch.from_numpy(col['indices']).to(device)
            vals = torch.from_numpy(col['vals']).to(device)
            e_sub = e[idx]
            # θ1 = -e^T A_{*,j} + sum_k 2btilde_{j,k}(xj - xk)
            theta1 = - torch.sum(e_sub * vals * R_diag[idx]) + prior_lin_flat[j]
            # θ2 = ||A_{*,j}||^2_R + sum_k 2btilde_{j,k}
            theta2 = A_norm2_R[j] + b_sum_flat[j]

            alpha = - theta1 / (theta2 + 1e-12)
            if x[j] + alpha < 0:
                alpha = -x[j]
            x[j] += alpha
            e[idx] -= vals * alpha
            total_update += abs(alpha)

        avg_update = total_update / float(N)

        # cost
        Ax = forward_project_from_columns(x, A_cols, M)
        data_term = 0.5 * torch.sum(R_diag * (y - Ax)**2).item()
        prior_term = torch.sum(b_sum_flat * (x**2)).item() * 1e-6  # optional diagnostic
        cost_history.append(data_term + prior_term)

        print(f"Iter {it+1}/{max_iters}  avg_update={avg_update:.3e}  data_term={data_term:.3e}")

        if avg_update < stop_thresh:
            break

    return x, cost_history
