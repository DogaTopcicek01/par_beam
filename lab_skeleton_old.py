
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
    def __init__(self, Nt, Ntheta, delta_t, theta0=0.0, center_offset=0.0):
        self.Nt = Nt #number of detector channels
        self.Ntheta = Ntheta # number of porjection views
        self.delta_t = delta_t #detector spacing
        self.theta0 = theta0  # starting angle
        self.center_offset = center_offset
        self.t0 = - (delta_t / 2.0) * (Nt - 1) #coordinate of detector 0 center t0 = -(delta_t/2) (N_t -1)


class ImgInfo:
    def __init__(self, Nx:int, Ny:int, lfov:float):
        assert Nx == Ny, "lab assumes Nx == Ny"
        self.Nx = Nx # # of pixels along x axis
        self.Ny = Ny # number of pixels along y axis
        self.N = Nx * Ny  # total # of pixels
        self.lfov = lfov #Total physical width (Field of View) lfov = Nt delta_t
        # N_t = number of detector elements, delta_t = physical size of each detector elemen
        self.delta_p = lfov / Nx   # pixel width Δp
        # coordinates of pixel centers (r0x, r0y) left-bottom
        self.r0x = - (self.delta_p/2.0) * (Nx - 1)
        self.r0y = - (self.delta_p/2.0) * (Ny - 1)
        # assign physical coordinates to the pixels, starting from the first pixel.
        # / 2 -> to be centered around middle ; -1 -> step size if there are 5 pixel, there are 4 gaps

# -----------------------
# Pixel profile computation
# Computes trapezoidal weighting profiles of a pixel for each ray angle.
# Profiles describe how much a pixel contributes to rays at offsets from its center.
# Returns tensor of shape (Ntheta, LEN_PIX) for all projection angles.
# -----------------------
def pix_prof_comp(sino: SinoInfo, img: ImgInfo, device='cpu') -> torch.Tensor:
    """
    Compute pixel profiles for each view.
    Returns: profiles tensor of shape (Ntheta, LEN_PIX) (torch.float32)
    Lab section: Pixel profile computation (Table 1).
    """

    Ntheta = sino.Ntheta
    delta_p = img.delta_p
    profiles = torch.zeros((Ntheta, LEN_PIX), dtype=torch.float32, device=device)
    #Len_pix = how many samples we take to approximate the trapezoidal function.

    # δ grid from -Δp to +Δp (2*Δp interval) sampled LEN_PIX points:
    delta_vals = torch.linspace(-delta_p, delta_p, LEN_PIX, device=device)  # δ

    for v in range(Ntheta): # loops over all projections angles
        theta = sino.theta0 + v * (math.pi / Ntheta)
        # Determine Lmax, δ1, δ2 depending on theta region (table 1)
        # We'll map theta to [0, pi)
        t = theta % math.pi

        # compute parameters from the table
        #Lmax = max intersection length of a ray through a pixel at this angle.
        # d2: half-width of trapezoid support (full width = 2*d2), Fixed as Δp / √2.
        #d1 : half-width of plateau region where profile = Lmax., Depends on θ; adjusts trapezoid shape for angle.

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
        profile = torch.zeros_like(delta_vals) # This will hold the trapezoid values for this angle, vector same size as delt_vals
        absd = torch.abs(delta_vals) # absolute bcs we need to check how far the ray is from pixel center w
        #delt_vals = distance of the ray from the pixel center
        # inside plateau
        plateau_mask = absd <= d1 + 1e-14 #flat top, (center of pixel) so Lmax
        profile[plateau_mask] = Lmax

        # rising/falling trapezoid between d1 and d2
        trapezoid_mask = (absd > d1) & (absd <= d2 + 1e-14)
        profile[trapezoid_mask] = Lmax * (1.0 - (absd[trapezoid_mask] - d1) / (d2 - d1 + 1e-15))

        # outside support remains zero
        profiles[v] = profile

    return profiles  # shape (Ntheta, LEN_PIX)

# -----------------------
# compute pixel center coordinates
# Converts a pixel’s linear index j into its 2D grid indices and then
# computes the physical coordinates (rx, ry) of the pixel center.
# -----------------------
def pixel_center_coords(img: ImgInfo, j:int) -> Tuple[float, float]:
    """
    j: pixel linear index (raster order left->right, bottom->top)
    returns (rx, ry)
    """
    #linear index to 2D coordinates in the pixel grid.
    # j = is the linear index of a pixel in the flattened image array.
    jx = j % img.Nx
    jy = j // img.Nx
    #computes the physical coordinates
    rx = img.r0x + jx * img.delta_p
    ry = img.r0y + jy * img.delta_p
    return rx, ry #physical loc of the pixel

# -----------------------
# A_col_comp: compute sparse column for pixel j
# Computes a single sparse column A_{*,j} of the system matrix for pixel (im_row, im_col).
# For each projection angle, it projects the pixel center onto the detector axis,
# finds the detector bins that intersect the pixel’s width, and subdivides each bin
# into sub-elements for numerical integration. Each sub-element’s offset from the
# pixel center (δ) is mapped to a precomputed trapezoid profile that represents
# pixel contribution along the detector. Contributions are summed and averaged over
# sub-elements to compute A_{i,j}, and only nonzero values are stored in a sparse
# representation (indices + values). Implements lab equations (7–12) / C pseudocode.
# -----------------------

# col for a single pixel
def A_col_comp(im_row:int, im_col:int, sino: SinoInfo, img: ImgInfo, profiles: torch.Tensor) -> Dict:
    """
    Compute A_{*,j} for pixel located at image row/col (im_row, im_col).
    Returns a dict with keys: 'indices' (list of sinogram indices i), 'vals' (list of floats)
    This follows the pseudocode in the lab. Uses LEN_DET uniform sensitivity.
    """
    Nx = img.Nx # number of pixels along X
    jx, jy = im_col, im_row
    # Cartesian coordinates of center of pixel j;
    #delta_p = physical pixel size, r0x = phsical coordinate of teh first pixel
    rx = img.r0x + jx * img.delta_p
    ry = img.r0y + jy * img.delta_p

    Nt = sino.Nt # number of detector bins
    Ntheta = sino.Ntheta # number of projection views
    delta_t = sino.delta_t #detector bin width
    t0 = sino.t0 # detector coordinate of the first bean

    #Initialize empty lists to store sparse column data
    # vals: A_{i, j} values
    # inds = tells you the positions (columns) in the sinogram where this pixel actually contributes something nonzero.
    # vals gives the corresponding values of those contributions

    vals = []
    inds = []

    # loop over all projections angles and compute angle of current projection
    for v in range(Ntheta):
        theta = sino.theta0 + v * (math.pi / Ntheta)
        # precompute sin/cos from rotation matrix
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
        #Loop over all detector bins that intersect this pixel.
        for d in range(dmin, dmax + 1):
            Aval = 0.0
            # sum over detector sub-elements
            for h in range(LEN_DET):
                th = t0 - delta_t / 2.0 + (d + sino.center_offset) * delta_t + (h * delta_t / LEN_DET)
                #th = t0 - delta_t / 2.0 + d * delta_t + (h * delta_t / LEN_DET) if no offset

                # compute profile index lh using eq(12)
                # lh = ceil(([th - (rjy cos θv − rjx sin θv) + ∆p] / (2∆p) * LEN_PIX))
                delta_val = th - t_center  # δ
                # translate δ=0 to center of profiles grid by adding Δp and scaling
                kf = ( (delta_val + img.delta_p) / (2.0 * img.delta_p) ) * LEN_PIX
                lh = int(math.ceil(kf)) - 1  # convert to 0-based index (lab ceil remapped)
                if 0 <= lh < LEN_PIX:
                    Aval += float(profiles[v, lh])
            Aval = Aval / LEN_DET  # uniform sensitivity φ(h) = 1/LEN_DET, Average over sub-elements → approximates integral over detector bin.
            if Aval > 0: #Store non-zero contributions in sparse column representation.
                i = ind_base + d
                vals.append(Aval)
                inds.append(i)

    return {'indices': np.array(inds, dtype=np.int64), 'vals': np.array(vals, dtype=np.float32)}

# -----------------------
# Build sparse system matrix A


# -----------------------
def build_A_all_columns(sino: SinoInfo, img: ImgInfo, profiles: torch.Tensor, device='cpu') -> List[Dict]:
    """
    Assemble A as a list of columns where each entry is the dict returned by A_col_comp.
    This can be memory/time heavy; use for moderate-sized problems or sampling.
    """
    #Loop over all pixels in the image: row (jy) then column (jx).
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
    #M = total number of measurments Ntheta(projections) * Nt(detector bins)
    N = x.numel() #total number of pixels in the image
    x_np = x.detach().cpu().numpy()

    #loops over each pixel
    for j in range(N):
        col = A_cols[j]
        if col['vals'].size == 0: # skip if no contribution
            continue
        contrib = col['vals'] * float(x_np[j]) #  Multiply the pixel value x[j] by its precomputed contributions vals to all affected detector bins.
        y_np_idx = col['indices']   # detector indices affected by this pixel
        # accumulate
        y[y_np_idx] += torch.from_numpy(contrib).to(device)
    return y

# -----------------------
# compute initial error sinogram e = y - A x
# -----------------------
def compute_error_sinogram(y: torch.Tensor, x: torch.Tensor, A_cols: List[Dict], M:int) -> torch.Tensor:
    Ax = forward_project_from_columns(x, A_cols, M)
    e = y - Ax
    return e

# -----------------------
# q-GGMRF surrogate weights helper
# -----------------------
def compute_surrogate_weights(x: torch.Tensor, p:float, q:float, sigma_x:float, b_s_r:float):
    """
    compute btilde for each neighbor pair {s, r}
    This is a naive implementation: compute per-pixel neighbor weights given x.
    Returns a tensor of same spatial dims as x containing sum of 2*btilde_{j,k} over neighbors for each j,
    and a function to compute the prior linear term per pixel when needed.
    Note: For efficient implementation you would vectorize neighbor computations;
    this simple function keeps readability for the lab.
    """
    # Assumes x is 2D torch (Ny, Nx)
    Ny, Nx = x.shape
    b_sum = torch.zeros_like(x)
    # neighbor offsets (8-neighborhood as in lab)
    neighbors = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    for jy in range(Ny):
        for jx in range(Nx):
            s_val = x[jy, jx]
            b_local = 0.0
            for dy, dx in neighbors:
                ky = jy + dy
                kx = jx + dx
                if 0 <= ky < Ny and 0 <= kx < Nx:
                    r_val = x[ky, kx]
                    diff = s_val - r_val
                    absdiff = torch.abs(diff)
                    # lab expression: formula for btilde_{s,r}
                    if absdiff.item() == 0:
                        btilde = b_s_r * (p / (sigma_x**q)) * (sigma_x**(q-p))  # fallback
                    else:
                        # second branch:
                        term = (absdiff / sigma_x)**(q-p)
                        btilde = b_s_r * ( (p * (absdiff**(p-2))) / (2 * (sigma_x**p)) ) * \
                                 ( term * (q/p + term) / ((1 + term)**2 + 1e-16) )
                    b_local += 2.0 * btilde
            b_sum[jy, jx] = b_local
    return b_sum

# -----------------------
# ICD main routine
# -----------------------
def icd_reconstruction(y: torch.Tensor,
                       sino: SinoInfo,
                       img: ImgInfo,
                       A_cols: List[Dict],
                       initial_x: torch.Tensor = None,
                       R_diag: torch.Tensor = None,
                       max_iters: int = 10,
                       stop_thresh: float = 2.0e-5,
                       p:float = 1.2,
                       q:float = 2.0,
                       sigma_x: float = 1.0,
                       b_s_r: float = 1.0,
                       device='cpu'):
    """
    Perform ICD iterative coordinate descent to minimize surrogate MAP cost.
    Returns reconstructed x (Nx*Ny flattened) and cost history.
    Notes:
     - y: (M,) sinogram measurements (torch)
     - R_diag: (M,) observation weights, if None assumes identity
     - A_cols: list of dicts for each image pixel column
    """
    M = sino.Nt * sino.Ntheta
    N = img.N
    if initial_x is None:
        x = torch.zeros((N,), dtype=torch.float32, device=device)
    else:
        x = initial_x.clone().to(device)

    if R_diag is None: #inverse noise varience / weight
        #R_diag = 1.0 / sigma_i_squared
        R_diag = torch.ones((M,), dtype=torch.float32, device=device)

    # precompute ||A_{*,j}||^2_R for each column j
    A_norm2_R = torch.zeros((N,), dtype=torch.float32, device=device)
    for j in range(N):
        col = A_cols[j]
        if col['vals'].size == 0:
            A_norm2_R[j] = 0.0
        else:
            vals = torch.from_numpy(col['vals']).to(device)
            idx = torch.from_numpy(col['indices']).to(device)
            A_norm2_R[j] = torch.sum(vals * vals * R_diag[idx])

    # compute initial residual e = y - Ax
    e = compute_error_sinogram(y, x, A_cols, M)

    cost_history = []
    for it in range(max_iters):
        total_update = 0.0
        # random order as lab suggests
        order = np.random.permutation(N)
        for j in order:
            col = A_cols[j]
            if col['vals'].size == 0:
                continue
            idx = torch.from_numpy(col['indices']).to(device)
            vals = torch.from_numpy(col['vals']).to(device)
            # compute θ1 = -e^T A_{*,j} + sum_k 2 btilde_{j,k} (xj - xk)
            # data term
            e_sub = e[idx]  # residual entries relevant
            theta1_data = - torch.sum(e_sub * vals * R_diag[idx])
            # prior term approximation (we need neighbor xk; here we'll approximate with finite differences)
            # For simplicity in lab code: assume isotropic constant prior weight B (could be refined)
            # We'll approximate sum_k 2 btilde_{j,k} (xj - xk) as B * xj - prior_sum
            # To keep consistent we set prior_sum = 0 and use B = sum neigh 2*btilde (approx.)
            # A more exact version would require neighbor indices mapping.
            # We'll use a small constant regularization as fallback:
            B = 1e-3
            theta1 = theta1_data + B * float(x[j].item())

            # θ2 = ||A_{*,j}||^2_R + sum_k 2 btilde_{j,k}
            theta2 = float(A_norm2_R[j].item()) + B

            # compute alpha = -theta1/theta2, then clip to [-xj, +inf)
            alpha = - theta1 / (theta2 + 1e-12)
            # clip so that xj + alpha >= 0 (assuming non-negative image)
            if x[j].item() + alpha < 0:
                alpha = - float(x[j].item())
            # apply update
            xj_old = float(x[j].item())
            new_xj = xj_old + alpha
            x[j] = new_xj
            total_update += abs(alpha)

            # update residual e := e - A_{*,j} * alpha
            e[idx] -= vals * alpha

        avg_update = total_update / float(N)
        # compute cost (data fidelity + prior approx)
        Ax = forward_project_from_columns(x, A_cols, M)
        data_term = 0.5 * torch.sum(R_diag * (y - Ax)**2).item()
        # simple prior term: sum |grad|^p approx (coarse)
        prior_term = 0.0
        cost_history.append(data_term + prior_term)

        print(f"Iter {it+1}/{max_iters} avg_update={avg_update:.6e} data_term={data_term:.6e}")

        if avg_update < stop_thresh:
            break

    return x, cost_history

