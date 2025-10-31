import torch
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

OUT_A = r"C:\Users\dqz\Desktop\parallel_beam\A_parallel_params_radia.npz"
SINOGRAM_PATH = r"C:\Users\dqz\Downloads\xradia_slice0700.2Dsinodata"
WEIGHT_PATH = r"C:\Users\dqz\Downloads\xradia_slice0700.2Dweightdata"
OUTPUT_PATH = r"C:\Users\dqz\Desktop\reconstruction_radia.npy"

num_iters = 300
lr = 0.3
Positivity = True

# ---------------- QGGMRF PRIOR PARAMETERS ----------------
p = 1.2
q = 2.0
T = 1.0
SigmaX = 16.0
b_nearest = 1.0
b_diag = 0.707
b_interslice = 1.0
InitImageValue = 0.0202527

# ---------------- LOAD A-MATRIX ----------------
A_data = np.load(OUT_A)
rows = torch.tensor(A_data["row_idx_all"], dtype=torch.int64, device=device)
cols = torch.tensor(A_data["col_idx_all"], dtype=torch.int64, device=device)
vals = torch.tensor(A_data["val_all"], dtype=torch.float32, device=device)
Nrows = int(A_data["Nrows"])
Ncols = int(A_data["Ncols"])
Nx = int(A_data["Nx"])
Ny = int(A_data["Ny"])
Nz = int(A_data.get("Nz", 1))
NViews = int(A_data["NViews"])
NChan = int(A_data["NChannels"])

# ---------------- Determine slice(s) from A-matrix ----------------
# Each column index corresponds to a voxel in (x, y, z) flattened
z_coords = cols // (Nx * Ny)
unique_slices = z_coords.unique()
slice_idx = int(unique_slices[0].item())
print(f"Sinogram will be assigned to slice: {slice_idx} (from A-matrix)")

A_sparse = torch.sparse_coo_tensor(
    indices=torch.stack([rows, cols]),
    values=vals,
    size=(Nrows, Ncols)
).coalesce().to(device)
print(f"A-matrix loaded | shape={A_sparse.shape} | nnz={vals.numel()}")

# ---------------- LOAD SINOGRAM ----------------
with open(SINOGRAM_PATH, "rb") as f:
    sino_flat = np.frombuffer(f.read(), dtype=np.float32)

# Assign sinogram to the slice determined from A-matrix
sino = np.zeros((NViews, Nz, NChan), dtype=np.float32)
sino[:, slice_idx, :] = sino_flat.reshape(NViews, NChan)
y = torch.tensor(sino, dtype=torch.float32, device=device).flatten()

# ---------------- LOAD WEIGHTS ----------------
with open(WEIGHT_PATH, "rb") as f:
    weight_flat = np.frombuffer(f.read(), dtype=np.float32)

weights = np.zeros((NViews, Nz, NChan), dtype=np.float32)
weights[:, slice_idx, :] = weight_flat.reshape(NViews, NChan)
W = torch.tensor(weights, dtype=torch.float32, device=device).flatten()

# ---------------- PLOT SINOGRAM ----------------
plt.figure(figsize=(10,5))
plt.imshow(sino[:, slice_idx, :], cmap='gray', aspect='auto')
plt.title(f"Input Sinogram (slice {slice_idx})")
plt.xlabel("Detector channels")
plt.ylabel("Views")
plt.colorbar()
plt.show()

# ---------------- INITIALIZE VOLUME ----------------
x = torch.full((Ncols,), InitImageValue, dtype=torch.float32, device=device, requires_grad=True)

# ---------------- QGGMRF functions ----------------
def qggmrf_potential_tensor(delta, p, q, T, SigmaX, pow_sigmaX_p, eps=1e-10):
    abs_delta = torch.abs(delta) + eps
    temp = torch.pow(abs_delta / (T * SigmaX), (q - p))
    GGMRF_Pot = torch.pow(abs_delta, p) / (p * pow_sigmaX_p)
    return GGMRF_Pot * (temp / (1.0 + temp))

def MAPCostFunction3D_fixed_order(x,
                      p=2.0, q=2.0, T=1.0, SigmaX=1.0,
                      b_nearest=0.01, b_diag=0.01, b_interslice=0.01,
                      eps=1e-10, return_components=False):
    pow_sigmaX_p = SigmaX ** p
    plusx = torch.roll(x, shifts=-1, dims=0)
    plusy = torch.roll(x, shifts=-1, dims=1)
    plusz = torch.roll(x, shifts=-1, dims=2)
    delta_x = x - plusx
    delta_y = x - plusy
    delta_diag1 = x - torch.roll(x, shifts=(-1, -1), dims=(0, 1))
    delta_diag2 = x - torch.roll(x, shifts=(-1, 1), dims=(0, 1))
    delta_interslice = x - plusz
    pot_x = qggmrf_potential_tensor(delta_x, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_y = qggmrf_potential_tensor(delta_y, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_diag1 = qggmrf_potential_tensor(delta_diag1, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_diag2 = qggmrf_potential_tensor(delta_diag2, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_interslice = qggmrf_potential_tensor(delta_interslice, p, q, T, SigmaX, pow_sigmaX_p, eps)
    nlogprior_nearest = torch.sum(pot_x) + torch.sum(pot_y)
    nlogprior_diag    = torch.sum(pot_diag1) + torch.sum(pot_diag2)
    nlogprior_interslice = torch.sum(pot_interslice)
    reg = (b_nearest * nlogprior_nearest +
           b_diag * nlogprior_diag +
           b_interslice * nlogprior_interslice)
    if return_components:
        return reg, nlogprior_nearest, nlogprior_diag, nlogprior_interslice
    else:
        return reg

# ---------------- OPTIMIZER ----------------
optimizer = torch.optim.Adam([x], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

# ---------------- RECONSTRUCTION LOOP ----------------
for it in range(1, num_iters + 1):
    optimizer.zero_grad()
    Ax = torch.sparse.mm(A_sparse, x.unsqueeze(1)).squeeze(1)
    e = y - Ax
    data_term = 0.5 * torch.sum(W * (e ** 2))
    x_3d = x.view(Nx, Ny, Nz)
    reg_term = MAPCostFunction3D_fixed_order(
        x_3d, p=p, q=q, T=T, SigmaX=SigmaX,
        b_nearest=b_nearest, b_diag=b_diag, b_interslice=b_interslice
    )
    loss = data_term + reg_term
    loss.backward()
    torch.nn.utils.clip_grad_norm_([x], 1.0)
    optimizer.step()
    if Positivity:
        x.data.clamp_(0.0)
    scheduler.step()
    if it % 10 == 0:
        print(f"Iter {it:3d} | Loss: {loss.item():.3e} | Data: {data_term.item():.3e} | Reg: {reg_term.item():.3e} | LR: {scheduler.get_last_lr()[0]:.3g}")

# ---------------- RESHAPE AND SAVE ----------------
recon = x.detach().cpu().numpy().reshape((Nz, Ny, Nx))
np.save(OUTPUT_PATH, recon)
print(f"Reconstruction saved to {OUTPUT_PATH}")

# ---------------- PLOT ----------------

import numpy as np
import matplotlib.pyplot as plt

# Example slice
sl = recon[0] if recon.shape[0] == 1 else recon[recon.shape[0] // 2]

# Percentiles
p2, p98 = np.percentile(sl, [2, 98])
if p2 == p98:  # fallback
    p2, p98 = sl.min(), sl.max()

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 1️⃣ Raw min/max
im0 = axes[0].imshow(sl, cmap='gray', origin='upper')
axes[0].set_title("Raw min/max")
plt.colorbar(im0, ax=axes[0])

# 2️⃣ Percentile-based
im1 = axes[1].imshow(sl, cmap='gray', origin='upper', vmin=p2, vmax=p98)
axes[1].set_title("2nd-98th percentile")
plt.colorbar(im1, ax=axes[1])

plt.show()
