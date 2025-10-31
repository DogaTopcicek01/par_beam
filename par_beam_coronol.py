import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch
import os
from scipy.sparse import coo_matrix
import time

# -----------------------------
# Parameters
# -----------------------------
dicom_folder = r"C:\Users\dqz\Downloads\Al 176 (1)\20240711 Battery Foil-Tab Weld Al 176 16bit DICONDE 1 Slices"
A_path = r"C:\Users\dqz\Desktop\parallel_beam\A_parallel_params_subsampled_coronal.npz"
coronal_idx = 1725          # choose which coronal slice (X index)
num_iters = 200
output_dir = r"C:\Users\dqz\Desktop\recon_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Linear scaling parameters
P_air = 6415
P_Al = 25238
mu_Al = 0.075  # mm^-1

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load DICOM slices and form 3D volume
# -----------------------------
dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])
vol_slices = []

for f in dicom_files:
    ds = pydicom.dcmread(f)
    vol_slices.append(ds.pixel_array.astype(np.float32))

volume = np.stack(vol_slices, axis=0)  # shape (Nz, Ny, Nx)
Nz, Ny, Nx = volume.shape
print(f"Full 3D volume shape: Nz={Nz}, Ny={Ny}, Nx={Nx}")

# -----------------------------
# Extract coronal slice as 3D phantom (Nz=1)
# -----------------------------
phantom_coronal = volume[:, coronal_idx, :][np.newaxis, :, :]  # shape (1, Nz, Nx)
Nz_phantom, Ny_phantom, Nx_phantom = phantom_coronal.shape
print(f"Coronal phantom shape (3D, Nz=1): Nz={Nz_phantom}, Ny={Ny_phantom}, Nx={Nx_phantom}")

# Plot coronal phantom
plt.figure(figsize=(6,5))
plt.imshow(phantom_coronal[0], cmap='gray', origin='upper', aspect='auto')
plt.title(f"Coronal Phantom Slice (X={coronal_idx})")
plt.colorbar(label="Intensity")
plt.show()

# -----------------------------
# Scale to linear attenuation (mm^-1)
# -----------------------------
slope = mu_Al / (P_Al - P_air)
intercept = -slope * P_air
phantom_scaled = np.clip(phantom_coronal * slope + intercept, 0, None)

plt.figure()
plt.imshow(phantom_scaled[0], cmap='gray')
plt.title("Scaled Phantom (μ values, mm⁻¹)")
plt.colorbar()
plt.show()

# -----------------------------
# Load precomputed coronal A-matrix
# -----------------------------
A_data = np.load(A_path)
A = coo_matrix(
    (A_data["val_all"], (A_data["row_idx_all"], A_data["col_idx_all"])),
    shape=(int(A_data["Nrows"]), int(A_data["Ncols"]))
).tocsr()
print(f"A-matrix loaded: {A.shape}, nnz={A.nnz}")

# -----------------------------
# Compute sinogram
# -----------------------------
x_flat = phantom_scaled.flatten()
y = A @ x_flat
NViews = int(A_data["NViews"])
NChannels = int(A_data["NChannels"])
y = np.array(y).reshape(NViews, NChannels)

print(f"Sinogram shape: {y.shape}, min/max: {y.min():.4f}/{y.max():.4f}")

# Weight matrix
W = np.exp(-y)
print(f"Weight matrix min/max: {W.min():.4f}/{W.max():.4f}")

# Plot sinogram and weight
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(y, cmap="gray", aspect="auto")
axs[0].set_title("Sinogram (y)")
axs[1].imshow(W, cmap="gray", aspect="auto")
axs[1].set_title("Weight matrix (W = exp(-y))")
plt.tight_layout()
plt.show()

# -----------------------------
# Reconstruction function
# -----------------------------
def reconstruct_from_sinogram(y, W, recon_shape, A, num_iters=100, output_dir=None, name="recon"):
    # Convert A to torch sparse tensor
    A_coo = A.tocoo()
    indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    A_torch = torch.sparse_coo_tensor(indices, values, size=A.shape).to(device)

    # Initialize reconstruction
    x = torch.zeros(recon_shape, device=device, requires_grad=True)
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)
    W_torch = torch.tensor(W, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([x], lr=1e-2)
    losses = []

    start_time = time.time()
    for i in range(num_iters):
        optimizer.zero_grad()
        x_flat = x.flatten()
        Ax = torch.sparse.mm(A_torch, x_flat.unsqueeze(1)).squeeze()
        residual = y_torch.flatten() - Ax
        loss = torch.sum(W_torch.flatten() * (residual**2))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(min=0)
        losses.append(loss.item())

        if i % 50 == 0:
            print(f"{name} | Iter {i:4d} | Loss: {loss.item():.6f}")

    # Save reconstruction
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        recon_path = os.path.join(output_dir, f"{name}.npy")
        np.save(recon_path, x.detach().cpu().numpy())
        print(f"{name} saved to {recon_path}")

    print(f"Reconstruction shape: {x.shape}")
    print(f"Reconstruction min/max: {x.min().item():.6f}, {x.max().item():.6f}")

    return x.detach().cpu().numpy(), losses

# -----------------------------
# Perform reconstruction
# -----------------------------
recon, losses = reconstruct_from_sinogram(y, W, (Nz_phantom, Ny_phantom, Nx_phantom),
                                          A, num_iters=num_iters,
                                          output_dir=output_dir,
                                          name="recon_coronal")

# -----------------------------
# Visualize reconstruction
# -----------------------------
plt.figure(figsize=(6,5))
plt.imshow(recon[0], cmap='gray')
plt.title("Reconstruction (Coronal Slice)")
plt.colorbar(label="μ [mm^-1]")
plt.show()
