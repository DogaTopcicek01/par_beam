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
dicom_slice = r"C:\Users\dqz\Downloads\Al 176 (1)\20240711 Battery Foil-Tab Weld Al 176 16bit DICONDE 1 Slices\20240711 Battery Foil-Tab Weld Al 176 16bit DICONDE 1_0210.dcm"
A_path = r"C:\Users\dqz\Desktop\parallel_beam\A_parallel_params_subsampled_axial.npz"
row_range = (670, 770)      # crop rows
num_iters = 200
output_dir = r"C:\Users\dqz\Desktop\recon_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Linear scaling parameters
P_air = 6415
P_Al = 25238
mu_Al = 0.075  # mm^-1

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load precomputed A-matrix (already subsampled)
# -----------------------------
A_data = np.load(A_path)
A = coo_matrix(
    (A_data["val_all"], (A_data["row_idx_all"], A_data["col_idx_all"])),

    shape=(int(A_data["Nrows"]), int(A_data["Ncols"]))
).tocsr()
print(f"A-matrix loaded: {A.shape}, nnz={A.nnz}")

# -----------------------------
# Load and crop DICOM
# -----------------------------
ds = pydicom.dcmread(dicom_slice)
img_full = ds.pixel_array.astype(np.float32)
img_cropped = img_full[row_range[0]:row_range[1], :]
Ny, Nx = img_cropped.shape
Nz = 1
phantom = np.zeros((Nz, Ny, Nx))
phantom[0] = img_cropped

print(f"Full DICOM shape: {img_full.shape}")
print(f"Cropped phantom shape: {phantom.shape} (rows {row_range[0]}–{row_range[1]})")

# Plot full slice and cropped region
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.imshow(img_full, cmap="gray", origin="upper")
plt.axhline(row_range[0], color="r", linestyle="--")
plt.axhline(row_range[1], color="r", linestyle="--")
plt.title("Full DICOM with Cropping Range")
plt.subplot(1,2,2)
plt.imshow(phantom[0], cmap="gray", origin="upper")
plt.title("Cropped Phantom (Raw Intensities)")
plt.tight_layout()
plt.show()

# -----------------------------
# Scale to linear attenuation (mm^-1)
# -----------------------------
slope = mu_Al / (P_Al - P_air)
intercept = -slope * P_air
phantom_scaled = np.clip(phantom * slope + intercept, 0, None)
plt.imshow(phantom_scaled[0], cmap="gray")
plt.title("Scaled Phantom (μ values, mm⁻¹)")
plt.colorbar()
plt.show()

# -----------------------------
# Compute sinogram
# -----------------------------
x_flat = phantom_scaled.flatten()
y = A @ x_flat
NViews = int(A_data["NViews"])
NChannels = int(A_data["NChannels"])
y = np.array(y).reshape(NViews, NChannels)

# Print sinogram shape and value range
print(f"Sinogram shape: {y.shape}")
print(f"Sinogram min/max: {y.min():.4f}, {y.max():.4f}")

# Weight matrix
W = np.exp(-y)

# Print weight matrix range
print(f"Weight matrix min/max: {W.min():.4f}, {W.max():.4f}")

# Plot sinogram and weight matrix
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
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

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
    os.makedirs(output_dir, exist_ok=True)
    recon_path = os.path.join(output_dir, f"{name}.npy")
    np.save(recon_path, x.detach().cpu().numpy())
    print(f"{name} saved to {recon_path}, total time: {(time.time()-start_time)/60:.2f} min")
    print(f"Reconstruction shape: {x.shape}")
    print(f"Reconstruction min/max: {x.min().item():.6f}, {x.max().item():.6f}")

    return x.detach().cpu().numpy(), losses

# -----------------------------
# Perform reconstruction (no extra subsampling needed)
# -----------------------------
recon, losses = reconstruct_from_sinogram(y, W, (Nz, Ny, Nx), A, num_iters=num_iters,
                                          output_dir=output_dir, name="recon_subsampled_A")

# -----------------------------
# Visualize reconstruction
# -----------------------------
def plot_reconstruction_slices(recon, title="Reconstruction"):
    axial_slice = np.rot90(recon[0])  # rotate for display
    plt.figure(figsize=(6,5))
    plt.imshow(axial_slice, cmap='gray')
    plt.title(title)
    plt.colorbar(label="μ [mm^-1]")
    plt.show()

plot_reconstruction_slices(recon, "Reconstruction from Subsampled A-matrix")
