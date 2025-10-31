#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:05:28 2025

@author: 1dh
"""
import pydicom
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from par_matrix import SparseAMat_2D
import qggmrf
import os
torch.set_default_dtype(torch.float32)

# %% set up problem
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %% xradia
Nz = 1
Ny_full, Nx_full = 50, 746
N_det = 746  # num detector channels in sino
c_rot = 23 # center of rotation offset in detectors
del_p = 0.00977 # pixel width
del_t = 0.00977
ds_factor = 1
z0 = 0
Ncols = Ny_full * Nx_full // ds_factor ** 2

# load angles
del_theta = 1 # degrees
angles = torch.linspace(-90, 90-del_theta, steps=180//del_theta, )*np.pi/180
N_views = len(angles)

# --- LOAD FULL VOLUME ---
import glob

dicom_folder = r"C:\Users\dqz\Downloads\Al 176 (1)\20240711 Battery Foil-Tab Weld Al 176 16bit DICONDE 1 Slices"
dicom_files = sorted(glob.glob(os.path.join(dicom_folder, "*.dcm")))

# Read all slices
slices = [pydicom.dcmread(f).pixel_array.astype(np.float32) for f in dicom_files]
volume = np.stack(slices, axis=0)  # shape (Nz, Ny, Nx)

Nz, Ny_full, Nx_full = volume.shape
print(f"Loaded DICOM volume shape: {volume.shape}")

# --- SELECT A CORONAL SLICE ---
# Coronal slice = along y-axis → fix y index, vary z and x
y_idx = Ny_full // 2  # pick middle coronal slice, or adjust as needed
coronal_slice = volume[:, y_idx, :]  # shape (Nz, Nx)

# For your reconstruction setup, make it look like (1, Ny, Nx)
phantom = coronal_slice[np.newaxis, :, :]
print("Coronal phantom shape:", phantom.shape)

# ✅ FIX: Reset dimensions to match the coronal phantom
Nz, Ny, Nx = phantom.shape
print(f"Updated dimensions -> Nz={Nz}, Ny={Ny}, Nx={Nx}")

# --- OPTIONAL: visualize coronal slice ---
vmax = np.percentile(phantom, 99.7)
plt.imshow(phantom[0], vmin=0, vmax=vmax, origin='upper')
plt.colorbar()
plt.title(f'Coronal Slice at y={y_idx}')
plt.show()

# --- SCALING STEP ---
P_air = 6415
P_Al = 25238
mu_Al = 0.075  # mm^-1
slope = mu_Al / (P_Al - P_air)
intercept = -slope * P_air
phantom[0] = phantom[0] * slope + intercept
phantom[0] = np.clip(phantom[0], 0, None)

x_gt = phantom[:, ::ds_factor, ::ds_factor]
_, Ny, Nx = x_gt.shape
del_p = del_p * ds_factor

# Optional: check scaled phantom
plt.imshow(phantom[0], vmax=vmax, vmin=0)
plt.colorbar()
plt.title('Scaled Coronal Slice')
plt.show()


# %% qggmrf parameters
p = 1.2
q = 2.0
T = 1.0
SigmaX = 5
SigmaY = 1.0
b_nearest = 1.0
b_diag = 0.707
b_interslice = 1.0

# %% make A_matrix projector
subsample_factor = 9
angles_sub = angles[::subsample_factor]  # pick every 9th angle
N_views_sub = len(angles_sub)

tstart = time.time()
projector = SparseAMat_2D(
    Nz=Nz,
    Ny=Ny,
    Nx=Nx,
    angles=angles_sub,
    N_det=N_det,
    c_rot=c_rot,
    del_p=del_p,
    del_t=del_t,
    device=device,
)
projector.A_mat = projector.A_mat.to(dtype=torch.float32)
tstop = time.time()
print('Time taken to compute A matrix = {:.2f} seconds'.format(tstop-tstart))

# %% generate measurement data from phantom
x_gt_tensor = torch.from_numpy(x_gt).to(dtype=torch.float32, device=device)
y = projector(x_gt_tensor)  # shape: [Nz, N_views, N_det]

# Weights
w = torch.exp(y)
w_sqrt = torch.sqrt(w)

# %% Save projector and sinogram
output_dir = "xradia_projectors"
os.makedirs(output_dir, exist_ok=True)

sino_filename = os.path.join(output_dir, f"simulated_sinogram_{Nz:02d}slice.npy")
np.save(sino_filename, y.cpu().detach().numpy())
print(f"Simulated sinogram saved to {sino_filename}")

proj_filename_pt = os.path.join(output_dir, f"projector_{Ncols:05d}col.pt")
torch.save(projector.A_mat.cpu(), proj_filename_pt)
print(f"Sparse projector saved as .pt to {proj_filename_pt}")

# %% Optional visualization of sinogram and weights
y_cpu = y[0].cpu().detach().numpy()
w_cpu = w[0].cpu().detach().numpy()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(y_cpu, aspect='auto')
plt.colorbar()
plt.title('Simulated sinogram (y)')
plt.subplot(1,2,2)
plt.imshow(w_cpu, aspect='auto')
plt.colorbar()
plt.title('Weights (w = exp(y))')
plt.show()

# %% Initialize reconstruction variable
xhat = torch.randn((Nz, Ny, Nx), dtype=torch.float32, requires_grad=True)
plt.imshow(xhat.detach().numpy()[0])
plt.colorbar()
plt.title('Initial x')
plt.show()

# %% Define optimizer
lr_init = 5e+0
optimizer = torch.optim.Adam([xhat], lr=lr_init)
num_iterations = int(10e2)
anneal_epochs = int(num_iterations // 1.5)
scheduler = torch.optim.swa_utils.SWALR(
    optimizer, anneal_strategy="linear", anneal_epochs=anneal_epochs, swa_lr=1e-1
)
lrs = []

# %% Move data to device
projector = projector.to(device)
y = y.to(device)
xhat = xhat.to(device)

# %% Optimization loop
nloglike_vec = []
nlogprior_vec = []
nlogloss_vec = []

tstart = time.time()
for i in range(num_iterations):
    optimizer.zero_grad()
    Ax = projector(xhat).to(device)
    scaled_err = (y - Ax) * w_sqrt
    nloglike = torch.norm(scaled_err, p=2) ** 2
    nlogprior = qggmrf.qggmrf_loss(
        xhat, p=p, q=q, T=T, SigmaX=SigmaX, SigmaY=SigmaY,
        b_nearest=b_nearest, b_diag=b_diag, b_interslice=b_interslice
    )
   # loss = nloglike + nlogprior
    loss = nloglike  # only data likelihood, no prior

    loss.backward()
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()
    with torch.no_grad():
        xhat.clamp_min_(0)
    nloglike_vec.append(nloglike.item())
    nlogprior_vec.append(nlogprior.item())
    nlogloss_vec.append(loss.item())
    if i % 10 == 0:
        print(f"iter {i:04d}, Loss: {loss.item():.4f}, Likelihood: {nloglike.item():.4f}, Prior: {nlogprior.item():.4f}, lr: {lrs[-1]:.4f}")

print("Reconstruction time = {:.4f}s".format(time.time() - tstart))

# %% Save reconstruction
recon_filename = os.path.join(output_dir, f"reconstruction_{Nz:02d}slice.npy")
np.save(recon_filename, xhat.cpu().detach().numpy())
print(f"Reconstruction saved to {recon_filename}")

# %% Plot losses
plt.plot(range(num_iterations), lrs)
plt.show()
plt.semilogy(nloglike_vec[:150], label='likelihood')
plt.semilogy(nlogprior_vec[:150], label='prior')
plt.semilogy(nlogloss_vec[:150], label='loss')
plt.grid()
plt.legend()
plt.show()

# %% Plot densities
plt.imshow(y[0].cpu().detach().numpy())
plt.colorbar()
plt.title('Measurement')
plt.show()

plt.imshow(Ax[0].cpu().detach().numpy())
plt.colorbar()
plt.title('Projection of xhat')
plt.show()

plt.imshow(xhat[0].cpu().detach().numpy())
plt.colorbar()
plt.title('Reconstruction')
plt.show()

plt.imshow(xhat[0].cpu().detach().numpy(), vmin=0, vmax=31.544980907440188)
plt.colorbar()
plt.title('Reconstruction, clipped')
plt.show()
