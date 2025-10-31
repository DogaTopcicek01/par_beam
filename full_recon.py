import os
import numpy as np
import torch
import time
from functools import partial
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== SYSTEM CONFIGURATION ====================
device = 'cpu'
print(f"Using device: {device}")


# ==================== SHARED CORE FUNCTIONS ====================
def sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles, output_device, worker):
    """
    Batch the views (angles) and voxels/indices, send batches to the GPU to project, and collect the results.
    """
    max_views = 30
    max_pixels = 1000
    num_to_exclude = 0

    indices = indices[:len(indices)-num_to_exclude]
    angles = angles.to(worker)

    # Batch the views and pixels
    num_views = len(angles)
    view_batch_indices = torch.arange(start=0, end=num_views, step=max_views)
    view_batch_indices = torch.concatenate([view_batch_indices, num_views * torch.ones(1, dtype=torch.int32)])

    num_pixels = len(indices)
    pixel_batch_indices = torch.arange(start=0, end=num_pixels, step=max_pixels)
    pixel_batch_indices = torch.concatenate([pixel_batch_indices, num_pixels * torch.ones(1, dtype=torch.int32)])

    # Create the output sinogram
    sinogram = []

    # Loop over the view batches
    for j, view_index_start in enumerate(view_batch_indices[:-1]):
        # Send a batch of views to worker
        view_index_end = view_batch_indices[j+1]
        cur_view_batch = torch.zeros([view_index_end-view_index_start, sinogram_shape[1], sinogram_shape[2]],
                                     device=worker)
        cur_view_params_batch = angles[view_index_start:view_index_end]


        # Loop over pixel batches
        for k, pixel_index_start in enumerate(pixel_batch_indices[:-1]):
            # Send a batch of pixels to worker
            pixel_index_end = pixel_batch_indices[k+1]
            cur_voxel_batch = voxel_values[pixel_index_start:pixel_index_end].to(worker)
            cur_index_batch = indices[pixel_index_start:pixel_index_end].to(worker)

            def forward_project_pixel_batch_local(view, angle):
                # Add the forward projection to the given existing view
                return forward_project_pixel_batch_to_one_view(cur_voxel_batch, cur_index_batch, angle, view,
                                                               sinogram_shape, recon_shape)

            view_map = torch.vmap(forward_project_pixel_batch_local)
            cur_view_batch = view_map(cur_view_batch, cur_view_params_batch)

        sinogram.append(cur_view_batch.to(output_device))
    sinogram = torch.concatenate(sinogram)
    return sinogram

@torch.compile
def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, sinogram_view,
                                            sinogram_shape, recon_shape):
    """
    Apply a parallel beam transformation to a set of voxel cylinders. These cylinders are assumed to have
    slices aligned with detector rows, so that a parallel beam maps a cylinder slice to a detector row.
    This function returns the resulting sinogram view.

    """
    # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
    # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
    num_views, num_det_rows, num_det_channels = sinogram_shape
    psf_radius = 1
    delta_voxel = 0.00977140380691

    # Get the data needed for horizontal projection
    n_p, n_p_center, W_p_c, cos_alpha_p_xy = compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape)
    L_max = torch.clip(W_p_c, None, 1)

    # Do the projection
    for n_offset in torch.arange(start=-psf_radius, end=psf_radius+1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = torch.abs(n_p - n)
        L_p_c_n = torch.clamp((W_p_c + 1) / 2 - abs_delta_p_c_n, torch.zeros(1, device=device), L_max)

        A_chan_n = delta_voxel * L_p_c_n / cos_alpha_p_xy
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        n = torch.clip(n, 0, num_det_channels - 1)  # n to a valid range and then not add anything.
        update = A_chan_n.reshape((1, -1)) * voxel_values.T
        # Scatter add the values into the sinogram_view tensor
        indices = n.expand(num_det_rows, -1).type(torch.int64)  # Expand to match batch dim
        sinogram_view = sinogram_view.scatter_add(1, indices, update)

    return sinogram_view


def compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape):
    """
    Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.
    """

    cosine = torch.cos(angle)
    sine = torch.sin(angle)

    delta_voxel = 1.0 # spacing
    dvc = delta_voxel
    dvs = delta_voxel
    dvc *= cosine
    dvs *= sine

    num_views, num_det_rows, num_det_channels = sinogram_shape

    # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
    row_index, col_index = torch.unravel_index(pixel_indices, recon_shape[:-1])

    y_tilde = dvs * (row_index - (recon_shape[0] - 1) / 2.0)
    x_tilde = dvc * (col_index - (recon_shape[1] - 1) / 2.0)

    x_p = x_tilde - y_tilde

    det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

    # Calculate indices on the detector grid
    n_p = x_p + det_center_channel
    n_p_center = torch.round(n_p).type(torch.int32)

    # Compute cos alpha for row and columns
    cos_alpha_p_xy = torch.maximum(torch.abs(cosine), torch.abs(sine))

    # Compute projected voxel width along columns and rows (in fraction of detector size)
    W_p_c = cos_alpha_p_xy

    proj_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

    return proj_data

# ==================== SINOGRAM CREATION ====================
def load_dicom_chunk(dicom_folder, chunk_num, total_chunks=1, row_range=(720, 770)):
    """Load only the specific chunk of DICOM images we need."""
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_files.sort()

    # Load the first file to get the dimensions
    ds = pydicom.dcmread(dicom_files[0])
    num_rows = ds.Rows
    num_columns = ds.Columns
    num_slices = len(dicom_files)

    # Validate row range
    start_row, end_row = row_range
    if start_row < 0 or end_row > num_rows or start_row >= end_row:
        raise ValueError(f"Invalid row range {row_range}. Must be within 0-{num_rows} and start < end.")

    # Calculate chunk parameters
    total_rows = end_row - start_row
    chunk_size = total_rows // total_chunks
    chunk_start = start_row + (chunk_num - 1) * chunk_size
    chunk_end = start_row + chunk_num * chunk_size if chunk_num < total_chunks else end_row

    # Verify we're not getting empty slices
    if chunk_start >= chunk_end:
        raise ValueError(f"Calculated empty chunk: start={chunk_start}, end={chunk_end}")

    # Initialize the 3D array with only the selected chunk rows
    selected_rows = chunk_end - chunk_start
    dicom_chunk = np.zeros((num_columns, num_slices, selected_rows), dtype=np.float32)

    # Load only the needed rows from each DICOM file
    for i, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dicom_file)
        pixel_data = ds.pixel_array.T  # Transpose to get (columns, rows)

        # Verify slice dimensions
        if pixel_data.shape != (num_columns, num_rows):
            raise ValueError(f"DICOM file {dicom_file} has unexpected dimensions {pixel_data.shape}")

        # Get the specific rows we need
        chunk_slice = pixel_data[:, chunk_start:chunk_end]

        # Verify the slice matches our expected dimensions
        if chunk_slice.shape != (num_columns, selected_rows):
            raise ValueError(
                f"Slice shape mismatch: expected ({num_columns}, {selected_rows}), got {chunk_slice.shape}")

        dicom_chunk[:, i, :] = chunk_slice

    print(f"Loaded chunk {chunk_num} (rows {chunk_start}-{chunk_end}) with shape {dicom_chunk.shape}")
    return dicom_chunk

def linear_scaling(dicom_chunk):
    # Hardcoded values
    P_air = 6415
    P_Al = 25238
    mu_Al = 0.075  # mm^-1

    slope = mu_Al / (P_Al - P_air)
    intercept = -slope * P_air
    mu_chunk = slope * dicom_chunk + intercept

    # Verify the mapping
    mean_air_scaled = mu_chunk[dicom_chunk == 6000].mean()
    mean_al_scaled = mu_chunk[dicom_chunk == 25000].mean()

    print("After linear scaling:")
    print("  Mean air value (should be ~0):", mean_air_scaled, "mm^-1")
    print("  Mean aluminum value (should be ~0.075):", mean_al_scaled, "mm^-1")

    return mu_chunk


def set_sinogram_parameters(num_det_rows):
    """Set parameters for the sinogram with dynamic detector rows."""
    num_views = 180
    num_det_channels = 1000
    return num_views, num_det_rows, num_det_channels


def create_sinogram(dicom_folder, chunk_num, total_chunks=40, row_range=(330, 3100),
                    subsampling_factor=6, output_dir=None):
    """Create a sinogram from DICOM data with consistent angle handling."""
    # Load only the specific chunk we need

    dicom_chunk = load_dicom_chunk(dicom_folder, chunk_num, total_chunks, row_range)
    #scaling
    dicom_chunk = linear_scaling(dicom_chunk)  # now the chunk is in mm^-1


    # Set sinogram parameters for this chunk
    num_views_full, num_det_rows, num_det_channels = set_sinogram_parameters(dicom_chunk.shape[2])
    sinogram_shape_full = (num_views_full, num_det_rows, num_det_channels)

    # Full set of angles (consistent with reconstruction)
    num_views, num_det_rows, num_det_channels = set_sinogram_parameters(num_det_rows)

    start_angle = 0
    end_angle = torch.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    step_size = (end_angle - start_angle) / num_views
    angles_full = torch.linspace(start=start_angle, end=end_angle - step_size, steps=num_views)
    # Subsample angles
    angles = angles_full[::subsampling_factor]
    num_views = len(angles)
    print(f"Processing {num_views} views (every {subsampling_factor}th)")

    # Prepare reconstruction
    recon_shape = dicom_chunk.shape
    phantom = torch.tensor(dicom_chunk, dtype=torch.float32, device=device)

    # Flatten voxel indices for forward projection
    voxel_values = phantom.reshape(-1, recon_shape[2])
    indices = torch.arange(voxel_values.shape[0], device=device)

    # Forward projection
    print('\nStarting forward projection...')
    t0 = time.time()
    sinogram = sparse_forward_project(voxel_values, indices,
                                      (num_views, num_det_rows, num_det_channels),
                                      recon_shape, angles, output_device=device, worker=device)
    print('Elapsed time:', time.time() - t0)


    # Save results
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"sinogram_chunk_{chunk_num}_views_{num_views}_subsampled_{subsampling_factor}.npy"
    output_path = os.path.join(output_dir, output_filename)

    np.save(output_path, sinogram.cpu().numpy())
    print(f"\nSaved sinogram to {output_path}")

    return sinogram, angles, recon_shape



def neighbor_l1(x, G):
    diff_y = torch.abs(x[:, 1:, :] - x[:, :-1, :])  # vertical
    diff_z = torch.abs(x[:, :, 1:] - x[:, :, :-1])  # horizontal
    diff_diag1 = torch.abs(x[:, 1:, 1:] - x[:, :-1, :-1])  # diagonal \
    diff_diag2 = torch.abs(x[:, 1:, :-1] - x[:, :-1, 1:])  # diagonal /
    return G * (torch.sum(diff_z) + torch.sum(diff_y)) + (G / (2 ** 0.5)) * (
                torch.sum(diff_diag1) + torch.sum(diff_diag2))


def neighbor_l2(x, G):
    diff_y = (x[:, 1:, :] - x[:, :-1, :]) ** 2  # vertical
    diff_z = (x[:, :, 1:] - x[:, :, :-1]) ** 2  # horizontal
    diff_diag1 = (x[:, 1:, 1:] - x[:, :-1, :-1]) ** 2  # diagonal \
    diff_diag2 = (x[:, 1:, :-1] - x[:, :-1, 1:]) ** 2  # diagonal /
    return G * (torch.sum(diff_z) + torch.sum(diff_y)) + 0.5 * G * (torch.sum(diff_diag1) + torch.sum(diff_diag2))


def qggmrf_potential_tensor(delta, p, q, T, SigmaX, pow_sigmaX_p, eps=1e-10):
    abs_delta = torch.abs(delta) + eps
    temp = torch.pow(abs_delta / (T * SigmaX), (q - p))
    GGMRF_Pot = torch.pow(abs_delta, p) / (p * pow_sigmaX_p)
    return GGMRF_Pot * (temp / (1.0 + temp))


def MAPCostFunction3D(x, e, W,
                      p=2.0, q=2.0, T=1.0, SigmaX=1.0,
                      b_nearest=0.01, b_diag=0.01, b_interslice=0.01,
                      eps=1e-10, return_components=False):
    nloglike = 0.5 * torch.sum(W * (e ** 2))
    pow_sigmaX_p = SigmaX ** p
    plusx = torch.roll(x, shifts=-1, dims=2)
    plusy = torch.roll(x, shifts=-1, dims=1)
    plusz = torch.roll(x, shifts=-1, dims=0)
    delta_x = x - plusx
    delta_y = x - plusy
    delta_diag1 = x - torch.roll(x, shifts=(-1, 1), dims=(1, 2))
    delta_diag2 = x - torch.roll(x, shifts=(-1, -1), dims=(1, 2))
    delta_interslice = x - plusz
    pot_x = qggmrf_potential_tensor(delta_x, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_y = qggmrf_potential_tensor(delta_y, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_diag1 = qggmrf_potential_tensor(delta_diag1, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_diag2 = qggmrf_potential_tensor(delta_diag2, p, q, T, SigmaX, pow_sigmaX_p, eps)
    pot_interslice = qggmrf_potential_tensor(delta_interslice, p, q, T, SigmaX, pow_sigmaX_p, eps)
    nlogprior_nearest = torch.sum(pot_x) + torch.sum(pot_y)
    nlogprior_diag = torch.sum(pot_diag1) + torch.sum(pot_diag2)
    nlogprior_interslice = torch.sum(pot_interslice)
    prior = b_nearest * nlogprior_nearest + b_diag * nlogprior_diag + b_interslice * nlogprior_interslice
    total_cost = nloglike + prior
    if return_components:
        return total_cost, nloglike, nlogprior_nearest, nlogprior_diag, nlogprior_interslice
    else:
        return total_cost


def reconstruct_chunk(sinogram, angles, recon_shape, output_dir=None,
                      use_l1=False, use_l2=False, use_map=False,
                      G_l1=0.01, G_l2=0.01,
                      map_params=None,
                      num_iters=100):
    """Full reconstruction pipeline for one chunk with optional MAP prior"""
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    y = sinogram
    x = torch.zeros(recon_shape, device=device, requires_grad=True)

    # Weight Matrix Calculation (done once)
    W = torch.exp(-y)
    # === Weight Matrix and Sinogram Visualization (once before loop) ===
    row_idx = y.shape[1] // 2  # Center row index
    sino_slice = y[:, row_idx, :].cpu().numpy()  # Shape: (views, channels)
    weight_slice = W[:, row_idx, :].cpu().numpy()

    plt.figure(figsize=(12, 5))

    # 1. Channel-View Sinogram (X=views, Y=channels)
    plt.subplot(121)
    plt.imshow(sino_slice.T, cmap='gray', aspect='auto')
    plt.title(f"Channel-View Sinogram\nRow {row_idx}")
    plt.xlabel(f"View Index (0-{y.shape[0] - 1})")
    plt.ylabel(f"Detector Channel (0-{y.shape[2] - 1})")
    plt.colorbar(label='Attenuation')

    # 2. Weight Matrix (same orientation as sinogram)
    plt.subplot(122)
    plt.imshow(weight_slice.T, cmap='grey', aspect='auto')
    plt.title(f"Weight Matrix\nRow {row_idx}")
    plt.xlabel(f"View Index (0-{y.shape[0] - 1})")
    plt.ylabel(f"Detector Channel (0-{y.shape[2] - 1})")
    plt.colorbar(label='Weight Value')

    plt.tight_layout()
    plt.show()


    # Training Initialization
    losses = []
    optimizer = torch.optim.Adam([x], lr=.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.0015))

    start_time = time.time()

    for iteration in range(num_iters):
        optimizer.zero_grad()

        indices = torch.arange(recon_shape[0] * recon_shape[1], dtype=torch.long, device=device)
        Ax = sparse_forward_project(
            x.reshape(-1, recon_shape[2]), indices, y.shape, recon_shape, angles, device, device
        )
        residual = y - Ax

        # Compute Data Fidelity Loss
        loss = torch.sum( W * (residual ** 2))
        reg_info = []
        l1_term_val = 0
        l2_term_val = 0
        map_term_val = 0

        # Add L1 Regularization
        if use_l1:
            l1_term = neighbor_l1(x, G=G_l1)
            loss += l1_term
            reg_info.append(f"L1({G_l1:.0e})")
            l1_term_val = l1_term.item()

        # Add L2 Regularization
        if use_l2:
            l2_term = neighbor_l2(x, G=G_l2)
            loss += l2_term
            reg_info.append(f"L2({G_l2:.0e})")
            l2_term_val = l2_term.item()

        # Add MAP Prior
        if use_map and map_params is not None:
            map_term = MAPCostFunction3D(
                x, residual, W,
                p=map_params.get("p", 2.0),
                q=map_params.get("q", 2.0),
                T=map_params.get("T", 1.0),
                SigmaX=map_params.get("SigmaX", 1.0),
                b_nearest=map_params.get("b_nearest", 0.01),
                b_diag=map_params.get("b_diag", 0.01),
                b_interslice=map_params.get("b_interslice", 0.01),
                return_components=False
            )
            loss += map_term
            reg_info.append("MAP")
            map_term_val = map_term.item()

        losses.append(loss.item())

        # Backprop and step
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(min=0.0)
        scheduler.step()

        # Print info every 10 iterations
        if iteration % 10 == 0:
            print(f"Iter {iteration:4d} | Loss: {loss.item():.6f} | "
                  f"L1: {l1_term_val:.6f} | L2: {l2_term_val:.6f} | MAP: {map_term_val:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save reconstruction
    os.makedirs(output_dir, exist_ok=True)
    # With this:
    base_name = f"chunk_{recon_shape[0]}_{recon_shape[1]}_{recon_shape[2]}"
    reg_str = "_" + "_".join(reg_info) if reg_info else "_no_reg"

    # Add MAP parameters if available
    if use_map and map_params is not None:
        map_str = "_map"
        for key in ["p", "q", "T", "SigmaX", "b_nearest", "b_diag", "b_interslice"]:
            if key in map_params:
                map_str += f"_{key}{map_params[key]:g}"
        reg_str += map_str

    output_path = os.path.join(output_dir, f"recon_q_test{base_name}{reg_str}.npy")
    recon = x.detach().cpu().numpy()
    np.save(output_path, recon)

    print(f"\nReconstruction saved to {output_path}")
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

    return recon, losses


# ==================== COMPLETE PIPELINE ====================
def complete_ct_pipeline(dicom_folder, chunk_num, total_chunks=1, row_range=(1724, 1727),
                         subsampling_factor=6, output_dir=None,
                         use_l1=False, use_l2=False, use_map=False,
                         G_l1=0.01, G_l2=0.01, map_params=None, num_iters=100):
    """
    Complete CT pipeline from DICOM to reconstruction.
    Handles angles consistently between sinogram creation and reconstruction.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    # Step 1: Create sinogram
    print("=" * 60)
    print("STEP 1: Creating sinogram from DICOM data")
    print("=" * 60)

    sinogram, angles, recon_shape = create_sinogram(
        dicom_folder, chunk_num, total_chunks, row_range,
        subsampling_factor, output_dir
    )
    print(f"Sinogram range: min = {sinogram.min():.6f}, max = {sinogram.max():.6f} mm^-1")

    # Step 2: Reconstruct from sinogram
    print("\n" + "=" * 60)
    print("STEP 2: Reconstructing from sinogram")
    print("=" * 60)

    reconstruction, losses = reconstruct_chunk(
        sinogram= sinogram,
        angles = angles,
        recon_shape=recon_shape,
        output_dir=output_dir,
        use_l1=use_l1,
        use_l2=use_l2,
        use_map=use_map,
        G_l1=G_l1,
        G_l2=G_l2,
        map_params=map_params,
        num_iters=num_iters
    )

    print(f"Reconstruction range: min = {reconstruction.min():.6f}, max = {reconstruction.max():.6f} mm^-1")

    # Step 3: Visualize results
    print("\n" + "=" * 60)
    print("STEP 3: Visualizing results")
    print("=" * 60)

    # Step 3: Visualize results with consistent value range and colorbars
    if reconstruction is not None:
        # Extract slices
        axial_slice = reconstruction[reconstruction.shape[0] // 2]
        coronal_slice = reconstruction[:, reconstruction.shape[1] // 2, :]
        sagittal_slice = reconstruction[:, :, reconstruction.shape[2] // 2]

        # Rotate the axial slice 90 degrees
        axial_slice_rot = np.rot90(axial_slice)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(axial_slice_rot, cmap='gray')
        axs[0].set_title("Axial Slice (Rotated 90°)")
        axs[1].imshow(coronal_slice, cmap='gray')
        axs[1].set_title("Coronal Slice")
        axs[2].imshow(sagittal_slice, cmap='gray')
        axs[2].set_title("Sagittal Slice")
        plt.show()

    return reconstruction, losses


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":

    dicom_folder = r"C:\Users\dqz\Desktop\20240711 Battery Foil-Tab Weld Al 176 16bit DICONDE 1 Slices"
    chunk_to_process = 1
    output_directory = r"C:\Users\dqz\Desktop"

    # MAP parameters
    map_params = {
        "p": 1.2,
        "q": 2.0,
        "T": 0.001,
        "SigmaX": 50.0,
        "b_nearest": 1.0,
        "b_diag": 0.707,
        "b_interslice": 1.0,
    }


    reconstruction, losses = complete_ct_pipeline(
        dicom_folder=dicom_folder,
        chunk_num=chunk_to_process,
        output_dir=output_directory,
        subsampling_factor=9,
        use_l1=True,
        use_l2=False,
        use_map=False,
        G_l1=0.05,
        G_l2=0.1,
        map_params=map_params,
        num_iters=1000
    )
