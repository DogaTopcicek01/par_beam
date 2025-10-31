import numpy as np
import torch
import matplotlib.pyplot as plt
from demo import SinoInfo, ImgInfo, pix_prof_comp, build_A_all_columns, icd_reconstruction_qggmrf

# ==========================================================
# 1. Sinogram / Scanner Parameters  (System Geometry)
# ==========================================================
sino_params = {
    "sino_path": r"C:\Users\dqz\Downloads\shepp\shepp\input\shepp.sino",
    "wght_path": r"C:\Users\dqz\Downloads\shepp\shepp\input\shepp.wght",
    "Ntheta": 288,
    "Nt": 512, #
    "delta_t": 0.9765625,
    "theta0": 0.0,
    "center_offset": 6.0,
}

# ==========================================================
# 2. Image Parameters (Reconstruction Grid)
# ==========================================================
img_params = {
    "Nx": 128,
    "Ny": 128,
}

# ==========================================================
# 3. Prior Model Parameters (q-GGMRF)
# ==========================================================
prior_params = {
    "p": 1.2,
    "q": 2.0,
    "T": 0.00038,
    "sigma_x": 0.8,
    "b_vert_horz": 0.14,
    "b_diag": 0.11,
}

# ==========================================================
# 4. Load Sinogram and Weights
# ==========================================================
print("--> Loading sinogram and weights...")
Ntheta = sino_params["Ntheta"]
Nt = sino_params["Nt"]

y_np = np.fromfile(sino_params["sino_path"], dtype=np.float32).reshape(Ntheta * Nt)
R_np = np.fromfile(sino_params["wght_path"], dtype=np.float32).reshape(Ntheta * Nt)

y = torch.from_numpy(y_np).float()
R_diag = torch.from_numpy(R_np).float()

print(f"Loaded sinogram shape: {Ntheta}×{Nt}, total={y.numel()}")

# ==========================================================
# 5. Initialize Geometry Classes
# ==========================================================
lfov = Nt * sino_params["delta_t"]

# Convert center-of-rotation offset from channels to physical distance
channel_offset = sino_params["center_offset"]      # in detector channels
delta_t = sino_params["delta_t"]                  # channel width in mm
detector_offset = channel_offset * delta_t        # convert to mm

# Initialize geometry
sino = SinoInfo(
    Nt=sino_params["Nt"],
    Ntheta=sino_params["Ntheta"],
    delta_t=delta_t,
    theta0=sino_params["theta0"],
    detector_offset=detector_offset               # pass in mm
)

img = ImgInfo(
    Nx=img_params["Nx"],
    Ny=img_params["Ny"],
    lfov=lfov
)

# ==========================================================
# 6. Compute Pixel Profiles and A-Matrix
# ==========================================================
print("--> Computing pixel profiles...")
profiles = pix_prof_comp(sino, img)

print("--> Building system matrix (this may take a few minutes)...")
A_cols = build_A_all_columns(sino, img, profiles)

# ==========================================================
# 7. Run Reconstruction (ICD)
# ==========================================================
print("--> Starting ICD iterations...")
#x_init = torch.zeros((img.N,), dtype=torch.float32)
x_init = torch.full((img.N,), 0.0202527, dtype=torch.float32, device='cpu')


x_rec, cost_history = icd_reconstruction_qggmrf(

    y=y,
    sino=sino,
    img=img,
    A_cols=A_cols,
    initial_x=x_init,
    R_diag=R_diag,
    max_iters=10,
    p=prior_params["p"],
    q=prior_params["q"],
    sigma_x=prior_params["sigma_x"],
    b_s_r=prior_params["b_vert_horz"]
)

# ==========================================================
# 8. Save Results
# ==========================================================
print("--> Saving results...")
out_path = r"C:\Users\dqz\Downloads\shepp\shepp\output\shepp_mbir.rec"
x_rec.cpu().numpy().astype(np.float32).tofile(out_path)

np.savetxt(
    r"C:\Users\dqz\Downloads\shepp\shepp\output\shepp_mbir_cost.txt",
    np.array(cost_history)
)

print("✅ Reconstruction complete! Output written to:", out_path)

# ==========================================================
# 9. Visualization
# ==========================================================
print("--> Displaying results...")
x_img = x_rec.cpu().numpy().reshape(img_params["Ny"], img_params["Nx"])

plt.figure(figsize=(12, 5))

# Reconstructed image
plt.subplot(1, 2, 1)
plt.imshow(x_img, cmap='gray')
plt.title("Reconstructed Shepp-Logan Phantom")
plt.axis('off')

# Cost history
plt.subplot(1, 2, 2)
plt.plot(cost_history, 'o-', linewidth=2)
plt.title("ICD Cost Function Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost")

plt.tight_layout()
plt.show()
