# %%

import os
import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing.mnist_loader import run_mnist_radon_preprocessing
from src.preprocessing.masking import generate_poisson_subsampled_mask

SEED = 42
n_samples = 4
n_angles = 360
n_detectors = 40

# 1. Pipeline configuration
config = {
    "n_samples": n_samples,
    "n_angles": n_angles,
    "data_dir": "./data",
    "random_state": SEED,
}

# 2. Run preprocessing
print("Running preprocessing stage...")
prep_data = run_mnist_radon_preprocessing(config)

# 3. Extract sample data
images = prep_data["ground_truth"]["images"]
labels = prep_data["ground_truth"]["labels"]
sinograms_flat = prep_data["tensors"]["data_proj"]
theta = prep_data["radon_metadata"]["theta"]

# 4. Generate sparse mask for all samples once
mask = generate_poisson_subsampled_mask(
    n_samples=n_samples,
    n_features=n_angles,
    sampling_mode='strided',
    random_state=SEED,
)

# 5. Create n_samples x 4 subplot grid
fig, axes = plt.subplots(
    nrows=n_samples, 
    ncols=4, 
    figsize=(17, 3.5 * n_samples),
    squeeze=False  # Guarantees 2D array shape (n_samples, 4)
)

for idx in range(n_samples):
    img_2d = images[idx].reshape(28, 28)
    sinogram_2d = sinograms_flat[idx].reshape(n_detectors, n_angles)
    bulk_sinogram = sinogram_2d.sum(axis=0)
    
    nth_mask = mask[idx]
    sampled_bulk_sinogram = bulk_sinogram[nth_mask]
    n_observed = nth_mask.sum()

    # Column 1: Ground Truth Digit
    im0 = axes[idx, 0].imshow(np.flipud(img_2d), cmap="gray", origin="lower")
    axes[idx, 0].set_title(f"Sample {idx} (Label: {labels[idx]})")
    axes[idx, 0].set_xlabel("x (pixels)")
    axes[idx, 0].set_ylabel("y (pixels)")
    plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)

    # Column 2: Full 2D Sinogram
    im1 = axes[idx, 1].imshow(
        sinogram_2d, 
        cmap="gray", 
        aspect="auto", 
        extent=[theta[0], theta[-1], 0, n_detectors],
        origin="lower"
    )
    axes[idx, 1].set_title(f"Radon Sinogram ({n_angles} Angles)")
    axes[idx, 1].set_xlabel(r"Projection Angle $\theta$ (deg)")
    axes[idx, 1].set_ylabel("Detector Bin")
    plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

    # Column 3: Full 1D Bulk Signal
    axes[idx, 2].plot(theta, bulk_sinogram, lw=1, marker='o', ms=2)
    axes[idx, 2].set_title("Full 1D Bulk Sinogram")
    axes[idx, 2].set_xlabel(r"Projection Angle $\theta$ (deg)")
    axes[idx, 2].set_ylabel("Summed Intensity")

    # Column 4: Sparse-Sampled Bulk Input
    axes[idx, 3].plot(theta, bulk_sinogram, lw=1, marker='o', ms=2, alpha=0.1, label="Unobserved")
    axes[idx, 3].plot(theta[nth_mask], sampled_bulk_sinogram, lw=0, marker='o', ms=3, color='red', label="Observed")
    axes[idx, 3].set_title(f"Sparse Input ({n_observed}/{n_angles} Views)")
    axes[idx, 3].set_xlabel(r"Projection Angle $\theta$ (deg)")
    axes[idx, 3].set_ylabel("Summed Intensity")

plt.tight_layout()

# 6. Automatic Save to Root Directory
output_path = "proof_of_concept_sparse_radon.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Proof of concept plot saved successfully to: {os.path.abspath(output_path)}")

plt.show()