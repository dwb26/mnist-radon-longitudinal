# src/preprocessing/mnist_loader

import numpy as np
# from scipy import sparse
from skimage.transform import radon
from sklearn.datasets import fetch_openml
import logging

logger = logging.getLogger(__name__)

def load_mnist_numpy(data_dir: str = "./data") -> tuple[np.ndarray, np.ndarray, tuple]:
    """Loads raw MNIST images directly into NumPy arrays using scikit-learn."""
    logger.info("Fetching MNIST dataset via OpenML...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, data_home=data_dir)
    
    images_flat = X.astype(np.float32) / 255.0
    labels = y.astype(int)
    image_shape = (28, 28)
    
    return images_flat, labels, image_shape

def compute_radon_projections(
    images_flat: np.ndarray, 
    image_shape: tuple, 
    theta: np.ndarray,
    ) -> np.ndarray:
    """
    Computes parallel-beam Radon transform sinograms for a batch of 2D images.

    Geometry & Forward Operator:
        Treats each image as an in-plane 2D cross-sectional slice. For each 
        angle in `theta`, parallel X-ray paths pass across the image plane 
        and integrate pixel intensities onto a 1D detector array with length 
        equal to max(image_shape) (28 bins for MNIST).

        Stacking the 1D projection profiles across all angles forms a 2D sinogram 
        matrix of shape (n_detectors, n_angles) = (28, len(theta)). This matrix is 
        flattened into a 1D measurement vector of length 28 * len(theta).

    Parameters:
        images_flat: np.ndarray of shape (N, 784)
            Batch of flattened 2D image vectors (e.g., 28x28 MNIST digits).
        image_shape: tuple (height, width)
            Spatial dimensions to reshape vectors back into 2D grids (typically (28, 28)).
        theta: np.ndarray
            1D array of projection angles in degrees (e.g., np.linspace(0., 180., 40, endpoint=False)).

    Returns:
        sinograms_flat: np.ndarray of shape (N, n_detectors * len(theta))
            Batch of vectorized measurement sinograms (y = Ax). For 40 angles, 
            each row yields a vector of length 28 * 40 = 1120.
    """
    N = images_flat.shape[0]
    sinograms = []
    
    logger.info(f"Computing Radon projections for {N} images across {len(theta)} angles...")
    for i in range(N):
        img = images_flat[i].reshape(image_shape)
        sinogram = radon(img, theta=theta, circle=False)
        sinograms.append(sinogram.ravel())
        
    return np.array(sinograms, dtype=np.float32)

def run_mnist_radon_preprocessing(config: dict) -> dict:
    """Stage 1: Preprocessing outputting pure NumPy/SciPy objects."""
    n_samples = config.get("n_samples", 1000)
    n_angles = config.get("n_angles", 40)
    data_dir = config.get("data_dir", "./data")
    
    # 1. Load data as NumPy arrays
    images_flat, labels, image_shape = load_mnist_numpy(data_dir=data_dir)
    
    if n_samples and n_samples < len(images_flat):
        images_flat = images_flat[:n_samples]
        labels = labels[:n_samples]
        
    # 2. Compute Radon projections
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinograms_flat = compute_radon_projections(
        images_flat=images_flat,
        image_shape=image_shape,
        theta=theta,
        )
    
    n_responses = sinograms_flat.shape[1]
    # print(f"sinograms_flat.shape = {sinograms_flat.shape}")
    
    # 3. Construct payload using NumPy arrays/ SciPy CSR matrices
    # (need to convert dense array to sparse array if solver expects sparse matrices)
    data_proj = sinograms_flat
    
    prep_data = {
        "tensors": {
            "data_proj": data_proj,     # NumPy array or scipy.sparse.csr_matrix(data_proj)
            "data_vec_sq": np.sum(sinograms_flat ** 2, axis=1),
            "proj_sq_diag": np.ones((images_flat.shape[0], n_responses), dtype=np.float32),
            "n_responses": n_responses,
        },
        "ground_truth": {
            "images": images_flat,
            "labels": labels,
            "image_shape": image_shape
        },
        "radon_metadata": {
            "theta": theta,
            "n_angles": n_angles,
        }
    }
    
    logger.info("Preprocesing complete. Data projection shape %s", sinograms_flat.shape)
    return prep_data