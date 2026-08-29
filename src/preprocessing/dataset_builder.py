import pandas as pd
import numpy as np
from src.preprocessing.mnist_loader import run_mnist_radon_preprocessing
from src.preprocessing.masking import generate_poisson_subsampled_mask

def build_longitudinal_mnist_dataframe(config: dict) -> pd.DataFrame:
    """
        Runs Radon preprocessing and subsampling, then formats observed data into 
        a longitudinal DataFrame compatible with the axSpA pipeline.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - n_samples (int)
            - n_angles (int)
            - random_state (int)
            - data_dir (str)
            - lam (float, optional)
            - sampling_mode (str, optional)
            
        Returns
        -------
        df : pd.DataFrame
            Long-format DataFrame with columns:
            ['COMPAS_UOB_ID', 'angle', 'response_type', 'response_value']
    """
    # 1. Run forward physics pipeline
    prep_data = run_mnist_radon_preprocessing(config=config)
    
    sinograms_flat = prep_data["tensors"]["data_proj"]
    theta = prep_data["radon_metadata"]["theta"]
    labels = prep_data["ground_truth"]["labels"]
    
    n_samples = config["n_samples"]
    n_angles = len(theta)
    seed = config.get("random_state", 42)
    lam = config.get("lam", 20.0)
    sampling_mode = config.get("sampling_mode", "strided")
    
    # 2. Compute 1D bulk projections: shape (n_samples, n_angles)
    # Reshape (N, n_detectors * n_angles) -> sum over detectors
    n_detectors = sinograms_flat.shape[1] // n_angles
    bulk_sinograms = sinograms_flat.reshape(n_samples, n_detectors, n_angles).sum(axis=1)
    
    # 3. Generate observation mask R: shape (n_samples, n_angles)
    mask = generate_poisson_subsampled_mask(
        n_samples=n_samples,
        n_features=n_angles,
        lam=lam,
        sampling_mode=sampling_mode,
        random_state=seed,
    )
    
    # Construct tidy long-format records
    records = []
    
    for i in range(n_samples):
        obs_indices = np.where(mask[i])[0]  # Indices of observed angles

        # theta is naturally sorted, so iterating through obs_indices
        # guarantees strict non-decreasing order for 'angle'
        for idx in obs_indices:
            records.append({
                "COMPAS_UOB_ID": i,
                "angle": theta[idx],
                "response_type": "mnist",
                "response_value": bulk_sinograms[i, idx],
                "label": labels[i],
            })
            
    df = pd.DataFrame(records)
    df.attrs["active_time_column"] = "angle"
    df.attrs["active_responses"] = ['mnist']
    
    # Ensure correct data types
    df = df.astype({
        "COMPAS_UOB_ID": int,
        "response_type": str,
        "angle": float,
        "response_value": float,
    })
    
    return df