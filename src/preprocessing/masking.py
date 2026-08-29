from typing import Optional, Union
import numpy as np

def generate_poisson_subsampled_mask(
    n_samples: int,
    n_features: int = 360,
    lam: float = 20.0,
    sampling_mode: str = "block",
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """Generates a binary observation mask R of shape (n_samples, n_features)

    Parameters
    ----------
    n_samples : int
        Number of samples (N).
    n_features : int
        Total feature dimension (D), e.g., 360 angles.
    lam : float
        Mean parameter for Poisson count distribution (expected number of observed angles).
    sampling_mode : str
        Sampling strategy: 'block' (missing wedge), 'strided', or 'uniform'.

    Returns:
        np.ndarray: shape (n_samples, n_features)
    """
    # Initialise modern NumPy Random Number Generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    
    mask = np.zeros((n_samples, n_features), dtype=bool)
    
    for i in range(n_samples):
        # Draw count of observed angles, clipped to [1, n_features]
        n_i = np.clip(rng.poisson(lam), 1, n_features)
        
        if sampling_mode == "block":
            max_start = n_features - n_i
            start_idx = rng.integers(0, max_start + 1) if max_start > 0 else 0
            observed_indices = np.arange(start_idx, start_idx + n_i)
            
        elif sampling_mode == "strided":
            base_indices = np.linspace(0, n_features - 1, n_i, dtype=int)
            jitter = rng.integers(-1, 2, size=n_i)
            observed_indices = np.clip(base_indices + jitter, 0, n_features - 1)
            observed_indices = np.unique(observed_indices)
            
        elif sampling_mode == "uniform":
            observed_indices = rng.choice(n_features, size=n_i, replace=False)
            
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        
        mask[i, observed_indices] = True
        
    return mask       
    