import numpy as np
import pandas as pd

def compute_time_mesh(
    df: pd.DataFrame,
    step_size: float = 1/180,
) -> np.ndarray:
    """Based on the time variable and step size, computes the time mesh for
    clustering.

    Parameters
    ----------
        df : pd.DataFrame
            Long-format DataFrame with time_column containing the time samples
        step_size : float
            Describes the number of points in the time mesh that model the resolution

    Returns
    -------
        np.ndarray
            A 1D array representing the uniform time mesh.
    """
    time_column = df.attrs.get("active_time_column")
    if not time_column:
        raise KeyError(f"DataFrame has no 'active_time_column' set in df.attrs")
    
    # We have already extracted the DataFrame based on features (female non-smoker on HCD, for example)
    # However, we should still remove rows where the time_column value is potentially NaN
    valid_times = df[time_column].dropna()
    if valid_times.empty:
        raise ValueError(f"The tracking column '{time_column}' contains no valid numeric entries.")
    
    min_time = int(np.floor(float(valid_times.min())))
    max_time = int(np.ceil(float(valid_times.max())))
    num_steps = int(np.round((max_time - min_time) / step_size)) + 1
    
    return np.linspace(min_time, max_time, num=num_steps, endpoint=True)
    

def prepare_clustering_arrays(
    df: pd.DataFrame,
    time_mesh: np.ndarray,
    id_key: str = "COMPAS_UOB_ID"
) -> dict[str, np.ndarray | int]:
    """Converts a long-format clinical DataFrame into clean, uniform NumPy tensors
    for the MissingDataKMeans clustering kernels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format clinical DataFrame. Expects df.attrs["active_time_column"]
    time_mesh : np.ndarray pr jnp.ndarray
        1D array representing the uniform discretised time grid of length M
    id_key : str, optional
        Name of column identifying the patient IDs. COMPAS_UOB_ID is the axSpA case
        
    Returns
    -------
    In the comments, M is the total mesh size, N is the number of patients, and F is the 
    number of features (responses).
    
    dict[str, jnp.darray | np.ndarray]
        - 'data_proj': NumPy array of shape (N, F * M) representing S^T @ x
        - 'proj_sq_diag': NumPy array of shape (N, F * M), where each row is the diag of the
           patient-specific S^T @ S (S = Sampling matrix)
    """
    # 1. Enforce Pipeline Contract via Metadata
    target_time_col = df.attrs.get("active_time_column")
    target_responses = df.attrs.get("active_responses")

    if not target_time_col or not target_responses:
        raise KeyError("DataFrame must be processed by extract_task_subset first (missing df.attrs).")

    # 2. Extract valid Patient IDs *before* response filtering
    #    Drop rows where the ID itself or the time column is NaN
    valid_id_df = df.dropna(subset=[id_key, target_time_col])
    
    if valid_id_df.empty:
        raise ValueError("No valid rows remaining in DataFrame after dropping NaNs.")

    ids = valid_id_df[id_key].unique()
    n_patients = len(ids)
    patient_to_idx = {pid: idx for idx, pid in enumerate(ids)}

    # 3. Filter for active responses and drop missing values
    clean_df = df[df["response_type"].isin(target_responses)].copy()
    clean_df = clean_df.dropna(subset=[id_key, target_time_col, "response_value"])

    time_mesh_np = np.asarray(time_mesh, dtype=np.float32)
    n_mesh = len(time_mesh_np)
    n_responses = len(target_responses)
    total_mesh_dim = n_responses * n_mesh

    # Allocations (Zeros for all N patients)
    data_proj_np = np.zeros((n_patients, total_mesh_dim), dtype=np.float32)
    proj_sq_diag_np = np.zeros((n_patients, total_mesh_dim), dtype=np.float32)
    data_vec_sq_np = np.zeros(n_patients, dtype=np.float32)

    # 4. Populate tensors only if matching active observations exist
    if not clean_df.empty:
        response_to_idx = {resp: idx for idx, resp in enumerate(target_responses)}

        # Midpoint binning
        boundaries = (time_mesh_np[:-1] + time_mesh_np[1:]) / 2.0 if n_mesh > 1 else np.array([], dtype=np.float32)
        times = clean_df[target_time_col].to_numpy(dtype=np.float32)
        mesh_indices = np.searchsorted(boundaries, times).astype(np.int64)

        # Flat indexing with explicit integer types
        response_indices = clean_df["response_type"].map(response_to_idx).to_numpy(dtype=np.int64)
        global_mesh_indices = (response_indices * n_mesh) + mesh_indices
        p_indices = clean_df[id_key].map(patient_to_idx).to_numpy(dtype=np.int64)
        values = clean_df["response_value"].to_numpy(dtype=np.float32)

        # Accumulate observed data
        np.add.at(data_proj_np, (p_indices, global_mesh_indices), values)
        np.add.at(proj_sq_diag_np, (p_indices, global_mesh_indices), 1.0)
        np.add.at(data_vec_sq_np, p_indices, values ** 2)

    return {
        "data_proj": data_proj_np,         # d_n = S_n^T y_n (N, F * M)
        "proj_sq_diag": proj_sq_diag_np,   # diag(S_n^T S_n) (N, F * M)
        "data_vec_sq": data_vec_sq_np,     # ||y_n||^2 
        "n_responses": n_responses,
        "ids": ids
    }

# Alias for backward compatibility
prepare_clustering_tensors = prepare_clustering_arrays