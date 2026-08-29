import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_pipeline_plots(
    model, 
    prep_data, 
    output_dir: Path, 
    n_sample_trajectories: int = 40
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    df = prep_data.task_df
    time_mesh = prep_data.time_mesh
    
    # Check common attribute names for MissingDataKMeans
    centroids = getattr(model, "mu", getattr(model, "cluster_centers_", None))
    cluster_labels = getattr(model, "labels_", getattr(model, "assignments_", None))
    
    # Default to 10 colors for trajectory coloring
    K = centroids.shape[0] if centroids is not None else (len(np.unique(cluster_labels)) if cluster_labels is not None else 10)
    colors = plt.cm.tab10(np.linspace(0, 1, min(K, 10)))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # --- Subplot 1: Centroid Profiles ---
    if centroids is not None:
        for k in range(centroids.shape[0]):
            axes[0].plot(
                time_mesh, 
                centroids[k], 
                label=f"Cluster {k}", 
                lw=2, 
                color=colors[k % 10]
            )
        axes[0].set_title(r"Learned Cluster Centroids $\mu_k(\theta)$")
        axes[0].set_xlabel(r"Projection Angle $\theta^\circ$")
        axes[0].set_ylabel("Summed Intensity")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        axes[0].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[0].text(0.5, 0.5, "Centroids (model.mu) Not Found", ha="center", va="center")
        axes[0].set_title("Learned Cluster Centroids")

    # --- Subplot 2: Sparse Observations Overlaid ---
    unique_ids = df["COMPAS_UOB_ID"].unique()
    sampled_ids = np.random.choice(
        unique_ids, 
        size=min(len(unique_ids), n_sample_trajectories), 
        replace=False
    )
    
    for uid in sampled_ids:
        sub_df = df[df["COMPAS_UOB_ID"] == uid]
        k_assigned = cluster_labels[uid] if cluster_labels is not None else 0
        
        axes[1].plot(
            sub_df["angle"],
            sub_df["response_value"],
            "o-",
            alpha=0.5,
            markersize=3,
            color=colors[k_assigned % 10]
        )
        
    axes[1].set_title(f"Sparse Patient Trajectories (Subsample N={len(sampled_ids)})")
    axes[1].set_xlabel(r"Projection Angle $\theta^\circ$")
    axes[1].set_ylabel("Summed Intensity")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    
    save_path = output_dir / "cluster_trajectories_diagnostic.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()