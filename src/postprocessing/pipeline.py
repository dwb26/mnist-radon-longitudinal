# src/preprocessing/pipeline.py

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.clustering.sparse_trajectory_kmeans import MissingDataKMeans
from src.utils.data_mgmt import PreprocessingOutput

logger = logging.getLogger(__name__)


class ExperimentExporter:
    """Handles directory creation, postprocessing, and artifact serialization 
    for a completed clustering experiment run.
    """

    def __init__(
        self,
        model: MissingDataKMeans,
        prep_data: PreprocessingOutput,
        config: dict[str, Any],
        output_base_dir: str | Path = "data/clustering_results",
        id_key: str = "COMPAS_UOB_ID",
        sweep_results: pd.DataFrame | None = None
    ):
        self.model = model
        self.prep_data = prep_data
        self.config = config
        self.output_base_dir = Path(output_base_dir)
        self.id_key = id_key

        # Derived attributes
        self.time_col = self.config.get("time_column", "time")
        self.features_by_sheet = self.config.get("features_by_sheet", {})
        
        # Build destination directory on initialization
        self.exp_dir = self._generate_experiment_dir()
        self.sweep_results = sweep_results

    def _generate_experiment_dir(self) -> Path:
        """Generates a deterministic experiment directory path."""
        include_timestamp = self.config.get("include_timestamp", False)
        
        # Flatten and sort response feature names for consistent paths
        responses = [r for group in self.features_by_sheet.values() for r in group]
        resp_str = "_".join(sorted(responses))
        
        dir_name = f"{self.time_col}_{resp_str}_K={self.model.K}_n_init={self.model.n_init}"
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{dir_name}_{timestamp}"
            
        exp_dir = self.output_base_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _save_dataframes(self) -> pd.DataFrame:
        """Assigns cluster labels to patient trajectories and writes Parquet/CSV artifacts."""
        labels = self.model.labels_
        df = self.prep_data.train_df
        ids = self.prep_data.tensors["ids"]

        label_map = pd.DataFrame({
            self.id_key: ids,
            "cluster": labels
        })

        if "cluster" in df.columns:
            df = df.drop(columns=["cluster"])
        df = df.merge(label_map, on=self.id_key, how="left")

        # Save training data with assigned clusters
        df.to_parquet(self.exp_dir / "train_df.parquet", index=False)

        # Save distinct patient-to-cluster lookup table
        patient_clusters = df.drop_duplicates(subset=[self.id_key])[[self.id_key, "cluster"]]
        patient_clusters.to_csv(self.exp_dir / "patient_cluster_assignments.csv", index=False)

        # Save test data if available
        if self.prep_data.test_df is not None:
            test_out_path = self.exp_dir / "test_df.parquet"
            self.prep_data.test_df.to_parquet(test_out_path, index=False)
            logger.info(f"Saved held-out test cohort to {test_out_path.name}")
            
        if self.sweep_results is not None:
            clust_cfg = self.config.get("clustering", {})
            method = clust_cfg.get("compute_K", None)
            sweep_path = self.exp_dir / "model_selection"
            sweep_path.mkdir(parents=True, exist_ok=True)
            self.sweep_results.to_parquet(sweep_path / f"{method}_method_k_sweep_diagnostics.parquet", index=False)

        return df

    def _save_numeric_arrays(self) -> None:
        """Saves compressed numeric arrays (.npz)."""
        array_filepath = self.exp_dir / "arrays.npz"
        np.savez_compressed(
            array_filepath,
            centroids=self.model.centroids_,
            labels=self.model.labels_,
            ids=self.prep_data.tensors["ids"],
            time_mesh=self.prep_data.time_mesh
        )
        logger.info(f"Saved numeric arrays to {array_filepath.name}")

    def _save_metadata_and_diagnostics(self) -> None:
        """Serializes hyperparameter, dataset, and optimization diagnostics to JSON."""
        diagnostics = self.model.diagnostics_
        responses = [r for group in self.features_by_sheet.values() for r in group]

        metadata = {
            "experiment_name": str(self.exp_dir),
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "K": self.model.K,
                "n_responses": self.model.n_responses,
                "lambda_reg": self.model.lambda_reg,
                "beta": self.model.beta,
                "n_init": self.model.n_init,
                "max_iter": self.model.max_iter,
                "tol": self.model.tol,
                "init": self.model.init,
                "random_state": self.model.random_state
            },
            "dataset_metadata": {
                "time_column": self.time_col,
                "active_responses": sorted(responses),
                "n_patients": len(self.model.labels_),
                "mesh_size": len(self.prep_data.time_mesh)
            },
            "diagnostics": {
                "converged": bool(diagnostics.converged),
                "iterations_run": int(diagnostics.iterations_run),
                "final_wcss": float(diagnostics.final_wcss),
                "final_loss": float(diagnostics.final_loss)
            }
        }

        metadata_filepath = self.exp_dir / "metadata.json"
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Saved experiment metadata to {metadata_filepath.name}")

    def export(self) -> Path:
        """Executes full postprocessing pipeline and writes all experiment artifacts."""
        logger.info(f"Exporting clustering experiment artifacts to {self.exp_dir}...")
        self._save_dataframes()
        self._save_numeric_arrays()
        self._save_metadata_and_diagnostics()
        return self.exp_dir