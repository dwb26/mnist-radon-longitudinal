import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.dataset_builder import build_longitudinal_mnist_dataframe
from src.utils.tensor_ops import compute_time_mesh, prepare_clustering_arrays
from src.utils.data_mgmt import PreprocessingOutput
from src.clustering.sparse_trajectory_kmeans import MissingDataKMeans
from src.postprocessing.pipeline import ExperimentExporter

import logging

logger = logging.getLogger(__name__)

def run_pipeline(config: dict) -> Path:
    """Main pipeline driver: Loads data, generates sparse Radon observations,
    formats into axSpA-compatible DataFrame, and passes to estimator.
    """
    logger.info("=" * 60)
    logger.info("Starting Sparse Radon Longitudinal Pipeline")
    logger.info("=" * 60)
    
    logger.info("\n=== Stage 1: Preprocessing ===")
    task_df = build_longitudinal_mnist_dataframe(config)
    time_mesh = compute_time_mesh(df=task_df, step_size=0.5)
    # NEED TRAIN STEP HERE MAYBE
    train_tensors = prepare_clustering_arrays(task_df, time_mesh=time_mesh)
    prep_data = PreprocessingOutput(
        task_df=task_df,
        train_df=task_df,
        test_df=None,
        time_mesh=time_mesh,
        tensors=train_tensors,
    )
    
    logger.info("\n=== Stage 2: Clustering ===")
    model = MissingDataKMeans(
        K=config.get("K", 10),
        n_responses=config.get("n_responses", 1),
        n_init=config.get("n_init", 1),
        init="random",
        lambda_reg=config.get("lambda_reg", 25000.0),
    )
    model.fit(
        data_proj=prep_data.tensors["data_proj"],
        proj_sq_diag=prep_data.tensors["proj_sq_diag"],
        data_vec_sq=prep_data.tensors["data_vec_sq"],
    )
    
    logger.info("\n=== Stage 3: Postprocessing ===")
    exporter = ExperimentExporter(
        model=model,
        prep_data=prep_data,
        config=config,
        output_base_dir="data/clustering_results",
    )
    exp_dir = exporter.export()
        
    logger.info("\nPipeline execution completed successfully.")
    
    # Generate diagnostic plots inside the exported experiment folder
    from src.visualisation.plotting import generate_pipeline_plots
    generate_pipeline_plots(model=model, prep_data=prep_data, output_dir=exp_dir)
    logger.info(f"--> Diagnostic plots saved to {exp_dir}")

    return exp_dir


if __name__ == "__main__":
    # Configure logging output format and level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Example default config input
    default_config = {
        "K": 10,
        "n_responses": 1,
        "n_init": 1,
        "lambda_reg": 75000.0,
        "n_samples": 100,
        "n_angles": 360,
        "data_dir": "./data",
        "random_state": 42,
        "lam": 20.0,
        "sampling_mode": "strided",
        "save_csv": True,
        "output_dir": "./outputs"
    }
        
    prep_data = run_pipeline(default_config)