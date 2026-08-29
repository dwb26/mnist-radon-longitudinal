# src/utils/data_mgmt.py

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional

import logging
logger = logging.getLogger(__name__)
@dataclass
class PreprocessingOutput:
    """Artefacts produced by the Preprocessing stage."""
    task_df: pd.DataFrame
    train_df: pd.DataFrame
    time_mesh: np.ndarray
    tensors: dict[str, np.ndarray]
    test_df: Optional[pd.DataFrame] = None
        
@dataclass
class FitDiagnostics:
    """Dataclass to track convergence metrics across iterations."""
    converged: bool = False
    iterations_run: int = 0
    final_wcss: float = float("inf")
    final_loss: float = float("inf")    

def parse_k_range(k_config) -> range | list[int]:
    """Parses standard YAML inputs into an executable sequence of K values."""
    if isinstance(k_config, dict):
        return range(k_config["start"], k_config["stop"], k_config.get("step", 1))
    elif isinstance(k_config, list):
        return k_config
    elif isinstance(k_config, int):
        return [k_config]
    raise TypeError(f"Invalid type for k_range in config: {type(k_config)}")