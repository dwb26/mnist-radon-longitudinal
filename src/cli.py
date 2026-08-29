import sys
from pathlib import Path
import yaml
import argparse
import json

from pipelines.main_pipeline import run_pipeline

import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Loads configuration from either a JSON or YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path.resolve()}")

    if config_path.stat().st_size == 0:
        raise ValueError(f"Config file is empty: {config_path.resolve()}")

    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run Sparse Radon Longitudinal Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON or YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
