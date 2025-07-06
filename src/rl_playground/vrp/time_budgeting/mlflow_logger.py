import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow

LOG_DIR_PATH = Path("tmp")


def setup_mlflow_logger(level=logging.INFO):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_path = LOG_DIR_PATH / f"stdout-{timestamp}.log"
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    return log_file_path


def upload_log_to_mlflow(log_file_path):
    if mlflow.active_run() and os.path.exists(log_file_path):
        mlflow.log_artifact(log_file_path)
        os.remove(log_file_path)
