"""Repository-relative paths shared by future project modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

DATABASE_PATH = DATA_DIR / "face_auth.db"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best.pt"

MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
