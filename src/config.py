from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_IMAGE_SIZE = 160

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CELEBA_ROOT = DATA_DIR / "celeba"
CELEBA_RAW_DIR = CELEBA_ROOT / "raw"
CELEBA_MANIFESTS_DIR = CELEBA_ROOT / "manifests"

DATABASE_PATH = DATA_DIR / "face_auth.db"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best.pt"

MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "face-recognition-arcface"
