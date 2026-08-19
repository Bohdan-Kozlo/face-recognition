import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = Path(
    os.environ.get("FACE_RECOGNITION_CHECKPOINTS_DIR", PROJECT_ROOT / "checkpoints")
)

CELEBA_ROOT = DATA_DIR / "celeba"
CELEBA_RAW_DIR = CELEBA_ROOT / "raw"
CELEBA_MANIFESTS_DIR = CELEBA_ROOT / "manifests"

DATABASE_PATH = DATA_DIR / "face_auth.db"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "face-recognition-arcface"
