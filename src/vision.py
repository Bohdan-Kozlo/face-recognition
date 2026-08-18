from __future__ import annotations

import io
import threading
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import FACE_IMAGE_SIZE, YUNET_MODEL_PATH

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
ARCFACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class VisionErrorCode(StrEnum):
    FILE_TOO_LARGE = "file_too_large"
    INVALID_IMAGE = "invalid_image"
    UNSUPPORTED_IMAGE_FORMAT = "unsupported_image_format"
    FACE_NOT_FOUND = "face_not_found"
    MULTIPLE_FACES = "multiple_faces"
    FACE_TOO_SMALL = "face_too_small"
    IMAGE_TOO_BLURRY = "image_too_blurry"
    ALIGNMENT_FAILED = "alignment_failed"


class VisionError(ValueError):
    def __init__(self, code: VisionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VisionConfig:
    max_file_size_bytes: int = 10 * 1024 * 1024
    max_input_side: int = 1280
    output_size: int = FACE_IMAGE_SIZE
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000
    minimum_face_side: float = 40.0
    blur_threshold: float = 80.0


class FacePreprocessor:
    def __init__(
        self,
        model_path: Path = YUNET_MODEL_PATH,
        config: VisionConfig | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model does not exist: {model_path}")

        config = config or VisionConfig()
        self._config = config
        self._detector_lock = threading.Lock()
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(model_path),
                config="",
                input_size=(320, 320),
                score_threshold=config.score_threshold,
                nms_threshold=config.nms_threshold,
                top_k=config.top_k,
            )
        except cv2.error as error:
            raise RuntimeError(f"Could not load YuNet model: {model_path}") from error

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        image = _decode_image(image_bytes, self._config.max_file_size_bytes)
        image = _resize_for_detection(image, self._config.max_input_side)
        detection = self._detect_exactly_one_face(image)

        width, height = float(detection[2]), float(detection[3])
        if min(width, height) < self._config.minimum_face_side:
            raise VisionError(
                VisionErrorCode.FACE_TOO_SMALL,
                f"Detected face side is below {self._config.minimum_face_side:g} pixels",
            )

        landmarks = detection[4:14].reshape(5, 2).astype(np.float32)
        crop = _align_face(image, landmarks, self._config.output_size)
        blur_score = _calculate_blur_score(crop)
        if blur_score < self._config.blur_threshold:
            raise VisionError(
                VisionErrorCode.IMAGE_TOO_BLURRY,
                f"Aligned face blur score {blur_score:.2f} is below "
                f"{self._config.blur_threshold:g}",
            )
        return crop

    def _detect_exactly_one_face(self, image: np.ndarray) -> np.ndarray:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = bgr_image.shape[:2]

        try:
            with self._detector_lock:
                self._detector.setInputSize((width, height))
                _, faces = self._detector.detect(bgr_image)
        except cv2.error as error:
            raise VisionError(
                VisionErrorCode.INVALID_IMAGE,
                "YuNet could not process the decoded image",
            ) from error

        if faces is None or len(faces) == 0:
            raise VisionError(VisionErrorCode.FACE_NOT_FOUND, "No face was detected")
        if len(faces) > 1:
            raise VisionError(
                VisionErrorCode.MULTIPLE_FACES,
                f"Expected one face, detected {len(faces)}",
            )
        return np.asarray(faces[0], dtype=np.float32)


def _decode_image(image_bytes: bytes, max_file_size_bytes: int) -> np.ndarray:
    if not image_bytes:
        raise VisionError(VisionErrorCode.INVALID_IMAGE, "Image is empty")
    if len(image_bytes) > max_file_size_bytes:
        raise VisionError(
            VisionErrorCode.FILE_TOO_LARGE,
            f"Image exceeds the {max_file_size_bytes // (1024 * 1024)} MB limit",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as encoded_image:
                image_format = encoded_image.format
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    raise VisionError(
                        VisionErrorCode.UNSUPPORTED_IMAGE_FORMAT,
                        f"Expected JPEG or PNG content, got {image_format or 'unknown'}",
                    )
                oriented_image = ImageOps.exif_transpose(encoded_image)
                rgb_image = oriented_image.convert("RGB")
                rgb_image.load()
                return np.asarray(rgb_image, dtype=np.uint8)
    except VisionError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ) as error:
        raise VisionError(VisionErrorCode.INVALID_IMAGE, "Image content is invalid") from error


def _resize_for_detection(image: np.ndarray, max_input_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_input_side:
        return image

    scale = max_input_side / largest_side
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int) -> np.ndarray:
    template = ARCFACE_112_TEMPLATE * (output_size / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if transform is None:
        raise VisionError(
            VisionErrorCode.ALIGNMENT_FAILED,
            "Could not estimate a face alignment transform",
        )

    crop = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop


def _calculate_blur_score(image: np.ndarray) -> float:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
