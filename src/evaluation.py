"""Evaluate a trained face embedder on identity-disjoint manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    det_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CELEBA_RAW_DIR
from data import CelebAFaceDataset
from model import FaceEmbedder, validate_checkpoint_metadata


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    manifest_path: Path
    checkpoint_path: Path
    dataset_root: Path = CELEBA_RAW_DIR
    batch_size: int = 32
    num_workers: int = 2
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class VerificationEvaluation:
    threshold: float
    metrics: dict[str, float]
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray


def evaluate_checkpoint(
    config: EvaluationConfig,
    *,
    threshold: float | None = None,
) -> VerificationEvaluation:
    """Evaluate one genuine and one impostor pair per identity."""

    device = _resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=True)
    try:
        validate_checkpoint_metadata(checkpoint)
    except ValueError as error:
        raise ValueError(f"Cannot evaluate checkpoint: {error}") from error
    model = FaceEmbedder(load_pretrained_weights=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CelebAFaceDataset(
        config.manifest_path,
        config.dataset_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    return evaluate_model(model, loader, device, threshold=threshold)


def evaluate_model(
    model: FaceEmbedder,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    threshold: float | None = None,
) -> VerificationEvaluation:
    """Evaluate one genuine and one impostor pair per identity."""

    model.eval()
    genuine_scores, impostor_scores = _collect_pair_scores(model, loader, device)
    selected_threshold = (
        _select_threshold(genuine_scores, impostor_scores) if threshold is None else threshold
    )
    metrics = _calculate_metrics(genuine_scores, impostor_scores, selected_threshold)
    return VerificationEvaluation(
        threshold=selected_threshold,
        metrics=metrics,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
    )


def _collect_pair_scores(
    model: FaceEmbedder,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings_by_identity: dict[int, list[torch.Tensor]] = {}

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Evaluation"):
            embeddings = model(images.to(device, non_blocking=True)).float().cpu()
            if not torch.isfinite(embeddings).all():
                raise ValueError("Evaluation embeddings are not finite")
            for embedding, label in zip(embeddings, labels.tolist(), strict=True):
                embeddings_by_identity.setdefault(label, []).append(embedding)

    identities = sorted(
        identity for identity, embeddings in embeddings_by_identity.items() if len(embeddings) >= 2
    )
    if len(identities) < 2:
        raise ValueError("Evaluation needs at least two identities with two images each")

    genuine_scores = np.asarray(
        [
            torch.dot(embeddings_by_identity[identity][0], embeddings_by_identity[identity][1])
            for identity in identities
        ],
        dtype=np.float32,
    )
    impostor_scores = np.asarray(
        [
            torch.dot(
                embeddings_by_identity[identity][0],
                embeddings_by_identity[identities[(index + 1) % len(identities)]][0],
            )
            for index, identity in enumerate(identities)
        ],
        dtype=np.float32,
    )
    return genuine_scores, impostor_scores


def _select_threshold(genuine: np.ndarray, impostor: np.ndarray) -> float:
    labels, scores = _verification_labels_and_scores(genuine, impostor)
    false_acceptance_rates, false_rejection_rates, thresholds = det_curve(labels, scores)
    best_index = int(np.argmin(false_acceptance_rates + false_rejection_rates))
    return float(thresholds[best_index])


def _calculate_metrics(
    genuine: np.ndarray,
    impostor: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels, scores = _verification_labels_and_scores(genuine, impostor)
    predictions = scores >= threshold
    normalized_confusion = confusion_matrix(labels, predictions, labels=[0, 1], normalize="true")
    false_acceptance_rate = float(normalized_confusion[0, 1])
    false_rejection_rate = float(normalized_confusion[1, 0])
    far_curve, frr_curve, _ = det_curve(labels, scores)
    eer_index = int(np.argmin(np.abs(far_curve - frr_curve)))

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=cast(Any, 0))),
        "recall_tar": float(recall_score(labels, predictions, zero_division=cast(Any, 0))),
        "f1": float(f1_score(labels, predictions, zero_division=cast(Any, 0))),
        "far": false_acceptance_rate,
        "frr": false_rejection_rate,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": float((far_curve[eer_index] + frr_curve[eer_index]) / 2),
        "genuine_similarity_mean": float(genuine.mean()),
        "impostor_similarity_mean": float(impostor.mean()),
        "similarity_gap": float(genuine.mean() - impostor.mean()),
        "evaluated_identities": float(len(genuine)),
    }


def _verification_labels_and_scores(
    genuine: np.ndarray,
    impostor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate(
        [np.ones(len(genuine), dtype=np.int8), np.zeros(len(impostor), dtype=np.int8)]
    )
    return labels, np.concatenate([genuine, impostor])


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
