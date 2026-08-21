"""Evaluate a trained face embedder on identity-disjoint manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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
    identity_limit: int | None = None
    batch_limit: int | None = None
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
    model = FaceEmbedder(initialization="scratch").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CelebAFaceDataset(
        config.manifest_path,
        config.dataset_root,
        training=False,
        identity_limit=config.identity_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    genuine_scores, impostor_scores = _collect_pair_scores(
        model,
        loader,
        device,
        config.batch_limit,
    )
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
    batch_limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings_by_identity: dict[int, list[torch.Tensor]] = {}
    total = min(len(loader), batch_limit) if batch_limit is not None else len(loader)

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(
            tqdm(loader, total=total, desc="Evaluation")
        ):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            embeddings = model(images.to(device, non_blocking=True)).float().cpu()
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


def _candidate_thresholds(genuine: np.ndarray, impostor: np.ndarray) -> np.ndarray:
    scores = np.unique(np.concatenate([genuine, impostor]))
    if len(scores) == 1:
        return scores
    midpoints = (scores[:-1] + scores[1:]) / 2
    return np.concatenate([[scores[0] - 1e-6], midpoints, [scores[-1] + 1e-6]])


def _error_rates(
    genuine: np.ndarray,
    impostor: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    false_acceptance_rates = (impostor[:, None] >= thresholds).mean(axis=0)
    false_rejection_rates = (genuine[:, None] < thresholds).mean(axis=0)
    return false_acceptance_rates, false_rejection_rates


def _select_threshold(genuine: np.ndarray, impostor: np.ndarray) -> float:
    thresholds = _candidate_thresholds(genuine, impostor)
    false_acceptance_rates, false_rejection_rates = _error_rates(
        genuine,
        impostor,
        thresholds,
    )
    best_index = int(np.argmin(false_acceptance_rates + false_rejection_rates))
    return float(thresholds[best_index])


def _calculate_metrics(
    genuine: np.ndarray,
    impostor: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    true_positives = int((genuine >= threshold).sum())
    false_negatives = len(genuine) - true_positives
    false_positives = int((impostor >= threshold).sum())
    true_negatives = len(impostor) - false_positives

    total = len(genuine) + len(impostor)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / len(genuine)
    false_acceptance_rate = false_positives / len(impostor)
    false_rejection_rate = false_negatives / len(genuine)

    thresholds = _candidate_thresholds(genuine, impostor)
    far_curve, frr_curve = _error_rates(genuine, impostor, thresholds)
    eer_index = int(np.argmin(np.abs(far_curve - frr_curve)))
    eer = float((far_curve[eer_index] + frr_curve[eer_index]) / 2)
    auc = float(
        (genuine[:, None] > impostor[None, :]).mean()
        + 0.5 * (genuine[:, None] == impostor[None, :]).mean()
    )

    return {
        "threshold": threshold,
        "accuracy": (true_positives + true_negatives) / total,
        "precision": precision,
        "recall_tar": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "far": false_acceptance_rate,
        "frr": false_rejection_rate,
        "roc_auc": auc,
        "eer": eer,
        "genuine_similarity_mean": float(genuine.mean()),
        "impostor_similarity_mean": float(impostor.mean()),
        "evaluated_identities": float(len(genuine)),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
