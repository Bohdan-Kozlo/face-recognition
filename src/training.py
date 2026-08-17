"""CelebA training workflow with local MLflow tracking."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    CELEBA_MANIFESTS_DIR,
    CELEBA_RAW_DIR,
    CHECKPOINTS_DIR,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_DATABASE_PATH,
    MLFLOW_EXPERIMENT_NAME,
)
from data import CelebAFaceDataset
from model import FaceEmbedder

LOGGER = logging.getLogger(__name__)
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
ARCFACE_MARGIN_DEGREES = 28.6
ARCFACE_SCALE = 64
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


class TrainingError(RuntimeError):
    """Raised when training configuration or data is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_manifest: Path = CELEBA_MANIFESTS_DIR / "train.csv"
    validation_manifest: Path = CELEBA_MANIFESTS_DIR / "validation.csv"
    dataset_root: Path = CELEBA_RAW_DIR
    run_name: str | None = None
    embedding_dim: int = 128
    arcface_margin_degrees: float = ARCFACE_MARGIN_DEGREES
    arcface_scale: int = ARCFACE_SCALE
    batch_size: int = 32
    learning_rate: float = 0.01
    epochs: int = 10
    seed: int = 24
    num_workers: int = 2
    identity_limit: int | None = None
    batch_limit: int | None = None
    validation_batch_limit: int | None = 50
    device: str = "auto"
    resume_from: Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_id: str
    completed_epochs: int
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class _DataLoaders:
    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    train_generator: torch.Generator
    number_of_classes: int
    train_images: int
    validation_images: int


def train(config: TrainingConfig) -> TrainingResult:
    """Train the face embedder and return the recorded run."""

    _validate_config(config)
    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)
    loaders = _create_data_loaders(config, device)

    model = FaceEmbedder(config.embedding_dim).to(device)
    loss_function = ArcFaceLoss(
        num_classes=loaders.number_of_classes,
        embedding_size=config.embedding_dim,
        margin=config.arcface_margin_degrees,
        scale=config.arcface_scale,
    ).to(device)
    optimizer = SGD(
        [
            {"params": model.parameters()},
            {"params": loss_function.parameters()},
        ],
        lr=config.learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.GradScaler("cuda", init_scale=128, enabled=device.type == "cuda")

    start_epoch = 0
    if config.resume_from is not None:
        start_epoch = _restore_checkpoint(
            config.resume_from,
            device,
            model,
            loss_function,
            optimizer,
            scheduler,
            scaler,
        )
    if start_epoch >= config.epochs:
        raise TrainingError(
            f"Checkpoint already completed {start_epoch} epochs, but epochs={config.epochs}"
        )

    experiment_id = _configure_mlflow()
    checkpoint_path = CHECKPOINTS_DIR / "last.pt"
    completed_epochs = start_epoch

    with mlflow.start_run(experiment_id=experiment_id, run_name=config.run_name) as active_run:
        _log_run_inputs(config, loaders, device)
        try:
            for epoch in range(start_epoch, config.epochs):
                epoch_seed = config.seed + epoch
                torch.manual_seed(epoch_seed)
                loaders.train_generator.manual_seed(epoch_seed)

                train_loss = _train_epoch(
                    model,
                    loss_function,
                    loaders.train,
                    optimizer,
                    scaler,
                    device,
                    config.batch_limit,
                    epoch,
                    config.epochs,
                )
                validation_metrics = _evaluate_embeddings(
                    model,
                    loaders.validation,
                    device,
                    config.validation_batch_limit,
                )

                completed_epochs = epoch + 1
                learning_rate = optimizer.param_groups[0]["lr"]
                scheduler.step()
                mlflow.log_metrics(
                    {
                        "train/loss": train_loss,
                        "train/learning_rate": learning_rate,
                        "system/amp_enabled": float(device.type == "cuda"),
                        **validation_metrics,
                    },
                    step=completed_epochs,
                )
                _save_checkpoint(
                    checkpoint_path,
                    completed_epochs,
                    model,
                    loss_function,
                    optimizer,
                    scheduler,
                    scaler,
                )
                LOGGER.info(
                    "Epoch %d/%d: loss=%.4f, validation_gap=%.4f",
                    completed_epochs,
                    config.epochs,
                    train_loss,
                    validation_metrics["validation/similarity_gap"],
                )
        except KeyboardInterrupt:
            _save_checkpoint(
                checkpoint_path,
                completed_epochs,
                model,
                loss_function,
                optimizer,
                scheduler,
                scaler,
            )
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            raise

        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        return TrainingResult(active_run.info.run_id, completed_epochs, checkpoint_path)


def _create_data_loaders(config: TrainingConfig, device: torch.device) -> _DataLoaders:
    train_dataset = CelebAFaceDataset(
        config.train_manifest,
        config.dataset_root,
        training=True,
        identity_limit=config.identity_limit,
        seed=config.seed,
    )
    validation_dataset = CelebAFaceDataset(
        config.validation_manifest,
        config.dataset_root,
        training=False,
        identity_limit=config.identity_limit,
        seed=config.seed,
    )
    if train_dataset.identity_ids.intersection(validation_dataset.identity_ids):
        raise TrainingError("Training and validation manifests contain overlapping identities")

    train_generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return _DataLoaders(
        train=DataLoader(
            train_dataset,
            shuffle=True,
            generator=train_generator,
            drop_last=True,
            **common,
        ),
        validation=DataLoader(validation_dataset, shuffle=False, **common),
        train_generator=train_generator,
        number_of_classes=train_dataset.number_of_classes,
        train_images=len(train_dataset),
        validation_images=len(validation_dataset),
    )


def _train_epoch(
    model: FaceEmbedder,
    loss_function: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: SGD,
    scaler: torch.GradScaler,
    device: torch.device,
    batch_limit: int | None,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    loss_function.train()
    total_loss = 0.0
    processed_batches = 0
    progress = tqdm(
        loader,
        desc=f"Train {epoch + 1}/{epochs}",
        total=_limited_length(len(loader), batch_limit),
    )

    for batch_index, (images, labels) in enumerate(progress):
        if batch_limit is not None and batch_index >= batch_limit:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            embeddings = model(images)
            loss = loss_function(embeddings, labels)

        if not torch.isfinite(loss):
            raise TrainingError(
                f"Training loss is not finite at batch {batch_index}; try batch_size=32"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss
        processed_batches += 1
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    if processed_batches == 0:
        raise TrainingError("Training loader did not produce any batches")
    return total_loss / processed_batches


def _evaluate_embeddings(
    model: FaceEmbedder,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_limit: int | None,
) -> dict[str, float]:
    model.eval()
    embeddings_by_identity: dict[int, list[torch.Tensor]] = {}
    progress = tqdm(
        loader,
        desc="Validation",
        total=_limited_length(len(loader), batch_limit),
    )

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(progress):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            embeddings = model(images.to(device, non_blocking=True)).float().cpu()
            if not torch.isfinite(embeddings).all():
                raise TrainingError("Validation embeddings are not finite")
            for embedding, label in zip(embeddings, labels.tolist(), strict=True):
                embeddings_by_identity.setdefault(label, []).append(embedding)

    identities = sorted(
        identity for identity, embeddings in embeddings_by_identity.items() if len(embeddings) >= 2
    )
    if len(identities) < 2:
        raise TrainingError(
            "Validation needs at least two identities with two images each; "
            "increase validation_batch_limit"
        )

    genuine = torch.stack(
        [
            torch.dot(embeddings_by_identity[identity][0], embeddings_by_identity[identity][1])
            for identity in identities
        ]
    )
    impostor = torch.stack(
        [
            torch.dot(
                embeddings_by_identity[identity][0],
                embeddings_by_identity[identities[(index + 1) % len(identities)]][0],
            )
            for index, identity in enumerate(identities)
        ]
    )
    genuine_mean = float(genuine.mean())
    impostor_mean = float(impostor.mean())
    return {
        "validation/genuine_similarity_mean": genuine_mean,
        "validation/impostor_similarity_mean": impostor_mean,
        "validation/similarity_gap": genuine_mean - impostor_mean,
        "validation/evaluated_identities": float(len(identities)),
    }


def _save_checkpoint(
    path: Path,
    completed_epochs: int,
    model: FaceEmbedder,
    loss_function: nn.Module,
    optimizer: SGD,
    scheduler: CosineAnnealingLR,
    scaler: torch.GradScaler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "completed_epochs": completed_epochs,
            "model_state_dict": model.state_dict(),
            "arcface_loss_state_dict": loss_function.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def _restore_checkpoint(
    path: Path,
    device: torch.device,
    model: FaceEmbedder,
    loss_function: nn.Module,
    optimizer: SGD,
    scheduler: CosineAnnealingLR,
    scaler: torch.GradScaler,
) -> int:
    if not path.is_file():
        raise TrainingError(f"Resume checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_function.load_state_dict(checkpoint["arcface_loss_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    return int(checkpoint["completed_epochs"])


def _configure_mlflow() -> str:
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        MLFLOW_EXPERIMENT_NAME,
        artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
    )


def _log_run_inputs(
    config: TrainingConfig,
    loaders: _DataLoaders,
    device: torch.device,
) -> None:
    config_values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    mlflow.log_params(
        {
            **config_values,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "number_of_classes": loaders.number_of_classes,
            "train_images": loaders.train_images,
            "validation_images": loaders.validation_images,
            "resolved_device": str(device),
        }
    )
    summary_path = config.train_manifest.parent / "summary.json"
    if summary_path.is_file():
        mlflow.log_artifact(str(summary_path), artifact_path="data")


def _validate_config(config: TrainingConfig) -> None:
    positive_values = {
        "embedding_dim": config.embedding_dim,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "arcface_margin_degrees": config.arcface_margin_degrees,
        "arcface_scale": config.arcface_scale,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise TrainingError(f"Configuration values must be positive: {', '.join(invalid)}")
    if config.num_workers < 0:
        raise TrainingError("num_workers cannot be negative")
    if config.batch_limit is not None and config.batch_limit <= 0:
        raise TrainingError("batch_limit must be positive")
    if config.validation_batch_limit is not None and config.validation_batch_limit <= 0:
        raise TrainingError("validation_batch_limit must be positive")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise TrainingError("Device must be auto, cpu, or cuda")
    return torch.device(requested)


def _limited_length(total: int, limit: int | None) -> int:
    return min(total, limit) if limit is not None else total


def _build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the CelebA ArcFace embedder")
    parser.add_argument("--train-manifest", type=Path, default=defaults.train_manifest)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=defaults.validation_manifest,
    )
    parser.add_argument("--dataset-root", type=Path, default=defaults.dataset_root)
    parser.add_argument("--run-name")
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument(
        "--arcface-margin-degrees",
        type=float,
        default=defaults.arcface_margin_degrees,
    )
    parser.add_argument("--arcface-scale", type=int, default=defaults.arcface_scale)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--identity-limit", type=int)
    parser.add_argument("--batch-limit", type=int)
    parser.add_argument(
        "--validation-batch-limit",
        type=int,
        default=defaults.validation_batch_limit,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=defaults.device)
    parser.add_argument("--resume-from", type=Path)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = TrainingConfig(**vars(_build_parser().parse_args()))
    try:
        result = train(config)
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted; the latest checkpoint is in checkpoints/last.pt")
        return 130
    except (TrainingError, OSError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1

    print(f"MLflow run: {result.run_id}")
    print(f"Checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
