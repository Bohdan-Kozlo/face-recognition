"""Fine-tune ResNet18 on CelebA with ArcFace and MLflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from torch import nn
from torch.optim import Adam, Optimizer
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
from evaluation import evaluate_model
from model import (
    BACKBONE_NAME,
    EMBEDDING_DIM,
    FaceEmbedder,
    FineTuningMode,
    checkpoint_metadata,
    validate_checkpoint_metadata,
)

DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
ARCFACE_MARGIN_DEGREES = 28.6
ARCFACE_SCALE = 64
BACKBONE_LEARNING_RATE = 1e-5
EMBEDDING_LEARNING_RATE = 1e-3
ARCFACE_LEARNING_RATE = 1e-3

type GradScaler = Any


class TrainingError(RuntimeError):
    """Raised when training configuration, data, or a checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_manifest: Path = CELEBA_MANIFESTS_DIR / "train.csv"
    validation_manifest: Path = CELEBA_MANIFESTS_DIR / "validation.csv"
    dataset_root: Path = CELEBA_RAW_DIR
    run_name: str | None = None
    epochs: int = 12
    batch_size: int = 32
    fine_tuning: FineTuningMode = "last-layer"
    seed: int = 24
    num_workers: int = 2
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
    _validate_config(config)
    device = _resolve_device(config.device)
    _seed_everything(config.seed)
    loaders = _create_data_loaders(config, device)

    model = FaceEmbedder().to(device)
    model.set_fine_tuning_mode(config.fine_tuning)
    loss_function = ArcFaceLoss(
        num_classes=loaders.number_of_classes,
        embedding_size=EMBEDDING_DIM,
        margin=ARCFACE_MARGIN_DEGREES,
        scale=ARCFACE_SCALE,
    ).to(device)
    optimizer = _create_optimizer(model, loss_function)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = _create_grad_scaler(device)

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
            config.fine_tuning,
        )
    if start_epoch >= config.epochs:
        raise TrainingError(
            f"Checkpoint completed {start_epoch} epochs, but epochs={config.epochs}"
        )

    checkpoint_path = CHECKPOINTS_DIR / f"resnet18-{config.fine_tuning}.pt"
    experiment_id = _configure_mlflow()
    completed_epochs = start_epoch

    with mlflow.start_run(experiment_id=experiment_id, run_name=config.run_name) as active_run:
        _log_run_inputs(config, loaders, device)
        try:
            for epoch in range(start_epoch, config.epochs):
                loaders.train_generator.manual_seed(config.seed + epoch)
                train_loss = _train_epoch(
                    model,
                    loss_function,
                    loaders.train,
                    optimizer,
                    scaler,
                    device,
                    epoch,
                    config.epochs,
                )
                validation = evaluate_model(model, loaders.validation, device)
                completed_epochs = epoch + 1

                metrics = {
                    "train/loss": train_loss,
                    **_learning_rate_metrics(optimizer),
                    **{f"validation/{name}": value for name, value in validation.metrics.items()},
                }
                mlflow.log_metrics(metrics, step=completed_epochs)
                scheduler.step()
                _save_checkpoint(
                    checkpoint_path,
                    completed_epochs,
                    model,
                    loss_function,
                    optimizer,
                    scheduler,
                    scaler,
                    config.fine_tuning,
                )
                print(
                    f"Epoch {completed_epochs}/{config.epochs}: "
                    f"loss={train_loss:.4f}, "
                    f"similarity_gap={validation.metrics['similarity_gap']:.4f}"
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
                config.fine_tuning,
            )
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            raise

        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        return TrainingResult(active_run.info.run_id, completed_epochs, checkpoint_path)


def _create_optimizer(model: FaceEmbedder, loss_function: nn.Module) -> Adam:
    parameter_groups = model.optimizer_parameter_groups(
        backbone_learning_rate=BACKBONE_LEARNING_RATE,
        embedding_learning_rate=EMBEDDING_LEARNING_RATE,
    )
    parameter_groups.append(
        {
            "name": "arcface",
            "params": loss_function.parameters(),
            "lr": ARCFACE_LEARNING_RATE,
        }
    )
    return Adam(parameter_groups)


def _learning_rate_metrics(optimizer: Optimizer) -> dict[str, float]:
    return {
        f"train/{group['name']}_learning_rate": float(group["lr"])
        for group in optimizer.param_groups
    }


def _create_grad_scaler(device: torch.device) -> GradScaler:
    torch_amp: Any = torch.amp
    return torch_amp.GradScaler(device.type, init_scale=128, enabled=device.type == "cuda")


def _create_data_loaders(config: TrainingConfig, device: torch.device) -> _DataLoaders:
    train_dataset = CelebAFaceDataset(config.train_manifest, config.dataset_root)
    validation_dataset = CelebAFaceDataset(config.validation_manifest, config.dataset_root)
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
    optimizer: Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    loss_function.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}")
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss = loss_function(model(images), labels)

        if not torch.isfinite(loss):
            raise TrainingError("Training loss is not finite")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    if not len(loader):
        raise TrainingError("Training loader did not produce any batches")
    return total_loss / len(loader)


def _save_checkpoint(
    path: Path,
    completed_epochs: int,
    model: FaceEmbedder,
    loss_function: nn.Module,
    optimizer: Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    fine_tuning: FineTuningMode,
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
            "metadata": checkpoint_metadata(fine_tuning=fine_tuning),
        },
        path,
    )


def _restore_checkpoint(
    path: Path,
    device: torch.device,
    model: FaceEmbedder,
    loss_function: nn.Module,
    optimizer: Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    fine_tuning: FineTuningMode,
) -> int:
    if not path.is_file():
        raise TrainingError(f"Resume checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    try:
        saved_fine_tuning = validate_checkpoint_metadata(checkpoint)
        if saved_fine_tuning != fine_tuning:
            raise ValueError(
                f"fine-tuning mode {saved_fine_tuning!r} does not match {fine_tuning!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        loss_function.load_state_dict(checkpoint["arcface_loss_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        return int(checkpoint["completed_epochs"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingError(f"Cannot resume checkpoint: {error}") from error


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
    mlflow.log_params(
        {
            "backbone": BACKBONE_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "fine_tuning": config.fine_tuning,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "device": str(device),
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "embedding_learning_rate": EMBEDDING_LEARNING_RATE,
            "arcface_learning_rate": ARCFACE_LEARNING_RATE,
            "arcface_margin_degrees": ARCFACE_MARGIN_DEGREES,
            "arcface_scale": ARCFACE_SCALE,
            "number_of_classes": loaders.number_of_classes,
            "train_images": loaders.train_images,
            "validation_images": loaders.validation_images,
        }
    )


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs <= 0:
        raise TrainingError("epochs must be positive")
    if config.batch_size <= 0:
        raise TrainingError("batch_size must be positive")
    if config.num_workers < 0:
        raise TrainingError("num_workers cannot be negative")
    if config.fine_tuning not in {"last-layer", "all"}:
        raise TrainingError("fine_tuning must be last-layer or all")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise TrainingError("Device must be auto, cpu, or cuda")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained ResNet18 on CelebA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fine-tuning", choices=("last-layer", "all"), required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    result = train(
        TrainingConfig(
            fine_tuning=args.fine_tuning,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            run_name=args.run_name,
            resume_from=args.resume_from,
        )
    )
    print(f"Completed epochs: {result.completed_epochs}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"MLflow run: {result.run_id}")


if __name__ == "__main__":
    main()
