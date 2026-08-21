"""CelebA training workflow with local MLflow tracking."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
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
from model import (
    BACKBONE_NAME,
    EMBEDDING_DIM,
    FaceEmbedder,
    FineTuningMode,
    WeightInitialization,
    checkpoint_metadata,
    validate_checkpoint_metadata,
)

DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
ARCFACE_MARGIN_DEGREES = 28.6
ARCFACE_SCALE = 64
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
PRETRAINED_BACKBONE_LEARNING_RATE = 1e-5
PRETRAINED_HEAD_LEARNING_RATE = 1e-3
SCRATCH_LEARNING_RATE = 1e-3

type GradScaler = Any


class TrainingError(RuntimeError):
    """Raised when training configuration or data is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_manifest: Path = CELEBA_MANIFESTS_DIR / "train.csv"
    validation_manifest: Path = CELEBA_MANIFESTS_DIR / "validation.csv"
    dataset_root: Path = CELEBA_RAW_DIR
    run_name: str | None = None
    batch_size: int = 32
    backbone_learning_rate: float | None = None
    embedding_learning_rate: float | None = None
    arcface_learning_rate: float | None = None
    epochs: int = 12
    fine_tuning: FineTuningMode = "last-layer"
    initialization: WeightInitialization = "imagenet"
    seed: int = 24
    deterministic: bool = False
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


def _create_optimizer(
    model: FaceEmbedder,
    loss_function: nn.Module,
    config: TrainingConfig,
) -> Adam:
    backbone_learning_rate, embedding_learning_rate, arcface_learning_rate = _learning_rates(config)
    return Adam(
        model.optimizer_parameter_groups(
            backbone_learning_rate=backbone_learning_rate,
            embedding_learning_rate=embedding_learning_rate,
        )
        + [
            {
                "name": "arcface",
                "params": loss_function.parameters(),
                "lr": arcface_learning_rate,
            },
        ],
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )


def _learning_rate_metrics(optimizer: Optimizer) -> dict[str, float]:
    return {
        f"train/{group['name']}_learning_rate": float(group["lr"])
        for group in optimizer.param_groups
    }


def _create_grad_scaler(device: torch.device) -> GradScaler:
    torch_amp: Any = torch.amp
    return torch_amp.GradScaler(device.type, init_scale=128, enabled=device.type == "cuda")


def _learning_rates(config: TrainingConfig) -> tuple[float, float, float]:
    defaults = (
        (PRETRAINED_BACKBONE_LEARNING_RATE,) + (PRETRAINED_HEAD_LEARNING_RATE,) * 2
        if config.initialization == "imagenet"
        else (SCRATCH_LEARNING_RATE,) * 3
    )
    return (
        config.backbone_learning_rate if config.backbone_learning_rate is not None else defaults[0],
        config.embedding_learning_rate
        if config.embedding_learning_rate is not None
        else defaults[1],
        config.arcface_learning_rate if config.arcface_learning_rate is not None else defaults[2],
    )


def train(config: TrainingConfig) -> TrainingResult:
    _validate_config(config)
    device = _resolve_device(config.device)
    _seed_everything(config.seed, deterministic=config.deterministic)
    loaders = _create_data_loaders(config, device)

    model = FaceEmbedder(initialization=config.initialization).to(device)
    model.set_fine_tuning_mode(config.fine_tuning)
    loss_function = ArcFaceLoss(
        num_classes=loaders.number_of_classes,
        embedding_size=EMBEDDING_DIM,
        margin=ARCFACE_MARGIN_DEGREES,
        scale=ARCFACE_SCALE,
    ).to(device)
    optimizer = _create_optimizer(model, loss_function, config)
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
            config.initialization,
        )
    if start_epoch >= config.epochs:
        raise TrainingError(
            f"Checkpoint already completed {start_epoch} epochs, but epochs={config.epochs}"
        )

    experiment_id = _configure_mlflow()
    checkpoint_path = CHECKPOINTS_DIR / f"resnet18-{config.initialization}-{config.fine_tuning}.pt"
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
                    config.batch_limit,
                    epoch,
                    config.epochs,
                    config.fine_tuning,
                )
                validation_metrics = _evaluate_embeddings(
                    model,
                    loaders.validation,
                    device,
                    config.validation_batch_limit,
                )

                completed_epochs = epoch + 1
                learning_rates = _learning_rate_metrics(optimizer)
                scheduler.step()
                mlflow.log_metrics(
                    {
                        "train/loss": train_loss,
                        **learning_rates,
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
                    config.fine_tuning,
                    config.initialization,
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
                config.initialization,
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
        "worker_init_fn": _seed_worker,
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
    batch_limit: int | None,
    epoch: int,
    epochs: int,
    fine_tuning: FineTuningMode,
) -> float:
    model.train()
    loss_function.train()
    total_loss = 0.0
    processed_batches = 0
    progress = tqdm(
        loader,
        desc=f"Train {epoch + 1}/{epochs} [{fine_tuning}]",
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
    optimizer: Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    fine_tuning: FineTuningMode,
    initialization: WeightInitialization,
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
            "metadata": checkpoint_metadata(
                fine_tuning=fine_tuning,
                initialization=initialization,
            ),
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
    initialization: WeightInitialization,
) -> int:
    if not path.is_file():
        raise TrainingError(f"Resume checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    try:
        saved_fine_tuning, saved_initialization = validate_checkpoint_metadata(checkpoint)
    except ValueError as error:
        raise TrainingError(f"Cannot resume checkpoint: {error}") from error
    if saved_fine_tuning != fine_tuning:
        raise TrainingError(
            "Checkpoint fine-tuning mode does not match the current run: "
            f"{saved_fine_tuning!r} != {fine_tuning!r}"
        )
    if saved_initialization != initialization:
        raise TrainingError(
            "Checkpoint initialization does not match the current run: "
            f"{saved_initialization!r} != {initialization!r}"
        )
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
            "backbone": BACKBONE_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "fine_tuning": config.fine_tuning,
            "initialization": config.initialization,
            "arcface_margin_degrees": ARCFACE_MARGIN_DEGREES,
            "arcface_scale": ARCFACE_SCALE,
            "optimizer": "Adam",
            "adam_beta1": ADAM_BETAS[0],
            "adam_beta2": ADAM_BETAS[1],
            "adam_eps": ADAM_EPS,
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
    positive_values: dict[str, float] = {
        "batch_size": float(config.batch_size),
        "epochs": float(config.epochs),
    }
    positive_values.update(
        {
            name: value
            for name, value in {
                "backbone_learning_rate": config.backbone_learning_rate,
                "embedding_learning_rate": config.embedding_learning_rate,
                "arcface_learning_rate": config.arcface_learning_rate,
            }.items()
            if value is not None
        }
    )
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise TrainingError(f"Configuration values must be positive: {', '.join(invalid)}")
    if config.num_workers < 0:
        raise TrainingError("num_workers cannot be negative")
    if config.batch_limit is not None and config.batch_limit <= 0:
        raise TrainingError("batch_limit must be positive")
    if config.validation_batch_limit is not None and config.validation_batch_limit <= 0:
        raise TrainingError("validation_batch_limit must be positive")
    if config.fine_tuning not in {"last-layer", "all"}:
        raise TrainingError("fine_tuning must be last-layer or all")
    if config.initialization not in {"imagenet", "scratch"}:
        raise TrainingError("initialization must be imagenet or scratch")
    if config.initialization == "scratch" and config.fine_tuning != "all":
        raise TrainingError("scratch initialization requires fine_tuning=all")


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


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained ResNet18 on CelebA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fine-tuning", choices=("last-layer", "all"), required=True)
    parser.add_argument("--initialization", choices=("imagenet", "scratch"), default="imagenet")
    parser.add_argument("--run-name", default="resnet18-local")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--identity-limit", type=int)
    parser.add_argument("--batch-limit", type=int)
    parser.add_argument("--validation-batch-limit", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument("--embedding-learning-rate", type=float)
    parser.add_argument("--arcface-learning-rate", type=float)
    args = parser.parse_args()

    result = train(
        TrainingConfig(
            run_name=args.run_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            resume_from=args.resume_from,
            fine_tuning=args.fine_tuning,
            initialization=args.initialization,
            identity_limit=args.identity_limit,
            batch_limit=args.batch_limit,
            validation_batch_limit=args.validation_batch_limit,
            deterministic=args.deterministic,
            backbone_learning_rate=args.backbone_learning_rate,
            embedding_learning_rate=args.embedding_learning_rate,
            arcface_learning_rate=args.arcface_learning_rate,
        )
    )
    print(f"Completed epochs: {result.completed_epochs}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"MLflow run: {result.run_id}")


if __name__ == "__main__":
    main()
