"""Prepare deterministic, identity-disjoint CelebA manifests."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import v2

from config import CELEBA_MANIFESTS_DIR, CELEBA_RAW_DIR
from model import IMAGE_SIZE

LOGGER = logging.getLogger(__name__)

IDENTITY_ANNOTATION_FILENAME = "identity_CelebA.txt"
IMAGE_DIRECTORY_NAME = "img_align_celeba"
MANIFEST_COLUMNS = ("image_path", "identity_id", "class_index")

type IdentityImages = dict[int, list[str]]
type FaceTransform = Callable[[Image.Image], torch.Tensor]


class DataPreparationError(ValueError):
    """Raised when CelebA input or preparation configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    dataset_root: Path = CELEBA_RAW_DIR
    output_dir: Path = CELEBA_MANIFESTS_DIR
    seed: int = 24
    min_images: int = 3
    limit_identities: int | None = None


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    image_path: Path
    identity_id: int
    class_index: int


class CelebAFaceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path = CELEBA_RAW_DIR,
        *,
        training: bool,
        identity_limit: int | None = None,
        seed: int = 24,
    ) -> None:
        records = _load_manifest_records(manifest_path)
        selected_records = _select_and_remap_records(records, identity_limit, seed)
        self.identity_ids = frozenset(record.identity_id for record in selected_records)
        self._records = _resolve_image_paths(selected_records, dataset_root)
        self._transform = build_face_transform(training=training)
        self.number_of_classes = len(self.identity_ids)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self._records[index]
        try:
            with Image.open(record.image_path) as image:
                rgb_image = image.convert("RGB")
                tensor = self._transform(rgb_image)
        except (OSError, UnidentifiedImageError) as error:
            raise DataPreparationError(f"Could not decode image: {record.image_path}") from error
        return tensor, torch.tensor(record.class_index, dtype=torch.long)


def build_face_transform(*, training: bool) -> FaceTransform:
    transforms: list[Callable[..., Any]] = [
        v2.ToImage(),
        v2.CenterCrop((178, 178)),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ]
    if training:
        transforms.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)],
                    p=0.5,
                ),
            ]
        )
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    composed = v2.Compose(transforms)
    return cast(FaceTransform, composed)


def _load_manifest_records(manifest_path: Path) -> list[ManifestRecord]:
    if not manifest_path.is_file():
        raise DataPreparationError(f"Manifest does not exist: {manifest_path}")

    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = set(MANIFEST_COLUMNS).difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataPreparationError(f"Manifest is missing columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            try:
                relative_path = Path(row["image_path"])
                identity_id = int(row["identity_id"])
                class_index = int(row["class_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise DataPreparationError(
                    f"Invalid manifest row at line {line_number}: {manifest_path}"
                ) from error
            if relative_path.is_absolute():
                raise DataPreparationError(
                    f"Manifest image path must be relative at line {line_number}"
                )
            records.append(ManifestRecord(relative_path, identity_id, class_index))

    if not records:
        raise DataPreparationError(f"Manifest is empty: {manifest_path}")
    return records


def _resolve_image_paths(records: list[ManifestRecord], dataset_root: Path) -> list[ManifestRecord]:
    resolved: list[ManifestRecord] = []
    for record in records:
        image_path = dataset_root / record.image_path
        if not image_path.is_file():
            raise DataPreparationError(f"Manifest image does not exist: {image_path}")
        resolved.append(ManifestRecord(image_path, record.identity_id, record.class_index))
    return resolved


def _select_and_remap_records(
    records: list[ManifestRecord],
    identity_limit: int | None,
    seed: int,
) -> list[ManifestRecord]:
    records_by_identity: defaultdict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        records_by_identity[record.identity_id].append(record)

    identity_ids = sorted(records_by_identity)
    if identity_limit is not None:
        if identity_limit < 2:
            raise DataPreparationError("Identity limit must be at least 2")
        random.Random(seed).shuffle(identity_ids)
        identity_ids = sorted(identity_ids[:identity_limit])

    class_indices = {
        identity_id: class_index for class_index, identity_id in enumerate(identity_ids)
    }
    remapped_by_identity = {
        identity_id: [
            ManifestRecord(record.image_path, identity_id, class_indices[identity_id])
            for record in records_by_identity[identity_id]
        ]
        for identity_id in identity_ids
    }

    # Interleaving keeps limited validation batches representative of multiple identities.
    interleaved: list[ManifestRecord] = []
    largest_identity = max(len(items) for items in remapped_by_identity.values())
    for image_index in range(largest_identity):
        for identity_id in identity_ids:
            identity_records = remapped_by_identity[identity_id]
            if image_index < len(identity_records):
                interleaved.append(identity_records[image_index])
    return interleaved


def prepare_celeba(config: PreparationConfig) -> dict[str, Any]:
    annotation_path = config.dataset_root / IDENTITY_ANNOTATION_FILENAME
    image_dir = config.dataset_root / IMAGE_DIRECTORY_NAME
    _validate_paths(annotation_path, image_dir)

    identities = _load_identities(annotation_path)
    eligible_ids = sorted(
        identity_id
        for identity_id, filenames in identities.items()
        if len(filenames) >= config.min_images
    )
    selected_ids = _select_identities(eligible_ids, config)
    splits = _split_identities(selected_ids)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, identity_ids in splits.items():
        _write_manifest(config.output_dir / f"{split_name}.csv", identity_ids, identities)

    summary: dict[str, Any] = {
        "seed": config.seed,
        "minimum_images_per_identity": config.min_images,
        "identity_limit": config.limit_identities,
        "total_images": sum(len(filenames) for filenames in identities.values()),
        "total_identities": len(identities),
        "eligible_identities": len(eligible_ids),
        "selected_identities": len(selected_ids),
        "splits": {
            split_name: {
                "identities": len(identity_ids),
                "images": sum(len(identities[identity_id]) for identity_id in identity_ids),
            }
            for split_name, identity_ids in splits.items()
        },
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_paths(annotation_path: Path, image_dir: Path) -> None:
    if not annotation_path.is_file():
        raise DataPreparationError(f"Identity annotations do not exist: {annotation_path}")
    if not image_dir.is_dir():
        raise DataPreparationError(f"Aligned image directory does not exist: {image_dir}")


def _load_identities(annotation_path: Path) -> IdentityImages:
    identities: defaultdict[int, list[str]] = defaultdict(list)

    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line_number, line in enumerate(annotation_file, start=1):
            try:
                filename, raw_identity_id = line.split()
                identity_id = int(raw_identity_id)
            except ValueError as error:
                raise DataPreparationError(
                    f"Invalid identity annotation at line {line_number}"
                ) from error
            identities[identity_id].append(filename)

    if not identities:
        raise DataPreparationError(f"Identity annotation file is empty: {annotation_path}")
    return {identity_id: sorted(filenames) for identity_id, filenames in identities.items()}


def _select_identities(
    eligible_ids: list[int],
    config: PreparationConfig,
) -> list[int]:
    selected_ids = eligible_ids.copy()
    random.Random(config.seed).shuffle(selected_ids)

    if config.limit_identities is not None:
        selected_ids = selected_ids[: config.limit_identities]
    if len(selected_ids) < 10:
        raise DataPreparationError(
            "At least 10 eligible identities are required for non-empty 80/10/10 splits"
        )
    return selected_ids


def _split_identities(identity_ids: list[int]) -> dict[str, list[int]]:
    train_end = int(len(identity_ids) * 0.8)
    validation_end = train_end + int(len(identity_ids) * 0.1)
    return {
        "train": identity_ids[:train_end],
        "validation": identity_ids[train_end:validation_end],
        "test": identity_ids[validation_end:],
    }


def _write_manifest(
    path: Path,
    identity_ids: list[int],
    identities: IdentityImages,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file, lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for class_index, identity_id in enumerate(sorted(identity_ids)):
            for filename in identities[identity_id]:
                writer.writerow((f"{IMAGE_DIRECTORY_NAME}/{filename}", identity_id, class_index))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = PreparationConfig()
    try:
        summary = prepare_celeba(config)
    except (DataPreparationError, OSError) as error:
        LOGGER.error("%s", error)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
