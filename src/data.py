"""Prepare deterministic, identity-disjoint CelebA manifests."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import CELEBA_MANIFESTS_DIR, CELEBA_RAW_DIR

LOGGER = logging.getLogger(__name__)

IDENTITY_ANNOTATION_FILENAME = "identity_CelebA.txt"
IMAGE_DIRECTORY_NAME = "img_align_celeba"
MANIFEST_COLUMNS = ("image_path", "identity_id", "class_index")

type IdentityImages = dict[int, list[str]]
type PreparationSummary = dict[str, Any]


class DataPreparationError(ValueError):
    """Raised when CelebA input or preparation configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    dataset_root: Path = CELEBA_RAW_DIR
    output_dir: Path = CELEBA_MANIFESTS_DIR
    seed: int = 24
    min_images: int = 3
    limit_identities: int | None = None


def prepare_celeba(config: PreparationConfig) -> PreparationSummary:
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

    summary: PreparationSummary = {
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
