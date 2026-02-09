from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


@dataclass
class SplitConfig:
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Suma train/val/test musi wynosić 1.0")


def list_images(folder: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in extensions]


def verify_images(paths: Iterable[Path]) -> list[Path]:
    valid: list[Path] = []
    for path in tqdm(list(paths), desc="Sprawdzanie obrazów"):
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(path)
        except Exception:
            continue
    return valid


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    config: SplitConfig,
    verify: bool = True,
) -> None:
    config.validate()
    random.seed(config.seed)

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    class_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"Brak folderów klas w {input_dir}")

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if verify:
            images = verify_images(images)
        if not images:
            continue

        random.shuffle(images)
        total = len(images)
        train_end = int(total * config.train_size)
        val_end = train_end + int(total * config.val_size)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, files in splits.items():
            target_dir = output_dir / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                shutil.copy2(file_path, target_dir / file_path.name)


