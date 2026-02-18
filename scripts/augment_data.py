from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from coal_classification.data import list_images


def build_transform(output_size: int, resnet_only: bool) -> transforms.Compose:
    ops: list[transforms.Transform] = [
        transforms.RandomResizedCrop(output_size if output_size > 0 else 224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if not resnet_only:
        ops.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            ]
        )

    if output_size > 0:
        ops.append(transforms.Resize((output_size, output_size)))

    return transforms.Compose(ops)


def save_augmented(
    image_path: Path,
    output_dir: Path,
    transform: transforms.Compose,
    copies: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            for idx in range(copies):
                aug = transform(img)
                out_name = f"{image_path.stem}_aug{idx + 1}{image_path.suffix.lower()}"
                aug.save(output_dir / out_name)
                created += 1
    except Exception:
        return 0

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline augmentacja danych (tworzy nowe obrazy)")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=2, help="Ile augmentacji na obraz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-size",
        type=int,
        default=0,
        help="Rozmiar wyjscia (0 = bez wymuszania, np. 224)",
    )
    parser.add_argument(
        "--include-originals",
        action="store_true",
        help="Kopiuje oryginalne obrazy do output-dir",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Zapisuje augmentacje w tym samym katalogu co wejscie (nadpisywanie nie wystepuje)",
    )
    parser.add_argument(
        "--resnet-only",
        action="store_true",
        help="Uzywa tylko augmentacji z ResNet (RandomResizedCrop + HorizontalFlip)",
    )
    args = parser.parse_args()

    if args.copies <= 0:
        raise ValueError("--copies musi byc > 0")

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except Exception:
        pass

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = input_dir if args.in_place else args.output_dir.expanduser().resolve()

    class_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"Brak folderow klas w {input_dir}")

    transform = build_transform(args.output_size, args.resnet_only)

    total_created = 0
    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            continue
        class_out = output_dir / class_dir.name
        if args.include_originals and not args.in_place:
            class_out.mkdir(parents=True, exist_ok=True)
            for image_path in images:
                shutil.copy2(image_path, class_out / image_path.name)
        for image_path in tqdm(images, desc=f"Augmentacja {class_dir.name}"):
            total_created += save_augmented(image_path, class_out, transform, args.copies)

    print(f"Utworzono {total_created} obrazow w {output_dir}")


if __name__ == "__main__":
    main()
