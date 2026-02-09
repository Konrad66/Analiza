from __future__ import annotations

import argparse
from pathlib import Path

from coal_classification.resnet import ResNetConfig, train_resnet


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening ResNet")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet"))
    args = parser.parse_args()

    config = ResNetConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    best_path = train_resnet(config)
    print(f"Najlepszy model zapisano w {best_path}")


if __name__ == "__main__":
    main()
