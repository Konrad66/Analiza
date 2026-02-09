from __future__ import annotations

import argparse
from pathlib import Path

from coal_classification.yolo import YoloConfig, train_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening YOLO (klasyfikacja)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt")
    args = parser.parse_args()

    config = YoloConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        model_name=args.model,
    )
    best_path = train_yolo(config)
    print(f"Najlepszy model zapisano w {best_path}")


if __name__ == "__main__":
    main()
