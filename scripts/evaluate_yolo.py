from __future__ import annotations

import argparse
from pathlib import Path

from coal_classification.yolo import evaluate_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja modelu YOLO")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_metrics = evaluate_yolo(args.yolo_weights, args.data_dir)
    report_path = output_dir / "yolo_metrics.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"top1: {yolo_metrics['top1']:.4f}\n")
        f.write(f"top5: {yolo_metrics['top5']:.4f}\n")

    print("YOLO metrics:")
    print(f"top1: {yolo_metrics['top1']:.4f}")
    print(f"top5: {yolo_metrics['top5']:.4f}")


if __name__ == "__main__":
    main()
