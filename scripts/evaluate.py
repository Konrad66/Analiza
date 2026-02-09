from __future__ import annotations

import argparse
from pathlib import Path

from coal_classification.metrics import plot_confusion_matrix
from coal_classification.resnet import evaluate_resnet
from coal_classification.yolo import evaluate_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja modeli ResNet i YOLO")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resnet-ckpt", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resnet_metrics = evaluate_resnet(args.resnet_ckpt, args.data_dir)
    with open(output_dir / "resnet_report.txt", "w", encoding="utf-8") as f:
        f.write(resnet_metrics["report"])
    print("ResNet metrics:")
    print(resnet_metrics["report"])

    yolo_metrics = evaluate_yolo(args.yolo_weights, args.data_dir)
    print("YOLO metrics:")
    print(yolo_metrics)

    # Confusion matrix for ResNet
    if "report" in resnet_metrics:
        # Placeholder: confusion matrix requires true/pred labels; extend if needed.
        pass


if __name__ == "__main__":
    main()
