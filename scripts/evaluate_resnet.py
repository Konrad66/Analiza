from __future__ import annotations

import argparse
from pathlib import Path

from coal_classification.metrics import plot_confusion_matrix
from coal_classification.resnet import evaluate_resnet


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja modelu ResNet")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resnet-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resnet_metrics = evaluate_resnet(args.resnet_ckpt, args.data_dir)
    with open(output_dir / "resnet_report.txt", "w", encoding="utf-8") as f:
        f.write(resnet_metrics["report"])

    metrics_path = output_dir / "resnet_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {resnet_metrics['accuracy']:.4f}\n")
        f.write(f"precision: {resnet_metrics['precision']:.4f}\n")
        f.write(f"recall: {resnet_metrics['recall']:.4f}\n")
        f.write(f"f1: {resnet_metrics['f1']:.4f}\n")

    cm_path = output_dir / "resnet_confusion_matrix.png"
    plot_confusion_matrix(
        resnet_metrics["y_true"],
        resnet_metrics["y_pred"],
        resnet_metrics["labels"],
        cm_path,
    )

    print("ResNet metrics:")
    print(resnet_metrics["report"])
    print(f"accuracy: {resnet_metrics['accuracy']:.4f}")
    print(f"precision: {resnet_metrics['precision']:.4f}")
    print(f"recall: {resnet_metrics['recall']:.4f}")
    print(f"f1: {resnet_metrics['f1']:.4f}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
