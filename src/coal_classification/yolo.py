from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass
class YoloConfig:
    data_dir: Path
    epochs: int = 25
    img_size: int = 224
    batch_size: int = 32
    output_dir: Path = Path("runs/yolo")
    model_name: str = "yolov8n-cls.pt"


def train_yolo(config: YoloConfig) -> Path:
    model = YOLO(config.model_name)

    results = model.train(
        data=str(config.data_dir),
        epochs=config.epochs,
        imgsz=config.img_size,
        batch=config.batch_size,
        project=str(config.output_dir),
        name="train",
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    return best_path


def evaluate_yolo(weights_path: Path, data_dir: Path) -> dict[str, float]:
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_dir))
    return {
        "top1": float(metrics.top1),
        "top5": float(metrics.top5),
    }

