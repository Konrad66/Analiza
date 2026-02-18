from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from .metrics import compute_metrics


@dataclass
class ResNetConfig:
    data_dir: Path
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_workers: int = 4
    output_dir: Path = Path("runs/resnet")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloaders(config: ResNetConfig) -> tuple[dict[str, DataLoader], list[str]]:
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(config.data_dir / x, data_transforms[x])
        for x in ["train", "val"]
    }
    class_names = image_datasets["train"].classes

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        for x in ["train", "val"]
    }
    return dataloaders, class_names


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_resnet(config: ResNetConfig) -> Path:
    dataloaders, class_names = build_dataloaders(config)
    device = torch.device(config.device)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_acc = 0.0
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"

    for epoch in range(config.epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} {epoch+1}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / max(total_samples, 1)
            epoch_acc = running_corrects / max(total_samples, 1)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({"model_state": model.state_dict(), "classes": class_names}, best_path)

            print(f"Epoch {epoch+1}/{config.epochs} {phase} loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

    return best_path


def evaluate_resnet(model_path: Path, data_dir: Path) -> dict[str, float | str | list[int] | list[str]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Ewaluacja ResNet"):
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    metrics = compute_metrics(y_true, y_pred, labels=classes)
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "report": metrics.report,
        "y_true": y_true,
        "y_pred": y_pred,
        "labels": classes,
    }

