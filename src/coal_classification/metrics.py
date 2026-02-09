from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class MetricsResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    report: str


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], labels: list[str]) -> MetricsResult:
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    return MetricsResult(accuracy, precision, recall, f1, report)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predykcja")
    plt.ylabel("Prawda")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

