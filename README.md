# Automatyczna klasyfikacja rodzajów węgla z użyciem YOLO i ResNet

Projekt demonstracyjny do pracy dyplomowej: **„Zastosowanie modeli YOLO i ResNet w automatycznej klasyfikacji rodzajów węgla”**.

Repozytorium zawiera kompletne skrypty do:
- przygotowania danych z Kaggle,
- trenowania klasyfikatora ResNet (PyTorch),
- trenowania YOLOv8 w trybie klasyfikacji (Ultralytics),
- oceny jakości modeli oraz generowania raportów.

> Zbiór danych: https://www.kaggle.com/datasets/pattnaiksatyajit/coal-classification

## Struktura katalogów

```
.
├── configs/               # przykładowe konfiguracje
├── data/
│   ├── raw/               # pobrane dane (np. z Kaggle)
│   └── processed/         # dane po podziale na train/val/test
├── scripts/               # skrypty CLI
└── src/coal_classification
    ├── __init__.py
    ├── data.py            # przygotowanie i podział danych
    ├── metrics.py         # metryki i raporty
    ├── resnet.py          # trening i ewaluacja ResNet
    └── yolo.py            # trening i ewaluacja YOLO
```

## Wymagania

- Python 3.10+
- NVIDIA GPU (opcjonalnie, ale zalecane)

Zainstaluj zależności:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Przygotowanie danych

1. Pobierz dane z Kaggle i rozpakuj do `data/raw/coal-classification`.
2. Uruchom podział na zbiory:

```bash
python scripts/prepare_data.py \
  --input-dir data/raw/coal-classification \
  --output-dir data/processed/coal-classification \
  --val-size 0.15 \
  --test-size 0.15
```

Skrypt tworzy strukturę zgodną z klasycznym image classification:

```
processed/coal-classification/
├── train/
├── val/
└── test/
    ├── Anthracite/
    ├── Bituminous/
    ├── Lignite/
    └── SubBituminous/
```

## Trening ResNet

```bash
python scripts/train_resnet.py \
  --data-dir data/processed/coal-classification \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir runs/resnet
```

## Trening YOLO (klasyfikacja)

Ultralytics YOLO wspiera tryb `classify`.

```bash
python scripts/train_yolo.py \
  --data-dir data/processed/coal-classification \
  --epochs 25 \
  --img-size 224 \
  --output-dir runs/yolo
```

## Ewaluacja

```bash
python scripts/evaluate.py \
  --data-dir data/processed/coal-classification/test \
  --resnet-ckpt runs/resnet/best.pt \
  --yolo-weights runs/yolo/weights/best.pt
```

## Notatki do pracy

- ResNet: klasyczny pipeline klasyfikacji obrazu z PyTorch.
- YOLO: tryb `classify` w Ultralytics umożliwia bezpośrednie porównanie z ResNet.
- Raporty: accuracy, precision, recall, F1-score, confusion matrix.

## Cytowanie

W pracy dyplomowej uwzględnij źródła:
- Kaggle dataset: S. Pattnaik, *Coal Classification Dataset*.
- Ultralytics YOLOv8
- PyTorch / Torchvision
