from __future__ import annotations

import argparse
from pathlib import Path

from coal_classification.data import SplitConfig, split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Podział danych na train/val/test")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify", action="store_true", help="Pomija weryfikację obrazów")
    args = parser.parse_args()

    train_size = 1.0 - args.val_size - args.test_size
    config = SplitConfig(
        train_size=train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    split_dataset(args.input_dir, args.output_dir, config, verify=not args.no_verify)


if __name__ == "__main__":
    main()
