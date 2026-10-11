import argparse

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

from constants import DEFAULT_MODEL_PATH


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_iris(return_X_y=True)
    return np.asarray(X), np.asarray(y)


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def save_model(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y = load_data()
    model = train_model(X, y)
    save_model(model, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
