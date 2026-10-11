from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from src.api.schemas.sentiment import SentimentPredictRequest

MODELS_DIR = Path("src") / "model"
ENCODER_PATH = MODELS_DIR / "sentence_transformer.model"
CLF_PATH = MODELS_DIR / "classifier.joblib"

CLASS_LABELS = ["negative", "neutral", "positive"]


@runtime_checkable
class Classifier(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def load_classifier(path: Path = CLF_PATH) -> Classifier:
    clf = joblib.load(path)
    if not isinstance(clf, Classifier):
        raise TypeError(f"Loaded model is not a Classifier, got {type(clf)}")

    return clf


def load_encoder(path: Path = ENCODER_PATH) -> SentenceTransformer:
    encoder = SentenceTransformer(model_name_or_path=str(path))
    return encoder


def predict(
    encoder: SentenceTransformer, clf: Classifier, request: SentimentPredictRequest
):
    embeddings = encoder.encode([request.text])
    class_idx = int(clf.predict(embeddings)[0])
    return CLASS_LABELS[class_idx]
