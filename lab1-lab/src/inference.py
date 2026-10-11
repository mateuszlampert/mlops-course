from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from src.api.models.iris import PredictRequest
from src.constants import DEFAULT_MODEL_PATH

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_model(path: Path = DEFAULT_MODEL_PATH) -> RandomForestClassifier:
    model = joblib.load(path)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError(
            f"Loaded model is not a RandomForestClassifier, got {type(model)}"
        )
    return model


def predict(model: RandomForestClassifier, request: PredictRequest):
    x = np.array(
        [
            [
                request.sepal_length,
                request.sepal_width,
                request.petal_length,
                request.petal_width,
            ]
        ]
    )
    class_idx = int(model.predict(x)[0])
    return CLASS_NAMES[class_idx]
