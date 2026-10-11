import pytest
from sentence_transformers import SentenceTransformer
from src.api.schemas.sentiment_predict import SentimentPredictRequest
from src.inference import predict

SAMPLES = [
    "This MLOPs course is great!",
    "I don't like this MLOPs course.",
    "This MLOPs course is okay, nothing special.",
]


def test_classifier_loads_and_has_predict(classifier):
    assert callable(getattr(classifier, "predict", None))


def test_encoder_loads(encoder):
    assert isinstance(encoder, SentenceTransformer)


@pytest.mark.parametrize("text", SAMPLES)
def test_predict_returns_label_and_confidence(encoder, classifier, text):
    result = predict(encoder, classifier, SentimentPredictRequest(text=text))

    assert result in ["negative", "neutral", "positive"]
