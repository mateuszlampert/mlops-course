import pytest
from pydantic import ValidationError
from src.api.schemas.sentiment_predict import SentimentPredictRequest


def test_valid_text():
    req = SentimentPredictRequest(text="asd asd asd")
    assert req.text == "asd asd asd"


@pytest.mark.parametrize("input", ["", "   ", "\n"])
def test_empty_text_rejected(input):
    with pytest.raises(ValidationError):
        SentimentPredictRequest(text=input)


@pytest.mark.parametrize("input", [None, 123, 4.5, ["a"]])
def test_non_string_rejected(input):
    with pytest.raises(ValidationError):
        SentimentPredictRequest(text=input)


def test_missing_text_rejected():
    with pytest.raises(ValidationError):
        SentimentPredictRequest()  # type: ignore
