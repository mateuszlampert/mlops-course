import json

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_predict_returns_valid_json():
    resp = client.post("/predict", json={"text": "Good course."})

    assert resp.status_code == 200
    body = json.loads(resp.text)
    assert isinstance(body["prediction"], str)
    assert body["prediction"] in ["negative", "neutral", "positive"]


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {"text": None}, {"text": 123}, {"asd": "asd"}, {}],
)
def test_invalid_input_returns_json_error(payload):
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    err = body["detail"][0]
    assert err["loc"][0] == "body"
    assert err["msg"]
