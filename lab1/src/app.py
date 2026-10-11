from fastapi import FastAPI

from src.api.schemas.sentiment import SentimentPredictRequest, SentimentPredictResponse
from src.inference import load_classifier, load_encoder
from src.inference import predict as _predict

app = FastAPI()

encoder = load_encoder()
clf = load_classifier()


@app.post("/predict")
def predict(request: SentimentPredictRequest) -> SentimentPredictResponse:
    prediction = _predict(encoder=encoder, clf=clf, request=request)
    return SentimentPredictResponse(prediction=prediction)
