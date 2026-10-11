from fastapi import FastAPI

from src.schemas.sentiment import SentimentPredictRequest, SentimentPredictResponse

app = FastAPI()


@app.post("/predict")
def predict(request: SentimentPredictRequest) -> SentimentPredictResponse:
    return SentimentPredictResponse(prediction="positive")
