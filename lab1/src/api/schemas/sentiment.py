from pydantic import BaseModel


class SentimentPredictRequest(BaseModel):
    text: str


class SentimentPredictResponse(BaseModel):
    prediction: str
