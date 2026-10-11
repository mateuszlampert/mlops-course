from pydantic import BaseModel, field_validator


class SentimentPredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must be a non-empty string")
        return v


class SentimentPredictResponse(BaseModel):
    prediction: str
