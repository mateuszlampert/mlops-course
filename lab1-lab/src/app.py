from fastapi import FastAPI
from src.api.models.iris import PredictRequest, PredictResponse
from src.inference import load_model, predict as _predict

app = FastAPI()
model = load_model()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = _predict(model, request)
    return PredictResponse(prediction=prediction)
