from fastapi import FastAPI
from datetime import datetime
import os
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Fraud API is running", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
def predict(data: dict):
    return {"risk_score": 0.75, "prediction": 1, "message": "Prediction received"}
