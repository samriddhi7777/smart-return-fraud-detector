from fastapi import FastAPI
from datetime import datetime
import uvicorn
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Fraud API is live!", "time": datetime.now().isoformat()}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.post("/predict")
def predict(data: dict):
    return {"risk_score": 0.5, "prediction": 0}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
