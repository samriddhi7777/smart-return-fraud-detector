"""
Smart Return Fraud Detector API - Minimal Working Version
"""

from fastapi import FastAPI
from datetime import datetime
import os

app = FastAPI(title="Fraud Detection API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Smart Return Fraud Detector API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict(data: dict):
    # Simple rule-based prediction
    risk_score = 0.5
    return {
        "risk_score": risk_score,
        "prediction": 1 if risk_score > 0.5 else 0,
        "risk_level": "Medium",
        "model_used": "rule-based-v1",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
