"""
Smart Return Fraud Detector API - Production Ready
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import random
from datetime import datetime

app = FastAPI(
    title="Smart Return Fraud Detector API",
    description="ML-based fraud detection for e-commerce returns",
    version="1.0.0"
)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Smart Return Fraud Detector API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - This info",
            "/health": "GET - Health check",
            "/predict": "POST - Predict fraud",
            "/docs": "GET - API Documentation"
        }
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fraud-detector-api",
        "timestamp": datetime.now().isoformat()
    }

# Prediction model
class PredictionRequest(BaseModel):
    account_age_days: float
    total_orders: float
    avg_order_value: float
    total_returns: float
    price: float
    days_since_delivery: float
    payment_method: str
    return_reason: str
    item_category: str
    return_method: str

@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict fraud risk for a return"""
    
    # Calculate risk score
    risk_score = 0.0
    
    # Rule 1: Return rate
    return_rate = request.total_returns / max(request.total_orders, 1)
    if return_rate > 0.6:
        risk_score += 0.4
    
    # Rule 2: High value
    if request.price > 200:
        risk_score += 0.2
    
    # Rule 3: Suspicious reason
    suspicious = ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind']
    if request.return_reason in suspicious:
        risk_score += 0.2
    
    # Rule 4: Near deadline
    if request.days_since_delivery > 25:
        risk_score += 0.2
    
    # Rule 5: Price anomaly
    if request.price > request.avg_order_value * 2.5:
        risk_score += 0.1
    
    risk_score = min(risk_score, 1.0)
    
    # Determine risk level
    if risk_score >= 0.7:
        risk_level = "High"
        prediction = 1
    elif risk_score >= 0.3:
        risk_level = "Medium"
        prediction = 1
    else:
        risk_level = "Low"
        prediction = 0
    
    return {
        "risk_score": risk_score,
        "prediction": prediction,
        "risk_level": risk_level,
        "model_used": "rule-based",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/stats")
def model_stats():
    return {
        "model_name": "rule-based",
        "version": "1.0.0",
        "status": "active"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
