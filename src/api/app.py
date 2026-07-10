"""
Smart Return Fraud Detector API
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from datetime import datetime

app = FastAPI(
    title="Smart Return Fraud Detector API",
    description="ML-based fraud detection for e-commerce returns",
    version="1.0.0"
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Smart Return Fraud Detector API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/predict": "POST - Predict fraud risk",
            "/docs": "GET - API Documentation"
        }
    }

# Health check
@app.get("/health")
async def health():
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
async def predict(request: PredictionRequest):
    """Predict fraud risk for a return"""
    
    risk_score = 0.0
    features = []
    
    # Return rate
    return_rate = request.total_returns / max(request.total_orders, 1)
    if return_rate > 0.6:
        risk_score += 0.35
        features.append({"feature": "High return rate", "value": return_rate, "impact": 0.35})
    
    # Price
    if request.price > 200:
        risk_score += 0.2
        features.append({"feature": "High value item", "value": request.price, "impact": 0.2})
    
    # Suspicious reason
    suspicious = ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind']
    if request.return_reason in suspicious:
        risk_score += 0.2
        features.append({"feature": "Suspicious reason", "value": request.return_reason, "impact": 0.2})
    
    # Near deadline
    if request.days_since_delivery > 25:
        risk_score += 0.15
        features.append({"feature": "Near deadline", "value": request.days_since_delivery, "impact": 0.15})
    
    risk_score = min(risk_score, 1.0)
    
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
        "risk_score": round(risk_score, 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "model_used": "rule-based-v1",
        "timestamp": datetime.now().isoformat(),
        "top_features": features[:5]
    }

@app.get("/model/stats")
async def model_stats():
    return {
        "model_name": "rule-based-v1",
        "version": "1.0.0",
        "status": "active",
        "deployed_at": datetime.now().isoformat()
    }

# Required for Render
app = app
