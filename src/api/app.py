"""
Smart Return Fraud Detector API - Simple Working Version
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import uvicorn

# Create FastAPI app
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
    """
    Predict fraud risk for a return
    """
    try:
        risk_score = 0.0
        features = []
        
        # 1. Return rate
        return_rate = request.total_returns / max(request.total_orders, 1)
        if return_rate > 0.6:
            risk_score += 0.35
            features.append({"feature": "High return rate", "value": return_rate, "impact": 0.35})
        elif return_rate > 0.3:
            risk_score += 0.15
            features.append({"feature": "Moderate return rate", "value": return_rate, "impact": 0.15})
        
        # 2. Price anomaly
        if request.price > 200:
            risk_score += 0.2
            features.append({"feature": "High value item", "value": request.price, "impact": 0.2})
        
        # 3. Price vs average
        price_ratio = request.price / max(request.avg_order_value, 1)
        if price_ratio > 2.5:
            risk_score += 0.15
            features.append({"feature": "Price 2.5x above average", "value": price_ratio, "impact": 0.15})
        
        # 4. Suspicious reason
        suspicious_reasons = ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind']
        if request.return_reason in suspicious_reasons:
            risk_score += 0.2
            features.append({"feature": "Suspicious return reason", "value": request.return_reason, "impact": 0.2})
        
        # 5. Near deadline
        if request.days_since_delivery > 25:
            risk_score += 0.15
            features.append({"feature": "Return near deadline", "value": request.days_since_delivery, "impact": 0.15})
        
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level and prediction
        if risk_score >= 0.7:
            risk_level = "High"
            prediction = 1
        elif risk_score >= 0.3:
            risk_level = "Medium"
            prediction = 1
        else:
            risk_level = "Low"
            prediction = 0
        
        # Sort features by impact
        features = sorted(features, key=lambda x: x['impact'], reverse=True)
        
        return {
            "risk_score": round(risk_score, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "model_used": "rule-based-v1",
            "timestamp": datetime.now().isoformat(),
            "top_features": features[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/stats")
async def model_stats():
    """Get model statistics"""
    return {
        "model_name": "rule-based-v1",
        "version": "1.0.0",
        "type": "rule-based",
        "status": "active",
        "deployed_at": datetime.now().isoformat()
    }

# For Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
