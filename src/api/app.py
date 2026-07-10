"""
Smart Return Fraud Detector API - Minimal Version
"""

from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="Smart Return Fraud Detector API",
    description="ML-based fraud detection for e-commerce returns",
    version="1.0.0"
)

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

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "fraud-detector-api",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: dict):
    """Predict fraud risk for a return"""
    
    risk_score = 0.0
    
    # Simple rule-based prediction
    total_returns = request.get('total_returns', 0)
    total_orders = request.get('total_orders', 1)
    return_rate = total_returns / max(total_orders, 1)
    
    if return_rate > 0.6:
        risk_score += 0.35
    
    price = request.get('price', 0)
    if price > 200:
        risk_score += 0.2
    
    reason = request.get('return_reason', '')
    suspicious = ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind']
    if reason in suspicious:
        risk_score += 0.2
    
    days = request.get('days_since_delivery', 0)
    if days > 25:
        risk_score += 0.15
    
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
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/stats")
async def model_stats():
    return {
        "model_name": "rule-based-v1",
        "version": "1.0.0",
        "status": "active",
        "deployed_at": datetime.now().isoformat()
    }
