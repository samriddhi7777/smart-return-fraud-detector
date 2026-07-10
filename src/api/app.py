"""
Smart Return Fraud Detector - FastAPI Service
REST API for fraud detection with SHAP explanations
"""

import pandas as pd
import numpy as np
import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(
    title="Smart Return Fraud Detector API",
    description="ML-based fraud detection for e-commerce returns with SHAP explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load models and data
MODEL_PATH = "models"
MODELS = {}
FEATURES = None

def load_models():
    """Load all trained models"""
    global MODELS, FEATURES
    
    try:
        # Load models
        model_files = ['xgboost.pkl', 'lightgbm.pkl', 'logistic_regression.pkl']
        for model_file in model_files:
            model_path = os.path.join(MODEL_PATH, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '')
                MODELS[model_name] = joblib.load(model_path)
                print(f"✓ Loaded model: {model_name}")
        
        # Load feature names for validation
        features_path = "data/processed/features.csv"
        if os.path.exists(features_path):
            FEATURES = pd.read_csv(features_path).columns.tolist()
            print(f"✓ Loaded {len(FEATURES)} features")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load models on startup
load_models()

# Pydantic models for request/response
class ReturnPredictionRequest(BaseModel):
    """Request model for single prediction"""
    account_age_days: float = Field(..., description="Customer account age in days")
    total_orders: float = Field(..., description="Total number of orders")
    avg_order_value: float = Field(..., description="Average order value in USD")
    total_returns: float = Field(..., description="Total number of returns")
    price: float = Field(..., description="Item price in USD")
    days_since_delivery: float = Field(..., description="Days between delivery and return request", ge=0, le=30)
    payment_method: str = Field(..., description="Payment method used")
    return_reason: str = Field(..., description="Reason for return")
    item_category: str = Field(..., description="Category of the item")
    return_method: str = Field(..., description="Return method (refund/replacement)")
    
    class Config:
        schema_extra = {
            "example": {
                "account_age_days": 180,
                "total_orders": 10,
                "avg_order_value": 150.0,
                "total_returns": 3,
                "price": 200.0,
                "days_since_delivery": 5,
                "payment_method": "credit_card",
                "return_reason": "Did not like it",
                "item_category": "Electronics",
                "return_method": "refund"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    risk_score: float = Field(..., description="Fraud probability (0-1)")
    prediction: int = Field(..., description="Prediction (0=legitimate, 1=fraud)")
    risk_level: str = Field(..., description="Risk level: Low/Medium/High")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    model_used: str = Field(..., description="Model that made the prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    returns: List[ReturnPredictionRequest] = Field(..., description="List of returns to predict")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of predictions")
    fraud_count: int = Field(..., description="Number of predicted fraud cases")
    fraud_rate: float = Field(..., description="Fraud rate among predictions")

class ModelStatsResponse(BaseModel):
    """Response model for model statistics"""
    model_name: str
    features_count: int
    is_trained: bool

# Helper functions
def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Calculate derived features
    df['fraudulent_returns'] = df['total_returns'] * 0.3
    df['fraud_rate'] = df['total_returns'] / df['total_orders'].clip(1)
    df['recent_returns_90d'] = df['total_returns'].clip(0, 5)
    df['avg_days_to_return'] = 10
    df['return_rate'] = df['total_returns'] / df['total_orders'].clip(1)
    df['high_return_rate'] = (df['return_rate'] > 0.6).astype(int)
    df['order_frequency'] = df['total_orders'] / (df['account_age_days'] / 30 + 1)
    df['quantity'] = 1
    df['price_to_avg_ratio'] = df['price'] / df['avg_order_value'].clip(1)
    df['price_anomaly'] = (df['price_to_avg_ratio'] > 2.5).astype(int)
    
    # Category risk
    category_risk = {
        'Electronics': 0.08, 'Formalwear': 0.15, 'Jewelry': 0.10,
        'Home Goods': 0.04, 'Books': 0.02, 'Beauty': 0.03,
        'Toys': 0.03, 'Sporting Goods': 0.05, 'Footwear': 0.07,
        'Handbags': 0.12
    }
    df['category_risk'] = df['item_category'].map(category_risk).fillna(0.05)
    
    # Payment risk
    payment_risk = {'credit_card': 0.3, 'debit_card': 0.2, 'paypal': 0.1, 'store_credit': 0.15}
    df['payment_risk'] = df['payment_method'].map(payment_risk).fillna(0.2)
    
    df['high_value'] = (df['price'] > 200).astype(int)
    df['order_risk_score'] = (
        df['category_risk'] * 0.5 +
        df['price_to_avg_ratio'].clip(0, 5) / 10 * 0.3 +
        df['payment_risk'] * 0.2
    )
    
    # Behavioral features
    df['return_day_of_week'] = 3  # Mid-week default
    df['return_month'] = 6  # June default
    df['return_hour'] = 12  # Noon default
    df['near_deadline'] = (df['days_since_delivery'] > 25).astype(int)
    
    suspicious_reasons = ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind']
    df['suspicious_reason'] = df['return_reason'].isin(suspicious_reasons).astype(int)
    df['is_replacement'] = (df['return_method'] == 'replacement').astype(int)
    df['returns_last_30d'] = df['total_returns'].clip(0, 3)
    df['multi_return_burst'] = (df['returns_last_30d'] > 2).astype(int)
    df['shared_address'] = 0
    
    # Encode categorical variables
    # Map categorical values to numbers (simplified)
    category_map = {
        'Electronics': 0, 'Formalwear': 1, 'Jewelry': 2, 'Home Goods': 3,
        'Books': 4, 'Beauty': 5, 'Toys': 6, 'Sporting Goods': 7, 
        'Footwear': 8, 'Handbags': 9
    }
    df['item_category'] = df['item_category'].map(category_map).fillna(0)
    
    payment_map = {'credit_card': 0, 'debit_card': 1, 'paypal': 2, 'store_credit': 3}
    df['payment_method'] = df['payment_method'].map(payment_map).fillna(0)
    
    reason_map = {
        'Defective product': 0, 'Wrong item sent': 1, 'Damaged during shipping': 2,
        'Not as described': 3, 'Size issue': 4, 'Did not like it': 5,
        'No longer needed': 6, 'Found better price': 7, 'Changed mind': 8
    }
    df['return_reason'] = df['return_reason'].map(reason_map).fillna(0)
    df['return_method'] = (df['return_method'] == 'refund').astype(int)
    
    # Ensure all columns exist
    if FEATURES:
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURES]
    
    return df

def get_risk_level(score: float) -> str:
    """Get risk level from score"""
    if score >= 0.7:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"

def get_top_features(df: pd.DataFrame, model, n: int = 5) -> List[Dict[str, Any]]:
    """Get top contributing features for prediction"""
    try:
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = df.columns
            
            # Get top features
            top_idx = np.argsort(importances)[-n:][::-1]
            features = []
            for idx in top_idx:
                features.append({
                    "feature": feature_names[idx],
                    "importance": float(importances[idx]),
                    "value": float(df.iloc[0, idx])
                })
            return features
    except:
        pass
    
    # Fallback: return dummy features
    return [
        {"feature": "price", "importance": 0.3, "value": float(df['price'].iloc[0])},
        {"feature": "return_rate", "importance": 0.25, "value": float(df['return_rate'].iloc[0])},
        {"feature": "suspicious_reason", "importance": 0.2, "value": float(df['suspicious_reason'].iloc[0])}
    ]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Return Fraud Detector API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict fraud for a single return",
            "/predict/batch": "POST - Predict fraud for multiple returns",
            "/model/stats": "GET - Get model statistics",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - API documentation (ReDoc)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS) > 0,
        "models": list(MODELS.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/stats")
async def get_model_stats():
    """Get model statistics"""
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    stats = []
    for model_name, model in MODELS.items():
        stats.append({
            "model_name": model_name,
            "features_count": len(FEATURES) if FEATURES else 0,
            "is_trained": True,
            "type": str(type(model).__name__)
        })
    
    return {
        "models": stats,
        "total_models": len(MODELS),
        "features_count": len(FEATURES) if FEATURES else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReturnPredictionRequest):
    """
    Predict fraud risk for a single return
    
    Returns risk score, prediction, and top contributing features
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded. Please train models first.")
    
    try:
        # Convert request to dict
        data = request.dict()
        
        # Preprocess
        df = preprocess_input(data)
        
        # Use XGBoost by default
        model = MODELS.get('xgboost')
        if model is None:
            model = list(MODELS.values())[0]
            model_name = list(MODELS.keys())[0]
        else:
            model_name = 'xgboost'
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0][1]
            pred = 1 if proba >= 0.5 else 0
        else:
            pred = model.predict(df)[0]
            proba = pred
        
        # Get risk level
        risk_level = get_risk_level(proba)
        
        # Get top features
        top_features = get_top_features(df, model)
        
        return PredictionResponse(
            risk_score=float(proba),
            prediction=int(pred),
            risk_level=risk_level,
            top_features=top_features,
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud risk for multiple returns
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded. Please train models first.")
    
    try:
        predictions = []
        fraud_count = 0
        
        for return_data in request.returns:
            # Get prediction for each return
            data = return_data.dict()
            df = preprocess_input(data)
            
            # Use XGBoost
            model = MODELS.get('xgboost', list(MODELS.values())[0])
            model_name = 'xgboost' if 'xgboost' in MODELS else list(MODELS.keys())[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)[0][1]
                pred = 1 if proba >= 0.5 else 0
            else:
                pred = model.predict(df)[0]
                proba = pred
            
            risk_level = get_risk_level(proba)
            top_features = get_top_features(df, model)
            
            if pred == 1:
                fraud_count += 1
            
            predictions.append(PredictionResponse(
                risk_score=float(proba),
                prediction=int(pred),
                risk_level=risk_level,
                top_features=top_features,
                model_used=model_name,
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            fraud_count=fraud_count,
            fraud_rate=fraud_count / len(predictions) if predictions else 0
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/features")
async def get_features():
    """Get list of features used by the model"""
    return {
        "features": FEATURES if FEATURES else [],
        "count": len(FEATURES) if FEATURES else 0
    }
@app.post("/explain")
async def get_explanation(request: ReturnPredictionRequest):
    """
    Get SHAP explanation for a prediction
    Returns top features contributing to the prediction
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded. Please train models first.")
    
    try:
        # Load SHAP explainer (use XGBoost by default)
        shap_explainer = joblib.load('models/shap_xgboost.pkl')
        model = MODELS.get('xgboost')
        
        if model is None:
            model = list(MODELS.values())[0]
        
        # Preprocess input
        data = request.dict()
        df = preprocess_input(data)
        
        # Get SHAP values
        shap_values = shap_explainer.shap_values(df)
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0][1]
            pred = 1 if proba >= 0.5 else 0
        else:
            pred = model.predict(df)[0]
            proba = pred
        
        # Format explanation
        explanation = []
        for i, feature in enumerate(df.columns):
            explanation.append({
                "feature": feature,
                "value": float(df.iloc[0, i]),
                "shap_value": float(shap_values[0][i]),
                "impact": "positive" if shap_values[0][i] > 0 else "negative"
            })
        
        # Sort by absolute SHAP value (most important first)
        explanation = sorted(explanation, key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            "prediction": int(pred),
            "risk_score": float(proba),
            "risk_level": get_risk_level(proba),
            "top_features": explanation[:10],
            "total_features": len(explanation),
            "model_used": "xgboost",
            "timestamp": datetime.now().isoformat()
        }
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="SHAP explainer not found. Please run training first: python src/models/train.py"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )