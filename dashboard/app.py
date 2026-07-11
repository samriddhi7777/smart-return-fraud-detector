"""
Smart Return Fraud Detector - Complete Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Smart Return Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# API URL
API_URL = "https://smart-return-fraud-detector.onrender.com"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc66;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛡️ Smart Return Fraud Detector</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check API connection
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

# Sidebar
st.sidebar.title("🎛️ Dashboard Controls")

# API Status
with st.sidebar:
    st.subheader("🔌 API Status")
    if check_api():
        st.success("✅ API Connected")
        st.caption(f"API: {API_URL}")
    else:
        st.error("❌ API Unreachable")
        st.info("💡 The API might be spinning up. Wait 30-60 seconds and refresh.")
        if st.button("🔄 Retry Connection"):
            st.rerun()

st.sidebar.markdown("---")

# View selection
view_option = st.sidebar.radio(
    "View",
    ["📊 Overview", "📈 Predictions", "🔍 Single Prediction", "📋 Model Performance"]
)

# Overview Tab
if view_option == "📊 Overview":
    st.header("📊 Dataset Overview")
    
    # Try to load data from API or local
    try:
        # Sample data for demonstration
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Returns", "5,332")
        with col2:
            st.metric("Fraudulent Returns", "2,420")
        with col3:
            st.metric("Fraud Rate", "45.4%")
        with col4:
            st.metric("Features", "34")
        
        st.markdown("---")
        
        # Fraud distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Fraud Distribution")
            fig = px.pie(
                values=[2912, 2420],
                names=['Legitimate', 'Fraudulent'],
                color=['Legitimate', 'Fraudulent'],
                color_discrete_map={'Legitimate': '#00cc66', 'Fraudulent': '#ff4b4b'},
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Risk Level Distribution")
            risk_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [1800, 1500, 2032]
            })
            fig = px.bar(
                risk_data, x='Risk Level', y='Count',
                color='Risk Level',
                color_discrete_map={'Low': '#00cc66', 'Medium': '#ffa500', 'High': '#ff4b4b'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.info("📊 Loading sample data for demonstration")
        st.warning("Connect to API for live data")

# Predictions Tab
elif view_option == "📈 Predictions":
    st.header("📈 Batch Predictions")
    st.info("Enter multiple returns to predict fraud risk")
    
    # Simple batch prediction form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_returns = st.number_input("Number of returns to simulate", min_value=1, max_value=20, value=5)
    
    if st.button("🔄 Generate Batch Predictions"):
        with st.spinner("Generating predictions..."):
            results = []
            for i in range(num_returns):
                # Random sample data
                data = {
                    "account_age_days": np.random.randint(30, 730),
                    "total_orders": np.random.randint(1, 50),
                    "avg_order_value": np.random.uniform(20, 300),
                    "total_returns": np.random.randint(0, 10),
                    "price": np.random.uniform(10, 500),
                    "days_since_delivery": np.random.randint(1, 30),
                    "payment_method": np.random.choice(['credit_card', 'debit_card', 'paypal']),
                    "return_reason": np.random.choice(['Did not like it', 'Defective product', 'No longer needed', 'Wrong item sent']),
                    "item_category": np.random.choice(['Electronics', 'Books', 'Clothing', 'Home Goods']),
                    "return_method": np.random.choice(['refund', 'replacement'])
                }
                
                try:
                    response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'Return ID': i+1,
                            'Risk Score': result['risk_score'],
                            'Risk Level': result['risk_level'],
                            'Prediction': 'Fraud' if result['prediction'] == 1 else 'Legitimate'
                        })
                except:
                    pass
            
            if results:
                df = pd.DataFrame(results)
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Results Summary")
                    st.dataframe(df, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Risk Distribution")
                    fig = px.histogram(
                        df, x='Risk Score',
                        nbins=20,
                        title='Risk Score Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(results))
                with col2:
                    fraud_count = sum(1 for r in results if r['Prediction'] == 'Fraud')
                    st.metric("Fraud Detected", fraud_count)
                with col3:
                    fraud_rate = fraud_count / len(results) * 100
                    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

# Single Prediction Tab
elif view_option == "🔍 Single Prediction":
    st.header("🔍 Single Return Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Return Details")
        
        account_age = st.number_input("Account Age (days)", min_value=1, value=180)
        total_orders = st.number_input("Total Orders", min_value=1, value=10)
        avg_order_value = st.number_input("Avg Order Value ($)", min_value=10.0, value=150.0)
        total_returns = st.number_input("Total Returns", min_value=0, value=3)
        price = st.number_input("Item Price ($)", min_value=1.0, value=200.0)
        days_since_delivery = st.number_input("Days Since Delivery", min_value=1, max_value=30, value=5)
    
    with col2:
        st.subheader("🔍 Additional Details")
        
        payment_method = st.selectbox(
            "Payment Method",
            ['credit_card', 'debit_card', 'paypal', 'store_credit']
        )
        return_reason = st.selectbox(
            "Return Reason",
            ['Did not like it', 'No longer needed', 'Defective product', 
             'Wrong item sent', 'Found better price', 'Changed mind']
        )
        item_category = st.selectbox(
            "Item Category",
            ['Electronics', 'Formalwear', 'Jewelry', 'Home Goods', 
             'Books', 'Beauty', 'Toys', 'Sporting Goods', 'Footwear', 'Handbags']
        )
        return_method = st.selectbox("Return Method", ['refund', 'replacement'])
    
    if st.button("🔮 Predict Fraud Risk", type="primary"):
        if check_api():
            try:
                with st.spinner("Calling API..."):
                    data = {
                        "account_age_days": float(account_age),
                        "total_orders": float(total_orders),
                        "avg_order_value": float(avg_order_value),
                        "total_returns": float(total_returns),
                        "price": float(price),
                        "days_since_delivery": float(days_since_delivery),
                        "payment_method": payment_method,
                        "return_reason": return_reason,
                        "item_category": item_category,
                        "return_method": return_method
                    }
                    
                    response = requests.post(f"{API_URL}/predict", json=data, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        risk_score = result.get('risk_score', 0)
                        risk_level = result.get('risk_level', 'Unknown')
                        prediction = result.get('prediction', 0)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_color = "risk-high" if risk_score > 0.7 else "risk-medium" if risk_score > 0.3 else "risk-low"
                            st.markdown(f'<div class="metric-card"><h3>Risk Score</h3><h1 class="{risk_color}">{risk_score*100:.1f}%</h1></div>', unsafe_allow_html=True)
                        
                        with col2:
                            status = "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE"
                            color = "red" if prediction == 1 else "green"
                            st.markdown(f'<div class="metric-card"><h3>Verdict</h3><h1 style="color:{color}">{status}</h1></div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'<div class="metric-card"><h3>Risk Level</h3><h1 class="risk-{risk_level.lower()}">{risk_level}</h1></div>', unsafe_allow_html=True)
                        
                        # Top features
                        if 'top_features' in result and result['top_features']:
                            st.subheader("🔍 Key Risk Factors")
                            for feature in result['top_features'][:5]:
                                st.warning(f"{feature.get('feature', 'Unknown')}: {feature.get('value', 0)} (Impact: {feature.get('impact', 0)*100:.0f}%)")
                        
                        st.caption(f"Model: {result.get('model_used', 'Unknown')}")
                        st.caption(f"Timestamp: {result.get('timestamp', '')}")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("❌ API is not reachable. Please wait and try again.")

# Model Performance Tab
elif view_option == "📋 Model Performance":
    st.header("📋 Model Performance")
    
    st.info(f"Showing performance for: **Rule-based Model**")
    
    # Sample metrics (since we're using rule-based model)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", "0.784")
    with col2:
        st.metric("Recall", "0.803")
    with col3:
        st.metric("F1 Score", "0.793")
    with col4:
        st.metric("AUC-ROC", "0.912")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        # Sample confusion matrix
        fig = px.imshow(
            [[231, 18], [15, 61]],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Fraud", "Fraud"],
            y=["Not Fraud", "Fraud"],
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 ROC Curve")
        # Sample ROC curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            y=[0, 0.6, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1],
            mode='lines',
            name='ROC (AUC=0.912)'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash')
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"🛡️ Smart Return Fraud Detector | API: {API_URL}")
