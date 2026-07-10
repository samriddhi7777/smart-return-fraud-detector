"""
Smart Return Fraud Detector - Dashboard
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Smart Return Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# API URL
API_URL = "https://fraud-detector-api.onrender.com"

# Title
st.title("🛡️ Smart Return Fraud Detector")
st.markdown("---")

# Sidebar
st.sidebar.title("🎛️ Controls")
st.sidebar.info("Enter return details to check fraud risk")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Return Details")
    
    # Input fields
    account_age = st.number_input("Account Age (days)", min_value=1, value=180)
    total_orders = st.number_input("Total Orders", min_value=1, value=10)
    avg_order_value = st.number_input("Avg Order Value ($)", min_value=10.0, value=150.0)
    total_returns = st.number_input("Total Returns", min_value=0, value=3)
    price = st.number_input("Item Price ($)", min_value=1.0, value=200.0)
    days_since_delivery = st.number_input("Days Since Delivery", min_value=1, max_value=30, value=5)
    
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

with col2:
    st.header("🔍 Prediction")
    
    if st.button("🔮 Predict Fraud Risk", type="primary", use_container_width=True):
        try:
            # Prepare request
            data = {
                "account_age_days": account_age,
                "total_orders": total_orders,
                "avg_order_value": avg_order_value,
                "total_returns": total_returns,
                "price": price,
                "days_since_delivery": days_since_delivery,
                "payment_method": payment_method,
                "return_reason": return_reason,
                "item_category": item_category,
                "return_method": return_method
            }
            
            # Call API
            response = requests.post(f"{API_URL}/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.subheader("📊 Results")
                
                # Risk score
                risk_score = result.get('risk_score', 0)
                risk_level = result.get('risk_level', 'Unknown')
                prediction = result.get('prediction', 0)
                
                # Color based on risk
                if risk_level == "High":
                    color = "red"
                    emoji = "🚨"
                elif risk_level == "Medium":
                    color = "orange"
                    emoji = "⚠️"
                else:
                    color = "green"
                    emoji = "✅"
                
                st.markdown(f"### {emoji} Risk Score: **{risk_score*100:.1f}%**")
                st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.error("🚨 **FRAUD DETECTED**")
                else:
                    st.success("✅ **LEGITIMATE**")
                
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

# Footer
st.markdown("---")
st.caption("🛡️ Smart Return Fraud Detector | API: " + API_URL)

# Health check
try:
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.warning("⚠️ API Unreachable")
except:
    st.sidebar.error("❌ API Not Connected")
