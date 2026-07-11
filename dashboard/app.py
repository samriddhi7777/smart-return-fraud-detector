"""
Smart Return Fraud Detector - Dashboard
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Smart Return Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# API URL - UPDATE THIS TO YOUR ACTUAL RENDER URL
API_URL = "https://smart-return-fraud-detector.onrender.com"

st.title("🛡️ Smart Return Fraud Detector")
st.markdown("---")

# Check API connection
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

# API Status in Sidebar
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.subheader("🔌 API Status")

if check_api():
    st.sidebar.success("✅ API Connected")
    st.sidebar.caption(f"API: {API_URL}")
else:
    st.sidebar.error("❌ API Unreachable")
    st.sidebar.info("💡 The API might be spinning up (free tier). Wait 30-60 seconds and refresh.")
    if st.sidebar.button("🔄 Retry Connection"):
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Enter return details to check fraud risk")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Return Details")
    
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
        if not check_api():
            st.error("❌ API is not reachable. Please wait and try again.")
        else:
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
                        
                        risk_score = result.get('risk_score', 0)
                        risk_level = result.get('risk_level', 'Unknown')
                        prediction = result.get('prediction', 0)
                        
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

st.markdown("---")
st.caption(f"🛡️ Smart Return Fraud Detector | API: {API_URL}")
