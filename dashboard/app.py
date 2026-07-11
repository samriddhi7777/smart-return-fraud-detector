"""
Smart Return Fraud Detector - Dashboard
"""

import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Smart Return Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# API URL
API_URL = "https://smart-return-fraud-detector.onrender.com"

st.title("🛡️ Smart Return Fraud Detector")
st.markdown("---")

# Check API
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

# Sidebar
st.sidebar.title("🎛️ Dashboard Controls")

if check_api():
    st.sidebar.success("✅ API Connected")
    st.sidebar.caption(f"API: {API_URL}")
else:
    st.sidebar.error("❌ API Unreachable")
    st.sidebar.info("Wait 30-60 seconds and refresh")

st.sidebar.markdown("---")
st.sidebar.info("Enter return details to check fraud risk")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Return Details")
    
    account_age = st.number_input("Account Age (days)", min_value=1, value=180)
    total_orders = st.number_input("Total Orders", min_value=1, value=10)
    total_returns = st.number_input("Total Returns", min_value=0, value=3)
    price = st.number_input("Item Price ($)", min_value=1.0, value=200.0)

with col2:
    st.subheader("🔍 Additional Details")
    
    return_reason = st.selectbox(
        "Return Reason",
        ['Did not like it', 'Defective product', 'No longer needed', 'Wrong item sent']
    )
    
    item_category = st.selectbox(
        "Item Category",
        ['Electronics', 'Books', 'Clothing', 'Home Goods']
    )
    
    payment_method = st.selectbox(
        "Payment Method",
        ['credit_card', 'debit_card', 'paypal']
    )
    
    if st.button("🔮 Predict Fraud Risk", type="primary", use_container_width=True):
        if not check_api():
            st.error("❌ API not reachable")
        else:
            try:
                data = {
                    "account_age_days": float(account_age),
                    "total_orders": float(total_orders),
                    "avg_order_value": 150.0,
                    "total_returns": float(total_returns),
                    "price": float(price),
                    "days_since_delivery": 5,
                    "payment_method": payment_method,
                    "return_reason": return_reason,
                    "item_category": item_category,
                    "return_method": "refund"
                }
                
                r = requests.post(f"{API_URL}/predict", json=data, timeout=30)
                
                if r.status_code == 200:
                    result = r.json()
                    
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    risk = result.get('risk_score', 0)
                    level = result.get('risk_level', 'Unknown')
                    pred = result.get('prediction', 0)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risk Score", f"{risk*100:.1f}%")
                    with col2:
                        st.metric("Risk Level", level)
                    with col3:
                        if pred == 1:
                            st.metric("Verdict", "🚨 FRAUD")
                        else:
                            st.metric("Verdict", "✅ LEGITIMATE")
                    
                    if 'top_features' in result:
                        st.subheader("🔍 Key Risk Factors")
                        for feature in result['top_features'][:3]:
                            st.write(f"- {feature.get('feature')}: {feature.get('value')} (Impact: {feature.get('impact', 0)*100:.0f}%)")
                            
                else:
                    st.error(f"API Error: {r.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption(f"🛡️ Smart Return Fraud Detector | API: {API_URL}")
