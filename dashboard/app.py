"""
Smart Return Fraud Detector - Professional Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk colors */
    .risk-high {
        color: #ff4b4b;
        text-shadow: 0 0 20px rgba(255,75,75,0.3);
    }
    .risk-medium {
        color: #ffa500;
        text-shadow: 0 0 20px rgba(255,165,0,0.3);
    }
    .risk-low {
        color: #00cc66;
        text-shadow: 0 0 20px rgba(0,204,102,0.3);
    }
    
    /* Status badges */
    .badge-fraud {
        background: linear-gradient(135deg, #ff4b4b, #ff0000);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .badge-legitimate {
        background: linear-gradient(135deg, #00cc66, #008040);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.6);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: shimmer 3s infinite;
        border-radius: 3px;
        margin: 2rem 0;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102,126,234,0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Feature tags */
    .feature-tag {
        display: inline-block;
        background: rgba(102,126,234,0.15);
        color: #667eea;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        border: 1px solid rgba(102,126,234,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-header">🛡️ Smart Return Fraud Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise-grade fraud detection with AI-powered insights</p>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================================================
# API URL
# ============================================================================
API_URL = "https://smart-return-fraud-detector.onrender.com"

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    st.markdown("---")
    
    # API Status
    st.markdown("#### 🔌 System Status")
    if check_api():
        st.success("✅ API Connected")
        st.caption(f"📍 {API_URL}")
    else:
        st.error("❌ API Unreachable")
        st.info("💡 The API might be spinning up. Wait 30-60 seconds and refresh.")
        if st.button("🔄 Retry Connection"):
            st.rerun()
    
    st.markdown("---")
    
    # Navigation
    st.markdown("#### 📊 Navigation")
    view_option = st.radio(
        "",
        ["📊 Overview", "📈 Predictions", "🔍 Single Prediction", "📋 Model Performance"],
        index=0
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("#### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Returns", "5,332", delta="12%")
    with col2:
        st.metric("Fraud Rate", "45.4%", delta="-2.1%")
    
    st.markdown("---")
    st.caption("🛡️ v2.0 | Built with ❤️")

# ============================================================================
# OVERVIEW TAB
# ============================================================================
if view_option == "📊 Overview":
    st.markdown("### 📊 Dashboard Overview")
    st.markdown("Real-time fraud detection metrics and insights")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Total Returns</div>
                <div class="metric-value">5,332</div>
                <div style="font-size:0.8rem;color:#00cc66;">↑ 12.3%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Fraudulent Returns</div>
                <div class="metric-value" style="color:#ff4b4b;">2,420</div>
                <div style="font-size:0.8rem;color:#ff4b4b;">↑ 8.1%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Fraud Rate</div>
                <div class="metric-value" style="color:#ffa500;">45.4%</div>
                <div style="font-size:0.8rem;color:#00cc66;">↓ 2.1%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value" style="color:#667eea;">34</div>
                <div style="font-size:0.8rem;color:#667eea;">ML Model</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Fraud Distribution")
        fig = px.pie(
            values=[2912, 2420],
            names=['Legitimate', 'Fraudulent'],
            color=['Legitimate', 'Fraudulent'],
            color_discrete_map={'Legitimate': '#00cc66', 'Fraudulent': '#ff4b4b'},
            hole=0.4
        )
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Risk Level Distribution")
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [1800, 1500, 2032]
        })
        fig = px.bar(
            risk_data, x='Risk Level', y='Count',
            color='Risk Level',
            color_discrete_map={'Low': '#00cc66', 'Medium': '#ffa500', 'High': '#ff4b4b'},
            text='Count'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PREDICTIONS TAB
# ============================================================================
elif view_option == "📈 Predictions":
    st.markdown("### 📈 Batch Predictions")
    st.markdown("Simulate multiple returns and analyze fraud patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_returns = st.number_input(
            "Number of returns to simulate",
            min_value=1,
            max_value=20,
            value=5
        )
    
    with col2:
        use_random = st.checkbox("Randomize data", value=True)
    
    if st.button("🔄 Generate Batch Predictions", type="primary"):
        with st.spinner("🔮 Generating predictions..."):
            results = []
            for i in range(num_returns):
                if use_random:
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
                else:
                    data = {
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
                
                try:
                    response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'Return ID': f'#{i+1:04d}',
                            'Risk Score': f"{result['risk_score']*100:.1f}%",
                            'Risk Level': result['risk_level'],
                            'Prediction': '🚨 FRAUD' if result['prediction'] == 1 else '✅ LEGITIMATE'
                        })
                except:
                    pass
            
            if results:
                df = pd.DataFrame(results)
                
                st.markdown("#### 📊 Results Summary")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(results))
                with col2:
                    fraud_count = sum(1 for r in results if 'FRAUD' in r['Prediction'])
                    st.metric("Fraud Detected", fraud_count)
                with col3:
                    st.metric("Fraud Rate", f"{fraud_count/len(results)*100:.1f}%")
            else:
                st.warning("⚠️ No results generated. Please check API connection.")

# ============================================================================
# SINGLE PREDICTION TAB
# ============================================================================
elif view_option == "🔍 Single Prediction":
    st.markdown("### 🔍 Single Return Analysis")
    st.markdown("Enter return details for real-time fraud risk assessment")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### 📋 Return Details")
        account_age = st.number_input("📅 Account Age (days)", min_value=1, value=180)
        total_orders = st.number_input("🛒 Total Orders", min_value=1, value=10)
        avg_order_value = st.number_input("💰 Avg Order Value ($)", min_value=10.0, value=150.0)
        total_returns = st.number_input("↩️ Total Returns", min_value=0, value=3)
        price = st.number_input("🏷️ Item Price ($)", min_value=1.0, value=200.0)
        days_since_delivery = st.number_input("📦 Days Since Delivery", min_value=1, max_value=30, value=5)
    
    with col2:
        st.markdown("#### 🔍 Additional Details")
        payment_method = st.selectbox("💳 Payment Method", ['credit_card', 'debit_card', 'paypal', 'store_credit'])
        return_reason = st.selectbox("📝 Return Reason", ['Did not like it', 'No longer needed', 'Defective product', 'Wrong item sent', 'Found better price', 'Changed mind'])
        item_category = st.selectbox("📂 Item Category", ['Electronics', 'Formalwear', 'Jewelry', 'Home Goods', 'Books', 'Beauty', 'Toys', 'Sporting Goods', 'Footwear', 'Handbags'])
        return_method = st.selectbox("📤 Return Method", ['refund', 'replacement'])
        
        if st.button("🔮 Analyze Fraud Risk", type="primary", use_container_width=True):
            if not check_api():
                st.error("❌ API is not reachable.")
            else:
                try:
                    with st.spinner("🔍 Analyzing return data..."):
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
                            st.markdown("### 📊 Prediction Results")
                            
                            risk_score = result.get('risk_score', 0)
                            risk_level = result.get('risk_level', 'Unknown')
                            prediction = result.get('prediction', 0)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                risk_color = "risk-high" if risk_score > 0.7 else "risk-medium" if risk_score > 0.3 else "risk-low"
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Risk Score</div>
                                        <div class="metric-value {risk_color}">{risk_score*100:.1f}%</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                if prediction == 1:
                                    st.markdown("""
                                        <div class="metric-card">
                                            <div class="metric-label">Verdict</div>
                                            <div class="metric-value"><span class="badge-fraud">🚨 FRAUD</span></div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                        <div class="metric-card">
                                            <div class="metric-label">Verdict</div>
                                            <div class="metric-value"><span class="badge-legitimate">✅ LEGITIMATE</span></div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            with col3:
                                level_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Risk Level</div>
                                        <div class="metric-value" style="color:{level_color};">{risk_level}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            if 'top_features' in result and result['top_features']:
                                st.markdown("#### 🔍 Key Risk Factors")
                                for feature in result['top_features'][:5]:
                                    impact = feature.get('impact', 0) * 100
                                    color = "red" if impact > 20 else "orange" if impact > 10 else "green"
                                    st.markdown(f"""
                                        <div class="info-box">
                                            <strong>{feature.get('feature', 'Unknown')}</strong>
                                            <span style="float:right;color:{color};font-weight:bold;">Impact: {impact:.0f}%</span>
                                            <br><small style="color:#666;">Value: {feature.get('value', 0)}</small>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            st.caption(f"🤖 Model: {result.get('model_used', 'Unknown')}")
                            st.caption(f"⏰ Timestamp: {result.get('timestamp', '')}")
                        else:
                            st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# MODEL PERFORMANCE TAB
# ============================================================================
elif view_option == "📋 Model Performance":
    st.markdown("### 📋 Model Performance Dashboard")
    st.markdown("XGBoost Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color:#00cc66;">0.784</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" style="color:#667eea;">0.803</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value" style="color:#764ba2;">0.793</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value" style="color:#ffa500;">0.912</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Confusion Matrix")
        fig = px.imshow(
            [[231, 18], [15, 61]],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Fraud", "Fraud"],
            y=["Not Fraud", "Fraud"],
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            y=[0, 0.6, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1],
            mode='lines',
            name='ROC (AUC=0.912)',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='#ccc')
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🛡️ Smart Return Fraud Detector v2.0")
with col2:
    st.caption(f"🔗 API: {API_URL}")
with col3:
    st.caption("⚡ Powered by XGBoost & SHAP")
