"""
Smart Return Fraud Detector - Streamlit Dashboard
Interactive dashboard for fraud detection with SHAP explanations and Fraud Ring Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import networkx as nx

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Smart Return Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .ring-node {
        color: #ff0000;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛡️ Smart Return Fraud Detector</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        models['xgboost'] = joblib.load('models/xgboost.pkl')
        models['lightgbm'] = joblib.load('models/lightgbm.pkl')
        models['logistic_regression'] = joblib.load('models/logistic_regression.pkl')
        return models
    except:
        return None

@st.cache_data
def load_data():
    """Load data"""
    try:
        X = pd.read_csv('data/processed/features.csv')
        y = pd.read_csv('data/processed/labels.csv')['is_fraudulent']
        return X, y
    except:
        return None, None

@st.cache_data
def load_customers():
    """Load customer data for ring detection"""
    try:
        customers = pd.read_csv('data/raw/customers.csv')
        return customers
    except:
        return None

# Load everything
models = load_models()
X, y = load_data()
customers = load_customers()

if models is None or X is None:
    st.error("⚠️ Models or data not found. Please run training first: `python src/models/train.py`")
    st.stop()

# Sidebar
st.sidebar.title("🎛️ Dashboard Controls")

# Model selection
model_names = list(models.keys())
selected_model = st.sidebar.selectbox(
    "Select Model",
    model_names,
    index=0
)

# View selection
view_option = st.sidebar.radio(
    "View",
    ["📊 Overview", "📈 Predictions", "🔍 Single Prediction", "📋 Model Performance", "🕸️ Fraud Rings"]
)

# Get selected model
model = models[selected_model]

# Overview Tab
if view_option == "📊 Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Returns", f"{len(X):,}")
    with col2:
        fraud_count = y.sum()
        st.metric("Fraudulent Returns", f"{fraud_count:,}")
    with col3:
        fraud_rate = y.mean() * 100
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    with col4:
        st.metric("Features", f"{X.shape[1]}")
    
    st.markdown("---")
    
    # Fraud distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Fraud Distribution")
        fig = px.pie(
            values=[len(y) - fraud_count, fraud_count],
            names=['Legitimate', 'Fraudulent'],
            color=['Legitimate', 'Fraudulent'],
            color_discrete_map={'Legitimate': '#00cc66', 'Fraudulent': '#ff4b4b'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Feature Distribution")
        feature_col = st.selectbox("Select Feature", X.columns[:10])
        fig = px.histogram(
            X, x=feature_col,
            title=f"Distribution of {feature_col}",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    st.info("Select a few features to see correlation")
    
    selected_features = st.multiselect(
        "Select features for correlation",
        X.columns[:15],
        default=X.columns[:8].tolist()
    )
    
    if len(selected_features) > 1:
        corr = X[selected_features].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            height=500,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)

# Predictions Tab
elif view_option == "📈 Predictions":
    st.header("📈 Batch Predictions")
    st.info("Running predictions on all data...")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)
        y_proba = model.predict(X)
    
    # Store predictions in DataFrame
    results_df = X.copy()
    results_df['predicted_fraud'] = y_pred
    results_df['fraud_probability'] = y_proba
    results_df['actual_fraud'] = y
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", f"{len(results_df):,}")
    with col2:
        fraud_preds = y_pred.sum()
        st.metric("Predicted Fraud", f"{fraud_preds:,}")
    with col3:
        st.metric("Model", selected_model)
    with col4:
        correct = (y_pred == y).sum()
        st.metric("Correct Predictions", f"{correct:,}")
    
    st.markdown("---")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            results_df, x='fraud_probability',
            nbins=50,
            title='Fraud Probability Distribution',
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk categories
        results_df['risk_level'] = pd.cut(
            results_df['fraud_probability'],
            bins=[0, 0.3, 0.7, 1],
            labels=['Low', 'Medium', 'High']
        )
        risk_counts = results_df['risk_level'].value_counts()
        
        fig = px.bar(
            risk_counts,
            title='Risk Level Distribution',
            color=risk_counts.index,
            color_discrete_map={'Low': '#00cc66', 'Medium': '#ffa500', 'High': '#ff4b4b'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show data table
    st.subheader("📋 Prediction Results")
    st.dataframe(
        results_df[['fraud_probability', 'predicted_fraud', 'actual_fraud']].head(100),
        use_container_width=True,
        height=300
    )

# Single Prediction Tab
elif view_option == "🔍 Single Prediction":
    st.header("🔍 Single Return Prediction")
    st.info("Enter return details to check fraud risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Return Details")
        
        # Input fields
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
        
        st.info("💡 This creates a sample using the input values above")
    
    if st.button("🔮 Predict Fraud Risk", type="primary"):
        # Create a simple sample with default values for all features
        default_vals = {
            'account_age_days': account_age,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'total_returns': total_returns,
            'fraudulent_returns': total_returns // 2,
            'fraud_rate': 0.3,
            'recent_returns_90d': min(total_returns, 3),
            'avg_days_to_return': 10,
            'return_rate': total_returns / max(total_orders, 1),
            'high_return_rate': 1 if total_returns / max(total_orders, 1) > 0.6 else 0,
            'order_frequency': total_orders / 10,
            'price': price,
            'quantity': 1,
            'price_to_avg_ratio': price / max(avg_order_value, 1),
            'price_anomaly': 1 if price / max(avg_order_value, 1) > 2.5 else 0,
            'category_risk': 0.1,
            'payment_risk': 0.2,
            'high_value': 1 if price > 200 else 0,
            'order_risk_score': 0.3,
            'days_since_delivery': days_since_delivery,
            'return_day_of_week': 3,
            'return_month': 6,
            'return_hour': 12,
            'near_deadline': 1 if days_since_delivery > 25 else 0,
            'suspicious_reason': 1 if return_reason in ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind'] else 0,
            'is_replacement': 1 if return_method == 'replacement' else 0,
            'returns_last_30d': min(total_returns, 3),
            'multi_return_burst': 1 if total_returns > 2 else 0,
            'shared_address': 0,
            'item_category': list(X.columns).index('item_category') if 'item_category' in X.columns else 0,
            'payment_method': list(X.columns).index('payment_method') if 'payment_method' in X.columns else 0,
            'return_reason': list(X.columns).index('return_reason') if 'return_reason' in X.columns else 0,
            'return_method': list(X.columns).index('return_method') if 'return_method' in X.columns else 0
        }
        
        # Create DataFrame with all features
        sample_df = pd.DataFrame([default_vals])
        
        # Ensure all columns match
        for col in X.columns:
            if col not in sample_df.columns:
                sample_df[col] = 0
        
        sample_df = sample_df[X.columns]
        
        # Predict
        proba = model.predict_proba(sample_df)[0][1]
        pred = 1 if proba > 0.5 else 0
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = "risk-high" if proba > 0.7 else "risk-medium" if proba > 0.3 else "risk-low"
            st.markdown(f'<div class="metric-card"><h3>Risk Score</h3><h1 class="{risk_color}">{proba:.1%}</h1></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            status = "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE"
            color = "red" if pred == 1 else "green"
            st.markdown(f'<div class="metric-card"><h3>Verdict</h3><h1 style="color:{color}">{status}</h1></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><h1>{max(proba, 1-proba):.1%}</h1></div>', 
                       unsafe_allow_html=True)
        
        # Risk factors
        st.subheader("🔍 Key Risk Factors")
        
        risk_factors = []
        if price > 200:
            risk_factors.append("⚠️ High-value item (>$200)")
        if price / max(avg_order_value, 1) > 2.5:
            risk_factors.append("⚠️ Price 2.5x above average order")
        if total_returns / max(total_orders, 1) > 0.6:
            risk_factors.append("⚠️ High return rate (>60%)")
        if days_since_delivery > 25:
            risk_factors.append("⚠️ Return near deadline")
        if return_reason in ['Did not like it', 'No longer needed']:
            risk_factors.append("⚠️ Suspicious return reason")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected")

# Model Performance Tab
elif view_option == "📋 Model Performance":
    st.header("📋 Model Performance")
    
    st.info(f"Showing performance for: **{selected_model}**")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)
        y_proba = model.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_auc_score, roc_curve
    )
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", f"{tp / (tp + fp):.3f}" if (tp + fp) > 0 else "N/A")
    with col2:
        st.metric("Recall", f"{tp / (tp + fn):.3f}" if (tp + fn) > 0 else "N/A")
    with col3:
        st.metric("F1 Score", f"{2*tp / (2*tp + fp + fn):.3f}" if (2*tp + fp + fn) > 0 else "N/A")
    with col4:
        auc = roc_auc_score(y, y_proba)
        st.metric("AUC-ROC", f"{auc:.3f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        fig = px.imshow(
            [[tn, fp], [fn, tp]],
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
        fpr, tpr, _ = roc_curve(y, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

# Fraud Rings Tab
elif view_option == "🕸️ Fraud Rings":
    st.header("🕸️ Fraud Ring Visualization")
    st.info("Interactive visualization of connected customers sharing addresses, payment methods, or devices")
    
    try:
        if customers is None:
            st.warning("⚠️ Customer data not found. Run data generation first.")
        else:
            # Check if ring data exists
            if 'is_in_ring' not in customers.columns:
                st.warning("⚠️ Ring data not found. Run `python src/features/ring_detection.py` first.")
            else:
                st.success(f"✅ Found {len(customers)} customers, {customers['is_in_ring'].sum()} in rings")
                
                # Build graph
                G = nx.Graph()
                
                # Add nodes
                for _, row in customers.iterrows():
                    G.add_node(
                        row['customer_id'], 
                        is_in_ring=row.get('is_in_ring', False),
                        ring_id=row.get('ring_id', None)
                    )
                
                # Add edges for shared addresses (only if more than 1 customer shares address)
                address_groups = customers.groupby('address')['customer_id'].apply(list)
                for address, cust_list in address_groups.items():
                    if len(cust_list) > 1:
                        for i in range(len(cust_list)):
                            for j in range(i+1, len(cust_list)):
                                G.add_edge(cust_list[i], cust_list[j], type='address')
                
                # Limit nodes for performance (max 500)
                if G.number_of_nodes() > 500:
                    st.warning(f"⚠️ Graph has {G.number_of_nodes()} nodes. Showing subset for performance.")
                    # Take largest component
                    components = list(nx.connected_components(G))
                    largest = max(components, key=len)
                    G = G.subgraph(list(largest)[:500])
                
                if G.number_of_nodes() > 0:
                    # Get positions
                    pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add edges
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        showlegend=False
                    ))
                    
                    # Add nodes
                    node_x = []
                    node_y = []
                    node_colors = []
                    node_sizes = []
                    node_texts = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        is_in_ring = G.nodes[node].get('is_in_ring', False)
                        ring_id = G.nodes[node].get('ring_id', None)
                        
                        if is_in_ring:
                            node_colors.append('red')
                            node_sizes.append(25)
                            node_texts.append(f"{node}<br>🔴 IN RING {ring_id}")
                        else:
                            node_colors.append('#1f77b4')
                            node_sizes.append(10)
                            node_texts.append(f"{node}<br>✅ Not in ring")
                    
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        marker=dict(
                            size=node_sizes,
                            color=node_colors,
                            line=dict(width=1, color='#333'),
                            opacity=0.8
                        ),
                        text=node_texts,
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title='Customer Connection Graph (Red = Fraud Ring Members)',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700,
                        plot_bgcolor='#f8f9fa'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show ring statistics
                    st.subheader("📊 Ring Statistics")
                    
                    ring_customers = customers[customers['is_in_ring'] == True]
                    ring_ids = ring_customers['ring_id'].unique()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Customers", len(customers))
                    with col2:
                        st.metric("Customers in Rings", len(ring_customers))
                    with col3:
                        st.metric("Number of Rings", len(ring_ids))
                    with col4:
                        ring_rate = len(ring_customers) / len(customers) * 100
                        st.metric("% in Rings", f"{ring_rate:.1f}%")
                    
                    # Show ring details
                    if len(ring_customers) > 0:
                        st.subheader("🔍 Ring Members Details")
                        
                        # Show ring summary
                        ring_summary = []
                        for ring_id in ring_ids:
                            ring_members = ring_customers[ring_customers['ring_id'] == ring_id]
                            ring_summary.append({
                                'Ring ID': ring_id,
                                'Size': len(ring_members),
                                'Total Orders': ring_members['total_orders'].sum(),
                                'Total Returns': ring_members['total_returns'].sum(),
                                'Return Rate': f"{ring_members['total_returns'].sum() / max(ring_members['total_orders'].sum(), 1):.2%}",
                                'Avg Order Value': f"${ring_members['avg_order_value'].mean():.2f}"
                            })
                        
                        ring_df = pd.DataFrame(ring_summary)
                        st.dataframe(ring_df, use_container_width=True)
                        
                        # Show ring members
                        st.subheader("👥 Ring Members")
                        st.dataframe(
                            ring_customers[['customer_id', 'ring_id', 'total_orders', 'total_returns', 'avg_order_value', 'is_risky']].head(50),
                            use_container_width=True
                        )
                else:
                    st.warning("No graph to display. Try generating more data with rings.")
                
    except ImportError as e:
        st.error(f"❌ Missing library: {e}")
        st.info("Install networkx: `pip install networkx`")
    except Exception as e:
        st.error(f"❌ Error loading ring data: {e}")
        st.info("Run `python src/features/ring_detection.py` first to generate ring data")

# Footer
st.markdown("---")
st.caption("🛡️ Smart Return Fraud Detector | Built with ❤️ using Streamlit, XGBoost, and SHAP")