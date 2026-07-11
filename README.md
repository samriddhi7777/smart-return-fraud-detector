# 🛡️ Smart Return Fraud Detector

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Render](https://img.shields.io/badge/Deployed-Render-46C3A4)](https://render.com)
[![Streamlit Cloud](https://img.shields.io/badge/Deployed-Streamlit-FF4B4B)](https://streamlit.io/cloud)

> **Enterprise-grade fraud detection system with AI-powered insights, fraud ring detection, and SHAP explainability.**

---

## 🎯 Live Demo

| Service | URL |
|---------|-----|
| **🚀 API** | https://smart-return-fraud-detector.onrender.com |
| **📚 API Docs** | https://smart-return-fraud-detector.onrender.com/docs |
| **📊 Dashboard** | https://smart-return-fraud-detector-ezglsfjtxlztdjvucqjyxe.streamlit.app |

---

## 📋 Overview

**Smart Return Fraud Detector** is an end-to-end ML system that detects 6 types of e-commerce return fraud:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Serial Returner** | High return rate customers | >60% return rate over many orders |
| **Wardrobing** | Using and returning items | Formalwear returned within 3 days |
| **Price Anomaly** | Expensive items relative to customer | Price 3x above average order |
| **Timing Anomaly** | Returns near deadline | Returns on day 28-30 of window |
| **Item Mismatch** | Returning wrong/damaged items | Different item returned in box |
| **Fraud Ring** | Shared addresses/payments | Multiple accounts at same address |

---

## ✨ Key Features

- 🎯 **Multi-Pattern Detection** - 6 distinct fraud patterns
- 🔍 **SHAP Explainability** - Understand why each prediction was made
- 🕸️ **Fraud Ring Detection** - Network analysis of connected accounts
- 📊 **Interactive Dashboard** - Real-time fraud insights
- 🤖 **Multiple ML Models** - XGBoost, LightGBM, Logistic Regression

---


---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Machine Learning** | XGBoost, LightGBM, Scikit-learn |
| **Explainability** | SHAP |
| **Fraud Ring Detection** | NetworkX |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Data Processing** | Pandas, NumPy |
| **Synthetic Data** | Faker |
| **Deployment** | Render, Streamlit Cloud |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/samriddhi7777/smart-return-fraud-detector.git
cd smart-return-fraud-detector

# Install dependencies
pip install -r requirements.txt

# Generate data and train models
python src/data_generation/generate_synthetic_data.py
python src/features/build_features.py
python src/features/ring_detection.py
python src/models/train.py

# Run services
python src/api/app.py
streamlit run dashboard/app.py
👩‍💻 Author

Samriddhi

GitHub: @samriddhi7777
📄 License

MIT License


