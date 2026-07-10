"""
Model Training Pipeline for Return Fraud Detection
Trains XGBoost, LightGBM, and Logistic Regression with SMOTE and SHAP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    def __init__(self, X, y, test_size=0.2):
        self.X = X
        self.y = y
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        smote = SMOTE(random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
        print(f"Resampled training set: {len(self.X_train_resampled)} samples")
        
        self.models = {}
        self.results = {}
    
    def evaluate_model(self, y_pred, y_pred_proba, model_name):
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN: {cm[0,0]}  FP: {cm[0,1]}")
        print(f"    FN: {cm[1,0]}  TP: {cm[1,1]}")
        
        return {'auc_roc': auc_roc, 'avg_precision': avg_precision}
    
    def train_logistic_regression(self):
        print("\n" + "=" * 50)
        print("Training Logistic Regression...")
        
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        results = self.evaluate_model(y_pred, y_pred_proba, "Logistic Regression")
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return results
    
    def train_xgboost(self):
        print("\n" + "=" * 50)
        print("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        results = self.evaluate_model(y_pred, y_pred_proba, "XGBoost")
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        # SAVE SHAP EXPLAINER FOR XGBOOST
        import shap
        print("\n📊 Creating SHAP explainer for XGBoost...")
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, 'models/shap_xgboost.pkl')
        print("✓ SHAP explainer saved: models/shap_xgboost.pkl")
        
        return results
    
    def train_lightgbm(self):
        print("\n" + "=" * 50)
        print("Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        results = self.evaluate_model(y_pred, y_pred_proba, "LightGBM")
        self.models['lightgbm'] = model
        self.results['lightgbm'] = results
        
        # SAVE SHAP EXPLAINER FOR LIGHTGBM
        import shap
        print("\n📊 Creating SHAP explainer for LightGBM...")
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, 'models/shap_lightgbm.pkl')
        print("✓ SHAP explainer saved: models/shap_lightgbm.pkl")
        
        return results
    
    def run_full_pipeline(self):
        print("\n" + "=" * 50)
        print("STARTING MODEL TRAINING")
        print("=" * 50)
        
        self.train_logistic_regression()
        self.train_xgboost()
        self.train_lightgbm()
        
        os.makedirs('models', exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}.pkl')
            print(f"✓ Model saved: models/{name}.pkl")

def main():
    print("=" * 60)
    print("SMART RETURN FRAUD DETECTOR - MODEL TRAINING")
    print("=" * 60)
    
    X = pd.read_csv('data/processed/features.csv')
    y = pd.read_csv('data/processed/labels.csv')['is_fraudulent']
    
    print(f"\nLoaded {len(X)} samples with {X.shape[1]} features")
    print(f"Fraud rate: {y.mean():.2%}")
    
    trainer = FraudModelTrainer(X, y, test_size=0.2)
    trainer.run_full_pipeline()
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()