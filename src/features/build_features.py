"""
Feature Engineering Pipeline for Return Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, customers_df, orders_df, returns_df):
        self.customers = customers_df.copy()
        self.orders = orders_df.copy()
        self.returns = returns_df.copy()
        
        if 'order_date' in self.orders.columns:
            self.orders['order_date'] = pd.to_datetime(self.orders['order_date'])
        if 'delivery_date' in self.orders.columns:
            self.orders['delivery_date'] = pd.to_datetime(self.orders['delivery_date'])
        if 'return_date' in self.returns.columns:
            self.returns['return_date'] = pd.to_datetime(self.returns['return_date'])
    
    def engineer_customer_features(self):
        customer_features = self.customers[['customer_id']].copy()
        
        customer_returns = self.returns.groupby('customer_id').agg({
            'return_id': 'count',
            'is_fraudulent': ['sum', 'mean']
        }).reset_index()
        customer_returns.columns = ['customer_id', 'total_returns', 'fraudulent_returns', 'fraud_rate']
        
        # Only keep customer_id for merging, drop duplicates
        customer_features = customer_features.merge(
            self.customers[['customer_id', 'account_age_days', 'total_orders', 'avg_order_value']], 
            on='customer_id', how='left'
        )
        
        customer_features = customer_features.merge(customer_returns, on='customer_id', how='left').fillna(0)
        
        recent_cutoff = self.returns['return_date'].max() - pd.Timedelta(days=90)
        recent_returns = self.returns[self.returns['return_date'] >= recent_cutoff]
        recent_return_counts = recent_returns.groupby('customer_id').size().reset_index()
        recent_return_counts.columns = ['customer_id', 'recent_returns_90d']
        
        customer_features = customer_features.merge(recent_return_counts, on='customer_id', how='left').fillna(0)
        
        avg_days = self.returns.groupby('customer_id')['days_since_delivery'].mean().reset_index()
        avg_days.columns = ['customer_id', 'avg_days_to_return']
        customer_features = customer_features.merge(avg_days, on='customer_id', how='left').fillna(0)
        
        customer_features['return_rate'] = customer_features['total_returns'] / customer_features['total_orders'].clip(1)
        customer_features['high_return_rate'] = (customer_features['return_rate'] > 0.6).astype(int)
        customer_features['order_frequency'] = customer_features['total_orders'] / (customer_features['account_age_days'] / 30 + 1)
        
        return customer_features
    
    def engineer_order_features(self):
        order_features = self.orders[['order_id', 'customer_id', 'item_category', 
                                     'price', 'quantity', 'payment_method']].copy()
        
        customer_avg = self.customers[['customer_id', 'avg_order_value']].copy()
        order_features = order_features.merge(customer_avg, on='customer_id', how='left')
        
        order_features['price_to_avg_ratio'] = order_features['price'] / order_features['avg_order_value'].clip(1)
        order_features['price_anomaly'] = (order_features['price_to_avg_ratio'] > 2.5).astype(int)
        
        category_risk = {
            'Electronics': 0.08, 'Formalwear': 0.15, 'Jewelry': 0.10,
            'Home Goods': 0.04, 'Books': 0.02, 'Beauty': 0.03,
            'Toys': 0.03, 'Sporting Goods': 0.05, 'Footwear': 0.07,
            'Handbags': 0.12
        }
        order_features['category_risk'] = order_features['item_category'].map(category_risk).fillna(0.05)
        
        payment_risk = {'credit_card': 0.3, 'debit_card': 0.2, 'paypal': 0.1, 'store_credit': 0.15}
        order_features['payment_risk'] = order_features['payment_method'].map(payment_risk).fillna(0.2)
        
        order_features['high_value'] = (order_features['price'] > 200).astype(int)
        order_features['order_risk_score'] = (
            order_features['category_risk'] * 0.5 +
            order_features['price_to_avg_ratio'].clip(0, 5) / 10 * 0.3 +
            order_features['payment_risk'] * 0.2
        )
        
        return order_features
    
    def engineer_behavioral_features(self):
        behavioral = self.returns[['return_id', 'order_id', 'customer_id', 
                                  'return_date', 'days_since_delivery',
                                  'return_reason', 'return_method']].copy()
        
        behavioral['return_day_of_week'] = behavioral['return_date'].dt.dayofweek
        behavioral['return_month'] = behavioral['return_date'].dt.month
        behavioral['return_hour'] = np.random.randint(0, 24, len(behavioral))
        
        behavioral['near_deadline'] = (behavioral['days_since_delivery'] > 25).astype(int)
        
        suspicious_reasons = ['Did not like it', 'No longer needed', 'Found better price',
                            'Changed mind', 'Better deal elsewhere']
        behavioral['suspicious_reason'] = behavioral['return_reason'].isin(suspicious_reasons).astype(int)
        behavioral['is_replacement'] = (behavioral['return_method'] == 'replacement').astype(int)
        
        recent_cutoff = behavioral['return_date'].max() - pd.Timedelta(days=30)
        recent_returns = behavioral[behavioral['return_date'] >= recent_cutoff]
        return_clusters = recent_returns.groupby('customer_id').size().reset_index()
        return_clusters.columns = ['customer_id', 'returns_last_30d']
        
        behavioral = behavioral.merge(return_clusters, on='customer_id', how='left').fillna(0)
        behavioral['multi_return_burst'] = (behavioral['returns_last_30d'] > 2).astype(int)
        
        # Drop datetime column
        behavioral = behavioral.drop(['return_date'], axis=1)
        
        return behavioral
    
    def create_network_features(self):
        # Just create a simple network feature without merging customer_id again
        network_features = self.returns[['return_id', 'customer_id']].copy()
        
        # Simple shared address feature - count how many customers share same address
        if 'address' in self.customers.columns:
            address_counts = self.customers.groupby('address')['customer_id'].nunique().reset_index()
            address_counts.columns = ['address', 'customers_at_address']
            
            # Merge address info
            network_with_address = network_features.merge(
                self.customers[['customer_id', 'address']], 
                on='customer_id', how='left'
            )
            network_with_address = network_with_address.merge(address_counts, on='address', how='left')
            network_with_address['shared_address'] = (network_with_address['customers_at_address'] > 1).astype(int)
            
            # Keep only needed columns
            network_features = network_with_address[['return_id', 'shared_address']]
        else:
            network_features['shared_address'] = 0
            network_features = network_features[['return_id', 'shared_address']]
        
        return network_features
    
    def build_full_feature_set(self):
        print("Building customer features...")
        customer_feats = self.engineer_customer_features()
        
        print("Building order features...")
        order_feats = self.engineer_order_features()
        
        print("Building behavioral features...")
        behavioral_feats = self.engineer_behavioral_features()
        
        print("Building network features...")
        network_feats = self.create_network_features()
        
        print("Merging features...")
        # Start with returns - only select needed columns
        features = self.returns[['return_id', 'order_id', 'customer_id', 'is_fraudulent']].copy()
        
        # Merge customer features (keep customer_id for merging)
        features = features.merge(customer_feats, on='customer_id', how='left')
        
        # Merge order features (drop customer_id from order features to avoid duplicates)
        order_feats_clean = order_feats.drop(['customer_id'], axis=1)
        features = features.merge(order_feats_clean, on='order_id', how='left')
        
        # Merge behavioral features (drop customer_id to avoid duplicates)
        behavioral_feats_clean = behavioral_feats.drop(['customer_id', 'order_id'], axis=1)
        features = features.merge(behavioral_feats_clean, on='return_id', how='left')
        
        # Merge network features (only has return_id and shared_address)
        features = features.merge(network_feats, on='return_id', how='left')
        
        # Separate label
        y = features['is_fraudulent'].values
        features = features.drop(['is_fraudulent', 'return_id', 'order_id', 'customer_id'], axis=1)
        
        # Convert any datetime columns to numeric
        for col in features.columns:
            if features[col].dtype == 'datetime64[ns]':
                features[col] = features[col].astype('int64') // 10**9
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))
        
        # Convert all columns to numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        print(f"\nFeature set created:")
        print(f"  - Shape: {features.shape}")
        print(f"  - Features: {len(features.columns)}")
        print(f"  - Fraud rate: {y.mean():.2%}")
        print(f"  - Feature columns: {list(features.columns)}")
        
        return features, y

def main():
    print("=" * 60)
    print("SMART RETURN FRAUD DETECTOR - FEATURE ENGINEERING")
    print("=" * 60)
    
    customers = pd.read_csv('data/raw/customers.csv')
    orders = pd.read_csv('data/raw/orders.csv')
    returns = pd.read_csv('data/raw/returns.csv')
    
    # Convert dates
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['delivery_date'] = pd.to_datetime(orders['delivery_date'])
    returns['return_date'] = pd.to_datetime(returns['return_date'])
    
    engineer = FeatureEngineer(customers, orders, returns)
    X, y = engineer.build_full_feature_set()
    
    os.makedirs('data/processed', exist_ok=True)
    X.to_csv('data/processed/features.csv', index=False)
    pd.Series(y).to_csv('data/processed/labels.csv', index=False, header=['is_fraudulent'])
    
    print("\n✓ Processed data saved to data/processed/")
    print(f"  - Features shape: {X.shape}")
    print(f"  - All columns are numeric: {all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)}")

if __name__ == "__main__":
    main()