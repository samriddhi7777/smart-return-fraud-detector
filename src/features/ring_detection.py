"""
Fraud Ring Detection using NetworkX Graph Analysis
Optimized for performance with sampling
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import json
import time

class FraudRingDetector:
    """
    Detect fraud rings by building a graph of connected customers
    """
    
    def __init__(self, customers_df, orders_df, returns_df):
        self.customers = customers_df.copy()
        self.orders = orders_df.copy()
        self.returns = returns_df.copy()
        self.graph = nx.Graph()
        self.rings = []
        self.ring_features = {}
        
    def build_graph(self):
        """
        Build network graph where customers are connected if they share:
        - Address
        - Device ID (only if explicitly shared)
        """
        print("🕸️ Building fraud ring graph...")
        start_time = time.time()
        
        # Add customer nodes
        for _, customer in self.customers.iterrows():
            self.graph.add_node(
                customer['customer_id'],
                address=customer.get('address', ''),
                payment_method=customer.get('payment_method', ''),
                device_id=customer.get('device_id', '')
            )
        
        edge_count = 0
        
        # 1. Shared Address (most important for rings)
        address_groups = self.customers.groupby('address')['customer_id'].apply(list)
        print(f"  Processing {len(address_groups)} unique addresses...")
        
        for address, customers in address_groups.items():
            if len(customers) > 1:
                # Only connect if address is shared by 2+ customers
                for i in range(len(customers)):
                    for j in range(i+1, len(customers)):
                        self.graph.add_edge(customers[i], customers[j], 
                                          relation='shared_address',
                                          attribute=address[:50])  # Truncate address
                        edge_count += 1
        
        # 2. Shared Device (only if explicitly in data)
        if 'device_id' in self.customers.columns:
            device_groups = self.customers.groupby('device_id')['customer_id'].apply(list)
            print(f"  Processing devices...")
            
            for device, customers in device_groups.items():
                if len(customers) > 1 and device != 'device_unknown' and not device.startswith('device_'):
                    for i in range(len(customers)):
                        for j in range(i+1, len(customers)):
                            self.graph.add_edge(customers[i], customers[j],
                                              relation='shared_device',
                                              attribute=device)
                            edge_count += 1
        
        elapsed = time.time() - start_time
        print(f"✓ Graph built: {self.graph.number_of_nodes()} nodes, {edge_count} edges in {elapsed:.2f}s")
        return self.graph
    
    def detect_rings(self, min_ring_size=3):
        """
        Find connected components and identify fraud rings
        """
        print("🔍 Detecting fraud rings...")
        start_time = time.time()
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        self.rings = []
        
        for component in components:
            if len(component) >= min_ring_size:
                # Calculate ring statistics
                component_customers = self.customers[
                    self.customers['customer_id'].isin(component)
                ]
                
                # Get return data for these customers
                component_returns = self.returns[
                    self.returns['customer_id'].isin(component)
                ]
                
                ring_data = {
                    'ring_id': f'RING_{len(self.rings):03d}',
                    'size': len(component),
                    'customers': list(component),
                    'total_orders': component_customers['total_orders'].sum(),
                    'total_returns': component_customers['total_returns'].sum(),
                    'return_rate': component_customers['total_returns'].sum() / 
                                  max(component_customers['total_orders'].sum(), 1),
                    'avg_account_age': component_customers['account_age_days'].mean(),
                    'fraudulent_returns': component_returns['is_fraudulent'].sum() if len(component_returns) > 0 else 0
                }
                
                self.rings.append(ring_data)
        
        elapsed = time.time() - start_time
        print(f"✓ Found {len(self.rings)} fraud rings in {elapsed:.2f}s")
        return self.rings
    
    def extract_ring_features(self):
        """
        Extract ring-based features for each customer
        """
        print("📊 Extracting ring features...")
        
        # Initialize features
        self.ring_features = {}
        
        # Add each customer to ring feature dict
        for customer_id in self.customers['customer_id']:
            self.ring_features[customer_id] = {
                'ring_size': 0,
                'ring_return_rate': 0,
                'ring_fraud_rate': 0,
                'is_in_ring': 0,
                'ring_id': None
            }
        
        # Populate ring features
        for ring in self.rings:
            for customer_id in ring['customers']:
                self.ring_features[customer_id].update({
                    'ring_size': ring['size'],
                    'ring_return_rate': ring['return_rate'],
                    'ring_fraud_rate': ring['fraudulent_returns'] / max(ring['total_returns'], 1),
                    'is_in_ring': 1,
                    'ring_id': ring['ring_id']
                })
        
        # Convert to DataFrame
        ring_df = pd.DataFrame.from_dict(self.ring_features, orient='index')
        ring_df.index.name = 'customer_id'
        ring_df = ring_df.reset_index()
        
        print(f"✓ Ring features extracted for {len(ring_df)} customers")
        return ring_df
    
    def add_ring_features_to_data(self, features_df):
        """
        Add ring features to existing feature set
        """
        ring_df = self.extract_ring_features()
        
        # Create a copy to avoid modifying original
        merged = features_df.copy()
        
        # Add customer_id if it doesn't exist
        if 'customer_id' not in merged.columns:
            # Try to get customer_id from returns via return_id
            if 'return_id' in merged.columns and 'return_id' in self.returns.columns:
                temp_df = merged.merge(
                    self.returns[['return_id', 'customer_id']], 
                    on='return_id', 
                    how='left'
                )
                # Now merge with ring features
                temp_df = temp_df.merge(ring_df, on='customer_id', how='left')
                # Drop customer_id to keep original structure
                temp_df = temp_df.drop(['customer_id'], axis=1)
                merged = temp_df
            else:
                # No way to match, add zeros
                print("⚠️ Cannot match features to customers. Adding ring features as zeros.")
                for col in ['ring_size', 'ring_return_rate', 'ring_fraud_rate', 'is_in_ring']:
                    merged[col] = 0
        else:
            # Features already has customer_id
            merged = merged.merge(ring_df, on='customer_id', how='left')
        
        # Fill NaN values
        for col in ['ring_size', 'ring_return_rate', 'ring_fraud_rate', 'is_in_ring']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        return merged
    
    def run_full_pipeline(self, features_df):
        """
        Run complete ring detection pipeline
        """
        self.build_graph()
        self.detect_rings()
        result_df = self.add_ring_features_to_data(features_df)
        
        total_in_rings = sum(1 for r in self.rings for _ in r['customers'])
        
        print("\n🕸️ Ring Detection Summary:")
        print(f"  - Total customers: {len(self.customers)}")
        print(f"  - Customers in rings: {total_in_rings}")
        print(f"  - Number of rings: {len(self.rings)}")
        if self.rings:
            print(f"  - Average ring size: {np.mean([r['size'] for r in self.rings]):.1f}")
            print(f"  - Largest ring: {max([r['size'] for r in self.rings])}")
        
        return result_df

def main():
    """Test ring detection"""
    print("=" * 60)
    print("FRAUD RING DETECTION TEST")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading data...")
        customers = pd.read_csv('data/raw/customers.csv')
        orders = pd.read_csv('data/raw/orders.csv')
        returns = pd.read_csv('data/raw/returns.csv')
        
        print(f"  - Customers: {len(customers)}")
        print(f"  - Orders: {len(orders)}")
        print(f"  - Returns: {len(returns)}")
        
        # Check if features with rings already exists
        try:
            features = pd.read_csv('data/processed/features_with_rings.csv')
            print("✓ Loaded existing features with rings")
        except:
            features = pd.read_csv('data/processed/features.csv')
            print("✓ Loaded regular features")
        
        print(f"  - Features shape: {features.shape}")
        
        # Detect rings
        detector = FraudRingDetector(customers, orders, returns)
        features_with_rings = detector.run_full_pipeline(features)
        
        # Save updated features
        features_with_rings.to_csv('data/processed/features_with_rings.csv', index=False)
        print(f"\n✓ Features with ring data saved to data/processed/features_with_rings.csv")
        print(f"  - New shape: {features_with_rings.shape}")
        
        # Check if ring columns were added
        ring_cols = ['ring_size', 'ring_return_rate', 'ring_fraud_rate', 'is_in_ring']
        added_cols = [col for col in ring_cols if col in features_with_rings.columns]
        print(f"  - Ring columns added: {added_cols}")
        
        # Show ring summary
        ring_customers = customers[customers['is_in_ring'] == True] if 'is_in_ring' in customers.columns else pd.DataFrame()
        if len(ring_customers) > 0:
            print(f"\n📊 Ring Statistics:")
            print(f"  - Customers in rings: {len(ring_customers)}")
            print(f"  - Average returns from ring members: {ring_customers['total_returns'].mean():.2f}")
            print(f"  - Average orders from ring members: {ring_customers['total_orders'].mean():.2f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run data generation first: python src/data_generation/generate_synthetic_data.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()