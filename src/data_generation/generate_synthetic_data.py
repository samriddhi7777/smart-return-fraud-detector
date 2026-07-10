"""
Synthetic Data Generator for E-commerce Return Fraud Detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

fake = Faker()

class ReturnFraudDataGenerator:
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.category_risk = {
            'Electronics': 0.08,
            'Formalwear': 0.15,
            'Jewelry': 0.10,
            'Home Goods': 0.04,
            'Books': 0.02,
            'Beauty': 0.03,
            'Toys': 0.03,
            'Sporting Goods': 0.05,
            'Footwear': 0.07,
            'Handbags': 0.12
        }
        
        self.return_reasons = {
            'legitimate': [
                'Defective product', 'Wrong item sent', 'Damaged during shipping',
                'Not as described', 'Size issue'
            ],
            'fraud_indicators': [
                'Did not like it', 'No longer needed', 'Found better price',
                'Changed mind', 'Better deal elsewhere'
            ]
        }
        
        self.fraud_addresses = [fake.address() for _ in range(50)]
    
    def generate_customers(self, n_customers: int = 2000):
        """Generate customer base with ring patterns"""
        customers = []
        
        # Create 3-4 fraud rings with shared attributes
        rings = []
        for ring_id in range(4):
            ring_size = np.random.randint(5, 15)
            shared_address = fake.address()
            shared_payment = np.random.choice(['credit_card', 'debit_card', 'paypal'])
            shared_device = f"device_{np.random.randint(1000, 9999)}"
            
            rings.append({
                'ring_id': ring_id,
                'size': ring_size,
                'address': shared_address,
                'payment_method': shared_payment,
                'device_id': shared_device
            })
        
        # Keep track of which customers are in rings
        ring_membership = {}
        next_ring_member_id = 0
        
        for ring in rings:
            for i in range(ring['size']):
                ring_membership[next_ring_member_id] = ring['ring_id']
                next_ring_member_id += 1
        
        for i in range(n_customers):
            # Check if this customer is in a ring
            if i < next_ring_member_id:
                is_in_ring = True
                ring_id = ring_membership[i]
                ring_data = rings[ring_id]
            else:
                is_in_ring = False
                ring_data = None
            
            # Customer risk profile
            is_risky = np.random.random() < 0.15
            
            account_age_days = int(np.random.exponential(180))
            account_age_days = min(account_age_days, 730)
            
            total_orders = max(1, int(account_age_days / 30 * np.random.gamma(2, 1.5)))
            
            if is_risky or is_in_ring:
                total_orders = max(total_orders, int(total_orders * np.random.uniform(1.5, 3)))
                base_return_rate = np.random.uniform(0.30, 0.75)
            else:
                base_return_rate = np.random.uniform(0.05, 0.25)
            
            avg_order_value = np.random.exponential(100) + 30
            if is_risky or is_in_ring:
                avg_order_value *= np.random.uniform(1.0, 2.5)
            
            # Assign addresses - ring members share address
            if is_in_ring and ring_data:
                address = ring_data['address']
                payment_method = ring_data['payment_method']
                device_id = ring_data['device_id']
            else:
                address = fake.address()
                payment_method = np.random.choice(['credit_card', 'debit_card', 'paypal', 'store_credit'], 
                                                 p=[0.4, 0.2, 0.3, 0.1])
                device_id = f"device_{np.random.randint(1000, 9999)}"
            
            customers.append({
                'customer_id': f'CUST_{i:05d}',
                'account_age_days': account_age_days,
                'total_orders': total_orders,
                'total_returns': int(total_orders * base_return_rate),
                'avg_order_value': avg_order_value,
                'is_risky': is_risky,
                'email': fake.email(),
                'address': address,
                'payment_method': payment_method,
                'device_id': device_id,
                'is_in_ring': is_in_ring,
                'ring_id': ring_id if is_in_ring else None
            })
        
        return pd.DataFrame(customers)
    
    def generate_orders(self, customers_df, n_orders_per_customer_range=(1, 15)):
        orders = []
        order_counter = 0
        
        for _, customer in customers_df.iterrows():
            n_orders = np.random.randint(n_orders_per_customer_range[0], 
                                        n_orders_per_customer_range[1] + 1)
            
            if customer['total_orders'] > 0:
                n_orders = min(customer['total_orders'], 20)
            
            for _ in range(n_orders):
                category = np.random.choice(list(self.category_risk.keys()))
                
                if customer['is_risky']:
                    base_price = np.random.exponential(150) + 50
                    if category in ['Electronics', 'Formalwear', 'Jewelry']:
                        base_price *= np.random.uniform(1.5, 3)
                else:
                    base_price = np.random.exponential(80) + 20
                
                order_date = datetime.now() - timedelta(days=np.random.randint(0, 730))
                
                order_counter += 1
                orders.append({
                    'order_id': f'ORD_{order_counter:06d}',
                    'customer_id': customer['customer_id'],
                    'item_category': category,
                    'price': round(base_price, 2),
                    'quantity': np.random.randint(1, 4),
                    'order_date': order_date,
                    'payment_method': customer['payment_method'],
                    'delivery_date': order_date + timedelta(days=np.random.randint(2, 7))
                })
        
        return pd.DataFrame(orders)
    
    def generate_returns(self, orders_df, customers_df, return_rate_overall=0.15):
        returns = []
        return_counter = 0
        
        customer_return_rates = {}
        for _, customer in customers_df.iterrows():
            if customer['total_orders'] > 0:
                customer_return_rates[customer['customer_id']] = (
                    customer['total_returns'] / customer['total_orders']
                )
            else:
                customer_return_rates[customer['customer_id']] = 0.0
        
        for _, order in orders_df.iterrows():
            customer_id = order['customer_id']
            base_return_prob = customer_return_rates.get(customer_id, return_rate_overall)
            category_risk = self.category_risk.get(order['item_category'], 0.05)
            adjusted_prob = min(0.95, base_return_prob * (1 + category_risk))
            
            if np.random.random() < adjusted_prob:
                is_fraudulent = False
                fraud_type = None
                
                if customer_return_rates.get(customer_id, 0) > 0.60:
                    is_fraudulent = True
                    fraud_type = 'serial_returner'
                elif (order['price'] > 200 and 
                      order['item_category'] in ['Formalwear', 'Electronics'] and
                      np.random.random() < 0.3):
                    is_fraudulent = True
                    fraud_type = 'wardrobing'
                else:
                    customer_avg = customers_df[customers_df['customer_id'] == customer_id]['avg_order_value'].values
                    if len(customer_avg) > 0 and order['price'] > 300 and order['price'] > customer_avg[0] * 2.5:
                        is_fraudulent = True
                        fraud_type = 'price_anomaly'
                
                if not is_fraudulent and np.random.random() < 0.05:
                    is_fraudulent = True
                    fraud_type = 'timing_anomaly'
                
                if not is_fraudulent and np.random.random() < 0.03:
                    is_fraudulent = True
                    fraud_type = 'item_mismatch'
                
                days_since_delivery = np.random.randint(1, 30)
                
                if is_fraudulent and fraud_type == 'wardrobing':
                    days_since_delivery = np.random.randint(1, 4)
                elif is_fraudulent and fraud_type == 'timing_anomaly':
                    days_since_delivery = np.random.randint(27, 30)
                elif is_fraudulent and fraud_type == 'serial_returner':
                    days_since_delivery = np.random.randint(5, 25)
                
                if is_fraudulent:
                    reason = np.random.choice(self.return_reasons['fraud_indicators'])
                    item_condition = np.random.choice(['New', 'Like New'], p=[0.4, 0.6]) if fraud_type != 'item_mismatch' else 'Used - Damage'
                else:
                    reason = np.random.choice(self.return_reasons['legitimate'])
                    item_condition = np.random.choice(['New', 'Like New', 'Good', 'Fair'], p=[0.3, 0.4, 0.2, 0.1])
                
                if is_fraudulent and fraud_type == 'item_mismatch':
                    item_condition = np.random.choice(['Used - Damage', 'Missing parts', 'Wrong item'], p=[0.4, 0.3, 0.3])
                
                return_counter += 1
                returns.append({
                    'return_id': f'RET_{return_counter:06d}',
                    'order_id': order['order_id'],
                    'customer_id': customer_id,
                    'return_reason': reason,
                    'return_date': order['delivery_date'] + timedelta(days=days_since_delivery),
                    'days_since_delivery': days_since_delivery,
                    'return_method': np.random.choice(['refund', 'replacement'], p=[0.8, 0.2]),
                    'item_condition_reported': item_condition,
                    'is_fraudulent': is_fraudulent,
                    'fraud_type': fraud_type if is_fraudulent else None
                })
        
        return pd.DataFrame(returns)
    
    def generate_full_dataset(self, n_customers=2000, n_orders_per_customer_range=(1, 15), return_rate_overall=0.15):
        print("Generating customers...")
        customers_df = self.generate_customers(n_customers)
        
        print("Generating orders...")
        orders_df = self.generate_orders(customers_df, n_orders_per_customer_range)
        
        print("Generating returns...")
        returns_df = self.generate_returns(orders_df, customers_df, return_rate_overall)
        
        actual_fraud_rate = returns_df['is_fraudulent'].mean() if len(returns_df) > 0 else 0
        
        print(f"\n=== Dataset Statistics ===")
        print(f"  Customers: {len(customers_df):,}")
        print(f"  Orders: {len(orders_df):,}")
        print(f"  Returns: {len(returns_df):,}")
        print(f"  Fraud Rate: {actual_fraud_rate:.2%}")
        
        if len(returns_df) > 0:
            fraud_counts = returns_df[returns_df['is_fraudulent']]['fraud_type'].value_counts()
            print(f"\nFraud Type Distribution:")
            for fraud_type, count in fraud_counts.items():
                print(f"  - {fraud_type}: {count} ({count/len(returns_df)*100:.1f}%)")
        
        return customers_df, orders_df, returns_df

def main():
    print("=" * 60)
    print("SMART RETURN FRAUD DETECTOR - DATA GENERATION")
    print("=" * 60)
    
    os.makedirs('data/raw', exist_ok=True)
    
    generator = ReturnFraudDataGenerator(random_seed=42)
    customers, orders, returns = generator.generate_full_dataset(
        n_customers=2000,
        n_orders_per_customer_range=(1, 15),
        return_rate_overall=0.15
    )
    
    customers.to_csv('data/raw/customers.csv', index=False)
    orders.to_csv('data/raw/orders.csv', index=False)
    returns.to_csv('data/raw/returns.csv', index=False)
    
    print("\n✓ Data saved to data/raw/")
    print(f"  - customers.csv ({len(customers)} rows)")
    print(f"  - orders.csv ({len(orders)} rows)")
    print(f"  - returns.csv ({len(returns)} rows)")

if __name__ == "__main__":
    main()