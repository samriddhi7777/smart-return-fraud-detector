#!/bin/bash
mkdir -p data/raw data/processed models
python -c "
import pandas as pd
import numpy as np
from src.data_generation.generate_synthetic_data import ReturnFraudDataGenerator
import os

# Generate data
generator = ReturnFraudDataGenerator(random_seed=42)
customers, orders, returns = generator.generate_full_dataset(
    n_customers=2000,
    n_orders_per_customer_range=(1, 15),
    return_rate_overall=0.15
)

customers.to_csv('data/raw/customers.csv', index=False)
orders.to_csv('data/raw/orders.csv', index=False)
returns.to_csv('data/raw/returns.csv', index=False)

# Feature engineering
from src.features.build_features import FeatureEngineer
engineer = FeatureEngineer(customers, orders, returns)
X, y = engineer.build_full_feature_set()
X.to_csv('data/processed/features.csv', index=False)
pd.Series(y).to_csv('data/processed/labels.csv', index=False, header=['is_fraudulent'])

# Train models
from src.models.train import FraudModelTrainer
trainer = FraudModelTrainer(X, y, test_size=0.2)
trainer.run_full_pipeline()
print('✓ Setup complete!')
"
