"""Shared feature configuration for training and inference."""
FEATURES = ['transaction_amount', 'merchant_category', 'merchant_country', 'transaction_hour', 'card_present', 'customer_age', 'account_age_days', 'credit_limit', 'credit_score', 'transactions_last_1h', 'transactions_last_24h', 'amount_last_24h', 'distance_from_home_km', 'international_transaction', 'transaction_channel', 'device_type', 'used_chip', 'used_pin', 'avg_transaction_amount_30d', 'num_declined_last_30d']
CATEGORICAL_FEATURES = ['merchant_category', 'merchant_country', 'transaction_channel', 'device_type']
NUMERIC_FEATURES = ['transaction_amount', 'transaction_hour', 'card_present', 'customer_age', 'account_age_days', 'credit_limit', 'credit_score', 'transactions_last_1h', 'transactions_last_24h', 'amount_last_24h', 'distance_from_home_km', 'international_transaction', 'used_chip', 'used_pin', 'avg_transaction_amount_30d', 'num_declined_last_30d']
TARGET = "is_fraud"
MODEL_PATH = "models/fraud_model.joblib"
DATA_PATH = "data/credit_card_fraud_detection_2026.csv"
