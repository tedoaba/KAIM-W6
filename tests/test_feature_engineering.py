import os 
import sys

import pandas as pd
import pytest
import unittest
from pytest import approx
#append the relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from feature_engineering import (
    create_aggregate_features,
    extract_transaction_time_features,
    merge_aggregate_and_time_features,
    reorder_columns,
    encode_features,
    handle_missing_values,
    normalize_features
)

# Sample data for testing (simulating CSV data without reading a file)
@pytest.fixture
def sample_data():
    data = {
        "TransactionId": ["TransactionId_76871", "TransactionId_73770", "TransactionId_26203"],
        "BatchId": ["BatchId_36123", "BatchId_15642", "BatchId_53941"],
        "AccountId": ["AccountId_3957", "AccountId_4841", "AccountId_4229"],
        "SubscriptionId": ["SubscriptionId_887", "SubscriptionId_3829", "SubscriptionId_222"],
        "CustomerId": [4406, 4406, 4683],
        "CurrencyCode": ["UGX", "UGX", "UGX"],
        "CountryCode": [256, 256, 256],
        "ProviderId": ["ProviderId_6", "ProviderId_4", "ProviderId_6"],
        "ProductId": [10, 6, 1],
        "ProductCategory": ["airtime", "financial_services", "airtime"],
        "ChannelId": ["ChannelId_3", "ChannelId_2", "ChannelId_3"],
        "Amount": [1000.0, -20.0, 500.0],
        "Value": [1000, 20, 500],
        "TransactionStartTime": ["2018-11-15T02:18:49Z", "2018-11-15T02:19:08Z", "2018-11-15T02:44:21Z"],
        "PricingStrategy": ["2", "2", "2"],
        "FraudResult": [0, 0, 0]
    }

    df = pd.DataFrame(data)
    return df


def test_create_aggregate_features(sample_data):
    """Test aggregate feature creation."""
    aggregate_features = create_aggregate_features(sample_data)
    assert aggregate_features.shape[0] == 2  # Two unique customers
    assert 'Total_Transaction_Amount' in aggregate_features.columns
    assert aggregate_features['Total_Transaction_Amount'].sum() == 1480.0  # Sum of all Amounts


def test_extract_transaction_time_features(sample_data):
    """Test time-based feature extraction."""
    df_with_time_features = extract_transaction_time_features(sample_data)
    assert 'Transaction_Hour' in df_with_time_features.columns
    assert df_with_time_features['Transaction_Hour'].iloc[0] == 2  # Check the hour for first transaction


def test_merge_aggregate_and_time_features(sample_data):
    """Test merging aggregate features and time-based features."""
    aggregate_features = create_aggregate_features(sample_data)
    df_with_time_features = extract_transaction_time_features(sample_data)
    final_df = merge_aggregate_and_time_features(df_with_time_features, aggregate_features)
    assert 'Total_Transaction_Amount' in final_df.columns
    assert final_df.shape[0] == 3  # Ensure all transactions are present


def test_reorder_columns(sample_data):
    """Test reordering of columns."""
    aggregate_features = create_aggregate_features(sample_data)
    df_with_time_features = extract_transaction_time_features(sample_data)
    final_df = merge_aggregate_and_time_features(df_with_time_features, aggregate_features)
    reordered_df = reorder_columns(final_df)
    assert reordered_df.columns[-1] == 'FraudResult'  # FraudResult should be the last column


# def test_encode_features(sample_data):
#     """Test One-Hot and Label Encoding."""
#     encoded_df = encode_features(sample_data)
#     print(encoded_df.columns)  # Debug: Print column names to understand what's happening
#     assert any('ProductCategory_airtime' in col for col in encoded_df.columns)  # Flexible check


def test_handle_missing_values(sample_data):
    """Test handling of missing values."""
    sample_data.loc[0, 'Amount'] = None  # Introduce a missing value in 'Amount'
    final_df = handle_missing_values(sample_data)
    assert final_df['Amount'].isnull().sum() == 0  # No missing values after imputation

def test_normalize_features(sample_data):
    """Test normalization of features."""
    aggregate_features = create_aggregate_features(sample_data)
    df_with_time_features = extract_transaction_time_features(sample_data)
    final_df = merge_aggregate_and_time_features(df_with_time_features, aggregate_features)
    final_df = handle_missing_values(final_df)
    normalized_df = normalize_features(final_df)
    assert normalized_df['Total_Transaction_Amount'].max() == approx(1.0, rel=1e-9)  # Handle floating-point comparison
