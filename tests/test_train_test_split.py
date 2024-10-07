import os
import sys
import unittest
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from train_test_split import split_data
    
def test_split_data():
    # Create a sample DataFrame for testing
    # Note: Use mock data for your actual tests.
    # df = pd.DataFrame({
    #     'TransactionId': [1, 2, 3],
    #     'Risk_Label': [0, 1, 0],
    #     'RFMS_Score': [1, 2, 3],
    #     'BatchId': [1, 1, 1],
    #     'AccountId': [1, 2, 3],
    #     'SubscriptionId': [1, 2, 3],
    #     'CustomerId': [1, 2, 3],
    #     'CountryCode': ['US', 'US', 'US'],
    #     'ProviderId': [1, 1, 1],
    #     'ProductId': [1, 2, 3],
    #     'FraudResult': [0, 1, 0],
    #     'TransactionStartTime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    # })
    
    # Call the split_data function with the sample DataFrame
    # X_train, X_test, y_train, y_test = split_data(df)
    
    pass  # Replace with actual assertions after implementing the test
