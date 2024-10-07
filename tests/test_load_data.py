import os
import sys
import pytest
import unittest
import pandas as pd
#append the relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.load_data import load_data

def test_load_data():
    # Create a sample CSV file for testing
    test_data = {
        'TransactionId': [1, 2, 3],
        'Risk_Label': [0, 1, 0],
        'RFMS_Score': [1.5, 2.0, 3.5]
    }
    
    test_df = pd.DataFrame(test_data)
    test_filepath = 'test_data.csv'
    
    # Save the DataFrame to a CSV file
    test_df.to_csv(test_filepath, index=False)
    
    try:
        # Load the data using the load_data function
        loaded_df = load_data(test_filepath)
        
        # Check if the loaded DataFrame matches the original DataFrame
        pd.testing.assert_frame_equal(test_df, loaded_df)
    finally:
        # Remove the test CSV file after the test
        if os.path.exists(test_filepath):
            os.remove(test_filepath)
