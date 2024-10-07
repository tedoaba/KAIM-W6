import os
import sys
import pandas as pd
import numpy as np
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from woe_binning import (
    calculate_rfms_components, 
    apply_risk_label, 
    bin_rfms_score, 
    apply_woe_binning, 
    process_rfms_binning
)
                         

# Sample DataFrame creation
def create_sample_data():
    data = {
        'TransactionId': ['TransactionId_1', 'TransactionId_2', 'TransactionId_3'],
        'BatchId': ['BatchId_1', 'BatchId_2', 'BatchId_3'],
        'AccountId': [1001, 1002, 1003],
        'SubscriptionId': [2001, 2002, 2003],
        'CustomerId': [3001, 3002, 3003],
        'Transaction_Year': [2022, 2021, 2020],  # For Recency calculation
        'Transaction_Count': [10, 15, 5],        # For Frequency
        'Total_Transaction_Amount': [500, 1500, 800],  # For Monetary
        'Std_Deviation_Transaction_Amount': [100, 300, 150],  # For Stability
    }
    df = pd.DataFrame(data)
    return df

# Testing the calculate_rfms_components function
def test_calculate_rfms_components():
    df = create_sample_data()
    rfms_df = calculate_rfms_components(df)
    print("RFMS Components:")
    print(rfms_df[['Recency', 'Frequency', 'Monetary', 'Stability', 'RFMS_Score']])

# Testing the apply_risk_label function
def test_apply_risk_label():
    df = create_sample_data()
    rfms_df = calculate_rfms_components(df)
    risk_df = apply_risk_label(rfms_df, threshold=0.5)
    print("Risk Labels:")
    print(risk_df[['RFMS_Score', 'Risk_Label']])

# Testing the bin_rfms_score function
def test_bin_rfms_score():
    df = create_sample_data()
    rfms_df = calculate_rfms_components(df)
    binned_df = bin_rfms_score(rfms_df, n_bins=3)
    print("Binned RFMS Scores:")
    print(binned_df[['RFMS_Score', 'RFMS_Binned']])

# Testing the apply_woe_binning function
def test_apply_woe_binning():
    df = create_sample_data()
    rfms_df = calculate_rfms_components(df)
    risk_df = apply_risk_label(rfms_df, threshold=0.5)
    binned_df = bin_rfms_score(risk_df, n_bins=3)
    woe_df = apply_woe_binning(binned_df, 'RFMS_Binned', 'Risk_Label')
    print("WoE Binning:")
    print(woe_df)

# Testing the full RFMS processing workflow
def test_process_rfms_binning():
    df = create_sample_data()
    final_df, woe_df = process_rfms_binning(df)
    print("Final DataFrame:")
    print(final_df)
    print("WoE DataFrame:")
    print(woe_df)

# Run tests
test_calculate_rfms_components()
test_apply_risk_label()
test_bin_rfms_score()
test_apply_woe_binning()
test_process_rfms_binning()
