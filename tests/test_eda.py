import os
import sys
import pytest
import unittest
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda import (
    dataset_overview,
    summary_statistics,
    plot_numerical_histograms,
    plot_numerical_boxplots,
    plot_pairplots,
    plot_categorical_distributions,
    plot_categorical_vs_target,
    correlation_analysis,
    identify_missing_values,
    plot_outliers,
    plot_boxplots,
)
import matplotlib.pyplot as plt

class TestEDA(unittest.TestCase):

    def setUp(self):
        # Sample dataset provided in the prompt
        data = {
            'TransactionId': ['TransactionId_76871', 'TransactionId_73770', 'TransactionId_26203', 'TransactionId_380', 'TransactionId_28195'],
            'BatchId': ['BatchId_36123', 'BatchId_15642', 'BatchId_53941', 'BatchId_102363', 'BatchId_38780'],
            'AccountId': ['AccountId_3957', 'AccountId_4841', 'AccountId_4229', 'AccountId_648', 'AccountId_4841'],
            'SubscriptionId': ['SubscriptionId_887', 'SubscriptionId_3829', 'SubscriptionId_222', 'SubscriptionId_2185', 'SubscriptionId_3829'],
            'CustomerId': ['CustomerId_4406', 'CustomerId_4406', 'CustomerId_4683', 'CustomerId_988', 'CustomerId_988'],
            'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX', 'UGX'],
            'CountryCode': [256, 256, 256, 256, 256],
            'ProviderId': ['ProviderId_6', 'ProviderId_4', 'ProviderId_6', 'ProviderId_1', 'ProviderId_4'],
            'ProductId': ['ProductId_10', 'ProductId_6', 'ProductId_1', 'ProductId_21', 'ProductId_6'],
            'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility_bill', 'financial_services'],
            'ChannelId': ['ChannelId_3', 'ChannelId_2', 'ChannelId_3', 'ChannelId_3', 'ChannelId_2'],
            'Amount': [1000.0, -20.0, 500.0, 20000.0, -644.0],
            'Value': [1000, 20, 500, 21800, 644],
            'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-11-15T02:19:08Z', '2018-11-15T02:44:21Z', '2018-11-15T03:32:55Z', '2018-11-15T03:34:21Z'],
            'PricingStrategy': [2, 2, 2, 2, 2],
            'FraudResult': [0, 0, 0, 0, 0]
        }
        self.df = pd.DataFrame(data)
    
    def test_dataset_overview(self):
        """Test dataset overview output"""
        print("\nTesting dataset_overview()...")
        dataset_overview(self.df)
        self.assertEqual(self.df.shape, (5, 16))

    def test_summary_statistics(self):
        """Test summary statistics output"""
        print("\nTesting summary_statistics()...")
        summary_statistics(self.df)
        self.assertEqual(self.df['Amount'].mean(), 4167.2)

    def test_plot_numerical_histograms(self):
        """Test numerical histograms plotting"""
        print("\nTesting plot_numerical_histograms()...")
        plt.figure()
        plot_numerical_histograms(self.df)
        plt.close()

    def test_plot_numerical_boxplots(self):
        """Test numerical boxplots plotting"""
        print("\nTesting plot_numerical_boxplots()...")
        plt.figure()
        plot_numerical_boxplots(self.df)
        plt.close()

    def test_plot_pairplots(self):
        """Test pair plots of numerical columns"""
        print("\nTesting plot_pairplots()...")
        plt.figure()
        plot_pairplots(self.df)
        plt.close()

    def test_plot_categorical_distributions(self):
        """Test categorical distributions plotting"""
        print("\nTesting plot_categorical_distributions()...")
        plt.figure()
        plot_categorical_distributions(self.df)
        plt.close()

    def test_plot_categorical_vs_target(self):
        """Test categorical variables vs target variable plotting"""
        print("\nTesting plot_categorical_vs_target()...")
        plt.figure()
        plot_categorical_vs_target(self.df, target_col='FraudResult')
        plt.close()

    def test_correlation_analysis(self):
        """Test correlation matrix plotting"""
        print("\nTesting correlation_analysis()...")
        plt.figure()
        correlation_analysis(self.df)
        plt.close()

    def test_identify_missing_values(self):
        """Test missing values identification"""
        print("\nTesting identify_missing_values()...")
        identify_missing_values(self.df)
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_plot_outliers(self):
        """Test outliers boxplot plotting"""
        print("\nTesting plot_outliers()...")
        plt.figure()
        plot_outliers(self.df)
        plt.close()

    def test_plot_boxplots(self):
        """Test boxplots of numerical columns vs target"""
        print("\nTesting plot_boxplots()...")
        plt.figure()
        plot_boxplots(self.df, target_col='FraudResult')
        plt.close()

if __name__ == '__main__':
    unittest.main()
