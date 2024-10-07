import os 
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

def create_aggregate_features(df):
    """
    Create aggregate features based on the transactions per customer.
    """
    aggregate_features = df.groupby('CustomerId').agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('TransactionId', 'count'),
        Std_Deviation_Transaction_Amount=('Amount', 'std')
    ).reset_index()

    return aggregate_features

def extract_transaction_time_features(df):
    """
    Extract time-based features from the TransactionStartTime column.
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    return df

def merge_aggregate_and_time_features(df, aggregate_features):
    """
    Merge the aggregate features with the extracted time-based features.

    """
    final_df = df.merge(aggregate_features, on='CustomerId', how='left')
    return final_df

def reorder_columns(final_df):
    """
    Reorder the columns to place 'FraudResult' at the end of the DataFrame.

    """
    column_order = [col for col in final_df.columns if col != 'FraudResult'] + ['FraudResult']
    final_df = final_df[column_order]
    return final_df

def encode_features(df):
    """
    Perform One-Hot and Label Encoding on the specified columns.
    """
    # One-Hot Encoding
    one_hot_columns = ['ProductCategory', 'ChannelId', 'CurrencyCode']
    df_one_hot = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

    # Label Encoding
    label_columns = ['ProviderId', 'PricingStrategy', 'CountryCode']
    label_encoder = LabelEncoder()
    for col in label_columns:
        df_one_hot[col] = label_encoder.fit_transform(df_one_hot[col])

    return df_one_hot

def handle_missing_values(final_df):
    """
    Handle missing values by imputing with median for numeric columns and mode for categorical columns.
    """
    for column in final_df.select_dtypes(include=['float64', 'int64']).columns:
        median_value = final_df[column].median()
        final_df[column].fillna(median_value, inplace=True)

    for column in final_df.select_dtypes(include=['object']).columns:
        mode_value = final_df[column].mode()[0]
        final_df[column].fillna(mode_value, inplace=True)

    return final_df

def normalize_features(final_df):
    """
    Normalize numerical columns using Min-Max scaling.
    """
    numerical_columns = final_df.select_dtypes(include=['float64', 'int64']).columns
    min_max_scaler = MinMaxScaler()
    final_df[numerical_columns] = min_max_scaler.fit_transform(final_df[numerical_columns])
    return final_df
