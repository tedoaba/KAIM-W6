import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer

def preprocess_transactions(df):
    # Aggregate features by CustomerId
    aggregate_features = df.groupby('CustomerId').agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('TransactionId', 'count')
    ).reset_index()

    # Convert 'TransactionStartTime' to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Extract features from 'TransactionStartTime'
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    # Merge aggregate features with the original dataframe
    final_df = df.merge(aggregate_features, on='CustomerId', how='left')

    # Label encoding for 'PricingStrategy'
    label_columns = ['PricingStrategy']
    label_encoder = LabelEncoder()
    for col in label_columns:
        final_df[col] = label_encoder.fit_transform(final_df[col])

    # Separate numerical columns for normalization
    numerical_columns = final_df.select_dtypes(include=['float64', 'int64']).columns
    min_max_scaler = MinMaxScaler()
    final_df[numerical_columns] = min_max_scaler.fit_transform(final_df[numerical_columns])

    # Recency, Frequency, Monetary, and Stability (RFMS) calculations
    final_df['Recency'] = final_df.groupby('CustomerId')['Transaction_Year'].transform('max')
    final_df['Frequency'] = final_df['Transaction_Count']
    final_df['Monetary'] = final_df['Total_Transaction_Amount']

    # Normalize Recency, Frequency, Monetary, and Stability
    final_df['Recency'] = (final_df['Recency'] - final_df['Recency'].min()) / (final_df['Recency'].max() - final_df['Recency'].min())
    final_df['Frequency'] = (final_df['Frequency'] - final_df['Frequency'].min()) / (final_df['Frequency'].max() - final_df['Frequency'].min())
    final_df['Monetary'] = (final_df['Monetary'] - final_df['Monetary'].min()) / (final_df['Monetary'].max() - final_df['Monetary'].min())
    # Compute RFMS Score (simple average)
    final_df['RFMS_Score'] = (final_df['Recency'] + final_df['Frequency'] + final_df['Monetary']) / 3

    # Fill missing values
    for column in final_df.select_dtypes(include=['float64', 'int64']).columns:
        median_value = final_df[column].median()
        final_df[column].fillna(0, inplace=True)

    for column in final_df.select_dtypes(include=['object']).columns:
        mode_value = final_df[column].mode()[0]
        final_df[column].fillna(mode_value, inplace=True)

    # Binning RFMS Score into discrete categories
    n_bins = 5  # Number of bins
    kbin = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    final_df['RFMS_Binned'] = kbin.fit_transform(final_df[['RFMS_Score']])

    # Define features (X) by dropping 'TransactionId' and 'CustomerId'
    X = final_df.drop(['TransactionId', 'CustomerId', 'TransactionStartTime'], axis=1)

    return X
