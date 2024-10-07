from sklearn.model_selection import train_test_split

def split_data(final_df):
    # Define features and target variable
    X = final_df.drop(['TransactionId', 'Risk_Label', 'RFMS_Score', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'CountryCode', 'ProviderId','ProductId','FraudResult', 'TransactionStartTime'], axis=1)  # Drop non-feature columns
    y = final_df['Risk_Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Display sizes of train and test sets
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test
