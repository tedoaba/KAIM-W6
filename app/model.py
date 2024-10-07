import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Example dataset
data = pd.DataFrame({
    'TransactionId': [1, 2, 3, 4, 5],
    'BatchId': [101, 102, 103, 104, 105],
    'AccountId': [1001, 1002, 1003, 1004, 1005],
    'SubscriptionId': [2001, 2002, 2003, 2004, 2005],
    'CustomerId': [3001, 3002, 3003, 3004, 3005],
    'CurrencyCode': ['USD', 'USD', 'USD', 'USD', 'USD'],
    'CountryCode': ['US', 'US', 'US', 'US', 'US'],
    'ProviderId': [1, 1, 2, 2, 1],
    'ProductId': [4001, 4002, 4003, 4004, 4005],
    'ProductCategory': ['A', 'B', 'A', 'C', 'B'],
    'ChannelId': [1, 2, 1, 3, 2],
    'Amount': [100.0, 150.0, 200.0, 250.0, 300.0],
    'Value': [10.0, 15.0, 20.0, 25.0, 30.0],
    'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', 
                             '2023-01-03 12:00:00', '2023-01-04 13:00:00', 
                             '2023-01-05 14:00:00'],  # Sample timestamps
    'PricingStrategy': ['A', 'B', 'A', 'C', 'B'],
    'FraudResult': [0, 1, 0, 1, 0]  # Target column (FraudResult)
})

# Define feature columns (X) and target column (y)
X = data[['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
           'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
           'ChannelId', 'Amount', 'Value', 'TransactionStartTime', 'PricingStrategy']]

X = data.drop(['TransactionStartTime'], axis=1)
# Target column for fraud detection
y = data['FraudResult']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, columns=['CurrencyCode', 'CountryCode', 'ProductCategory', 'PricingStrategy'], drop_first=True)

# Train a Linear Regression model (or choose an appropriate model for classification)
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
