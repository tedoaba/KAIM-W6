from flask import Flask, render_template, request
import pickle
import pandas as pd
from data_preprocess import preprocess_transactions, load_model

app = Flask(__name__)

# Load the model when starting the app
model = load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data for the new columns
        transaction_id = int(request.form['TransactionId'])
        batch_id = int(request.form['BatchId'])
        account_id = int(request.form['AccountId'])
        subscription_id = int(request.form['SubscriptionId'])
        customer_id = int(request.form['CustomerId'])
        currency_code = request.form['CurrencyCode']  # Assume string
        country_code = request.form['CountryCode']  # Assume string
        provider_id = int(request.form['ProviderId'])
        product_id = int(request.form['ProductId'])
        product_category = request.form['ProductCategory']  # Assume string
        channel_id = int(request.form['ChannelId'])
        amount = float(request.form['Amount'])
        value = float(request.form['Value'])
        transaction_start_time = request.form['TransactionStartTime']  # Keep as string or datetime, depends on format
        pricing_strategy = request.form['PricingStrategy']  # Assume string

        # Prepare input for the model as a DataFrame
        input_data = {
            'TransactionId': [transaction_id],
            'BatchId': [batch_id],
            'AccountId': [account_id],
            'SubscriptionId': [subscription_id],
            'CustomerId': [customer_id],
            'CurrencyCode': [currency_code],
            'CountryCode': [country_code],
            'ProviderId': [provider_id],
            'ProductId': [product_id],
            'ProductCategory': [product_category],
            'ChannelId': [channel_id],
            'Amount': [amount],
            'Value': [value],
            'TransactionStartTime': [transaction_start_time],
            'PricingStrategy': [pricing_strategy]
        }

        # Convert input data to DataFrame
        input_features = pd.DataFrame(input_data)

        df = input_features.drop(['BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode', 'CountryCode', 'ChannelId', 'ProviderId','ProductId', 'ProductCategory'], axis=1)

        X = preprocess_transactions(df)

        # Make prediction
        prediction = model.predict(X)[0]

        # Render the result.html template
        return render_template('result.html', transaction_id=transaction_id, batch_id=batch_id,
                               account_id=account_id, subscription_id=subscription_id, customer_id=customer_id,
                               currency_code=currency_code, country_code=country_code, provider_id=provider_id,
                               product_id=product_id, product_category=product_category, channel_id=channel_id,
                               amount=amount, value=value, transaction_start_time=transaction_start_time,
                               pricing_strategy=pricing_strategy, 
                               prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
