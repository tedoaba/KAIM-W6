import os
import sys
# Append the correct src path for custom module imports
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../data'))

from load_data import load_data
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
from feature_engineering import (
    create_aggregate_features,
    extract_transaction_time_features,
    merge_aggregate_and_time_features,
    reorder_columns,
    encode_features,
    handle_missing_values,
    normalize_features
)
from woe_binning import process_rfms_binning
from train_test_split import split_data

from modeling import (
    train_and_evaluate_logistic_regression,
    train_and_evaluate_random_forest,
    train_and_evaluate_xgboost,
    train_and_evaluate_adaboost,
    train_and_evaluate_decision_tree
)

def main():
    # Load the data
    df = load_data('../data/data.csv')

    # Perform EDA
    dataset_overview(df)
    summary_statistics(df)
    plot_numerical_histograms(df)
    plot_numerical_boxplots(df)
    plot_pairplots(df)
    # plot_categorical_distributions(df)
    # plot_categorical_vs_target(df)
    correlation_analysis(df)
    identify_missing_values(df)
    plot_outliers(df)
    plot_boxplots(df)

    # Create aggregate features
    aggregate_features = create_aggregate_features(df)

    # Extract time-based features
    df = extract_transaction_time_features(df)

    # Merge aggregate features with extracted time-based features
    final_df = merge_aggregate_and_time_features(df, aggregate_features)

    # Reorder columns to place 'FraudResult' at the end
    final_df = reorder_columns(final_df)

    # Handle missing values
    final_df = handle_missing_values(final_df)

    # Encode categorical features
    final_df = encode_features(final_df)

    # Normalize numerical features
    final_df = normalize_features(final_df)

    # Display the final DataFrame
    print("Final DataFrame after feature engineering:\n", final_df.head())

    """Main function to execute the RFMS analysis."""
    final_df, woe_df = process_rfms_binning(final_df)
    print(final_df.head())
    print(woe_df.head())
    

    X_train, X_test, y_train, y_test = split_data(final_df)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Train and evaluate Logistic Regression
    train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Train and evaluate Random Forest
    train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)

    # Train and evaluate XGBoost
    train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)

    # Train and evaluate AdaBoost
    train_and_evaluate_adaboost(X_train, X_test, y_train, y_test)

    # Train and evaluate Decision Tree
    train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()

