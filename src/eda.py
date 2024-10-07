# scripts/eda.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dataset_overview(df):
    print("Overview of the Dataset:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nData types of each column:")
    print(df.dtypes)
    print("\nFirst few rows of the dataset:")
    print(df.head())

def summary_statistics(df):
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nValue counts for categorical variables:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

def plot_numerical_histograms(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 2, i)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_numerical_boxplots(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(data=df, x=col)
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()

def plot_pairplots(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    sns.pairplot(df[numerical_cols])
    plt.suptitle('Pair Plot of Numerical Features', y=1.02)
    plt.show()

def plot_categorical_distributions(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Frequency')
        plt.ylabel(col)
        plt.show()

def plot_categorical_vs_target(df, target_col='FraudResult'):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=col, hue=target_col, data=df, palette='viridis')
        plt.title(f'Distribution of {col} by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend(title=target_col)
        plt.show()

def correlation_analysis(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

def identify_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    missing_values_table = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_values_percentage
    })
    missing_values_table = missing_values_table[missing_values_table['Missing Values'] > 0]
    print("Missing Values in the Dataset:")
    print(missing_values_table)

def plot_outliers(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(4, 3, i)
        sns.boxplot(data=df, y=col)
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, target_col='FraudResult'):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if col != target_col:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f'Box Plot of {col} by {target_col}')
            plt.xlabel(target_col)
            plt.ylabel(col)
            plt.show()
