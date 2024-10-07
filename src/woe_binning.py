import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

def calculate_rfms_components(final_df):
    """Calculate RFMS components."""
    # Calculate RFMS score components
    final_df['Recency'] = final_df.groupby('CustomerId')['Transaction_Year'].transform('max')
    final_df['Frequency'] = final_df['Transaction_Count']
    final_df['Monetary'] = final_df['Total_Transaction_Amount']
    final_df['Stability'] = final_df['Std_Deviation_Transaction_Amount']
    
    # Normalize RFMS components to bring them to the same scale
    for col in ['Recency', 'Frequency', 'Monetary', 'Stability']:
        final_df[col] = (final_df[col] - final_df[col].min()) / (final_df[col].max() - final_df[col].min())
    
    # Compute RFMS score (simple average of components, can adjust this formula)
    final_df['RFMS_Score'] = final_df[['Recency', 'Frequency', 'Monetary', 'Stability']].mean(axis=1)
    return final_df

# Function to apply risk labeling based on a threshold
def apply_risk_label(final_df, threshold=0.5):
    final_df['Risk_Label'] = final_df['RFMS_Score'].apply(lambda x: 1 if x > threshold else 0)  # 1: Low Risk, 0: High Risk
    return final_df

# Function to bin RFMS score
def bin_rfms_score(final_df, n_bins=5):
    kbin = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    final_df['RFMS_Binned'] = kbin.fit_transform(final_df[['RFMS_Score']])
    return final_df

# Function to calculate WoE (Weight of Evidence)
def calculate_woe(df, feature, target):
    woe_df = df.groupby(feature)[target].agg(['count', 'sum'])
    woe_df['good'] = woe_df['sum']  # Number of good risks
    woe_df['bad'] = woe_df['count'] - woe_df['sum']  # Number of bad risks
    # Calculate WoE and replace infinity values with 0
    woe_df['woe'] = np.log(woe_df['good'] / woe_df['bad']).replace([np.inf, -np.inf], 0)
    return woe_df

# Function to apply WoE binning
def apply_woe_binning(df, feature, target):
    woe_df = calculate_woe(df, feature, target)
    return woe_df

# Visualization functions
def plot_rfms_score_distribution(final_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(final_df['RFMS_Score'], bins=30, kde=True)
    plt.title('Distribution of RFMS Scores')
    plt.xlabel('RFMS Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.show()

def visualize_rfms_space(final_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(final_df['Monetary'], final_df['Frequency'], c=final_df['RFMS_Score'], cmap='viridis')
    plt.colorbar(label='RFMS Score')
    plt.title('RFMS Space: Frequency vs Monetary with RFMS Scores')
    plt.xlabel('Monetary')
    plt.ylabel('Frequency')
    plt.show()

# Main function that ties everything together
def process_rfms_binning(final_df):
    # Calculate RFMS components
    final_df = calculate_rfms_components(final_df)
    
    # Visualize RFMS distribution and RFMS space
    plot_rfms_score_distribution(final_df)
    visualize_rfms_space(final_df)
    
    # Apply risk labeling
    final_df = apply_risk_label(final_df, threshold=0.5)
    
    # Bin RFMS score into discrete categories
    final_df = bin_rfms_score(final_df, n_bins=5)
    
    # Apply WoE binning
    woe_df = apply_woe_binning(final_df, 'RFMS_Binned', 'Risk_Label')
    print(woe_df)
    
    return final_df, woe_df