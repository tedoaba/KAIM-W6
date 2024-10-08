# KAIM Week 6 Challenges

## Credit Scoring Model Development

## Overview

This repository contains all the necessary notebooks, scripts, and resources required to build and deploy a **Credit Scoring Model** for Bati Bank. The model assesses credit risk, predicts fraudulent transactions, and classifies users based on their likelihood of default. The entire workflow adheres to **Basel II regulatory standards** and uses advanced machine learning techniques and feature engineering.

The project is divided into several tasks, from **exploratory data analysis** to **model deployment** on **Render**.


## Folder Structure

```bash

├── data/                       
├── notebooks/                  
│   ├── kaim-week6-task-2.ipynb         
│   ├── kaim-week6-task-3.ipynb         
│   ├── kaim-week6-task-4.ipynb         
├── src/  
│   ├── load_data.py                 
│   ├── eda.py
│   ├── 
├── scripts/ 
│   ├── main.py
├── tests/ 
│   ├── test_load_data.py                 
├── app/                        
│   ├── app.py                 
│   ├── model.pkl
│   ├── requirements.txt
├── .gitignore                   
├── README.md                   
├── requirements.txt

```

## Installation and Setup

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/tedoaba/KAIM-W6.git
cd KAIM-W6
```

### Install Dependencies

Ensure you have Python 3.8+ installed. Install the necessary libraries using:

If you want to set up a virtual environment:

```bash
# Create a virtual environment
python3 -m venv .week6
# Activate it
source .week6/Scripts/activate   # On Linux: venv\bin\activate
# Install dependencies
pip install -r requirements.txt
```

Dependencies include:
- **pandas, numpy**: Data manipulation and numerical operations.
- **scikit-learn**: Machine learning models and evaluation metrics.
- **seaborn, matplotlib**: Visualization tools.
- **flask**: For model deployment via a web API.
- **xverse, woe**: Advanced feature engineering and Weight of Evidence transformations.

## Tasks Breakdown

### **Task 2: Exploratory Data Analysis (EDA)**

- Dataset structure overview.
- Summary statistics for numerical and categorical variables.
- Visualization of distributions and correlations.
- Identifying missing values and outliers.

> Notebook: [task2_EDA.ipynb](notebooks/kaim-week6-task-2.ipynb)

### **Task 3: Feature Engineering**

- Creation of aggregate features like `TotalTransactionAmount`, `TransactionCount`, and `SpendScore`.
- Extraction of date-time features from `TransactionStartTime`.
- Categorical encoding and handling missing values.
- Normalization and scaling of numerical features.

> Notebook: [task3_feature_engineering.ipynb](notebooks/kaim-week6-task-3.ipynb)

### **Task 3 (Extended): Default Estimator and WoE Binning**

- Creation of RFMS features (Recency, Frequency, Monetary, SpendScore).
- Classifying customers into **Good** and **Bad** risk categories using SpendScore.
- Applying **Weight of Evidence (WoE)** binning for better interpretability of key features.

> Notebook: [task3_extended_woe_binning.ipynb](notebooks/kaim-week6-task-3-woe.ipynb)

### **Task 4: Model Selection and Training**

- Training and evaluation of models: **Logistic Regression, Decision Trees, Random Forest, Gradient Boosting**.
- Hyperparameter tuning using **Grid Search** for Random Forest and Gradient Boosting.
- Evaluation metrics include **Accuracy, Precision, Recall, F1 Score, ROC-AUC**, and confusion matrix.

> Notebook: [task4_modelling.ipynb](notebooks/kaim-week6-task-4.ipynb)


## Model Deployment

The model is deployed using **Flask** on **Render**, enabling real-time credit scoring predictions via an API. The deployed model takes transaction data and outputs a classification (Good/Bad risk) or a default probability score.

### Deployment Folder Structure

The `app/` directory contains files required for deployment:
- **main.py**: The Flask web application that serves the trained model.
- **xgb_model.pkl**: The trained model saved as a pickle file.
- **requirements.txt**: Lists the necessary libraries for deployment.

### Running Locally

To run the Flask app locally:

1. **Navigate to the `app/` directory**:
   ```bash
   cd app
   ```

2. **Install app-specific dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:
   ```bash
   python app.py
   ```

4. The API will be hosted locally at `http://127.0.0.1:5000/`.

### Deployed on Render

The model is deployed live on Render for real-time predictions. You can send transaction data to the endpoint and get predictions on credit risk. The URL will look something like this:

```bash
https://kaim-w6.onrender.com/
```

### API Endpoints


## Results and Outputs

Results from model evaluation, such as confusion matrices, performance metrics, and hyperparameter tuning results, are saved in the `results/` directory. This includes:
- **Accuracy, Precision, Recall, F1 Scores** for each model.
- **Feature importance plots** that show key features influencing predictions.
- **Confusion matrices** to visualize classification errors.

## How to Run Notebooks

You can run the notebooks locally or on platforms like **Kaggle** or **Google Colab**. Follow these steps:

1. Download the dataset and place it in the `data/` directory.
2. Open any notebook from the `notebooks/` folder.
3. Run all cells to execute the EDA, feature engineering, and model training steps.

## Contributions

Contributions to improve the project are always welcome! Feel free to open issues or pull requests if you have suggestions or enhancements to propose.

## Future Work

Future improvements and extensions for this project include:
- **Model Monitoring**: Implement model drift detection to ensure performance over time.
- **Explainability**: Add tools like **SHAP** or **LIME** for feature-level explainability.
- **Additional Data**: Incorporate more external data sources (e.g., credit history, behavioral data) to improve predictions.
- **Deployment Scaling**: Scale the API for production using tools like Docker or Kubernetes.
