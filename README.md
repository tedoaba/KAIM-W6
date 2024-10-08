# KAIM Week 6 Challenges

## Credit Scoring Model Development

## Overview

This repository contains all the necessary notebooks, scripts, and resources required to build and deploy a **Credit Scoring Model** for Bati Bank. The model assesses credit risk, predicts fraudulent transactions, and classifies users based on their likelihood of default. The entire workflow adheres to **Basel II regulatory standards** and uses advanced machine learning techniques and feature engineering.

The project is divided into several tasks, from **exploratory data analysis** to **model deployment** on **Render**.

### **Check it out yourself:** [Deployed API](https://kaim-w6.onrender.com/)

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

### **Task 4: Model Selection and Training**

- Training and evaluation of models: **Logistic Regression, Decision Trees, Random Forest, Gradient Boosting**.
- Hyperparameter tuning using **Grid Search** for Random Forest and Gradient Boosting.
- Evaluation metrics include **Accuracy, Precision, Recall, F1 Score, ROC-AUC**, and confusion matrix.

> Notebook: [task4_modelling.ipynb](notebooks/kaim-week6-task-4.ipynb)

## How to Run Notebooks

You can run the notebooks locally or on platforms like **Kaggle** or **Google Colab**. Follow these steps:

1. Download the dataset and place it in the `data/` directory.
2. Open any notebook from the `notebooks/` folder.
3. Run all cells to execute the EDA, feature engineering, and model training steps.

## Contributions

Contributions to improve the project are always welcome! Feel free to open issues or pull requests if you have suggestions or enhancements to propose.
