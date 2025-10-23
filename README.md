# Project: Credit-Card-Fraud-Detection

## Description
This project aims to detect fraudulent credit card transactions using machine learning — specifically the Random Forest algorithm.
It uses historical transaction data to classify each transaction as fraudulent (1) or legitimate (0).
By integrating the model with a Flask web app, users can upload transaction data, view analysis results, and get fraud predictions directly through a browser interface.

## Business Use-case Overview
Credit card fraud costs financial institutions billions every year.With accurate fraud detection, banks and payment processors can:

1. Prevent financial losses by identifying suspicious transactions early.
2. Protect customer trust by blocking fraudulent activities in real-time.
3. Automate detection to reduce manual review workloads.
4. Continuously adapt to new fraud patterns with retrainable models.

The Random Forest algorithm is robust and interpretable, handling complex feature interactions while maintaining high predictive accuracy.

## Installation
This project requires Python 3.8+ and the following Python libraries:

1. numpy
2. pandas
3. matplotlib
4. seaborn
5. scikit-learn
6. flask
7. joblib

It is recommended to install these packages using pip or via the Anaconda distribution of Python.

## Data
The dataset contains anonymized transaction details.

Columns:
Time – Seconds elapsed since the first transaction.
V1–V28 – PCA-transformed features for confidentiality.
Amount – Transaction amount.
Class – Target variable (0 = legitimate, 1 = fraud).
Source: Public dataset such as Kaggle’s Credit Card Fraud Detection Dataset.

## Model
The project uses the Random Forest Classifier, an ensemble learning method that builds multiple decision trees and aggregates their outputs for robust classification.

Model Training Steps:

1. Load and preprocess data.
2. Handle class imbalance using SMOTE or undersampling.
3. Split data into training and testing sets.
4. Train the Random Forest model on the training data.
5. Evaluate model performance using Precision, Recall, F1-score, ROC-AUC.
6. Visualize feature importance and confusion matrix.

## Run
To run the project:
1. Navigate to the project folder
2. Load the dataset of the path
3. Run the Flask app:
   Flask run app.py
4. Interact with the app:
   1. Home Page
   2. Prediction Page
   3. Model Page

## Conclusion
This Credit Card Fraud Detection with Flask project provides a complete solution for fraud classification and web deployment.
By combining Random Forest and Flask, it offers a scalable, easy-to-use platform for detecting fraudulent transactions.

## Key Outcomes:
Detects fraudulent transactions with high precision.
Handles imbalanced data using SMOTE, Nearmiss
Provides an interactive web interface for real-time predictions.
