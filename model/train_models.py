"""
Training script for ML Classification Models
Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository (via sklearn)

This script trains 5 different classification models on the Breast Cancer dataset
and evaluates each using 6 metrics. It also generates the test_data.csv file 
and saves trained model files for use in the Streamlit app.

NAME: MEGHA SINGH
BITS ID: 2025AC05122
Date: 14-08-2026
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load the breast cancer dataset and prepare train/test splits."""
    
    # loading the dataset
    data = load_breast_cancer()
    
    # creating a dataframe for easier handling
    feature_names = data.feature_names.tolist()
    df = pd.DataFrame(data.data, columns=feature_names)
    df['diagnosis'] = data.target  # 1 = Benign, 0 = Malignant
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts()}")
    print(f"  Benign (1): {(df['diagnosis'] == 1).sum()}")
    print(f"  Malignant (0): {(df['diagnosis'] == 0).sum()}")
    
    # separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # stratified split to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # save the test data as csv for the streamlit app
    test_df = X_test.copy()
    test_df['diagnosis'] = y_test.values
    test_df.to_csv('../test_data.csv', index=False)
    print("\ntest_data.csv saved successfully!")
    
    return X_train, X_test, y_train, y_test, feature_names


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # save scaler for the streamlit app
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to model/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, scaler


def get_models():
    """Initialize all classification models with their hyperparameters."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=5000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,          # limiting depth to reduce overfitting
            min_samples_split=5,
            random_state=42
        ),
        'kNN': KNeighborsClassifier(
            n_neighbors=7,        # tried k=3,5,7 - 7 gave best results
            weights='uniform',
            metric='minkowski'
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Calculate all 6 required evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    
    # get probability scores for AUC calculation
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_prob), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    return metrics, y_pred, y_prob


def train_and_evaluate_all(models, X_train, X_test, y_train, y_test):
    """Train all models and collect evaluation results."""
    
    all_results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        # train the model
        model.fit(X_train, y_train)
        
        # evaluate
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        all_results[name] = metrics
        
        # print metrics
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")
        
        # print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
        
        # print classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
        
        # save the trained model
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, filename)
        print(f"  Model saved to model/{filename}")
    
    return all_results


def print_comparison_table(results):
    """Print a formatted comparison table of all models."""
    
    print("\n" + "="*85)
    print("MODEL COMPARISON TABLE")
    print("="*85)
    
    # header
    header = f"{'Model':<22} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}"
    print(header)
    print("-"*85)
    
    for model_name, metrics in results.items():
        row = f"{model_name:<22} {metrics['Accuracy']:>10.4f} {metrics['AUC']:>10.4f} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} {metrics['F1']:>10.4f} {metrics['MCC']:>10.4f}"
        print(row)
    
    print("-"*85)
    
    # find overall winner based on F1 score (good balance of precision and recall)
    best_model = max(results, key=lambda x: results[x]['F1'])
    print(f"\nOverall Winner: {best_model} (highest F1 Score: {results[best_model]['F1']})")


if __name__ == '__main__':
    
    print("="*60)
    print("  ML Assignment 2 - Classification Model Training")
    print("  Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print("="*60)
    
    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Step 2: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 3: Initialize models
    models = get_models()
    
    # Step 4: Train and evaluate all models
    results = train_and_evaluate_all(
        models, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Step 5: Print comparison
    print_comparison_table(results)
    
    # Save results for reference
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results.csv')
    print("\nResults saved to model/model_results.csv")
    print("\nTraining complete! All models saved in model/ directory.")
