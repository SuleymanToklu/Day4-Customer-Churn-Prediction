import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def run_training_pipeline():
    print("--- Training Pipeline Started ---")

    print("1/6 - Loading data...")
    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    except FileNotFoundError:
        print("HATA: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' bulunamadı.")
        return

    print("2/6 - Preprocessing data...")
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    print("3/6 - Splitting data...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("4/6 - Training model...")
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=ratio)
    model.fit(X_train, y_train)

    print("5/6 - Evaluating model and calculating confidence intervals...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    n_iterations = 1000
    n_size = int(len(X_test) * 0.80)
    
    boot_accuracies = []
    boot_f1_scores = []
    
    for i in range(n_iterations):
        X_boot, y_boot = resample(X_test, y_test, n_samples=n_size, random_state=i)
        
        y_pred_boot = model.predict(X_boot)
        
        boot_accuracies.append(accuracy_score(y_boot, y_pred_boot))
        boot_f1_scores.append(f1_score(y_boot, y_pred_boot))

    alpha = 0.95
    p_lower = ((1.0 - alpha) / 2.0) * 100
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    accuracy_ci = (np.percentile(boot_accuracies, p_lower), np.percentile(boot_accuracies, p_upper))
    f1_ci = (np.percentile(boot_f1_scores, p_lower), np.percentile(boot_f1_scores, p_upper))
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'accuracy_confidence_interval': accuracy_ci,
        'f1_score_confidence_interval': f1_ci
    }

    print("6/6 - Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(list(X.columns), 'model_columns.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4) 
    
    print("\n--- Training Pipeline Completed Successfully! ---")
    print(f"Accuracy: {accuracy:.4f} (95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
    print(f"F1 Score: {f1:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")


if __name__ == "__main__":
    run_training_pipeline()