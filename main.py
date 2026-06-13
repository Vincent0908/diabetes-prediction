"""
Diabetes Prediction — ML Pipeline
Runs EDA, preprocessing, training, evaluation, and saves all outputs.

Usage:
    python main.py
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.eda import run_eda
from src.preprocessing import run_preprocessing_pipeline
from src.train import train_logistic_regression, train_neural_network
from src.evaluate import (
    evaluate_model, plot_confusion_matrix,
    plot_roc_curves, plot_training_history
)

DATA_PATH    = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')
LR_PATH      = os.path.join(MODELS_DIR, 'logistic_model.pkl')
NN_PATH      = os.path.join(MODELS_DIR, 'nn_model.keras')
METRICS_PATH = os.path.join(OUTPUTS_DIR, 'metrics_summary.json')


def main():
    print("\n=== DIABETES PREDICTION — ML PIPELINE ===\n")

    print("--- Step 1: EDA ---")
    run_eda(DATA_PATH, OUTPUTS_DIR)

    print("\n--- Step 2: Preprocessing ---")
    df = load_data(DATA_PATH)
    data = run_preprocessing_pipeline(df, MODELS_DIR)

    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_test  = data['y_test']

    print("\n--- Step 3a: Logistic Regression ---")
    lr_model = train_logistic_regression(X_train, y_train, model_path=LR_PATH)

    print("\n--- Step 3b: Neural Network ---")
    nn_model, history = train_neural_network(
        X_train, y_train, X_test, y_test, model_path=NN_PATH
    )

    print("\n--- Step 4: Evaluation ---")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")
    nn_metrics = evaluate_model(nn_model, X_test, y_test, model_name="Neural Network", is_keras=True)

    print("\n--- Step 5: Saving plots ---")
    plot_confusion_matrix(lr_model, X_test, y_test, "Logistic Regression",
                          os.path.join(OUTPUTS_DIR, 'cm_logistic.png'))
    plot_confusion_matrix(nn_model, X_test, y_test, "Neural Network",
                          os.path.join(OUTPUTS_DIR, 'cm_nn.png'), is_keras=True)
    plot_roc_curves(
        models_data=[
            {'model': lr_model, 'name': 'Logistic Regression', 'is_keras': False},
            {'model': nn_model, 'name': 'Neural Network',       'is_keras': True}
        ],
        X_test=X_test, y_test=y_test,
        output_path=os.path.join(OUTPUTS_DIR, 'roc_comparison.png')
    )
    plot_training_history(history, os.path.join(OUTPUTS_DIR, 'nn_training_history.png'))

    summary = {'logistic_regression': lr_metrics, 'neural_network': nn_metrics}
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVE] metrics -> {METRICS_PATH}")

    print("\n=== PIPELINE COMPLETE ===")
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*50}")
    for name, m in [("Logistic Regression", lr_metrics), ("Neural Network", nn_metrics)]:
        print(f"  {name:<25} {m['Accuracy']:>10.4f} {m['F1-Score']:>10.4f} {m['ROC-AUC']:>10.4f}")
    print(f"\n  streamlit run app.py\n")


if __name__ == "__main__":
    main()
