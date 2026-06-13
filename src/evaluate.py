import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)


def get_predictions(model, X_test, is_keras: bool = False):
    if is_keras:
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_true, y_prob), 4)
    }


def evaluate_model(model, X_test, y_test, model_name: str = "Model", is_keras: bool = False) -> dict:
    y_pred, y_prob = get_predictions(model, X_test, is_keras)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print(f"\n{model_name}")
    print("-" * 40)
    for name, val in metrics.items():
        print(f"  {name:<12}: {val:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str, output_path: str, is_keras: bool = False):
    y_pred, _ = get_predictions(model, X_test, is_keras)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ['No Diabetes', 'Diabetes']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')

    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, pad=12)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] {output_path}")


def plot_roc_curves(models_data: list, X_test, y_test, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#2563EB', '#DC2626']

    for i, entry in enumerate(models_data):
        _, y_prob = get_predictions(entry['model'], X_test, entry.get('is_keras', False))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{entry['name']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] {output_path}")


def plot_training_history(history, output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history.history['loss']) + 1)

    ax1.plot(epochs, history.history['loss'], '#2563EB', label='Train')
    ax1.plot(epochs, history.history['val_loss'], '#DC2626', label='Val', linestyle='--')
    ax1.set_title('Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history.history['accuracy'], '#2563EB', label='Train')
    ax2.plot(epochs, history.history['val_accuracy'], '#DC2626', label='Val', linestyle='--')
    ax2.set_title('Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('Neural Network Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] {output_path}")
