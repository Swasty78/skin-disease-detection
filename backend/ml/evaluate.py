"""
Model evaluation utilities
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class names
CLASS_NAMES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis and Lichen Planus",
    "Seborrheic Keratoses and Benign Tumors",
    "Tinea Ringworm and Fungal Infections",
    "Warts Molluscum and Viral Infections"
]


def load_validation_data(data_dir: str, img_size: int = 224, batch_size: int = 32):
    """Load validation dataset."""
    data_path = Path(data_dir) / "IMG_CLASSES"
    
    val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )
    
    return val_ds


def evaluate_model(model_path: str, data_dir: str, output_dir: str = None):
    """
    Comprehensive model evaluation with metrics and visualizations.
    """
    output_dir = output_dir or str(Path(model_path).parent)
    output_path = Path(output_dir)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load data
    val_ds = load_validation_data(data_dir)
    
    # Collect predictions
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    logger.info("Running predictions...")
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_proba.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    logger.info("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Save report as JSON
    report_path = output_path / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.2%}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    cm_path = output_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    cm_norm_path = output_path / "confusion_matrix_normalized.png"
    plt.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Normalized confusion matrix saved to {cm_norm_path}")
    
    # Per-class accuracy bar chart
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(CLASS_NAMES)), per_class_accuracy, color='steelblue')
    plt.axhline(y=accuracy, color='red', linestyle='--', label=f'Overall: {accuracy:.2%}')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.legend()
    plt.tight_layout()
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
    
    acc_path = output_path / "per_class_accuracy.png"
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class accuracy chart saved to {acc_path}")
    
    # Summary
    summary = {
        "overall_accuracy": float(accuracy),
        "per_class_accuracy": {
            name: float(acc) for name, acc in zip(CLASS_NAMES, per_class_accuracy)
        },
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"])
    }
    
    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Overall Accuracy: {accuracy:.2%}")
    logger.info(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    logger.info(f"\nBest performing classes:")
    for name, acc in sorted(zip(CLASS_NAMES, per_class_accuracy), key=lambda x: -x[1])[:3]:
        logger.info(f"  - {name}: {acc:.2%}")
    logger.info(f"\nWorst performing classes:")
    for name, acc in sorted(zip(CLASS_NAMES, per_class_accuracy), key=lambda x: x[1])[:3]:
        logger.info(f"  - {name}: {acc:.2%}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data_dir, args.output_dir)
