"""
Skin Disease Classification Model Training Script
Uses EfficientNet-B3 with transfer learning from ImageNet.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB3
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disease classes (must match order in config.py)
DISEASE_CLASSES = [
    "1. Eczema 1677",
    "2. Melanoma 15.75k",
    "3. Atopic Dermatitis - 1.25k",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "5. Melanocytic Nevi (NV) - 7970",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k",
    "10. Warts Molluscum and other Viral Infections - 2103"
]

# Simplified class names for display
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

# Training constants
IMG_SIZE = 300  # EfficientNet-B3 native input size
BATCH_SIZE = 12  # Small batch = less CPU usage
AUTOTUNE = tf.data.AUTOTUNE

# CPU throttling - limit to 2 cores (adjust as needed)
NUM_CPU_THREADS = 4  # Set to 1 for minimum CPU, 4 for moderate

# Apply CPU limits BEFORE any TensorFlow operations
tf.config.threading.set_intra_op_parallelism_threads(NUM_CPU_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPU_THREADS)
os.environ['OMP_NUM_THREADS'] = str(NUM_CPU_THREADS)
os.environ['TF_NUM_INTEROP_THREADS'] = str(NUM_CPU_THREADS)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(NUM_CPU_THREADS)


def setup_gpu():
    """Configure GPU memory growth to prevent OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            logger.warning(f"GPU setup error: {e}")
    else:
        logger.warning("No GPU found. Training will be slow on CPU.")


def create_data_augmentation():
    """Create stronger data augmentation pipeline for training data."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.3),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


# Global augmentation layer for data pipeline
_augmentation_layer = None

def get_augmentation_layer():
    """Get or create augmentation layer."""
    global _augmentation_layer
    if _augmentation_layer is None:
        _augmentation_layer = create_data_augmentation()
    return _augmentation_layer


def preprocess_image(image, label):
    """Preprocess a single image for EfficientNet (no augmentation)."""
    # EfficientNet expects inputs in [0, 255] range
    image = tf.cast(image, tf.float32)
    return image, label


def augment_image(image, label):
    """Apply data augmentation to training images."""
    augmentation = get_augmentation_layer()
    image = augmentation(image, training=True)
    return image, label


def load_dataset(data_dir: str, validation_split: float = 0.2, seed: int = 42):
    """
    Load dataset from directory structure.
    Expects: data_dir/IMG_CLASSES/class_name/*.jpg
    """
    data_path = Path(data_dir) / "IMG_CLASSES"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download from Kaggle and extract to data/IMG_CLASSES/"
        )
    
    logger.info(f"Loading dataset from {data_path}")
    
    # Load training dataset
    train_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    
    # Load validation dataset
    val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    
    # Get class names from dataset
    class_names = train_ds.class_names
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    return train_ds, val_ds, class_names


def compute_class_weights(train_ds, num_classes: int):
    """Compute class weights to handle imbalanced dataset."""
    labels = []
    for _, label_batch in train_ds:
        labels.extend(label_batch.numpy())
    
    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict


def create_model(num_classes: int, trainable_base: bool = False):
    """
    Create EfficientNet-B3 model with custom classification head.
    
    NOTE: Data augmentation is applied in the data pipeline, NOT in the model.
    This ensures the saved model works correctly for inference.
    
    Args:
        num_classes: Number of output classes
        trainable_base: Whether to make base model trainable
    """
    # Load EfficientNet-B3 pretrained on ImageNet
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None
    )
    
    # Freeze or unfreeze base model
    base_model.trainable = trainable_base
    
    # Create model with custom head (NO augmentation layers)
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # EfficientNet preprocessing is built into the model
    x = base_model(inputs, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs, outputs, name="skin_disease_classifier")
    
    return model, base_model


def train_model(
    data_dir: str,
    output_dir: str,
    epochs_phase1: int = 15,
    epochs_phase2: int = 40,
    learning_rate_phase1: float = 1e-3,
    learning_rate_phase2: float = 5e-5,
    fine_tune_layers: int = 50
):
    """
    Train the model using two-phase transfer learning.
    
    Phase 1: Train only the custom head (frozen base)
    Phase 2: Fine-tune top layers of base model
    """
    setup_gpu()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    train_ds, val_ds, class_names = load_dataset(data_dir)
    num_classes = len(class_names)
    
    # Optimize dataset performance (limited parallelism for low CPU usage)
    # Apply augmentation ONLY to training data (in data pipeline, not model)
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=NUM_CPU_THREADS)
    train_ds = train_ds.map(augment_image, num_parallel_calls=NUM_CPU_THREADS)
    train_ds = train_ds.prefetch(1)  # Minimal prefetch
    
    # Validation data: preprocess only, NO augmentation
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=NUM_CPU_THREADS)
    val_ds = val_ds.prefetch(1)
    
    # Compute class weights
    class_weights = compute_class_weights(train_ds, num_classes)
    
    # Create model
    model, base_model = create_model(num_classes, trainable_base=False)
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_path / f"checkpoint_{timestamp}.keras"
    
    common_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=output_path / "logs" / timestamp,
            histogram_freq=1
        )
    ]
    
    # ===== PHASE 1: Train custom head =====
    logger.info("=" * 50)
    logger.info("PHASE 1: Training custom classification head")
    logger.info("=" * 50)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=common_callbacks
    )
    
    # ===== PHASE 2: Fine-tune base model =====
    logger.info("=" * 50)
    logger.info(f"PHASE 2: Fine-tuning top {fine_tune_layers} layers")
    logger.info("=" * 50)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_phase2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Update checkpoint path for phase 2
    checkpoint_path_p2 = output_path / f"checkpoint_finetune_{timestamp}.keras"
    common_callbacks[2] = callbacks.ModelCheckpoint(
        checkpoint_path_p2,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        class_weight=class_weights,
        callbacks=common_callbacks,
        initial_epoch=len(history1.history['loss'])
    )
    
    # Save final model
    final_model_path = output_path / "efficientnetb3_skin_disease.keras"
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save class names mapping
    class_names_path = output_path / "class_names.txt"
    with open(class_names_path, "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")
    logger.info(f"Class names saved to {class_names_path}")
    
    # Evaluate on validation set
    logger.info("=" * 50)
    logger.info("Final Evaluation")
    logger.info("=" * 50)
    
    val_loss, val_accuracy = model.evaluate(val_ds)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return model, history1, history2


def evaluate_model(model_path: str, data_dir: str):
    """Evaluate a trained model and generate metrics."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load validation data
    _, val_ds, class_names = load_dataset(data_dir)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=NUM_CPU_THREADS)
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification report
    logger.info("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = Path(model_path).parent / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    logger.info(f"Confusion matrix saved to {output_path}")
    
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)


def main():
    parser = argparse.ArgumentParser(description="Train skin disease classification model")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Directory containing IMG_CLASSES folder"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="ml/model_weights",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--epochs-phase1", 
        type=int, 
        default=10,
        help="Epochs for phase 1 (frozen base)"
    )
    parser.add_argument(
        "--epochs-phase2", 
        type=int, 
        default=20,
        help="Epochs for phase 2 (fine-tuning)"
    )
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        help="Path to model for evaluation only (skip training)"
    )
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        evaluate_model(args.evaluate_only, args.data_dir)
    else:
        train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs_phase1=args.epochs_phase1,
            epochs_phase2=args.epochs_phase2
        )


if __name__ == "__main__":
    main()
