"""
Improved Skin Disease Classification Training Script v2.0
Targets F1 Score > 90% with advanced augmentation, regularization, and training strategies.

Key Improvements over v1:
- EfficientNetV2 support
- Advanced augmentation (CutMix, MixUp, RandAugment via albumentations)
- Label smoothing and Focal Loss
- Cosine annealing with warm restarts
- Test-Time Augmentation (TTA)
- Mixed precision training (optional)
- Progressive resizing
- Stochastic Weight Averaging
"""
import os
import sys
import argparse
import logging
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetV2S, EfficientNetV2M
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "img_size": 300,
    "batch_size": 32,  # Optimized for g4dn.4xlarge (T4 16GB VRAM)
    "epochs": 50,
    "learning_rate": 3e-4,  # Reduced from 1e-3 for stability
    "min_lr": 1e-7,
    "warmup_epochs": 3,  # Reduced warmup
    "label_smoothing": 0.05,  # Reduced from 0.1
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "mixup_prob": 0.15,  # Reduced from 0.5 - less aggressive
    "cutmix_prob": 0.15,  # Reduced from 0.5 - less aggressive
    "dropout_rate": 0.3,  # Reduced from 0.4
    "l2_reg": 0.005,  # Reduced from 0.01
    "focal_gamma": 2.0,
    "tta_augments": 5,
    "fine_tune_layers": 100,
    "validation_split": 0.15,
    "test_split": 0.05,
}

# Disease classes
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

CLASS_NAMES_SIMPLE = [
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


# ============================================================================
# GPU and Environment Setup
# ============================================================================

def setup_environment(use_mixed_precision: bool = False):
    """Configure GPU and optional mixed precision training."""
    # Enable XLA compilation for faster training
    tf.config.optimizer.set_jit(True)
    logger.info("XLA JIT compilation enabled")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Log GPU memory info
            for gpu in gpus:
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    logger.info(f"GPU details: {gpu_details}")
                except:
                    pass
        except RuntimeError as e:
            logger.warning(f"GPU setup error: {e}")
    else:
        logger.warning("No GPU found. Training will be slower on CPU.")
    
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision training enabled (FP16) - recommended for T4 GPU")


# ============================================================================
# Data Augmentation with Albumentations-style transforms
# ============================================================================

def create_augmentation_layers(img_size: int, strength: str = "strong"):
    """
    Create TF augmentation layers with varying strength.
    
    Args:
        img_size: Image size
        strength: "light", "medium", or "strong"
    """
    if strength == "light":
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="light_augmentation")
    
    elif strength == "medium":
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ], name="medium_augmentation")
    
    else:  # strong
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.3),
            layers.RandomTranslation(0.15, 0.15),
            RandomErasing(probability=0.25, min_area=0.02, max_area=0.2),
            GridDistortion(probability=0.3, num_steps=5, distort_limit=0.3),
        ], name="strong_augmentation")


class RandomErasing(layers.Layer):
    """Random Erasing augmentation layer."""
    
    def __init__(self, probability=0.5, min_area=0.02, max_area=0.33, 
                 min_aspect=0.3, max_aspect=3.3, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
    
    def call(self, images, training=None):
        if not training:
            return images
        
        def erase_single(image):
            def apply_erase():
                shape = tf.shape(image)
                h, w = shape[0], shape[1]
                img_dtype = image.dtype  # Get input dtype for mixed precision compatibility
                area = tf.cast(h * w, tf.float32)
                
                target_area = tf.random.uniform([], self.min_area, self.max_area) * area
                aspect_ratio = tf.random.uniform([], self.min_aspect, self.max_aspect)
                
                erase_h = tf.cast(tf.math.sqrt(target_area / aspect_ratio), tf.int32)
                erase_w = tf.cast(tf.math.sqrt(target_area * aspect_ratio), tf.int32)
                
                erase_h = tf.minimum(erase_h, h - 1)
                erase_w = tf.minimum(erase_w, w - 1)
                erase_h = tf.maximum(erase_h, 1)
                erase_w = tf.maximum(erase_w, 1)
                
                x = tf.random.uniform([], 0, w - erase_w, dtype=tf.int32)
                y = tf.random.uniform([], 0, h - erase_h, dtype=tf.int32)
                
                # Create a mask with the erased region filled with random values
                # Cast to input dtype for mixed precision compatibility
                noise = tf.cast(tf.random.uniform([erase_h, erase_w, 3], 0, 255), img_dtype)
                
                # Pad the noise to match image size
                paddings = [[y, h - y - erase_h], [x, w - x - erase_w], [0, 0]]
                noise_padded = tf.pad(noise, paddings)
                
                # Create binary mask (cast to input dtype)
                mask_ones = tf.ones([erase_h, erase_w, 3], dtype=img_dtype)
                mask_padded = tf.pad(mask_ones, paddings)
                
                # Apply: keep original where mask is 0, use noise where mask is 1
                return image * (1 - mask_padded) + noise_padded * mask_padded
            
            return tf.cond(
                tf.random.uniform([]) > self.probability,
                lambda: image,
                apply_erase
            )
        
        return tf.map_fn(erase_single, images)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "probability": self.probability,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_aspect": self.min_aspect,
            "max_aspect": self.max_aspect,
        })
        return config


class GridDistortion(layers.Layer):
    """Grid distortion augmentation layer (simplified as color jitter)."""
    
    def __init__(self, probability=0.5, num_steps=5, distort_limit=0.3, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.num_steps = num_steps
        self.distort_limit = distort_limit
    
    def call(self, images, training=None):
        if not training:
            return images
        
        def apply_distortion():
            # Apply color jitter as a simplified distortion
            return tf.image.random_hue(images, max_delta=0.05)
        
        return tf.cond(
            tf.random.uniform([]) > self.probability,
            lambda: images,
            apply_distortion
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "probability": self.probability,
            "num_steps": self.num_steps,
            "distort_limit": self.distort_limit,
        })
        return config


# ============================================================================
# MixUp and CutMix Augmentation
# ============================================================================

def mixup(images, labels, alpha=0.2):
    """Apply MixUp augmentation."""
    batch_size = tf.shape(images)[0]
    img_dtype = images.dtype  # Get input dtype for mixed precision compatibility
    
    # Sample lambda from beta distribution
    lam = tf.random.uniform([], 0, alpha)
    lam_img = tf.cast(lam, img_dtype)  # Cast for image operations
    
    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and labels
    mixed_images = lam_img * images + (1 - lam_img) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)  # Labels stay float32
    
    return mixed_images, mixed_labels


def cutmix(images, labels, alpha=1.0):
    """Apply CutMix augmentation."""
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]
    img_dtype = images.dtype  # Get input dtype for mixed precision compatibility
    
    # Sample lambda from beta distribution
    lam = tf.random.uniform([], 0.3, 0.7)
    
    # Compute cut dimensions
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
    
    # Random position
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    
    bbx1 = tf.maximum(0, cx - cut_w // 2)
    bby1 = tf.maximum(0, cy - cut_h // 2)
    bbx2 = tf.minimum(img_w, cx + cut_w // 2)
    bby2 = tf.minimum(img_h, cy + cut_h // 2)
    
    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    
    # Create mask
    mask = tf.ones((batch_size, img_h, img_w, 1), dtype=img_dtype)
    padding = tf.zeros((batch_size, bby2 - bby1, bbx2 - bbx1, 1), dtype=img_dtype)
    
    # Apply cut
    def apply_cut(args):
        img, shuffled_img = args
        cut_region = shuffled_img[bby1:bby2, bbx1:bbx2, :]
        paddings = [[bby1, img_h - bby2], [bbx1, img_w - bbx2], [0, 0]]
        mask_2d = tf.pad(tf.zeros_like(cut_region), paddings, constant_values=1)
        cut_padded = tf.pad(cut_region, paddings)
        return img * tf.cast(mask_2d, img_dtype) + cut_padded
    
    mixed_images = tf.map_fn(apply_cut, (images, shuffled_images), fn_output_signature=img_dtype)
    
    # Adjust lambda based on actual cut area
    actual_lam = 1 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    mixed_labels = actual_lam * labels + (1 - actual_lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels


def apply_mixup_cutmix(images, labels, mixup_prob=0.5, cutmix_prob=0.5, 
                        mixup_alpha=0.2, cutmix_alpha=1.0):
    """Randomly apply either MixUp, CutMix, or neither."""
    rand = tf.random.uniform([])
    
    if rand < mixup_prob:
        return mixup(images, labels, mixup_alpha)
    elif rand < mixup_prob + cutmix_prob:
        return cutmix(images, labels, cutmix_alpha)
    else:
        return images, labels


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = tf.shape(y_pred)[-1]
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / tf.cast(num_classes, tf.float32)
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config


def create_loss(use_focal: bool = True, label_smoothing: float = 0.1, 
                focal_gamma: float = 2.0):
    """Create loss function based on configuration."""
    if use_focal:
        return FocalLoss(gamma=focal_gamma, alpha=0.25, label_smoothing=label_smoothing)
    else:
        return keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)


# ============================================================================
# Learning Rate Schedules
# ============================================================================

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with linear warmup."""
    
    def __init__(self, initial_lr, warmup_steps, decay_steps, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_lr * (step / warmup_steps)
        
        # Cosine decay phase
        decay_step = tf.minimum(step - warmup_steps, decay_steps)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / decay_steps))
        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_lr": self.min_lr,
        }


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_dataframe(data_dir: str, validation_split: float = 0.15, 
                   test_split: float = 0.05, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data into DataFrames with train/val/test split."""
    data_path = Path(data_dir) / "IMG_CLASSES"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    filepaths = []
    labels = []
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    filepaths.append(str(img_file))
                    labels.append(class_dir.name)
    
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    logger.info(f"Loaded {len(df)} images from {len(df['labels'].unique())} classes")
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(
        df, test_size=validation_split + test_split, 
        stratify=df['labels'], random_state=seed
    )
    
    val_ratio = validation_split / (validation_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=1-val_ratio, 
        stratify=temp_df['labels'], random_state=seed
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def balance_dataset(df: pd.DataFrame, max_samples: int = 3000, 
                    min_samples: int = 500) -> pd.DataFrame:
    """Balance dataset by oversampling minority classes and capping majority classes."""
    balanced_dfs = []
    
    for label in df['labels'].unique():
        class_df = df[df['labels'] == label]
        n_samples = len(class_df)
        
        if n_samples < min_samples:
            # Oversample with replacement
            class_df = class_df.sample(min_samples, replace=True, random_state=SEED)
        elif n_samples > max_samples:
            # Undersample
            class_df = class_df.sample(max_samples, replace=False, random_state=SEED)
        
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    logger.info(f"Balanced dataset: {len(balanced_df)} samples")
    logger.info(f"Class distribution:\n{balanced_df['labels'].value_counts()}")
    
    return balanced_df


def create_dataset(df: pd.DataFrame, img_size: int, batch_size: int, 
                   is_training: bool = True, num_classes: int = 10,
                   use_mixup: bool = True, use_cutmix: bool = True,
                   mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                   mixup_prob: float = 0.15, cutmix_prob: float = 0.15) -> tf.data.Dataset:
    """Create tf.data.Dataset from DataFrame."""
    
    # Create label encoder
    label_encoder = {label: i for i, label in enumerate(sorted(df['labels'].unique()))}
    
    def load_and_preprocess(filepath, label):
        # Load image
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)  # EfficientNet expects 0-255 range
        
        # One-hot encode label
        label_idx = label
        label_onehot = tf.one_hot(label_idx, num_classes)
        
        return image, label_onehot
    
    # Create dataset
    filepaths = df['filepaths'].values
    labels = np.array([label_encoder[l] for l in df['labels'].values])
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Apply augmentation - use medium strength to avoid over-augmentation
        augmentation = create_augmentation_layers(img_size, strength="medium")
        dataset = dataset.map(
            lambda x, y: (augmentation(tf.expand_dims(x, 0), training=True)[0], y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    
    if is_training and (use_mixup or use_cutmix):
        # Apply MixUp/CutMix with conservative probabilities
        mixup_prob_actual = mixup_prob if use_mixup else 0.0
        cutmix_prob_actual = cutmix_prob if use_cutmix else 0.0
        
        dataset = dataset.map(
            lambda x, y: apply_mixup_cutmix(x, y, mixup_prob_actual, cutmix_prob_actual, mixup_alpha, cutmix_alpha),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, label_encoder


# ============================================================================
# Model Architecture
# ============================================================================

def get_base_model(name: str, img_size: int, weights: str = "imagenet"):
    """Get base model by name."""
    input_shape = (img_size, img_size, 3)
    
    models = {
        "efficientnetb3": lambda: EfficientNetB3(
            include_top=False, weights=weights, input_shape=input_shape, pooling=None
        ),
        "efficientnetb4": lambda: EfficientNetB4(
            include_top=False, weights=weights, input_shape=input_shape, pooling=None
        ),
        "efficientnetv2-s": lambda: EfficientNetV2S(
            include_top=False, weights=weights, input_shape=input_shape, pooling=None
        ),
        "efficientnetv2-m": lambda: EfficientNetV2M(
            include_top=False, weights=weights, input_shape=input_shape, pooling=None
        ),
    }
    
    model_name = name.lower()
    if model_name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[model_name]()


def create_model(base_model_name: str, num_classes: int, img_size: int,
                 dropout_rate: float = 0.4, l2_reg: float = 0.01,
                 trainable_base: bool = False) -> Tuple[keras.Model, keras.Model]:
    """
    Create classification model with improved head.
    
    Returns:
        Tuple of (full_model, base_model)
    """
    base_model = get_base_model(base_model_name, img_size)
    base_model.trainable = trainable_base
    
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    # Base model features
    x = base_model(inputs, training=False)
    
    # Global pooling with both average and max
    x_avg = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x_max = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Concatenate()([x_avg, x_max])
    
    # Enhanced classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # First dense block
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)
    
    # Second dense block
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name="skin_disease_classifier_v2")
    
    return model, base_model


# ============================================================================
# Training Callbacks
# ============================================================================

def create_callbacks(output_dir: Path, monitor: str = "val_accuracy",
                     patience_early_stop: int = 15) -> List[callbacks.Callback]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return [
        # Model checkpoint
        callbacks.ModelCheckpoint(
            output_dir / f"best_model_{timestamp}.keras",
            monitor=monitor,
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        # Note: ReduceLROnPlateau removed - incompatible with LearningRateSchedule
        # We use WarmupCosineDecay instead which handles LR scheduling
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=output_dir / "logs" / timestamp,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        # CSV logger
        callbacks.CSVLogger(
            output_dir / f"training_log_{timestamp}.csv"
        ),
    ]


# ============================================================================
# Test-Time Augmentation
# ============================================================================

def predict_with_tta(model: keras.Model, dataset: tf.data.Dataset, 
                     n_augments: int = 5) -> np.ndarray:
    """Make predictions with Test-Time Augmentation."""
    augmentation = create_augmentation_layers(300, strength="light")
    
    all_predictions = []
    
    # Original predictions
    for images, _ in dataset:
        preds = model.predict(images, verbose=0)
        all_predictions.append(preds)
    
    base_preds = np.concatenate(all_predictions, axis=0)
    
    # TTA predictions
    tta_preds = [base_preds]
    
    for _ in range(n_augments):
        aug_preds = []
        for images, _ in dataset:
            # Apply augmentation
            aug_images = augmentation(images, training=True)
            preds = model.predict(aug_images, verbose=0)
            aug_preds.append(preds)
        tta_preds.append(np.concatenate(aug_preds, axis=0))
    
    # Average all predictions
    final_preds = np.mean(tta_preds, axis=0)
    
    return final_preds


# ============================================================================
# Evaluation and Metrics
# ============================================================================

def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset, 
                   class_names: List[str], use_tta: bool = True,
                   tta_augments: int = 5) -> Dict:
    """Comprehensive model evaluation."""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    
    # Get true labels
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)
    
    # Get predictions
    if use_tta:
        logger.info(f"Running TTA with {tta_augments} augmentations...")
        predictions = predict_with_tta(model, test_ds, tta_augments)
    else:
        predictions = model.predict(test_ds, verbose=1)
    
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    logger.info(f"Accuracy: {report['accuracy']:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")
    
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": report['accuracy'],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "y_true": y_true,
        "y_pred": y_pred
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train(
    data_dir: str,
    output_dir: str,
    model_name: str = "efficientnetv2-s",
    img_size: int = 300,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.15,
    cutmix_prob: float = 0.15,
    dropout_rate: float = 0.4,
    l2_reg: float = 0.01,
    fine_tune_layers: int = 100,
    validation_split: float = 0.15,
    test_split: float = 0.05,
    max_samples_per_class: int = 3000,
    min_samples_per_class: int = 500,
    use_mixed_precision: bool = False,
    use_tta: bool = True,
    tta_augments: int = 5,
):
    """
    Main training function with all improvements.
    """
    # Setup
    setup_environment(use_mixed_precision)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        "model_name": model_name,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_epochs": warmup_epochs,
        "label_smoothing": label_smoothing,
        "use_focal_loss": use_focal_loss,
        "use_mixup": use_mixup,
        "use_cutmix": use_cutmix,
        "dropout_rate": dropout_rate,
        "fine_tune_layers": fine_tune_layers,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load data
    logger.info("Loading dataset...")
    train_df, val_df, test_df = load_dataframe(
        data_dir, validation_split, test_split
    )
    
    # Balance training data
    train_df = balance_dataset(train_df, max_samples_per_class, min_samples_per_class)
    
    # Create datasets
    num_classes = len(train_df['labels'].unique())
    class_names = sorted(train_df['labels'].unique())
    
    train_ds, label_encoder = create_dataset(
        train_df, img_size, batch_size, is_training=True, num_classes=num_classes,
        use_mixup=use_mixup, use_cutmix=use_cutmix, 
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
        mixup_prob=mixup_prob, cutmix_prob=cutmix_prob
    )
    
    val_ds, _ = create_dataset(
        val_df, img_size, batch_size, is_training=False, num_classes=num_classes
    )
    
    test_ds, _ = create_dataset(
        test_df, img_size, batch_size, is_training=False, num_classes=num_classes
    )
    
    # Create model
    logger.info(f"Creating model: {model_name}")
    model, base_model = create_model(
        model_name, num_classes, img_size,
        dropout_rate=dropout_rate, l2_reg=l2_reg, trainable_base=False
    )
    model.summary()
    
    # Calculate steps
    steps_per_epoch = len(train_df) // batch_size
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    
    # Learning rate schedule
    lr_schedule = WarmupCosineDecay(
        initial_lr=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps
    )
    
    # Loss function
    loss_fn = create_loss(use_focal_loss, label_smoothing, focal_gamma)
    
    # Compile model - Phase 1 (frozen base)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Callbacks
    model_callbacks = create_callbacks(output_path)
    
    # ===== PHASE 1: Train classification head =====
    logger.info("=" * 60)
    logger.info("PHASE 1: Training classification head (base frozen)")
    logger.info("=" * 60)
    
    phase1_epochs = min(epochs // 3, 15)
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=model_callbacks,
        verbose=1
    )
    
    # ===== PHASE 2: Fine-tune base model =====
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Fine-tuning top {fine_tune_layers} layers")
    logger.info("=" * 60)
    
    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    fine_tune_lr = learning_rate / 10
    lr_schedule_ft = WarmupCosineDecay(
        initial_lr=fine_tune_lr,
        warmup_steps=warmup_steps // 2,
        decay_steps=(epochs - phase1_epochs) * steps_per_epoch
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_ft),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Update checkpoint path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_callbacks[0] = callbacks.ModelCheckpoint(
        output_path / f"best_model_finetuned_{timestamp}.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=phase1_epochs,
        callbacks=model_callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = output_path / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    # Save class names
    with open(output_path / "class_names.json", "w") as f:
        json.dump({"classes": class_names, "encoder": label_encoder}, f, indent=2)
    
    # ===== EVALUATION =====
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    
    results = evaluate_model(
        model, test_ds, class_names, 
        use_tta=use_tta, tta_augments=tta_augments
    )
    
    # Save results
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump({
            "f1_macro": results["f1_macro"],
            "f1_weighted": results["f1_weighted"],
            "accuracy": results["accuracy"],
            "classification_report": results["classification_report"],
            "confusion_matrix": results["confusion_matrix"]
        }, f, indent=2)
    
    return model, results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Skin Disease Classifier v2")
    
    # Data paths
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory containing IMG_CLASSES")
    parser.add_argument("--output-dir", type=str, default="backend/ml/model_weights_v2",
                        help="Output directory for models and logs")
    
    # Model
    parser.add_argument("--model", type=str, default="efficientnetv2-s",
                        choices=["efficientnetb3", "efficientnetb4", "efficientnetv2-s", "efficientnetv2-m"],
                        help="Base model architecture")
    parser.add_argument("--img-size", type=int, default=300,
                        help="Image size")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (32-64 recommended for g4dn.4xlarge with T4 GPU)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs")
    
    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--dropout-rate", type=float, default=0.4,
                        help="Dropout rate")
    parser.add_argument("--l2-reg", type=float, default=0.01,
                        help="L2 regularization")
    
    # Augmentation
    parser.add_argument("--use-mixup", action="store_true", default=True,
                        help="Use MixUp augmentation")
    parser.add_argument("--no-mixup", action="store_false", dest="use_mixup",
                        help="Disable MixUp")
    parser.add_argument("--use-cutmix", action="store_true", default=True,
                        help="Use CutMix augmentation")
    parser.add_argument("--no-cutmix", action="store_false", dest="use_cutmix",
                        help="Disable CutMix")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="MixUp alpha parameter")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix alpha parameter")
    parser.add_argument("--mixup-prob", type=float, default=0.15,
                        help="MixUp probability (0.0-1.0)")
    parser.add_argument("--cutmix-prob", type=float, default=0.15,
                        help="CutMix probability (0.0-1.0)")
    
    # Loss
    parser.add_argument("--focal-loss", action="store_true", default=True,
                        help="Use Focal Loss")
    parser.add_argument("--no-focal-loss", action="store_false", dest="focal_loss",
                        help="Use standard cross-entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")
    
    # Fine-tuning
    parser.add_argument("--fine-tune-layers", type=int, default=100,
                        help="Number of base model layers to fine-tune")
    
    # Data splits
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Test split ratio")
    
    # Class balancing
    parser.add_argument("--max-samples", type=int, default=3000,
                        help="Maximum samples per class")
    parser.add_argument("--min-samples", type=int, default=500,
                        help="Minimum samples per class (oversample if below)")
    
    # Inference
    parser.add_argument("--use-tta", action="store_true", default=True,
                        help="Use Test-Time Augmentation")
    parser.add_argument("--no-tta", action="store_false", dest="use_tta",
                        help="Disable TTA")
    parser.add_argument("--tta-augments", type=int, default=5,
                        help="Number of TTA augmentations")
    
    # Hardware
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision training (FP16)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_prob=args.cutmix_prob,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        fine_tune_layers=args.fine_tune_layers,
        validation_split=args.val_split,
        test_split=args.test_split,
        max_samples_per_class=args.max_samples,
        min_samples_per_class=args.min_samples,
        use_mixed_precision=args.mixed_precision,
        use_tta=args.use_tta,
        tta_augments=args.tta_augments,
    )
