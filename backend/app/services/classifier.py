"""
Skin Disease Classifier Service
Handles loading and running inference with the EfficientNet model.
"""
import os
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for TensorFlow
tf = None
keras = None


def _load_tensorflow():
    """Lazy load TensorFlow to speed up startup when not needed."""
    global tf, keras
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        keras = tf.keras
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
    return tf, keras


class SkinDiseaseClassifier:
    """
    Classifier service for skin disease detection using EfficientNet.
    Implements singleton pattern for efficient model loading.
    """
    
    def __init__(self, model_path: str, disease_classes: list[str]):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained Keras model
            disease_classes: List of disease class names in order
        """
        self.model_path = model_path
        self.disease_classes = disease_classes
        self.model = None
        self._loaded = False
        logger.info(f"SkinDiseaseClassifier initialized with {len(disease_classes)} classes")
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._loaded:
            return True
        
        try:
            tf, keras = _load_tensorflow()
            
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(f"Model file not found at {self.model_path}")
                logger.info("Using mock model for development. Train the model first for real predictions.")
                self._create_mock_model()
                return True
            
            logger.info(f"Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            self._loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock model")
            self._create_mock_model()
            return True
    
    def _create_mock_model(self):
        """Create a mock model for development/testing when real model isn't available."""
        tf, keras = _load_tensorflow()
        
        # Simple mock model that returns random predictions
        inputs = keras.layers.Input(shape=(224, 224, 3))
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = keras.layers.Dense(len(self.disease_classes), activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self._loaded = True
        logger.warning("Using MOCK model - predictions are random. Train the real model for actual use.")
    
    def predict(self, preprocessed_image: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Run inference on a preprocessed image.
        
        Args:
            preprocessed_image: Normalized image array of shape (1, 224, 224, 3)
            
        Returns:
            Tuple of (predicted_disease, confidence, all_probabilities)
        """
        if not self._loaded:
            self.load_model()
        
        # Run inference with training=False to disable augmentation layers
        # Using __call__ with training=False is more explicit than predict()
        tf, _ = _load_tensorflow()
        preprocessed_tensor = tf.constant(preprocessed_image, dtype=tf.float32)
        predictions = self.model(preprocessed_tensor, training=False)
        probabilities = predictions[0].numpy()
        
        # Get top prediction
        top_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top_idx])
        predicted_disease = self.disease_classes[top_idx]
        
        # Create probability dict for all classes
        all_probs = {
            disease: float(prob) 
            for disease, prob in zip(self.disease_classes, probabilities)
        }
        
        logger.info(f"Prediction: {predicted_disease} ({confidence:.2%})")
        return predicted_disease, confidence, all_probs
    
    def predict_top_k(self, preprocessed_image: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """
        Get top-k predictions.
        
        Args:
            preprocessed_image: Normalized image array
            k: Number of top predictions to return
            
        Returns:
            List of (disease, probability) tuples sorted by confidence
        """
        if not self._loaded:
            self.load_model()
        
        # Run inference with training=False to disable augmentation
        tf, _ = _load_tensorflow()
        preprocessed_tensor = tf.constant(preprocessed_image, dtype=tf.float32)
        predictions = self.model(preprocessed_tensor, training=False)
        probabilities = predictions[0].numpy()
        
        # Get top k indices
        top_indices = np.argsort(probabilities)[-k:][::-1]
        
        results = [
            (self.disease_classes[idx], float(probabilities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


# Singleton instance
_classifier: SkinDiseaseClassifier | None = None


def get_classifier() -> SkinDiseaseClassifier:
    """Get or create singleton classifier instance."""
    global _classifier
    if _classifier is None:
        from app.config import get_settings
        settings = get_settings()
        
        # Resolve model path relative to backend directory
        model_path = settings.model_path
        if not Path(model_path).is_absolute():
            # Resolve relative to backend/ directory (parent of app/)
            backend_dir = Path(__file__).parent.parent.parent
            model_path = str(backend_dir / model_path)
        
        _classifier = SkinDiseaseClassifier(
            model_path=model_path,
            disease_classes=settings.disease_classes
        )
    return _classifier


def preload_model():
    """Preload the model at startup."""
    classifier = get_classifier()
    classifier.load_model()
