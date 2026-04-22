"""
Image Preprocessing Service
Handles image validation, resizing, and normalization for model inference.
"""
import io
import hashlib
from PIL import Image
import numpy as np
from fastapi import UploadFile, HTTPException
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class ImageProcessor:
    """Service for preprocessing skin images before model inference."""
    
    def __init__(self, image_size: int = 300):
        """
        Initialize the image processor.
        
        Args:
            image_size: Target size for resizing (default 300 for EfficientNet-B3)
        """
        self.image_size = image_size
        logger.info(f"ImageProcessor initialized with size {image_size}x{image_size}")
    
    async def validate_image(self, file: UploadFile) -> bytes:
        """
        Validate uploaded image file.
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Image bytes if valid
            
        Raises:
            HTTPException: If image is invalid
        """
        # Check filename and extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        ext = "." + file.filename.lower().split(".")[-1] if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validate it's actually an image
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        return content
    
    def compute_hash(self, image_bytes: bytes) -> str:
        """Compute SHA256 hash of image for deduplication."""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        EfficientNet has built-in preprocessing, so we keep values in [0, 255].
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed numpy array ready for model input
            Shape: (1, image_size, image_size, 3)
        """
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize with high-quality resampling
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array as float32
        # EfficientNet expects [0, 255] range - it has built-in preprocessing
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.debug(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    
    def preprocess_for_display(self, image_bytes: bytes) -> Image.Image:
        """
        Preprocess image for display (resize only, no normalization).
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            PIL Image resized for display
        """
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize maintaining aspect ratio for display
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        return img


# Singleton instance
_image_processor: ImageProcessor | None = None


def get_image_processor() -> ImageProcessor:
    """Get or create singleton ImageProcessor instance."""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor(image_size=settings.image_size)
    return _image_processor
