"""
Skin Disease Detection API - FastAPI Application
Main entry point for the backend service.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.database import init_db
from app.routers.analyze import router as analyze_router
from app.services.classifier import preload_model
from app.services.llm_advisor import preload_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Skin Disease Detection API...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.info("Continuing without database - history features will be unavailable")
    
    # Preload ML model
    try:
        logger.info("Loading classification model...")
        preload_model()
        logger.info("Classification model loaded")
    except Exception as e:
        logger.error(f"Failed to load classification model: {e}")
    
    # Preload LLM (optional - can be lazy loaded)
    if not settings.debug:
        try:
            logger.info("Loading LLM model...")
            preload_llm()
            logger.info("LLM model loaded")
        except Exception as e:
            logger.warning(f"LLM preload failed (will load on first request): {e}")
    
    logger.info(f"API ready at http://{settings.backend_host}:{settings.backend_port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Skin Disease Detection API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## Skin Disease Detection & LLM Advisor API

This API analyzes skin images to detect potential skin diseases using deep learning 
and provides AI-generated recommendations.

### Features
- **Image Analysis**: Upload skin images for disease classification
- **10 Disease Classes**: Supports detection of common skin conditions
- **LLM Recommendations**: Get AI-generated treatment recommendations
- **History Tracking**: Store and retrieve past analyses

### Supported Diseases
1. Eczema
2. Melanoma
3. Atopic Dermatitis
4. Basal Cell Carcinoma (BCC)
5. Melanocytic Nevi (NV)
6. Benign Keratosis-like Lesions (BKL)
7. Psoriasis and Lichen Planus
8. Seborrheic Keratoses and Benign Tumors
9. Tinea Ringworm and Fungal Infections
10. Warts Molluscum and Viral Infections

### Disclaimer
This is an AI-powered tool for informational purposes only. It is NOT a substitute 
for professional medical advice, diagnosis, or treatment.
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze_router, prefix="/api/v1")

# Also mount at root for convenience
app.include_router(analyze_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug
    )
