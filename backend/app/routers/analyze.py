"""
API Router for skin analysis endpoints
"""
import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.models.database import get_db, Analysis
from app.models.schemas import (
    AnalysisResponse, 
    HealthResponse, 
    DiseasesListResponse,
    DiseaseInfo,
    AnalysisHistoryItem,
    ErrorResponse
)
from app.services.image_processor import get_image_processor
from app.services.classifier import get_classifier
from app.services.llm_advisor import get_llm_advisor
from app.config import get_settings, DISEASE_INFO

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["Skin Analysis"])


@router.post(
    "/analyze_skin",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Analyze skin image for disease detection",
    description="Upload a skin image to detect potential skin diseases and receive AI-generated recommendations."
)
async def analyze_skin(
    file: UploadFile = File(..., description="Skin image file (jpg, jpeg, png, webp)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a skin image for disease detection.
    
    This endpoint:
    1. Validates and preprocesses the uploaded image
    2. Runs classification using EfficientNet model
    3. Generates recommendations using LLM
    4. Stores results in database
    5. Returns complete analysis with recommendations
    """
    try:
        # Get service instances
        image_processor = get_image_processor()
        classifier = get_classifier()
        llm_advisor = get_llm_advisor()
        
        # Step 1: Validate and read image
        logger.info(f"Processing image: {file.filename}")
        image_bytes = await image_processor.validate_image(file)
        image_hash = image_processor.compute_hash(image_bytes)
        
        # Step 2: Preprocess image
        preprocessed = image_processor.preprocess(image_bytes)
        
        # Step 3: Run classification
        disease, confidence, _ = classifier.predict(preprocessed)
        logger.info(f"Classification result: {disease} ({confidence:.2%})")
        
        # Step 4: Generate LLM recommendations
        recommendations = await llm_advisor.generate_recommendations(disease, confidence)
        logger.info("LLM recommendations generated")
        
        # Step 5: Get severity from disease info
        disease_info = DISEASE_INFO.get(disease, {"severity": "unknown"})
        severity = disease_info.get("severity", "unknown")
        
        # Step 6: Store in database
        try:
            analysis = Analysis(
                image_hash=image_hash,
                disease=disease,
                confidence=confidence,
                recommendations=recommendations.get("recommendations", ""),
                next_steps=recommendations.get("next_steps", ""),
                tips=recommendations.get("tips", "")
            )
            db.add(analysis)
            await db.commit()
            logger.info(f"Analysis stored with ID: {analysis.id}")
        except Exception as db_error:
            logger.warning(f"Failed to store analysis in database: {db_error}")
            # Continue without database storage
        
        # Step 7: Return response
        return AnalysisResponse(
            disease=disease,
            confidence=round(confidence, 4),
            recommendations=recommendations.get("recommendations", ""),
            next_steps=recommendations.get("next_steps", ""),
            tips=recommendations.get("tips", ""),
            severity=severity
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the API and its components."
)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check API health and component status."""
    classifier = get_classifier()
    llm_advisor = get_llm_advisor()
    
    # Check database connection
    db_connected = False
    try:
        await db.execute(select(1))
        db_connected = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=classifier.is_loaded,
        llm_loaded=llm_advisor.is_loaded(),
        database_connected=db_connected
    )


@router.get(
    "/diseases",
    response_model=DiseasesListResponse,
    summary="List supported diseases",
    description="Get information about all skin diseases the model can detect."
)
async def list_diseases():
    """Get list of all supported skin diseases with information."""
    diseases = []
    for name in settings.disease_classes:
        info = DISEASE_INFO.get(name, {})
        diseases.append(DiseaseInfo(
            name=name,
            severity=info.get("severity", "unknown"),
            contagious=info.get("contagious", False),
            description=info.get("description", "")
        ))
    
    return DiseasesListResponse(diseases=diseases, total=len(diseases))


@router.get(
    "/history",
    response_model=list[AnalysisHistoryItem],
    summary="Get analysis history",
    description="Retrieve recent skin analysis history from the database."
)
async def get_history(
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get recent analysis history."""
    try:
        result = await db.execute(
            select(Analysis)
            .order_by(desc(Analysis.created_at))
            .offset(offset)
            .limit(min(limit, 100))  # Cap at 100
        )
        analyses = result.scalars().all()
        
        return [
            AnalysisHistoryItem(
                id=a.id,
                disease=a.disease,
                confidence=a.confidence,
                recommendations=a.recommendations or "",
                next_steps=a.next_steps or "",
                tips=a.tips or "",
                created_at=a.created_at,
                image_hash=a.image_hash
            )
            for a in analyses
        ]
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")
