"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class AnalysisRequest(BaseModel):
    """Request model for skin analysis (used for documentation)."""
    pass  # File is sent via multipart/form-data


class DiseaseResult(BaseModel):
    """Basic disease classification result."""
    disease: str = Field(..., description="Detected skin disease name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class AnalysisResponse(BaseModel):
    """Complete analysis response with LLM recommendations."""
    disease: str = Field(..., description="Detected skin disease name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    recommendations: str = Field(..., description="LLM-generated treatment recommendations")
    next_steps: str = Field(..., description="Recommended next steps for the patient")
    tips: str = Field(..., description="General care tips and advice")
    severity: str = Field(..., description="Severity level: benign, mild, mild-moderate, serious")
    disclaimer: str = Field(
        default="This is an AI-generated analysis for informational purposes only. "
                "It is not a substitute for professional medical advice, diagnosis, or treatment. "
                "Please consult a qualified dermatologist for proper evaluation.",
        description="Medical disclaimer"
    )


class AnalysisHistoryItem(BaseModel):
    """Analysis history entry."""
    id: int
    disease: str
    confidence: float
    recommendations: str
    next_steps: str
    tips: str
    created_at: datetime
    image_hash: Optional[str] = None

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}
    
    status: str = "healthy"
    version: str
    model_loaded: bool
    llm_loaded: bool
    database_connected: bool


class DiseaseInfo(BaseModel):
    """Information about a supported disease class."""
    name: str
    severity: str
    contagious: bool
    description: str


class DiseasesListResponse(BaseModel):
    """List of all supported disease classes."""
    diseases: list[DiseaseInfo]
    total: int


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_code: Optional[str] = None
