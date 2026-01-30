"""Health check routes."""
from fastapi import APIRouter

from app.config import get_settings
from app.services import DocumentService
from app.models.schemas import HealthResponse

router = APIRouter()
document_service = DocumentService()


@router.get("", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        vector_store=settings.effective_vector_store_type,
        ollama_url=settings.ollama_base_url,
        documents_count=document_service.get_document_count()
    )
