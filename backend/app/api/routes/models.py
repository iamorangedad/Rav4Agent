"""Model management routes."""
from fastapi import APIRouter

from app.services import ModelService
from app.models.schemas import (
    ModelsListResponse,
    RecommendedModelsResponse,
    ModelInfo
)

router = APIRouter()
model_service = ModelService()


@router.get("", response_model=ModelsListResponse)
async def get_models():
    """Get list of available models from Ollama."""
    models = model_service.list_models()
    return ModelsListResponse(
        models=[ModelInfo(**m) for m in models],
        count=len(models)
    )


@router.get("/recommended", response_model=RecommendedModelsResponse)
async def get_recommended_models():
    """Get recommended models for document chat."""
    return RecommendedModelsResponse(**model_service.get_recommended_models())
