"""API routes module."""
from fastapi import APIRouter

from app.api.routes import documents, chat, models, health, frontend, tasks

# Create main API router
api_router = APIRouter()

# Include all routes
api_router.include_router(frontend.router, tags=["frontend"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
