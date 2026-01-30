"""Frontend serving routes."""
import os
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings

settings = get_settings()
router = APIRouter()

# Ensure static directory exists
os.makedirs(settings.static_dir, exist_ok=True)


@router.get("/")
async def serve_index():
    """Serve the main frontend page."""
    index_path = os.path.join(settings.static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # Fallback: return API info if frontend not built yet
    return {
        "message": "Doc-Chat API",
        "docs": "/docs",
        "health": "/health"
    }
