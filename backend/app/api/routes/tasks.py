"""Task management routes for async document indexing."""
from fastapi import APIRouter, HTTPException

from app.services import get_indexing_service
from app.models.schemas import (
    TaskListResponse,
    IndexingTaskResponse,
    ProcessingStatusResponse
)

router = APIRouter()


@router.get("/tasks", response_model=TaskListResponse)
async def get_all_tasks():
    """Get all indexing tasks."""
    try:
        indexing_service = get_indexing_service()
        tasks = indexing_service.get_recent_tasks(20)
        
        pending_count = len(indexing_service.get_pending_tasks())
        processing_count = len(indexing_service.get_processing_tasks())
        
        return TaskListResponse(
            tasks=[IndexingTaskResponse(**t.to_dict()) for t in tasks],
            pending_count=pending_count,
            processing_count=processing_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=IndexingTaskResponse)
async def get_task(task_id: str):
    """Get a specific task by ID."""
    try:
        indexing_service = get_indexing_service()
        task = indexing_service.get_task(task_id)
        
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return IndexingTaskResponse(**task.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing-status", response_model=ProcessingStatusResponse)
async def get_processing_status():
    """Get current processing status summary."""
    try:
        indexing_service = get_indexing_service()
        
        pending_tasks = indexing_service.get_pending_tasks()
        processing_tasks = indexing_service.get_processing_tasks()
        
        return ProcessingStatusResponse(
            has_pending_tasks=len(pending_tasks) > 0,
            has_processing_tasks=len(processing_tasks) > 0,
            pending_count=len(pending_tasks),
            processing_count=len(processing_tasks),
            tasks=[IndexingTaskResponse(**t.to_dict()) 
                   for t in processing_tasks + pending_tasks[:5]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/old", response_model=dict)
async def clear_old_tasks(older_than_hours: int = 24):
    """Clear completed tasks older than specified hours."""
    try:
        indexing_service = get_indexing_service()
        count = indexing_service.clear_completed_tasks(older_than_hours)
        return {"message": f"Cleared {count} old tasks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
