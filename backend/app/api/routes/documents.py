"""Document management routes."""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services import DocumentService, ChatService, get_indexing_service
from app.services.indexing_service import TaskStatus
from app.models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DeleteResponse,
    VectorKnowledgeBaseStatusResponse,
    DocumentVectorStatus
)

router = APIRouter()
document_service = DocumentService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and start async indexing."""
    try:
        result = document_service.save_document(file.filename, file.file)
        
        # Create async indexing task
        indexing_service = get_indexing_service()
        task = indexing_service.create_task(result['filename'])
        
        return UploadResponse(
            message=f"File {result['filename']} uploaded successfully, indexing started",
            filename=result['filename'],
            size=result['size'],
            task_id=task.task_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=DocumentListResponse)
async def get_documents():
    """Get list of uploaded documents."""
    try:
        documents = document_service.list_documents()
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector-status", response_model=VectorKnowledgeBaseStatusResponse)
async def get_vector_knowledge_base_status():
    """Get vector knowledge base status for all documents."""
    try:
        indexing_service = get_indexing_service()
        
        # Get all uploaded documents
        uploaded_files = document_service.list_documents()
        
        # Get all indexing tasks
        all_tasks = indexing_service.get_all_tasks()
        
        # Create a mapping of filename to latest task
        file_to_task = {}
        for task in all_tasks:
            # Keep the most recent task for each file
            if task.filename not in file_to_task:
                file_to_task[task.filename] = task
            elif task.created_at > file_to_task[task.filename].created_at:
                file_to_task[task.filename] = task
        
        # Build status for each document
        document_statuses = []
        indexed_count = 0
        processing_count = 0
        pending_count = 0
        failed_count = 0
        
        for filename in uploaded_files:
            task = file_to_task.get(filename)
            
            if task is None:
                # File uploaded but no indexing task created
                status = DocumentVectorStatus(
                    filename=filename,
                    is_uploaded=True,
                    is_indexed=False,
                    indexing_status="none",
                    progress=0,
                    chunk_count=0,
                    message="Not indexed yet"
                )
            else:
                # Determine indexing state
                is_indexed = task.status == TaskStatus.COMPLETED
                is_processing = task.status == TaskStatus.PROCESSING
                is_pending = task.status == TaskStatus.PENDING
                is_failed = task.status == TaskStatus.FAILED
                
                # Count by status
                if is_indexed:
                    indexed_count += 1
                elif is_processing:
                    processing_count += 1
                elif is_pending:
                    pending_count += 1
                elif is_failed:
                    failed_count += 1
                
                # Set appropriate message
                if is_indexed:
                    message = "Ready for chat ✓"
                elif is_processing:
                    message = task.message or f"Indexing... {task.progress}%"
                elif is_pending:
                    message = "Waiting to index..."
                elif is_failed:
                    message = f"Failed: {task.error or 'Unknown error'}"
                else:
                    message = "Unknown status"
                
                status = DocumentVectorStatus(
                    filename=filename,
                    is_uploaded=True,
                    is_indexed=is_indexed,
                    indexing_status=task.status.value,
                    progress=task.progress if is_processing or is_indexed else 0,
                    chunk_count=0,  # Could be enhanced to track actual chunks
                    message=message
                )
            
            document_statuses.append(status)
        
        return VectorKnowledgeBaseStatusResponse(
            total_documents=len(uploaded_files),
            indexed_documents=indexed_count,
            processing_documents=processing_count,
            pending_documents=pending_count,
            failed_documents=failed_count,
            documents=document_statuses
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str):
    """Delete a document."""
    try:
        document_service.delete_document(filename)
        # Clear all conversations as documents changed
        chat_service = ChatService()
        chat_service.clear_all_conversations()
        
        return DeleteResponse(message=f"File {filename} deleted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
