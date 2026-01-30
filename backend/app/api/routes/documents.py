"""Document management routes."""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services import DocumentService, ChatService
from app.models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DeleteResponse
)

router = APIRouter()
document_service = DocumentService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document."""
    try:
        result = document_service.save_document(file.filename, file.file)
        return UploadResponse(
            message=f"File {result['filename']} uploaded successfully",
            filename=result['filename'],
            size=result['size']
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
