"""Document service for file operations."""
import os
import shutil
import logging
from typing import List

from app.config import get_settings
from app.utils.security import validate_filename, is_path_within_directory

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document operations."""
    
    def __init__(self, upload_dir: str = None):
        """
        Initialize document service.
        
        Args:
            upload_dir: Directory for uploaded documents
        """
        settings = get_settings()
        self.upload_dir = upload_dir or settings.upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def save_document(self, filename: str, file_content) -> dict:
        """
        Save uploaded document.
        
        Args:
            filename: Name of the file
            file_content: File content (file-like object)
            
        Returns:
            dict: File information
            
        Raises:
            ValueError: If filename is invalid
        """
        if not validate_filename(filename):
            logger.warning(f"[Upload] Invalid filename rejected: {filename}")
            raise ValueError("Invalid filename")
        
        file_path = os.path.join(self.upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_content, buffer)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"[Upload] File saved: {filename} ({file_size} bytes)")
        
        return {
            "filename": filename,
            "size": file_size,
        }
    
    def list_documents(self) -> List[str]:
        """
        List all uploaded documents.
        
        Returns:
            List[str]: List of filenames
        """
        try:
            return os.listdir(self.upload_dir)
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete a document.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            ValueError: If filename is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not validate_filename(filename):
            raise ValueError("Invalid filename")
        
        file_path = os.path.join(self.upload_dir, filename)
        
        # Security check: ensure file is within upload directory
        if not is_path_within_directory(file_path, self.upload_dir):
            raise ValueError("Invalid file path")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {filename}")
        
        os.remove(file_path)
        logger.info(f"[Delete] File deleted: {filename}")
        return True
    
    def document_exists(self, filename: str) -> bool:
        """Check if a document exists."""
        if not validate_filename(filename):
            return False
        file_path = os.path.join(self.upload_dir, filename)
        return os.path.exists(file_path)
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        try:
            return len(os.listdir(self.upload_dir))
        except Exception:
            return 0
