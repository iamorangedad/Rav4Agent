"""Async task service for document indexing using background threads."""
import os
import uuid
import time
import logging
import threading
from typing import Dict, Optional, Any
from enum import Enum
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.core.vector_store import create_vector_store, VectorStoreProvider
from app.services.model_service import ModelService
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexingTask:
    """Represents a document indexing task."""
    
    def __init__(
        self,
        task_id: str,
        filename: str,
        status: TaskStatus = TaskStatus.PENDING,
        progress: int = 0,
        message: str = "",
        created_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None
    ):
        self.task_id = task_id
        self.filename = filename
        self.status = status
        self.progress = progress
        self.message = message
        self.created_at = created_at or datetime.now()
        self.completed_at = completed_at
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }


class AsyncIndexingService:
    """Background service for async document indexing."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._tasks: Dict[str, IndexingTask] = {}
        self._task_lock = threading.Lock()
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Services
        settings = get_settings()
        self.upload_dir = settings.upload_dir
        self.model_service = ModelService()
        
        # Initialize vector store
        store_type = settings.effective_vector_store_type
        if store_type == "chroma":
            self.vector_store_provider = create_vector_store(
                "chroma",
                host=settings.chroma_host,
                port=settings.chroma_port,
                collection=settings.chroma_collection
            )
        else:
            self.vector_store_provider = create_vector_store(
                "simple",
                persist_dir=settings.chroma_dir
            )
        
        # Start background worker
        self._start_worker()
        logger.info("[AsyncIndexing] Service initialized")
    
    def _start_worker(self):
        """Start background worker thread."""
        if self._background_thread is not None and self._background_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._background_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="IndexingWorker"
        )
        self._background_thread.start()
        logger.info("[AsyncIndexing] Background worker started")
    
    def _worker_loop(self):
        """Background worker that processes pending tasks."""
        while not self._stop_event.is_set():
            pending_tasks = self.get_tasks_by_status(TaskStatus.PENDING)
            
            for task in pending_tasks:
                self._process_task(task)
            
            # Wait before checking for new tasks
            time.sleep(1)
    
    def _process_task(self, task: IndexingTask):
        """Process a single indexing task."""
        with self._task_lock:
            task.status = TaskStatus.PROCESSING
            task.message = "Starting indexing..."
            logger.info(f"[AsyncIndexing] Processing task {task.task_id} for {task.filename}")
        
        try:
            # Get file path
            file_path = os.path.join(self.upload_dir, task.filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Update progress
            with self._task_lock:
                task.progress = 10
                task.message = "Loading document..."
            
            # Load and parse document
            from llama_index.core import SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                input_files=[file_path],
                filename_as_id=True
            ).load_data()
            
            total_docs = len(documents)
            logger.info(f"[AsyncIndexing] Loaded {total_docs} document pages")
            
            # Update progress
            with self._task_lock:
                task.progress = 30
                task.message = f"Generating embeddings ({total_docs} pages)..."
            
            # Get embedding model
            settings = get_settings()
            embed_model = self.model_service.get_provider().get_embedding_model(
                settings.default_embedding_model
            )
            
            # Process in batches for stability
            batch_size = 5
            processed = 0
            
            from llama_index.core import VectorStoreIndex
            vector_store = self.vector_store_provider.get_vector_store()
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                
                with self._task_lock:
                    task.progress = 30 + int((processed / total_docs) * 50)
                    task.message = f"Processing batch {batch_num}/{total_batches}..."
                
                # Create index for batch
                try:
                    if i == 0:
                        index = VectorStoreIndex.from_documents(
                            batch,
                            vector_store=vector_store,
                            embed_model=embed_model,
                            show_progress=False
                        )
                    else:
                        # Add to existing index
                        for doc in batch:
                            index.insert(doc, show_progress=False)
                except Exception as e:
                    logger.warning(f"[AsyncIndexing] Batch failed, retrying: {e}")
                    time.sleep(2)
                    # Retry once
                    if i == 0:
                        index = VectorStoreIndex.from_documents(
                            batch,
                            vector_store=vector_store,
                            embed_model=embed_model,
                            show_progress=False
                        )
                    else:
                        for doc in batch:
                            index.insert(doc, show_progress=False)
                
                processed += len(batch)
                logger.info(f"[AsyncIndexing] Processed batch {batch_num}/{total_batches}")
            
            # Mark as completed
            with self._task_lock:
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                task.message = f"Completed! Indexed {total_docs} pages."
                task.completed_at = datetime.now()
            
            logger.info(f"[AsyncIndexing] Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"[AsyncIndexing] Task {task.task_id} failed: {e}")
            with self._task_lock:
                task.status = TaskStatus.FAILED
                task.message = f"Failed: {str(e)}"
                task.error = str(e)
                task.completed_at = datetime.now()
    
    def create_task(self, filename: str) -> IndexingTask:
        """Create a new indexing task."""
        task_id = str(uuid.uuid4())
        task = IndexingTask(
            task_id=task_id,
            filename=filename,
            status=TaskStatus.PENDING,
            message="Waiting in queue..."
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
        
        logger.info(f"[AsyncIndexing] Created task {task_id} for {filename}")
        return task
    
    def get_task(self, task_id: str) -> Optional[IndexingTask]:
        """Get a task by ID."""
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> list[IndexingTask]:
        """Get all tasks."""
        with self._task_lock:
            return list(self._tasks.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> list[IndexingTask]:
        """Get tasks by status."""
        with self._task_lock:
            return [t for t in self._tasks.values() if t.status == status]
    
    def get_pending_tasks(self) -> list[IndexingTask]:
        """Get all pending tasks."""
        return self.get_tasks_by_status(TaskStatus.PENDING)
    
    def get_processing_tasks(self) -> list[IndexingTask]:
        """Get all processing tasks."""
        return self.get_tasks_by_status(TaskStatus.PROCESSING)
    
    def get_recent_tasks(self, limit: int = 10) -> list[IndexingTask]:
        """Get recent tasks sorted by creation time."""
        with self._task_lock:
            tasks = sorted(self._tasks.values(), key=lambda x: x.created_at, reverse=True)
            return tasks[:limit]
    
    def clear_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clear completed tasks older than specified hours."""
        cutoff = datetime.now().timestamp() - (older_than_hours * 3600)
        removed = 0
        
        with self._task_lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    if task.created_at.timestamp() < cutoff:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1
        
        if removed > 0:
            logger.info(f"[AsyncIndexing] Cleared {removed} old tasks")
        
        return removed
    
    def shutdown(self):
        """Shutdown the service."""
        logger.info("[AsyncIndexing] Shutting down...")
        self._stop_event.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5)
        logger.info("[AsyncIndexing] Shutdown complete")


# Singleton instance
def get_indexing_service() -> AsyncIndexingService:
    """Get the async indexing service singleton."""
    return AsyncIndexingService()
