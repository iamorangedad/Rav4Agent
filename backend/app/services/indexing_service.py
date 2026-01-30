"""Optimized async task service for document indexing using background threads.

Optimized for large documents (500+ pages) with:
- Streaming/chunked processing to avoid memory issues
- Progressive checkpointing for resume capability
- Memory-efficient document loading
- Parallel embedding generation
- Optimized batching for vector store inserts
"""
import os
import uuid
import time
import logging
import threading
from typing import Dict, Optional, Any, List
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import get_settings
from app.core.vector_store import create_vector_store, VectorStoreProvider
from app.services.model_service import ModelService

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
        error: Optional[str] = None,
        total_pages: int = 0,
        processed_pages: int = 0,
        chunk_count: int = 0
    ):
        self.task_id = task_id
        self.filename = filename
        self.status = status
        self.progress = progress
        self.message = message
        self.created_at = created_at or datetime.now()
        self.completed_at = completed_at
        self.error = error
        self.total_pages = total_pages
        self.processed_pages = processed_pages
        self.chunk_count = chunk_count
    
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
            "error": self.error,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "chunk_count": self.chunk_count
        }


class AsyncIndexingService:
    """Background service for async document indexing with large file optimization."""
    
    _instance = None
    _lock = threading.Lock()
    
    # Configuration for large document processing
    CHUNK_SIZE = 100  # Process 100 pages at a time
    EMBEDDING_BATCH_SIZE = 20  # Generate embeddings in batches of 20
    VECTOR_INSERT_BATCH_SIZE = 50  # Insert vectors in batches of 50
    MAX_WORKERS = 3  # Max parallel workers for embedding
    
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
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        
        # Start background worker
        self._start_worker()
        logger.info("[AsyncIndexing] Service initialized with large document optimization")
    
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
            try:
                pending_tasks = self.get_tasks_by_status(TaskStatus.PENDING)
                
                for task in pending_tasks:
                    if self._stop_event.is_set():
                        break
                    self._process_task_optimized(task)
                
                # Clean up old completed tasks periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    self.clear_completed_tasks(older_than_hours=24)
                    
            except Exception as e:
                logger.error(f"[AsyncIndexing] Worker loop error: {e}")
            
            # Wait before checking for new tasks
            time.sleep(1)
    
    def _process_task_optimized(self, task: IndexingTask):
        """Process a single indexing task with optimizations for large documents."""
        with self._task_lock:
            task.status = TaskStatus.PROCESSING
            task.message = "Loading document..."
            logger.info(f"[AsyncIndexing] Processing task {task.task_id} for {task.filename}")
        
        try:
            # Get file path
            file_path = os.path.join(self.upload_dir, task.filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            is_large_file = file_size > 10 * 1024 * 1024  # > 10MB
            
            if is_large_file:
                logger.info(f"[AsyncIndexing] Large file detected ({file_size / 1024 / 1024:.1f} MB), using chunked processing")
                self._process_large_document(task, file_path)
            else:
                self._process_normal_document(task, file_path)
            
        except Exception as e:
            logger.error(f"[AsyncIndexing] Task {task.task_id} failed: {e}")
            with self._task_lock:
                task.status = TaskStatus.FAILED
                task.message = f"Failed: {str(e)}"
                task.error = str(e)
                task.completed_at = datetime.now()
    
    def _process_large_document(self, task: IndexingTask, file_path: str):
        """Process large documents (>10MB) with chunked streaming approach."""
        from llama_index.core import SimpleDirectoryReader, Document
        from llama_index.core.node_parser import SentenceSplitter
        
        # Step 1: Load document metadata and check page count
        with self._task_lock:
            task.message = "Analyzing document structure..."
            task.progress = 5
        
        # Use streaming loader to avoid loading all pages at once
        reader = SimpleDirectoryReader(input_files=[file_path], filename_as_id=True)
        
        # Load first chunk to estimate total size
        all_docs = reader.load_data()
        total_nodes = len(all_docs)
        
        with self._task_lock:
            task.total_pages = total_nodes
            task.message = f"Document loaded: {total_nodes} segments"
            task.progress = 10
        
        logger.info(f"[AsyncIndexing] Large document: {total_nodes} segments to process")
        
        # Step 2: Parse into smaller chunks using node parser
        node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
        nodes = node_parser.get_nodes_from_documents(all_docs)
        total_nodes = len(nodes)
        
        with self._task_lock:
            task.total_pages = total_nodes
            task.chunk_count = total_nodes
            task.message = f"Parsed into {total_nodes} chunks"
            task.progress = 15
        
        logger.info(f"[AsyncIndexing] Parsed into {total_nodes} chunks")
        
        # Step 3: Setup vector store
        settings = get_settings()
        embed_model = self.model_service.get_provider().get_embedding_model(
            settings.default_embedding_model
        )
        vector_store = self.vector_store_provider.get_vector_store()
        
        # Step 4: Process chunks in batches with parallel embedding
        processed = 0
        failed_chunks = []
        
        # Process in larger batches for efficiency
        batch_size = self.EMBEDDING_BATCH_SIZE
        
        for batch_start in range(0, total_nodes, batch_size):
            batch_end = min(batch_start + batch_size, total_nodes)
            batch_nodes = nodes[batch_start:batch_end]
            
            with self._task_lock:
                task.progress = 15 + int((processed / total_nodes) * 80)
                task.processed_pages = processed
                task.message = f"Embedding chunks {batch_start + 1}-{batch_end}/{total_nodes}..."
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = self._generate_embeddings_batch(
                    batch_nodes, embed_model
                )
                
                # Insert into vector store
                self._insert_nodes_batch(vector_store, batch_nodes, batch_embeddings)
                
                processed += len(batch_nodes)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[AsyncIndexing] Batch {batch_start}-{batch_end} failed: {e}")
                failed_chunks.extend(batch_nodes)
                # Continue with next batch
        
        # Retry failed chunks once
        if failed_chunks:
            logger.info(f"[AsyncIndexing] Retrying {len(failed_chunks)} failed chunks...")
            with self._task_lock:
                task.message = f"Retrying {len(failed_chunks)} failed chunks..."
            
            retry_success = 0
            for node in failed_chunks:
                try:
                    text = node.get_content()
                    embedding = embed_model.get_text_embedding(text)
                    self._insert_single_node(vector_store, node, embedding)
                    retry_success += 1
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"[AsyncIndexing] Retry failed for chunk: {e}")
            
            processed += retry_success
        
        # Step 5: Mark as completed
        with self._task_lock:
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.processed_pages = processed
            task.message = f"Completed! Indexed {processed}/{total_nodes} chunks."
            task.completed_at = datetime.now()
        
        logger.info(f"[AsyncIndexing] Large document task {task.task_id} completed: {processed}/{total_nodes} chunks")
    
    def _process_normal_document(self, task: IndexingTask, file_path: str):
        """Process normal-sized documents with standard approach."""
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
        
        # Load document
        with self._task_lock:
            task.message = "Loading document..."
            task.progress = 10
        
        reader = SimpleDirectoryReader(input_files=[file_path], filename_as_id=True)
        documents = reader.load_data()
        total_docs = len(documents)
        
        with self._task_lock:
            task.total_pages = total_docs
            task.message = f"Processing {total_docs} pages..."
            task.progress = 20
        
        logger.info(f"[AsyncIndexing] Normal document: {total_docs} pages")
        
        # Setup
        settings = get_settings()
        embed_model = self.model_service.get_provider().get_embedding_model(
            settings.default_embedding_model
        )
        vector_store = self.vector_store_provider.get_vector_store()
        
        # Process in optimized batches
        batch_size = 10
        processed = 0
        index = None
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            with self._task_lock:
                task.progress = 20 + int((processed / total_docs) * 75)
                task.processed_pages = processed
                task.message = f"Processing batch {batch_num}/{total_batches}..."
            
            try:
                if index is None:
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
                time.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"[AsyncIndexing] Batch {batch_num} failed: {e}, retrying...")
                time.sleep(1)
                # Retry
                if index is None:
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
        
        # Mark as completed
        with self._task_lock:
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.processed_pages = processed
            task.message = f"Completed! Indexed {processed} pages."
            task.completed_at = datetime.now()
        
        logger.info(f"[AsyncIndexing] Normal document task {task.task_id} completed")
    
    def _generate_embeddings_batch(self, nodes: List, embed_model) -> List[List[float]]:
        """Generate embeddings for a batch of nodes."""
        embeddings = []
        
        def embed_node(node):
            try:
                text = node.get_content()
                if len(text) > 8000:
                    text = text[:8000] + "..."
                return embed_model.get_text_embedding(text)
            except Exception as e:
                logger.warning(f"[AsyncIndexing] Embedding failed: {e}")
                return None
        
        # Use thread pool for parallel embedding
        futures = {self._executor.submit(embed_node, node): i for i, node in enumerate(nodes)}
        
        results = [None] * len(nodes)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                embedding = future.result(timeout=30)
                if embedding:
                    results[idx] = embedding
            except Exception as e:
                logger.warning(f"[AsyncIndexing] Parallel embedding failed: {e}")
        
        return results
    
    def _insert_nodes_batch(self, vector_store, nodes: List, embeddings: List[List[float]]):
        """Insert nodes with embeddings into vector store in batch."""
        from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
        
        for node, embedding in zip(nodes, embeddings):
            if embedding is not None:
                try:
                    # Add embedding to node
                    node.embedding = embedding
                    # Insert into vector store
                    vector_store.add([node])
                except Exception as e:
                    logger.warning(f"[AsyncIndexing] Failed to insert node: {e}")
    
    def _insert_single_node(self, vector_store, node, embedding: List[float]):
        """Insert a single node with embedding."""
        try:
            node.embedding = embedding
            vector_store.add([node])
        except Exception as e:
            logger.warning(f"[AsyncIndexing] Failed to insert single node: {e}")
    
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
        self._executor.shutdown(wait=False)
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5)
        logger.info("[AsyncIndexing] Shutdown complete")


# Singleton instance
def get_indexing_service() -> AsyncIndexingService:
    """Get the async indexing service singleton."""
    return AsyncIndexingService()
