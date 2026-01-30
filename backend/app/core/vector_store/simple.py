"""Simple in-memory vector store implementation."""
from typing import Optional, Any
import os

from app.core.vector_store.base import VectorStoreProvider, register_vector_store


class SimpleVectorStoreProvider(VectorStoreProvider):
    """Simple in-memory vector store provider."""
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize simple vector store.
        
        Args:
            persist_dir: Directory to persist data (optional)
        """
        self.persist_dir = persist_dir
        self._vector_store = None
        self._docstore = None
        self._index_store = None
        
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
    
    def get_vector_store(self) -> Any:
        """Get simple vector store instance."""
        if self._vector_store is None:
            from llama_index.core.vector_stores import SimpleVectorStore
            self._vector_store = SimpleVectorStore()
        return self._vector_store
    
    def get_docstore(self) -> Any:
        """Get simple document store instance."""
        if self._docstore is None:
            from llama_index.core.storage.docstore import SimpleDocumentStore
            self._docstore = SimpleDocumentStore()
        return self._docstore
    
    def get_index_store(self) -> Any:
        """Get simple index store instance."""
        if self._index_store is None:
            from llama_index.core.storage.index_store import SimpleIndexStore
            self._index_store = SimpleIndexStore()
        return self._index_store
    
    def is_available(self) -> bool:
        """Simple store is always available."""
        return True


# Register the provider
register_vector_store("simple", SimpleVectorStoreProvider)
