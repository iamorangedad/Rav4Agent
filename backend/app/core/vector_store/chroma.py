"""ChromaDB vector store implementation."""
import logging
from typing import Optional

from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from app.core.vector_store.base import VectorStoreProvider, register_vector_store

logger = logging.getLogger(__name__)


class ChromaVectorStoreProvider(VectorStoreProvider):
    """ChromaDB vector store provider."""
    
    def __init__(
        self,
        host: str = "chroma",
        port: int = 8000,
        collection: str = "documents",
        persist_dir: Optional[str] = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            host: ChromaDB host
            port: ChromaDB port
            collection: Collection name
            persist_dir: Local persistence directory (fallback)
        """
        self.host = host
        self.port = port
        self.collection_name = collection
        self.persist_dir = persist_dir
        self._vector_store = None
        self._docstore = None
        self._index_store = None
        self._client = None
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            import chromadb
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
        return self._client
    
    def get_vector_store(self):
        """Get ChromaDB vector store instance."""
        if self._vector_store is None:
            try:
                from llama_index.vector_stores.chroma import ChromaVectorStore
                client = self._get_client()
                chroma_collection = client.get_or_create_collection(self.collection_name)
                self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")
                logger.info("Falling back to simple vector store")
                self._vector_store = SimpleVectorStore()
        return self._vector_store
    
    def get_docstore(self) -> SimpleDocumentStore:
        """Get document store instance."""
        if self._docstore is None:
            self._docstore = SimpleDocumentStore()
        return self._docstore
    
    def get_index_store(self) -> SimpleIndexStore:
        """Get index store instance."""
        if self._index_store is None:
            self._index_store = SimpleIndexStore()
        return self._index_store
    
    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        try:
            client = self._get_client()
            client.heartbeat()
            return True
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}")
            return False


# Register the provider
register_vector_store("chroma", ChromaVectorStoreProvider)
