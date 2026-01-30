"""Vector store abstract base class and factory."""
from abc import ABC, abstractmethod
from typing import Optional, Type
from llama_index.core.vector_stores import VectorStore
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.storage.index_store import BaseIndexStore


class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""
    
    @abstractmethod
    def get_vector_store(self) -> VectorStore:
        """Get the vector store instance."""
        pass
    
    @abstractmethod
    def get_docstore(self) -> BaseDocumentStore:
        """Get the document store instance."""
        pass
    
    @abstractmethod
    def get_index_store(self) -> BaseIndexStore:
        """Get the index store instance."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the vector store is available."""
        pass


# Registry of vector store providers
_vector_store_registry: dict[str, Type[VectorStoreProvider]] = {}


def register_vector_store(name: str, provider_class: Type[VectorStoreProvider]):
    """Register a vector store provider."""
    _vector_store_registry[name] = provider_class


def create_vector_store(store_type: str, **kwargs) -> VectorStoreProvider:
    """
    Factory function to create vector store provider.
    
    Args:
        store_type: Type of vector store (e.g., 'simple', 'chroma', 'pinecone', 'qdrant')
        **kwargs: Provider-specific configuration
        
    Returns:
        VectorStoreProvider: Initialized vector store provider
        
    Raises:
        ValueError: If vector store type is not supported
    """
    if store_type not in _vector_store_registry:
        available = list(_vector_store_registry.keys())
        raise ValueError(
            f"Unknown vector store type: {store_type}. "
            f"Available types: {available}"
        )
    
    provider_class = _vector_store_registry[store_type]
    return provider_class(**kwargs)


def list_available_stores() -> list[str]:
    """List all available vector store types."""
    return list(_vector_store_registry.keys())
