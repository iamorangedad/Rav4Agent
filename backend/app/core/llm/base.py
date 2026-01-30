"""LLM abstract base class and factory."""
from abc import ABC, abstractmethod
from typing import Type, Optional
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self) -> LLM:
        """Get the LLM instance."""
        pass
    
    @abstractmethod
    def get_embedding_model(self, model_name: Optional[str] = None) -> BaseEmbedding:
        """Get the embedding model instance."""
        pass
    
    @abstractmethod
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists."""
        pass
    
    @abstractmethod
    def list_models(self) -> list[dict]:
        """List available models."""
        pass


# Registry of LLM providers
_llm_registry: dict[str, Type[LLMProvider]] = {}


def register_llm(name: str, provider_class: Type[LLMProvider]):
    """Register an LLM provider."""
    _llm_registry[name] = provider_class


def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM provider.
    
    Args:
        provider_type: Type of LLM provider (e.g., 'ollama', 'openai')
        **kwargs: Provider-specific configuration
        
    Returns:
        LLMProvider: Initialized LLM provider
        
    Raises:
        ValueError: If provider type is not supported
    """
    if provider_type not in _llm_registry:
        available = list(_llm_registry.keys())
        raise ValueError(
            f"Unknown LLM provider type: {provider_type}. "
            f"Available types: {available}"
        )
    
    provider_class = _llm_registry[provider_type]
    return provider_class(**kwargs)


def list_available_llm_providers() -> list[str]:
    """List all available LLM provider types."""
    return list(_llm_registry.keys())
