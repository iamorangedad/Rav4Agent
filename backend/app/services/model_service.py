"""Model service for LLM operations."""
import logging
from typing import List, Dict, Any

from app.config import get_settings
from app.core.llm import create_llm_provider
from app.utils.helpers import get_param_size, requires_gpu

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model management and queries."""
    
    def __init__(self):
        """Initialize model service."""
        settings = get_settings()
        self.provider = create_llm_provider(
            "ollama",
            base_url=settings.ollama_base_url,
            default_model=settings.default_model_name,
            default_embedding_model=settings.default_embedding_model
        )
        self.settings = settings
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List[Dict[str, Any]]: List of model information
        """
        return self.provider.list_models()
    
    def get_recommended_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recommended models for document chat.
        
        Returns:
            Dict with recommended LLM and embedding models
        """
        return {
            "recommended_models": [
                {
                    "name": self.settings.default_model_name,
                    "description": "Default model configured for this deployment",
                    "context_window": "128K",
                    "gpu_required": False,
                },
                {
                    "name": "llama3.1:8b",
                    "description": "Good balance of performance and resource usage",
                    "context_window": "128K",
                    "gpu_required": False,
                },
                {
                    "name": "mistral",
                    "description": "Excellent reasoning capabilities",
                    "context_window": "32K",
                    "gpu_required": False,
                },
                {
                    "name": "qwen2:7b",
                    "description": "Strong multilingual support",
                    "context_window": "128K",
                    "gpu_required": False,
                },
            ],
            "embedding_models": [
                {
                    "name": self.settings.default_embedding_model,
                    "description": "Default embedding model configured for this deployment",
                    "dimensions": 768,
                    "gpu_required": False,
                },
                {
                    "name": "mxbai-embed-large",
                    "description": "Large embeddings for better retrieval",
                    "dimensions": 1024,
                    "gpu_required": False,
                },
            ],
        }
    
    def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if model exists
        """
        return self.provider.check_model_exists(model_name)
    
    def get_provider(self):
        """Get the LLM provider instance."""
        return self.provider
