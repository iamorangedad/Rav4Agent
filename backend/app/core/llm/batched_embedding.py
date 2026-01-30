"""Batched embedding model with retry logic for large documents."""
import logging
import time
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


class BatchedOllamaEmbedding:
    """Wrapper for Ollama embedding with batching and retry for large documents.
    
    This is NOT a subclass of BaseEmbedding - it's a wrapper that delegates
to an underlying OllamaEmbedding while adding batching logic.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize batched Ollama embedding.
        
        Args:
            model_name: Model name
            base_url: Ollama base URL
            batch_size: Number of texts to embed in one batch
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        
        # Create underlying embedding model
        try:
            self._embed_model = OllamaEmbedding(
                model_name=model_name,
                base_url=base_url,
                **kwargs
            )
        except TypeError:
            # Fallback for older versions
            self._embed_model = OllamaEmbedding(
                model=model_name,
                base_url=base_url,
                **kwargs
            )
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying embed model."""
        return getattr(self._embed_model, name)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._embed_model.get_query_embedding(query)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        return self._embed_model.get_text_embedding(text)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Private method for query embedding."""
        return self._embed_model._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Private method for text embedding."""
        return self._embed_model._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching.
        
        Process in smaller batches to avoid overwhelming Ollama.
        """
        if not texts:
            return []
        
        all_embeddings = []
        total = len(texts)
        
        logger.info(f"[Embedding] Processing {total} texts in batches of {self._batch_size}")
        
        for i in range(0, total, self._batch_size):
            batch = texts[i:i + self._batch_size]
            batch_num = i // self._batch_size + 1
            total_batches = (total + self._batch_size - 1) // self._batch_size
            
            logger.info(f"[Embedding] Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Retry logic for each batch
            for attempt in range(self._max_retries):
                try:
                    batch_embeddings = self._embed_model._get_text_embeddings(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Small delay between batches to not overwhelm Ollama
                    if i + self._batch_size < total:
                        time.sleep(0.5)
                    
                    break
                except Exception as e:
                    logger.error(f"[Embedding] Batch {batch_num} failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                    
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_delay * (attempt + 1))
                    else:
                        logger.error(f"[Embedding] Batch {batch_num} failed after {self._max_retries} attempts")
                        raise
        
        logger.info(f"[Embedding] Successfully processed {len(all_embeddings)} embeddings")
        return all_embeddings
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for a query."""
        return await self._embed_model._aget_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding for a text."""
        return await self._embed_model._aget_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get embeddings with batching."""
        return self._get_text_embeddings(texts)
