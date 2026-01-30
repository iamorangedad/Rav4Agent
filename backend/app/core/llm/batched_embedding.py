"""Batched embedding model with retry logic for large documents."""
import logging
import time
from typing import List, Optional
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)


class BatchedOllamaEmbedding(BaseEmbedding):
    """Ollama embedding with batching and retry for large documents."""
    
    _embed_model: OllamaEmbedding = PrivateAttr()
    _batch_size: int = PrivateAttr()
    _max_retries: int = PrivateAttr()
    _retry_delay: float = PrivateAttr()
    
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
        # Set private attributes before super().__init__
        super().__init__(**kwargs)
        
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
        
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay
    
    @classmethod
    def class_name(cls) -> str:
        return "batched_ollama"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._embed_model._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with retry."""
        # Truncate text to max 2000 characters to prevent Ollama batch panic
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"[Embedding] Truncated text to {max_length} chars")
        
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return self._embed_model._get_text_embedding(text)
            except Exception as e:
                last_error = e
                logger.error(f"[Embedding] Text embedding failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                if attempt < self._max_retries - 1:
                    wait_time = self._retry_delay * (attempt + 1)
                    logger.info(f"[Embedding] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        # All retries failed, raise the last error
        if last_error:
            raise last_error
        return []  # Fallback (should never reach here)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts one by one to avoid overwhelming Ollama.
        
        Process individually instead of batches for better stability.
        """
        if not texts:
            return []
        
        all_embeddings = []
        total = len(texts)
        
        logger.info(f"[Embedding] Processing {total} texts one by one (max 2000 chars each)")
        
        for i, text in enumerate(texts):
            # Truncate text to max 2000 characters to prevent Ollama batch panic
            max_length = 2000
            if len(text) > max_length:
                text = text[:max_length] + "..."
                logger.warning(f"[Embedding] Truncated text {i+1} from {len(text)} to {max_length} chars")
            
            # Retry logic for each text
            for attempt in range(self._max_retries):
                try:
                    embedding = self._embed_model._get_text_embedding(text)
                    all_embeddings.append(embedding)
                    
                    # Small delay between texts to not overwhelm Ollama
                    if i < total - 1:
                        time.sleep(0.5)
                    
                    break
                except Exception as e:
                    logger.error(f"[Embedding] Text {i+1}/{total} failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                    
                    if attempt < self._max_retries - 1:
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.info(f"[Embedding] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"[Embedding] Text {i+1}/{total} failed after {self._max_retries} attempts")
                        raise
            
            # Log progress every 10 texts
            if (i + 1) % 10 == 0:
                logger.info(f"[Embedding] Progress: {i+1}/{total} texts processed")
        
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
