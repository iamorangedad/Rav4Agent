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
        """Get embedding for a text."""
        return self._embed_model._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching.
        
        Process in smaller batches to avoid overwhelming Ollama.
        Truncate long texts to prevent memory issues.
        """
        if not texts:
            return []
        
        # Truncate texts to max 8000 characters to prevent Ollama overload
        max_length = 8000
        truncated_texts = []
        for text in texts:
            if len(text) > max_length:
                truncated_texts.append(text[:max_length] + "...")
                logger.warning(f"[Embedding] Truncated text from {len(text)} to {max_length} chars")
            else:
                truncated_texts.append(text)
        
        all_embeddings = []
        total = len(truncated_texts)
        
        logger.info(f"[Embedding] Processing {total} texts in batches of {self._batch_size}")
        
        for i in range(0, total, self._batch_size):
            batch = truncated_texts[i:i + self._batch_size]
            batch_num = i // self._batch_size + 1
            total_batches = (total + self._batch_size - 1) // self._batch_size
            
            # Log batch details
            total_chars = sum(len(t) for t in batch)
            logger.info(f"[Embedding] Processing batch {batch_num}/{total_batches} ({len(batch)} texts, {total_chars} total chars)")
            
            # Retry logic for each batch
            for attempt in range(self._max_retries):
                try:
                    batch_embeddings = self._embed_model._get_text_embeddings(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Longer delay between batches to let Ollama recover
                    if i + self._batch_size < total:
                        logger.info(f"[Embedding] Waiting {self._retry_delay}s before next batch...")
                        time.sleep(self._retry_delay)
                    
                    break
                except Exception as e:
                    logger.error(f"[Embedding] Batch {batch_num} failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                    
                    if attempt < self._max_retries - 1:
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.info(f"[Embedding] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
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
