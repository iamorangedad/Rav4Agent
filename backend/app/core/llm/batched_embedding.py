"""Batched embedding model with retry logic for large documents.

Optimized for Ollama stability with aggressive error handling.
"""
import logging
import time
from typing import List, Optional, Any
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)


class BatchedOllamaEmbedding(BaseEmbedding):
    """Ollama embedding with aggressive error handling for large documents.
    
    This implementation prioritizes stability over completeness:
    - Failed chunks are skipped (return None) rather than failing entire batch
    - More aggressive text truncation
    - Longer delays between requests
    - Detailed logging of failures
    """
    
    _embed_model: OllamaEmbedding = PrivateAttr()
    _batch_size: int = PrivateAttr()
    _max_retries: int = PrivateAttr()
    _retry_delay: float = PrivateAttr()
    _failed_indices: List[int] = PrivateAttr()
    
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
        self._failed_indices = []
    
    @classmethod
    def class_name(cls) -> str:
        return "batched_ollama"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._embed_model._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text with retry and graceful failure.
        
        Returns None if all retries fail, rather than raising exception.
        """
        # Truncate text to max 1500 characters (more aggressive than before)
        max_length = 1500
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return self._embed_model._get_text_embedding(text)
            except Exception as e:
                last_error = e
                # Only log first failure to reduce noise
                if attempt == 0:
                    logger.warning(f"[Embedding] Failed, will retry: {str(e)[:100]}")
                
                if attempt < self._max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2.0 * (2 ** attempt)
                    logger.info(f"[Embedding] Waiting {wait_time}s before retry {attempt + 2}/{self._max_retries}...")
                    time.sleep(wait_time)
        
        # All retries failed - return None instead of raising
        logger.error(f"[Embedding] Failed after {self._max_retries} attempts, returning None: {str(last_error)[:100]}")
        return None
    
    def _get_text_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts with graceful failure handling.
        
        Failed texts return None instead of crashing the entire batch.
        This allows partial indexing of large documents.
        """
        if not texts:
            return []
        
        results: List[Optional[List[float]]] = []
        total = len(texts)
        success_count = 0
        fail_count = 0
        
        logger.info(f"[Embedding] Processing {total} texts (max 1500 chars each, graceful failures enabled)")
        
        for i, text in enumerate(texts):
            # Truncate text to max 1500 characters
            max_length = 1500
            original_len = len(text)
            if original_len > max_length:
                text = text[:max_length] + "..."
                if i < 3 or i % 50 == 0:  # Log first 3 and every 50th
                    logger.debug(f"[Embedding] Truncated text {i+1} from {original_len} to {max_length} chars")
            
            # Try to get embedding
            try:
                embedding = self._get_text_embedding(text)
                
                if embedding is not None:
                    results.append(embedding)
                    success_count += 1
                else:
                    # Failed but gracefully - add None placeholder
                    results.append(None)
                    fail_count += 1
                    self._failed_indices.append(i)
                    
                    # Log progress on failures
                    if fail_count <= 5 or fail_count % 10 == 0:
                        logger.warning(f"[Embedding] Chunk {i+1}/{total} failed ({fail_count} total failures), continuing...")
                
                # Delay between requests - longer for large documents
                # Progressive delay: 1s base + 0.5s per 100 chunks processed
                delay = 1.0 + (i // 100) * 0.5
                if i < total - 1:
                    time.sleep(min(delay, 3.0))  # Cap at 3 seconds
                
            except Exception as e:
                # Catch any unexpected errors
                logger.error(f"[Embedding] Unexpected error on chunk {i+1}: {str(e)[:100]}")
                results.append(None)
                fail_count += 1
                self._failed_indices.append(i)
            
            # Log progress periodically
            if (i + 1) % 25 == 0 or i == total - 1:
                progress = (i + 1) / total * 100
                logger.info(f"[Embedding] Progress: {i+1}/{total} ({progress:.1f}%) - Success: {success_count}, Failed: {fail_count}")
        
        # Summary
        success_rate = success_count / total * 100 if total > 0 else 0
        if fail_count > 0:
            logger.warning(f"[Embedding] Completed with {fail_count}/{total} failures ({success_rate:.1f}% success rate)")
            if len(self._failed_indices) <= 20:
                logger.warning(f"[Embedding] Failed indices: {self._failed_indices}")
        else:
            logger.info(f"[Embedding] Successfully processed all {total} embeddings")
        
        return results
    
    def get_failed_indices(self) -> List[int]:
        """Get list of indices that failed embedding."""
        return self._failed_indices.copy()
    
    def clear_failed_indices(self):
        """Clear the failed indices list."""
        self._failed_indices.clear()
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for a query."""
        return await self._embed_model._aget_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> Optional[List[float]]:
        """Async get embedding for a text."""
        try:
            return await self._embed_model._aget_text_embedding(text)
        except Exception as e:
            logger.error(f"[Embedding] Async text embedding failed: {e}")
            return None
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Async get embeddings with batching."""
        # For async, just delegate to sync version for simplicity
        return self._get_text_embeddings(texts)
