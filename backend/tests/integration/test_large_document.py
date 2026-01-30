"""Integration test for large document processing.
Test with OM0R028U.pdf (31MB, 15 pages) to verify robustness.
"""
import pytest
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test configuration
TEST_PDF_PATH = Path(__file__).parent.parent.parent.parent / "OM0R028U.pdf"


class TestLargeDocumentProcessing:
    """Test processing of large documents like OM0R028U.pdf"""
    
    @pytest.mark.slow
    def test_large_pdf_loading(self):
        """Test that large PDF can be loaded without memory issues."""
        if not TEST_PDF_PATH.exists():
            pytest.skip(f"Test file not found: {TEST_PDF_PATH}")
        
        from llama_index.core import SimpleDirectoryReader
        
        # Measure memory and time
        start_time = time.time()
        
        reader = SimpleDirectoryReader(
            input_files=[str(TEST_PDF_PATH)],
            filename_as_id=True
        )
        
        documents = reader.load_data()
        load_time = time.time() - start_time
        
        # Assertions
        assert len(documents) > 0, "Should load at least one document"
        assert load_time < 30, f"Loading took too long: {load_time}s"
        
        # Check total content size
        total_chars = sum(len(doc.text) for doc in documents)
        print(f"\nLoaded {len(documents)} documents, {total_chars} total characters in {load_time:.2f}s")
    
    @pytest.mark.slow  
    def test_large_document_chunking(self):
        """Test chunking of large document content."""
        if not TEST_PDF_PATH.exists():
            pytest.skip(f"Test file not found: {TEST_PDF_PATH}")
        
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
        
        # Load document
        reader = SimpleDirectoryReader(input_files=[str(TEST_PDF_PATH)])
        documents = reader.load_data()
        
        # Chunk with various sizes
        chunk_sizes = [500, 1000, 1500]
        
        for chunk_size in chunk_sizes:
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=min(200, chunk_size // 3)
            )
            
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Verify chunks
            assert len(nodes) > 0, f"Should create chunks with size {chunk_size}"
            
            # Check chunk sizes are reasonable
            for node in nodes[:10]:  # Sample first 10
                content = node.get_content()
                # Allow some tolerance for sentence boundaries
                assert len(content) <= chunk_size + 500, \
                    f"Chunk too large: {len(content)} > {chunk_size + 500}"
            
            print(f"\nChunk size {chunk_size}: created {len(nodes)} nodes")
    
    def test_embedding_with_very_long_texts(self):
        """Test embedding generation with very long texts (simulated)."""
        from app.core.llm.batched_embedding import BatchedOllamaEmbedding
        
        # Create embedding model with very conservative settings
        embed_model = BatchedOllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            batch_size=1,
            max_retries=3,
            retry_delay=2.0
        )
        
        # Test with progressively longer texts
        test_cases = [
            ("Short text", 100),
            ("Medium text", 1000),
            ("Long text", 1500),
            ("Very long text", 3000),
            ("Extreme text", 5000),
        ]
        
        for name, length in test_cases:
            text = "word " * (length // 5)  # Approximate length
            
            try:
                # Try to get embedding
                result = embed_model._get_text_embedding(text)
                
                if result is not None:
                    print(f"✓ {name} ({length} chars): SUCCESS")
                else:
                    print(f"⚠ {name} ({length} chars): Failed gracefully (returned None)")
                    
            except Exception as e:
                print(f"✗ {name} ({length} chars): EXCEPTION - {str(e)[:50]}")
    
    @pytest.mark.slow
    def test_full_indexing_pipeline_with_large_pdf(self):
        """Test complete indexing pipeline with large PDF."""
        if not TEST_PDF_PATH.exists():
            pytest.skip(f"Test file not found: {TEST_PDF_PATH}")
        
        from app.services.indexing_service import AsyncIndexingService, get_indexing_service
        
        # Copy file to uploads directory
        import shutil
        upload_dir = Path("/tmp/test_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        dest_file = upload_dir / TEST_PDF_PATH.name
        shutil.copy(TEST_PDF_PATH, dest_file)
        
        try:
            # Create indexing task
            service = get_indexing_service()
            service.upload_dir = str(upload_dir)
            
            task = service.create_task(TEST_PDF_PATH.name)
            
            # Process the task
            service._process_task_optimized(task)
            
            # Verify results
            assert task.status.name in ["COMPLETED", "FAILED"], \
                f"Task should complete or fail gracefully, got: {task.status}"
            
            if task.status.name == "COMPLETED":
                print(f"\n✓ Successfully indexed {task.processed_pages}/{task.total_pages} chunks")
            else:
                print(f"\n✗ Task failed: {task.error}")
                
        finally:
            # Cleanup
            if dest_file.exists():
                dest_file.unlink()


class TestEmbeddingRobustness:
    """Test embedding model robustness with edge cases."""
    
    def test_embedding_handles_various_text_types(self):
        """Test embedding with different text types."""
        from app.core.llm.batched_embedding import BatchedOllamaEmbedding
        
        embed_model = BatchedOllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            max_retries=2
        )
        
        test_texts = [
            "Simple English text",
            "Special chars: @#$%^&*()",
            "Unicode: 中文测试内容",
            "Code: def hello(): return 'world'",
            "Very short",
            "Numbers: 12345 67890 3.14159",
            "",  # Empty
            "a" * 1500,  # Exact limit
            "a" * 3000,  # Over limit
        ]
        
        results = []
        for text in test_texts:
            try:
                result = embed_model._get_text_embedding(text)
                results.append((len(text), result is not None))
            except Exception as e:
                results.append((len(text), False))
        
        # Should handle most cases without crashing
        success_rate = sum(1 for _, success in results if success) / len(results)
        print(f"\nEmbedding success rate: {success_rate*100:.1f}% ({sum(1 for _, s in results if s)}/{len(results)})")
        
        # At least 70% should succeed
        assert success_rate >= 0.7, f"Success rate too low: {success_rate*100:.1f}%"
    
    def test_batch_processing_with_mixed_success(self):
        """Test batch processing where some items fail."""
        from app.core.llm.batched_embedding import BatchedOllamaEmbedding
        
        embed_model = BatchedOllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            max_retries=2
        )
        
        # Create batch with varying lengths
        texts = [
            "Short 1",
            "a" * 1500,  # At limit
            "Short 2", 
            "b" * 2000,  # Over limit
            "Short 3",
            "c" * 3000,  # Way over limit
        ]
        
        results = embed_model._get_text_embeddings(texts)
        
        # Count successes and failures
        successes = sum(1 for r in results if r is not None)
        failures = sum(1 for r in results if r is None)
        
        print(f"\nBatch results: {successes} success, {failures} failure out of {len(texts)}")
        
        # Should process all without crashing
        assert len(results) == len(texts), "Should return result for each input"
        
        # At least some should succeed
        assert successes > 0, "Should have at least some successful embeddings"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
