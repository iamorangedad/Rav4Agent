"""Real environment integration tests for large document processing.

These tests run against the actual deployed environment:
- Real Ollama service at http://10.0.0.55:11434
- Real document processing pipeline
- Real vector storage

Requirements:
- Ollama must be running and accessible
- Test PDF OM0R028U.pdf must exist
- Environment must be properly configured

Run with: pytest tests/integration/test_real_environment.py -v -s --tb=short
"""
import pytest
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

# Test configuration
TEST_PDF_PATH = Path(__file__).parent.parent.parent.parent / "OM0R028U.pdf"
OLLAMA_URL = os.getenv("TEST_OLLAMA_URL", "http://10.0.0.55:11434")
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "300"))  # 5 minutes default


# Skip all tests if not in real environment mode
pytestmark = pytest.mark.skipif(
    os.getenv("TEST_REAL_ENV", "false").lower() != "true",
    reason="Set TEST_REAL_ENV=true to run real environment tests"
)


@pytest.fixture(scope="module")
def real_services():
    """Initialize real services for testing."""
    from app.config import get_settings
    from app.services.model_service import ModelService
    from app.core.vector_store import create_vector_store
    from app.services.large_document_processor import RobustLargeDocumentProcessor
    
    # Use real settings
    settings = get_settings()
    
    # Override for test
    settings.ollama_base_url = OLLAMA_URL
    
    # Initialize real services
    model_service = ModelService()
    vector_store_provider = create_vector_store("simple")
    
    # Get real embedding model
    embed_model = model_service.get_provider().get_embedding_model(
        settings.default_embedding_model
    )
    vector_store = vector_store_provider.get_vector_store()
    
    return {
        "settings": settings,
        "model_service": model_service,
        "embed_model": embed_model,
        "vector_store": vector_store,
        "vector_store_provider": vector_store_provider
    }


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create temporary upload directory."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


class TestRealEnvironmentSetup:
    """Verify real environment is accessible."""
    
    def test_ollama_connection(self):
        """Test that Ollama is accessible."""
        import httpx
        
        try:
            response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
            assert response.status_code == 200, f"Ollama returned {response.status_code}"
            
            data = response.json()
            models = data.get("models", [])
            assert len(models) > 0, "No models found in Ollama"
            
            model_names = [m.get("name", "") for m in models]
            print(f"\n✅ Ollama accessible with models: {model_names}")
            
        except Exception as e:
            pytest.fail(f"Cannot connect to Ollama at {OLLAMA_URL}: {e}")
    
    def test_embedding_model_available(self, real_services):
        """Test that embedding model is available."""
        settings = real_services["settings"]
        model_service = real_services["model_service"]
        
        model_name = settings.default_embedding_model
        exists = model_service.get_provider().check_model_exists(model_name)
        
        assert exists, f"Embedding model '{model_name}' not found in Ollama"
        print(f"\n✅ Embedding model '{model_name}' is available")
    
    def test_test_file_exists(self):
        """Verify test PDF exists."""
        assert TEST_PDF_PATH.exists(), f"Test file not found: {TEST_PDF_PATH}"
        file_size = TEST_PDF_PATH.stat().st_size
        print(f"\n✅ Test file exists: {TEST_PDF_PATH} ({file_size / 1024 / 1024:.1f} MB)")


class TestRealDocumentProcessing:
    """Test document processing with real services."""
    
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_real_pdf_loading(self):
        """Test loading PDF with real SimpleDirectoryReader."""
        from llama_index.core import SimpleDirectoryReader
        
        start_time = time.time()
        reader = SimpleDirectoryReader(
            input_files=[str(TEST_PDF_PATH)],
            filename_as_id=True
        )
        documents = reader.load_data()
        elapsed = time.time() - start_time
        
        assert len(documents) > 0, "Should load documents"
        total_chars = sum(len(doc.text) for doc in documents)
        
        print(f"\n✅ Loaded {len(documents)} docs, {total_chars} chars in {elapsed:.2f}s")
    
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_real_embedding_generation(self, real_services):
        """Test embedding generation with real Ollama."""
        embed_model = real_services["embed_model"]
        
        test_texts = [
            "Short text",
            "This is a medium length text with more content.",
            "word " * 200,  # ~1000 chars
        ]
        
        results = []
        for text in test_texts:
            start = time.time()
            try:
                embedding = embed_model.get_text_embedding(text)
                elapsed = time.time() - start
                results.append((len(text), embedding is not None, elapsed))
            except Exception as e:
                results.append((len(text), False, 0))
                print(f"\n⚠️ Embedding failed for {len(text)} chars: {e}")
        
        success_count = sum(1 for _, success, _ in results if success)
        success_rate = success_count / len(results)
        
        print(f"\n✅ Embedding success rate: {success_rate*100:.0f}% ({success_count}/{len(results)})")
        
        # At least 2/3 should succeed
        assert success_rate >= 0.66, f"Success rate too low: {success_rate*100:.1f}%"
    
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_real_vector_store_operations(self, real_services):
        """Test vector store with real embeddings."""
        from llama_index.core.schema import TextNode
        
        embed_model = real_services["embed_model"]
        vector_store = real_services["vector_store"]
        
        # Create test nodes
        nodes = [
            TextNode(id_=f"test-{i}", text=f"Test content {i}")
            for i in range(3)
        ]
        
        # Generate embeddings and add
        for node in nodes:
            try:
                node.embedding = embed_model.get_text_embedding(node.text)
                vector_store.add([node])
            except Exception as e:
                pytest.fail(f"Failed to add node: {e}")
        
        print(f"\n✅ Added {len(nodes)} nodes to vector store")
    
    @pytest.mark.slow
    @pytest.mark.timeout(TEST_TIMEOUT * 2)  # 10 minutes for full processing
    def test_full_large_document_pipeline(self, real_services, temp_upload_dir):
        """Test complete pipeline with real OM0R028U.pdf."""
        from app.services.large_document_processor import process_large_document_robust
        from llama_index.core.vector_stores import SimpleVectorStore
        
        # Setup
        embed_model = real_services["embed_model"]
        vector_store = SimpleVectorStore()
        
        # Copy PDF to temp dir
        dest_file = temp_upload_dir / TEST_PDF_PATH.name
        shutil.copy(TEST_PDF_PATH, dest_file)
        
        print(f"\n🚀 Starting full pipeline test with {TEST_PDF_PATH.name}")
        print(f"   File size: {dest_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Progress tracking
        progress_log = []
        def progress_callback(percent, message):
            progress_log.append((percent, message))
            if percent % 10 == 0 or percent == 95:
                print(f"   [{percent:3d}%] {message}")
        
        # Process
        start_time = time.time()
        
        try:
            result = process_large_document_robust(
                str(dest_file),
                embed_model,
                vector_store,
                progress_callback
            )
            
            elapsed = time.time() - start_time
            
            # Results
            print(f"\n{'='*60}")
            print("RESULTS:")
            print(f"{'='*60}")
            print(f"  Total chunks: {result['total_chunks']}")
            print(f"  Successful: {result['successful']}")
            print(f"  Failed: {result['failed']}")
            print(f"  Success rate: {result['success_rate']*100:.1f}%")
            print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            
            if result['failed'] > 0:
                print(f"  Failed indices: {result['failed_indices'][:10]}")
            
            # Assertions
            assert result['total_chunks'] > 0, "Should create chunks"
            assert result['success_rate'] >= 0.8, \
                f"Success rate {result['success_rate']*100:.1f}% < 80%"
            
            print(f"\n✅ TEST PASSED: {result['success_rate']*100:.1f}% success rate")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ TEST FAILED after {elapsed:.1f}s: {e}")
            raise


class TestRealIndexingService:
    """Test the actual indexing service with real documents."""
    
    @pytest.mark.slow
    @pytest.mark.timeout(TEST_TIMEOUT * 2)
    def test_real_indexing_task(self, real_services, temp_upload_dir):
        """Test AsyncIndexingService with real document."""
        from app.services.indexing_service import AsyncIndexingService, IndexingTask, TaskStatus
        
        # Copy PDF
        dest_file = temp_upload_dir / TEST_PDF_PATH.name
        shutil.copy(TEST_PDF_PATH, dest_file)
        
        # Create service instance
        service = AsyncIndexingService()
        service.upload_dir = str(temp_upload_dir)
        
        # Create and process task
        task = service.create_task(TEST_PDF_PATH.name)
        
        print(f"\n🚀 Processing task {task.task_id}")
        
        # Process
        start_time = time.time()
        service._process_task_optimized(task)
        elapsed = time.time() - start_time
        
        # Results
        print(f"\n{'='*60}")
        print("INDEXING SERVICE RESULTS:")
        print(f"{'='*60}")
        print(f"  Status: {task.status.value}")
        print(f"  Total chunks: {task.total_pages}")
        print(f"  Processed: {task.processed_pages}")
        print(f"  Success rate: {task.processed_pages/task.total_pages*100:.1f}%" if task.total_pages > 0 else "N/A")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Message: {task.message}")
        
        # Task should complete (even with partial success)
        assert task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED], \
            f"Unexpected status: {task.status}"
        
        if task.status == TaskStatus.COMPLETED:
            print(f"\n✅ INDEXING COMPLETED SUCCESSFULLY")
        else:
            print(f"\n⚠️ INDEXING FAILED: {task.error}")
            pytest.fail(f"Indexing failed: {task.error}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
