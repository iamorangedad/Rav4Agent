"""Test the robust large document processor with OM0R028U.pdf"""
import sys
import os
sys.path.insert(0, '/home/mac/Rav4Agent/backend')

from app.services.large_document_processor import RobustLargeDocumentProcessor, process_large_document_robust
from llama_index.core.vector_stores import SimpleVectorStore
from app.core.llm.batched_embedding import BatchedOllamaEmbedding
import time

# Test configuration
TEST_PDF = "/home/mac/Rav4Agent/OM0R028U.pdf"
OLLAMA_URL = "http://10.0.0.55:11434"

def test_with_robust_processor():
    """Test the new robust processor"""
    print("\n" + "="*60)
    print("Testing Robust Large Document Processor")
    print("="*60)
    
    if not os.path.exists(TEST_PDF):
        print(f"❌ Test file not found: {TEST_PDF}")
        return False
    
    print(f"📄 Test file: {TEST_PDF}")
    print(f"📏 File size: {os.path.getsize(TEST_PDF) / 1024 / 1024:.1f} MB")
    print()
    
    # Setup
    print("🔧 Setting up embedding model and vector store...")
    embed_model = BatchedOllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_URL,
        batch_size=1,
        max_retries=3,
        retry_delay=2.0
    )
    vector_store = SimpleVectorStore()
    
    # Progress callback
    def progress_callback(percent, message):
        print(f"  [{percent:3d}%] {message}")
    
    # Process
    print("\n🚀 Starting processing...")
    start_time = time.time()
    
    try:
        result = process_large_document_robust(
            TEST_PDF,
            embed_model,
            vector_store,
            progress_callback
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("Results:")
        print("="*60)
        print(f"  Total chunks: {result['total_chunks']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Success rate: {result['success_rate']*100:.1f}%")
        print(f"  Time elapsed: {elapsed:.1f}s")
        
        if result['failed'] > 0:
            print(f"  Failed indices (first 10): {result['failed_indices'][:10]}")
        
        # Success criteria: at least 80% success rate
        if result['success_rate'] >= 0.8:
            print("\n✅ TEST PASSED: Success rate >= 80%")
            return True
        else:
            print("\n❌ TEST FAILED: Success rate < 80%")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_robust_processor()
    sys.exit(0 if success else 1)
