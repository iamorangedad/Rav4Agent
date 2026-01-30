"""Pytest configuration for the project."""
import os
import sys
import pytest
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Set test environment variables
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("EMBEDDING_MODEL", "test-embedding")
os.environ.setdefault("VECTOR_STORE_TYPE", "simple")


@pytest.fixture(scope="session")
def test_upload_dir(tmp_path_factory):
    """Create a temporary upload directory for tests."""
    return tmp_path_factory.mktemp("uploads")


@pytest.fixture(scope="session")
def test_chroma_dir(tmp_path_factory):
    """Create a temporary chroma directory for tests."""
    return tmp_path_factory.mktemp("chroma_db")


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    This is a sample document for testing purposes.
    It contains multiple paragraphs and sentences.
    
    The first section discusses the importance of testing.
    Testing ensures that our code works as expected.
    It helps us catch bugs early in development.
    
    The second section talks about document processing.
    Documents need to be parsed and split into chunks.
    Each chunk should contain meaningful information.
    
    Finally, we discuss vectorization.
    Text needs to be converted to vector embeddings.
    These embeddings are stored in vector databases.
    """


@pytest.fixture
def sample_long_text():
    """Sample long text for chunking tests."""
    paragraphs = []
    for i in range(50):
        paragraphs.append(f"Paragraph {i+1}: " + "This is a test sentence. " * 20)
    return "\n\n".join(paragraphs)


@pytest.fixture
def mock_embedding():
    """Mock embedding vector for testing."""
    return [0.1] * 768  # 768-dimensional embedding


@pytest.fixture
def mock_embeddings_list():
    """List of mock embeddings for batch testing."""
    return [[0.1] * 768 for _ in range(10)]
