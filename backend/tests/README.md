# Unit Tests for Smart Document Assistant

This directory contains comprehensive unit tests for the key functionalities of the Smart Document Assistant.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit/
│   ├── test_document_parsing.py   # Document parsing and splitting tests
│   ├── test_node_vectorization.py # Node embedding/vectorization tests
│   ├── test_vector_storage.py     # Vector database storage tests
│   ├── test_vector_retrieval.py   # Vector similarity search tests
│   └── test_prompt_generation.py  # Prompt generation tests
└── integration/                   # Integration tests (future)
```

## Running Tests

### Run all tests
```bash
cd backend
pip install -r requirements-test.txt
pytest
```

### Run specific test categories
```bash
# Run only unit tests
pytest -m unit

# Run only vector-related tests
pytest -m vector

# Run only document processing tests
pytest -m document

# Exclude slow tests
pytest -m "not slow"
```

### Run with coverage
```bash
pytest --cov=app --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/unit/test_document_parsing.py -v
```

### Run specific test class
```bash
pytest tests/unit/test_document_parsing.py::TestDocumentSplitting -v
```

### Run specific test method
```bash
pytest tests/unit/test_document_parsing.py::TestDocumentSplitting::test_sentence_splitter_basic -v
```

## Test Categories

### 1. Document Parsing Tests (`test_document_parsing.py`)

Tests for document loading and text chunking:

- **Document Creation**: Creating documents from text with metadata
- **Sentence Splitting**: Splitting text into chunks with configurable size and overlap
- **Chunk Size Enforcement**: Verifying chunks respect size limits
- **Context Preservation**: Ensuring semantic context is preserved during splitting
- **Special Characters**: Handling Unicode, emojis, and special formatting
- **Large Documents**: Optimized splitting for documents with 50+ paragraphs

Key test classes:
- `TestDocumentParsing`: Basic document operations
- `TestDocumentSplitting`: Text chunking functionality
- `TestDocumentLoading`: File loading operations
- `TestChunkMetadata`: Metadata inheritance and relationships

### 2. Node Vectorization Tests (`test_node_vectorization.py`)

Tests for text embedding generation:

- **Single Embedding**: Generating embeddings for individual nodes
- **Batch Embedding**: Processing multiple nodes in parallel
- **Embedding Dimensions**: Consistency in vector dimensions
- **Vector Normalization**: L2 normalization for unit vectors
- **Similarity Calculation**: Cosine similarity between vectors
- **Parallel Processing**: Thread-safe concurrent embedding generation
- **Retry Logic**: Handling temporary embedding failures

Key test classes:
- `TestNodeVectorization`: Basic embedding operations
- `TestBatchVectorization`: Batch processing
- `TestEmbeddingModel`: Model interface validation
- `TestVectorConsistency`: Vector math and similarity
- `TestParallelEmbedding`: Concurrent operations

### 3. Vector Storage Tests (`test_vector_storage.py`)

Tests for vector database operations:

- **SimpleVectorStore**: In-memory storage operations
- **Vector Addition**: Adding single and batch vectors
- **Vector Deletion**: Removing vectors by ID
- **Persistence**: Saving and loading from disk
- **Provider Abstraction**: Factory pattern for different backends
- **ChromaDB Integration**: External vector store support

Key test classes:
- `TestSimpleVectorStore`: Core storage operations
- `TestVectorStoreProvider`: Provider abstraction
- `TestBatchVectorOperations`: Batch insert/delete
- `TestVectorStorePersistence`: Disk persistence
- `TestChromaVectorStore`: ChromaDB provider

### 4. Vector Retrieval Tests (`test_vector_retrieval.py`)

Tests for similarity search and retrieval:

- **Similarity Search**: Finding similar vectors with top-k
- **Query Embedding**: Converting queries to embeddings
- **Relevance Scoring**: Cosine similarity and ranking
- **Metadata Filtering**: Search with metadata constraints
- **Edge Cases**: Empty stores, no matches, all matches
- **Large Scale**: Performance with many vectors

Key test classes:
- `TestSimilaritySearch`: Core search functionality
- `TestQueryEmbedding`: Query processing
- `TestRelevanceScoring`: Ranking algorithms
- `TestRetrievalWithFilters`: Filtered searches
- `TestVectorSearchEdgeCases`: Edge cases

### 5. Prompt Generation Tests (`test_prompt_generation.py`)

Tests for LLM prompt construction:

- **Message Formatting**: User, assistant, system messages
- **Context Injection**: Adding retrieved documents to prompts
- **History Management**: Conversation history handling
- **System Prompts**: Role and persona definitions
- **Prompt Assembly**: Combining all components
- **Context Truncation**: Handling long contexts within token limits
- **Special Content**: Code blocks, special characters, formatting

Key test classes:
- `TestChatMessageFormatting`: Message structure
- `TestContextInjection`: Document context
- `TestConversationHistory`: History tracking
- `TestSystemPromptGeneration`: System instructions
- `TestPromptAssembly`: Complete prompts
- `TestPromptEdgeCases`: Edge cases

## Test Fixtures

Common fixtures available in `conftest.py`:

- `test_upload_dir`: Temporary directory for uploaded files
- `test_chroma_dir`: Temporary directory for ChromaDB persistence
- `sample_text_content`: Standard text sample for testing
- `sample_long_text`: 50-paragraph text for chunking tests
- `mock_embedding`: 768-dimensional mock embedding vector
- `mock_embeddings_list`: List of 10 mock embeddings

## Writing New Tests

### Test Structure
```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Test description."""
    
    def test_specific_scenario(self):
        """Test description with specific conditions."""
        # Arrange
        input_data = "test"
        
        # Act
        result = some_function(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case behavior."""
        pass
```

### Using Fixtures
```python
def test_with_fixture(self, sample_text_content, mock_embedding):
    """Test using shared fixtures."""
    # Use fixtures directly
    assert len(sample_text_content) > 0
    assert len(mock_embedding) == 768
```

### Mocking External Services
```python
@patch('app.services.model_service.ModelService')
def test_with_mock(self, mock_service):
    """Test with mocked dependencies."""
    mock_service.get_provider.return_value = Mock()
    # Test code here
```

## CI/CD Integration

For continuous integration:

```yaml
# Example GitHub Actions
- name: Run Tests
  run: |
    cd backend
    pip install -r requirements.txt
    pip install -r requirements-test.txt
    pytest --cov=app --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./backend/coverage.xml
```

## Coverage Goals

- **Document Parsing**: >90% coverage
- **Node Vectorization**: >85% coverage
- **Vector Storage**: >85% coverage
- **Vector Retrieval**: >85% coverage
- **Prompt Generation**: >90% coverage

## Debugging Tests

### Verbose output
```bash
pytest -vv --tb=long tests/unit/test_document_parsing.py
```

### Stop on first failure
```bash
pytest -x
```

### Interactive debugging
```bash
pytest --pdb
```

### Show local variables on failure
```bash
pytest -l
```

## Common Issues

1. **Import errors**: Make sure you're running from the `backend` directory
2. **Missing dependencies**: Install both `requirements.txt` and `requirements-test.txt`
3. **Slow tests**: Mark slow tests with `@pytest.mark.slow` and exclude with `-m "not slow"`
4. **Async tests**: Use `@pytest.mark.asyncio` for async test functions

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Cover both success and failure cases
3. Add edge case tests
4. Use descriptive test names
5. Add docstrings explaining the test scenario
6. Keep tests independent (no ordering dependencies)
