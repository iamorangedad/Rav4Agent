"""Unit tests for vector database storage functionality."""
import os
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from llama_index.core.schema import TextNode


class TestSimpleVectorStore:
    """Test SimpleVectorStore operations."""
    
    def test_vector_store_initialization(self):
        """Test initializing SimpleVectorStore."""
        from llama_index.core.vector_stores import SimpleVectorStore
        
        store = SimpleVectorStore()
        assert store is not None
        assert hasattr(store, 'add')
        assert hasattr(store, 'delete')
    
    def test_vector_store_add_single_node(self, mock_embedding):
        """Test adding a single node with embedding to vector store."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        from llama_index.core.vector_stores.types import TextNode as VSTextNode
        
        store = SimpleVectorStore()
        node = TextNode(
            id_="node-1",
            text="Test content for storage",
            embedding=mock_embedding
        )
        
        # Add node to store
        store.add([node])
        
        # Verify node was added by querying
        result = store.query(VectorStoreQuery(query_embedding=mock_embedding, similarity_top_k=1))
        assert len(result.nodes) > 0
    
    def test_vector_store_add_multiple_nodes(self, mock_embeddings_list):
        """Test adding multiple nodes with embeddings to vector store."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        nodes = []
        
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"node-{i}",
                text=f"Content {i}",
                embedding=embedding
            )
            nodes.append(node)
        
        # Add all nodes
        store.add(nodes)
        
        # Query and verify
        query_embedding = mock_embeddings_list[0]
        result = store.query(VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5))
        assert len(result.nodes) == 5
    
    def test_vector_store_delete_node(self, mock_embedding):
        """Test deleting a node from vector store."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        node = TextNode(
            id_="delete-test",
            text="Content to delete",
            embedding=mock_embedding
        )
        
        # Add and then delete
        store.add([node])
        store.delete(node.ref_doc_id)
        
        # Query should return empty or not include the deleted node
        result = store.query(VectorStoreQuery(query_embedding=mock_embedding, similarity_top_k=10))
        node_ids = [n.id_ for n in result.nodes]
        assert "delete-test" not in node_ids
    
    def test_vector_store_persist_and_load(self, mock_embeddings_list, tmp_path):
        """Test persisting vector store to disk and loading."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        persist_path = tmp_path / "vector_store.json"
        store = SimpleVectorStore()
        
        # Add nodes
        nodes = []
        for i, embedding in enumerate(mock_embeddings_list[:3]):
            node = TextNode(
                id_=f"persist-node-{i}",
                text=f"Persistent content {i}",
                embedding=embedding
            )
            nodes.append(node)
        
        store.add(nodes)
        
        # Persist to disk
        store.persist(persist_path=str(persist_path))
        
        # Verify file was created
        assert persist_path.exists()
        
        # Load from disk
        loaded_store = SimpleVectorStore.from_persist_path(str(persist_path))
        
        # Query loaded store
        result = loaded_store.query(VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=3
        ))
        assert len(result.nodes) == 3


class TestVectorStoreProvider:
    """Test vector store provider abstraction."""
    
    def test_simple_provider_initialization(self):
        """Test SimpleVectorStoreProvider initialization."""
        from app.core.vector_store import create_vector_store
        
        provider = create_vector_store("simple")
        
        assert provider is not None
        assert hasattr(provider, 'get_vector_store')
        assert hasattr(provider, 'get_docstore')
        assert hasattr(provider, 'get_index_store')
        assert hasattr(provider, 'is_available')
    
    def test_simple_provider_is_available(self):
        """Test that simple provider is always available."""
        from app.core.vector_store import create_vector_store
        
        provider = create_vector_store("simple")
        
        assert provider.is_available() is True
    
    def test_simple_provider_with_persist_dir(self, tmp_path):
        """Test SimpleVectorStoreProvider with persistence directory."""
        from app.core.vector_store import create_vector_store
        
        persist_dir = tmp_path / "persist"
        provider = create_vector_store("simple", persist_dir=str(persist_dir))
        
        # Directory should be created
        assert persist_dir.exists()
        assert provider.persist_dir == str(persist_dir)
    
    def test_provider_get_vector_store(self):
        """Test getting vector store from provider."""
        from app.core.vector_store import create_vector_store
        from llama_index.core.vector_stores import SimpleVectorStore
        
        provider = create_vector_store("simple")
        vector_store = provider.get_vector_store()
        
        assert vector_store is not None
        assert isinstance(vector_store, SimpleVectorStore)
    
    def test_provider_get_docstore(self):
        """Test getting document store from provider."""
        from app.core.vector_store import create_vector_store
        from llama_index.core.storage.docstore import SimpleDocumentStore
        
        provider = create_vector_store("simple")
        docstore = provider.get_docstore()
        
        assert docstore is not None
        assert isinstance(docstore, SimpleDocumentStore)
    
    def test_provider_get_index_store(self):
        """Test getting index store from provider."""
        from app.core.vector_store import create_vector_store
        from llama_index.core.storage.index_store import SimpleIndexStore
        
        provider = create_vector_store("simple")
        index_store = provider.get_index_store()
        
        assert index_store is not None
        assert isinstance(index_store, SimpleIndexStore)
    
    def test_provider_caches_instances(self):
        """Test that provider caches store instances."""
        from app.core.vector_store import create_vector_store
        
        provider = create_vector_store("simple")
        
        # Get stores multiple times
        store1 = provider.get_vector_store()
        store2 = provider.get_vector_store()
        
        # Should be the same instance (cached)
        assert store1 is store2


class TestVectorStoreFactory:
    """Test vector store factory and registration."""
    
    def test_create_simple_vector_store(self):
        """Test creating simple vector store via factory."""
        from app.core.vector_store import create_vector_store
        from app.core.vector_store.simple import SimpleVectorStoreProvider
        
        provider = create_vector_store("simple")
        
        assert isinstance(provider, SimpleVectorStoreProvider)
    
    def test_list_available_stores(self):
        """Test listing available vector store types."""
        from app.core.vector_store import list_available_stores
        
        stores = list_available_stores()
        
        assert "simple" in stores
        assert isinstance(stores, list)
    
    def test_create_unknown_store_raises_error(self):
        """Test that creating unknown store type raises ValueError."""
        from app.core.vector_store import create_vector_store
        
        with pytest.raises(ValueError) as exc_info:
            create_vector_store("unknown_store_type")
        
        assert "Unknown vector store type" in str(exc_info.value)


class TestBatchVectorOperations:
    """Test batch insertion and operations on vectors."""
    
    def test_batch_add_nodes(self, mock_embeddings_list):
        """Test adding nodes in batch."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        batch_size = 50
        nodes = []
        
        for i in range(batch_size):
            node = TextNode(
                id_=f"batch-node-{i}",
                text=f"Batch content {i}",
                embedding=mock_embeddings_list[i % len(mock_embeddings_list)]
            )
            nodes.append(node)
        
        # Add batch
        store.add(nodes)
        
        # Verify all were added
        result = store.query(VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=batch_size
        ))
        assert len(result.nodes) == batch_size
    
    def test_batch_add_with_metadata(self, mock_embeddings_list):
        """Test batch adding nodes with metadata."""
        from llama_index.core.vector_stores import SimpleVectorStore
        
        store = SimpleVectorStore()
        nodes = []
        
        for i in range(10):
            node = TextNode(
                id_=f"meta-node-{i}",
                text=f"Content with metadata {i}",
                embedding=mock_embeddings_list[i % len(mock_embeddings_list)],
                metadata={
                    "source": "test",
                    "index": i,
                    "category": "batch"
                }
            )
            nodes.append(node)
        
        store.add(nodes)
        
        # Verify metadata was preserved
        result = store.query(VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=10
        ))
        
        for node in result.nodes:
            assert node.metadata.get("source") == "test"
            assert node.metadata.get("category") == "batch"
    
    def test_large_batch_performance(self, mock_embeddings_list):
        """Test handling of large batch operations."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        large_batch_size = 200
        nodes = []
        
        # Create large batch
        for i in range(large_batch_size):
            embedding = mock_embeddings_list[i % len(mock_embeddings_list)]
            node = TextNode(
                id_=f"large-batch-{i}",
                text=f"Large batch content item {i}",
                embedding=embedding
            )
            nodes.append(node)
        
        # Add large batch
        store.add(nodes)
        
        # Query should return all nodes
        result = store.query(VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=large_batch_size
        ))
        assert len(result.nodes) == large_batch_size


class TestVectorStorePersistence:
    """Test persistence operations for vector stores."""
    
    def test_persist_to_custom_path(self, mock_embeddings_list, tmp_path):
        """Test persisting to a custom file path."""
        from llama_index.core.vector_stores import SimpleVectorStore
        
        store = SimpleVectorStore()
        nodes = [
            TextNode(
                id_=f"persist-{i}",
                text=f"Content {i}",
                embedding=mock_embeddings_list[i % len(mock_embeddings_list)]
            )
            for i in range(5)
        ]
        
        store.add(nodes)
        
        custom_path = tmp_path / "custom_store.json"
        store.persist(persist_path=str(custom_path))
        
        assert custom_path.exists()
        # File should contain valid JSON
        content = custom_path.read_text()
        data = json.loads(content)
        assert isinstance(data, dict)
    
    def test_persist_directory_creation(self, mock_embedding, tmp_path):
        """Test that persistence creates directories if needed."""
        from llama_index.core.vector_stores import SimpleVectorStore
        
        store = SimpleVectorStore()
        node = TextNode(
            id_="test-node",
            text="Test content",
            embedding=mock_embedding
        )
        store.add([node])
        
        nested_path = tmp_path / "nested" / "deep" / "store.json"
        store.persist(persist_path=str(nested_path))
        
        assert nested_path.exists()
    
    def test_load_from_nonexistent_file_fails(self, tmp_path):
        """Test that loading from nonexistent file raises error."""
        from llama_index.core.vector_stores import SimpleVectorStore
        
        nonexistent_path = tmp_path / "does_not_exist.json"
        
        with pytest.raises((FileNotFoundError, ValueError)):
            SimpleVectorStore.from_persist_path(str(nonexistent_path))
    
    def test_persist_and_load_empty_store(self, tmp_path):
        """Test persisting and loading an empty store."""
        from llama_index.core.vector_stores import SimpleVectorStore
        
        store = SimpleVectorStore()
        persist_path = tmp_path / "empty_store.json"
        
        store.persist(persist_path=str(persist_path))
        
        loaded = SimpleVectorStore.from_persist_path(str(persist_path))
        assert loaded is not None


class TestChromaVectorStore:
    """Test ChromaDB vector store provider."""
    
    @patch('app.core.vector_store.chroma.chromadb.HttpClient')
    def test_chroma_provider_initialization(self, mock_chroma_client):
        """Test ChromaVectorStoreProvider initialization."""
        from app.core.vector_store import create_vector_store
        
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        provider = create_vector_store(
            "chroma",
            host="localhost",
            port=8000,
            collection="test_collection"
        )
        
        assert provider.host == "localhost"
        assert provider.port == 8000
        assert provider.collection_name == "test_collection"
    
    @patch('app.core.vector_store.chroma.chromadb.HttpClient')
    def test_chroma_provider_is_available_true(self, mock_chroma_client):
        """Test is_available returns True when ChromaDB is accessible."""
        from app.core.vector_store import create_vector_store
        
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = True
        mock_chroma_client.return_value = mock_client
        
        provider = create_vector_store("chroma")
        
        assert provider.is_available() is True
        mock_client.heartbeat.assert_called_once()
    
    @patch('app.core.vector_store.chroma.chromadb.HttpClient')
    def test_chroma_provider_is_available_false(self, mock_chroma_client):
        """Test is_available returns False when ChromaDB is not accessible."""
        from app.core.vector_store import create_vector_store
        
        mock_client = MagicMock()
        mock_client.heartbeat.side_effect = Exception("Connection refused")
        mock_chroma_client.return_value = mock_client
        
        provider = create_vector_store("chroma")
        
        assert provider.is_available() is False
    
    def test_chroma_provider_fallback_to_simple(self):
        """Test that ChromaDB provider falls back to simple store on failure."""
        from app.core.vector_store.chroma import ChromaVectorStoreProvider
        from llama_index.core.vector_stores import SimpleVectorStore
        
        with patch('chromadb.HttpClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            provider = ChromaVectorStoreProvider()
            
            # When ChromaDB fails, should fall back to SimpleVectorStore
            store = provider.get_vector_store()
            assert isinstance(store, SimpleVectorStore)
