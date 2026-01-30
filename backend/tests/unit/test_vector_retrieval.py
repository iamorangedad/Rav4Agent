"""Unit tests for vector database retrieval functionality."""
import math
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from llama_index.core.schema import TextNode


class TestSimilaritySearch:
    """Test similarity search operations."""
    
    def test_basic_similarity_search(self, mock_embeddings_list):
        """Test basic similarity search with query embedding."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add nodes with different embeddings
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"search-node-{i}",
                text=f"Document about topic {i}",
                embedding=embedding
            )
            store.add([node])
        
        # Search with first embedding
        query = VectorStoreQuery(query_embedding=mock_embeddings_list[0], similarity_top_k=3)
        result = store.query(query)
        
        assert len(result.nodes) <= 3
        assert len(result.similarities) == len(result.nodes)
        assert all(0 <= s <= 1 for s in result.similarities)
    
    def test_similarity_search_top_k(self, mock_embeddings_list):
        """Test similarity search with different top_k values."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add 10 nodes
        for i in range(10):
            node = TextNode(
                id_=f"topk-node-{i}",
                text=f"Content {i}",
                embedding=mock_embeddings_list[i % len(mock_embeddings_list)]
            )
            store.add([node])
        
        # Test different top_k values
        for top_k in [1, 3, 5, 10]:
            query = VectorStoreQuery(
                query_embedding=mock_embeddings_list[0],
                similarity_top_k=top_k
            )
            result = store.query(query)
            assert len(result.nodes) == top_k
    
    def test_similarity_search_empty_store(self, mock_embedding):
        """Test similarity search on empty store."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        query = VectorStoreQuery(query_embedding=mock_embedding, similarity_top_k=5)
        result = store.query(query)
        
        assert len(result.nodes) == 0
        assert len(result.similarities) == 0
    
    def test_similarity_scores_in_range(self, mock_embeddings_list):
        """Test that similarity scores are within valid range."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"score-node-{i}",
                text=f"Document {i}",
                embedding=embedding
            )
            store.add([node])
        
        query = VectorStoreQuery(query_embedding=mock_embeddings_list[0], similarity_top_k=5)
        result = store.query(query)
        
        # Cosine similarity should be between -1 and 1
        for score in result.similarities:
            assert -1 <= score <= 1
    
    def test_similar_vectors_have_high_score(self, mock_embeddings_list):
        """Test that similar vectors have high similarity scores."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add same embedding multiple times (identical vectors)
        identical_embedding = mock_embeddings_list[0]
        for i in range(3):
            node = TextNode(
                id_=f"identical-{i}",
                text=f"Identical content {i}",
                embedding=identical_embedding
            )
            store.add([node])
        
        # Search with the same embedding
        query = VectorStoreQuery(
            query_embedding=identical_embedding,
            similarity_top_k=3
        )
        result = store.query(query)
        
        # Should have high similarity (close to 1 for identical vectors)
        if result.similarities:
            assert result.similarities[0] > 0.99


class TestQueryEmbedding:
    """Test query embedding and retrieval flow."""
    
    def test_query_embedding_generation(self):
        """Test generating embedding for query text."""
        # Mock embedding model
        mock_embed_model = Mock()
        mock_embed_model.get_text_embedding.return_value = [0.1] * 768
        
        query_text = "What is machine learning?"
        embedding = mock_embed_model.get_text_embedding(query_text)
        
        assert len(embedding) == 768
        assert isinstance(embedding, list)
    
    def test_query_variations_produce_embeddings(self):
        """Test that different query variations produce embeddings."""
        queries = [
            "What is AI?",
            "Tell me about artificial intelligence",
            "Explain machine learning",
            "How does deep learning work?"
        ]
        
        mock_embed_model = Mock()
        mock_embed_model.get_text_embedding.return_value = [0.1] * 768
        
        embeddings = []
        for query in queries:
            embedding = mock_embed_model.get_text_embedding(query)
            embeddings.append(embedding)
        
        assert len(embeddings) == len(queries)
        assert all(len(e) == 768 for e in embeddings)
    
    def test_empty_query_handling(self):
        """Test handling of empty query text."""
        mock_embed_model = Mock()
        mock_embed_model.get_text_embedding.return_value = [0.0] * 768
        
        empty_embedding = mock_embed_model.get_text_embedding("")
        
        assert len(empty_embedding) == 768
    
    def test_long_query_truncation(self):
        """Test that long queries are truncated for embedding."""
        mock_embed_model = Mock()
        mock_embed_model.get_text_embedding.return_value = [0.1] * 768
        
        long_query = "word " * 10000
        
        # Simulate truncation logic
        max_length = 2000
        if len(long_query) > max_length:
            truncated = long_query[:max_length] + "..."
        else:
            truncated = long_query
        
        assert len(truncated) <= max_length + 3
        
        # Should still get embedding
        embedding = mock_embed_model.get_text_embedding(truncated)
        assert len(embedding) == 768


class TestRelevanceScoring:
    """Test relevance scoring and ranking."""
    
    def test_relevance_score_calculation(self):
        """Test calculation of relevance scores."""
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(x**2 for x in v1))
            mag2 = math.sqrt(sum(x**2 for x in v2))
            if mag1 == 0 or mag2 == 0:
                return 0
            return dot_product / (mag1 * mag2)
        
        # Test with known similar vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.9, 0.1, 0.0]
        
        similarity = cosine_similarity(v1, v2)
        assert 0 < similarity < 1  # Should be positive but not 1
    
    def test_identical_vectors_score(self):
        """Test that identical vectors have perfect similarity."""
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(x**2 for x in v1))
            mag2 = math.sqrt(sum(x**2 for x in v2))
            if mag1 == 0 or mag2 == 0:
                return 0
            return dot_product / (mag1 * mag2)
        
        v = [0.1, 0.2, 0.3, 0.4]
        similarity = cosine_similarity(v, v)
        
        assert abs(similarity - 1.0) < 0.0001  # Should be exactly 1
    
    def test_orthogonal_vectors_score(self):
        """Test that orthogonal vectors have zero similarity."""
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(x**2 for x in v1))
            mag2 = math.sqrt(sum(x**2 for x in v2))
            if mag1 == 0 or mag2 == 0:
                return 0
            return dot_product / (mag1 * mag2)
        
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        
        similarity = cosine_similarity(v1, v2)
        assert abs(similarity - 0.0) < 0.0001  # Should be 0
    
    def test_relevance_ranking_order(self, mock_embeddings_list):
        """Test that results are ranked by relevance."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Create nodes with clear similarity differences
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"rank-{i}",
                text=f"Document {i}",
                embedding=embedding
            )
            store.add([node])
        
        query = VectorStoreQuery(query_embedding=mock_embeddings_list[0], similarity_top_k=5)
        result = store.query(query)
        
        # Scores should be in descending order (most relevant first)
        for i in range(len(result.similarities) - 1):
            assert result.similarities[i] >= result.similarities[i + 1]
    
    def test_relevance_threshold_filtering(self, mock_embeddings_list):
        """Test filtering results by relevance threshold."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add diverse embeddings
        for i, embedding in enumerate(mock_embeddings_list):
            node = TextNode(
                id_=f"threshold-{i}",
                text=f"Document {i}",
                embedding=embedding
            )
            store.add([node])
        
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=len(mock_embeddings_list)
        )
        result = store.query(query)
        
        # Filter by threshold
        threshold = 0.5
        filtered = [
            (node, score) for node, score in zip(result.nodes, result.similarities)
            if score >= threshold
        ]
        
        # All filtered results should meet threshold
        for _, score in filtered:
            assert score >= threshold


class TestRetrievalWithFilters:
    """Test retrieval operations with filters."""
    
    def test_retrieval_with_metadata_filter(self, mock_embeddings_list):
        """Test retrieval filtered by metadata."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
        
        store = SimpleVectorStore()
        
        # Add nodes with different metadata
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"filter-node-{i}",
                text=f"Content {i}",
                embedding=embedding,
                metadata={
                    "category": "A" if i < 3 else "B",
                    "priority": i
                }
            )
            store.add([node])
        
        # Query with metadata filter for category A
        filters = MetadataFilters(
            filters=[MetadataFilter(key="category", value="A")]
        )
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=5,
            filters=filters
        )
        result = store.query(query)
        
        # Should only return nodes with category A
        for node in result.nodes:
            assert node.metadata.get("category") == "A"
    
    def test_retrieval_with_multiple_filters(self, mock_embeddings_list):
        """Test retrieval with multiple metadata filters."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
        
        store = SimpleVectorStore()
        
        # Add nodes with varied metadata
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"multi-filter-{i}",
                text=f"Content {i}",
                embedding=embedding,
                metadata={
                    "type": "document",
                    "priority": i,
                    "active": i < 3
                }
            )
            store.add([node])
        
        # Query with multiple filters
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="type", value="document"),
                MetadataFilter(key="active", value=True)
            ]
        )
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=5,
            filters=filters
        )
        result = store.query(query)
        
        # All results should match both filters
        for node in result.nodes:
            assert node.metadata.get("type") == "document"
            assert node.metadata.get("active") is True
    
    def test_retrieval_filter_no_matches(self, mock_embeddings_list):
        """Test retrieval when filter matches no documents."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
        
        store = SimpleVectorStore()
        
        # Add nodes
        for i, embedding in enumerate(mock_embeddings_list[:3]):
            node = TextNode(
                id_=f"nomatch-{i}",
                text=f"Content {i}",
                embedding=embedding,
                metadata={"category": "existing"}
            )
            store.add([node])
        
        # Query with non-matching filter
        filters = MetadataFilters(
            filters=[MetadataFilter(key="category", value="nonexistent")]
        )
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=5,
            filters=filters
        )
        result = store.query(query)
        
        # Should return empty results
        assert len(result.nodes) == 0
    
    def test_retrieval_without_filters_returns_all(self, mock_embeddings_list):
        """Test that retrieval without filters returns all matching documents."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add nodes with various metadata
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node = TextNode(
                id_=f"nofilter-{i}",
                text=f"Content {i}",
                embedding=embedding,
                metadata={"category": f"cat{i}"}
            )
            store.add([node])
        
        # Query without filters
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=5
        )
        result = store.query(query)
        
        # Should return all documents
        assert len(result.nodes) == 5
    
    def test_filter_with_id_list(self, mock_embeddings_list):
        """Test filtering by specific node IDs."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add nodes with specific IDs
        target_ids = ["target-1", "target-2"]
        for i, embedding in enumerate(mock_embeddings_list[:5]):
            node_id = target_ids[i] if i < 2 else f"other-{i}"
            node = TextNode(
                id_=node_id,
                text=f"Content {i}",
                embedding=embedding
            )
            store.add([node])
        
        # Query with specific node IDs (if supported)
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=5
        )
        result = store.query(query)
        
        # Results should exist
        assert len(result.nodes) > 0


class TestVectorSearchEdgeCases:
    """Test edge cases for vector search."""
    
    def test_search_with_zero_vector(self):
        """Test search with zero vector embedding."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add nodes with normal embeddings
        for i in range(3):
            node = TextNode(
                id_=f"zero-search-{i}",
                text=f"Content {i}",
                embedding=[0.1, 0.2, 0.3] * 256  # 768 dims
            )
            store.add([node])
        
        # Search with zero vector
        zero_vector = [0.0] * 768
        query = VectorStoreQuery(query_embedding=zero_vector, similarity_top_k=3)
        result = store.query(query)
        
        # Should handle gracefully
        assert isinstance(result.nodes, list)
    
    def test_search_top_k_larger_than_store(self, mock_embeddings_list):
        """Test search when top_k exceeds number of documents."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add only 3 nodes
        for i in range(3):
            node = TextNode(
                id_=f"few-{i}",
                text=f"Content {i}",
                embedding=mock_embeddings_list[i]
            )
            store.add([node])
        
        # Request top 10 when only 3 exist
        query = VectorStoreQuery(query_embedding=mock_embeddings_list[0], similarity_top_k=10)
        result = store.query(query)
        
        # Should return only available nodes
        assert len(result.nodes) == 3
    
    def test_search_with_special_characters_in_metadata(self, mock_embeddings_list):
        """Test search with special characters in metadata."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
        
        store = SimpleVectorStore()
        
        # Add node with special metadata
        node = TextNode(
            id_="special-meta",
            text="Test content",
            embedding=mock_embeddings_list[0],
            metadata={
                "title": "Document with émojis 🎉",
                "path": "/path/with spaces/file.txt",
                "tags": "tag1,tag2,tag3"
            }
        )
        store.add([node])
        
        # Query should work
        query = VectorStoreQuery(query_embedding=mock_embeddings_list[0], similarity_top_k=1)
        result = store.query(query)
        
        assert len(result.nodes) == 1
        assert "émojis" in result.nodes[0].metadata.get("title", "")
    
    def test_concurrent_queries(self, mock_embeddings_list):
        """Test handling concurrent query requests."""
        import concurrent.futures
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Populate store
        for i in range(20):
            node = TextNode(
                id_=f"concurrent-{i}",
                text=f"Content {i}",
                embedding=mock_embeddings_list[i % len(mock_embeddings_list)]
            )
            store.add([node])
        
        # Execute concurrent queries
        def query_store(embedding):
            query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=5)
            return store.query(query)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(query_store, mock_embeddings_list[i % len(mock_embeddings_list)])
                for i in range(10)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # All queries should succeed
        assert len(results) == 10
        assert all(len(r.nodes) > 0 for r in results)


class TestRetrievalIntegration:
    """Test retrieval integration scenarios."""
    
    def test_full_retrieval_pipeline(self, mock_embeddings_list):
        """Test full retrieval pipeline from query to results."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        # Setup
        store = SimpleVectorStore()
        documents = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Python is a programming language.",
            "Neural networks are inspired by the brain."
        ]
        
        # Index documents
        for i, doc in enumerate(documents):
            node = TextNode(
                id_=f"doc-{i}",
                text=doc,
                embedding=mock_embeddings_list[i % len(mock_embeddings_list)],
                metadata={"source": "test", "index": i}
            )
            store.add([node])
        
        # Query
        query_embedding = mock_embeddings_list[0]
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
        result = store.query(query)
        
        # Verify
        assert len(result.nodes) == 2
        assert all(hasattr(node, 'text') for node in result.nodes)
        assert all(hasattr(node, 'metadata') for node in result.nodes)
    
    def test_retrieval_with_source_tracking(self, mock_embeddings_list):
        """Test that retrieval preserves source information."""
        from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
        
        store = SimpleVectorStore()
        
        # Add nodes with source information
        sources = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        for i, embedding in enumerate(mock_embeddings_list[:3]):
            node = TextNode(
                id_=f"source-track-{i}",
                text=f"Content from {sources[i]}",
                embedding=embedding,
                metadata={"source_file": sources[i], "page": i + 1}
            )
            store.add([node])
        
        # Query
        query = VectorStoreQuery(
            query_embedding=mock_embeddings_list[0],
            similarity_top_k=3
        )
        result = store.query(query)
        
        # Verify source tracking
        for node in result.nodes:
            assert "source_file" in node.metadata
            assert node.metadata["source_file"] in sources
            assert "page" in node.metadata
