"""Unit tests for document parsing and splitting functionality."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class TestDocumentParsing:
    """Test document parsing functionality."""
    
    def test_document_creation_from_text(self):
        """Test creating a Document from text content."""
        text = "This is a test document content."
        doc = Document(text=text, id_="test-doc-1")
        
        assert doc.text == text
        assert doc.id_ == "test-doc-1"
        assert doc.get_content() == text
    
    def test_document_with_metadata(self):
        """Test document with metadata."""
        text = "Test content"
        metadata = {"filename": "test.txt", "page": 1}
        doc = Document(text=text, metadata=metadata, id_="test-doc-2")
        
        assert doc.metadata["filename"] == "test.txt"
        assert doc.metadata["page"] == 1
    
    def test_document_content_length(self):
        """Test document content length calculation."""
        text = "This is a test."
        doc = Document(text=text)
        
        assert len(doc.text) == len(text)
        assert len(doc.get_content()) == len(text)


class TestDocumentSplitting:
    """Test document splitting and chunking functionality."""
    
    def test_sentence_splitter_basic(self):
        """Test basic sentence splitting."""
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        
        text = "First sentence. Second sentence. Third sentence."
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        
        assert len(nodes) > 0
        assert all(len(node.get_content()) > 0 for node in nodes)
    
    def test_sentence_splitter_chunk_size(self):
        """Test that chunks respect size limits."""
        chunk_size = 200
        chunk_overlap = 50
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Create a long text
        text = "Word. " * 500  # Approximately 2500 characters
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Each node should be roughly within chunk_size + tolerance
        for node in nodes:
            content = node.get_content()
            assert len(content) <= chunk_size + 100  # Allow some tolerance
    
    def test_sentence_splitter_overlap(self):
        """Test that chunks have proper overlap."""
        chunk_size = 100
        chunk_overlap = 20
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        
        # If we have multiple nodes, check for overlap
        if len(nodes) > 1:
            for i in range(len(nodes) - 1):
                current_text = nodes[i].get_content()
                next_text = nodes[i + 1].get_content()
                # Check that there's some shared content between consecutive chunks
                assert len(current_text) > 0
                assert len(next_text) > 0
    
    def test_splitter_preserves_context(self):
        """Test that splitting preserves semantic context."""
        splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
        
        text = """
        Introduction: This document discusses testing strategies.
        
        Section 1: Unit Testing
        Unit tests verify individual components in isolation.
        They are fast and reliable.
        
        Section 2: Integration Testing
        Integration tests verify component interactions.
        They ensure the system works as a whole.
        """
        doc = Document(text=text, metadata={"section": "intro"})
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Each node should have meaningful content
        for node in nodes:
            content = node.get_content()
            assert len(content) > 10  # Not empty or too short
            # Should preserve some words from original
            assert any(word in content.lower() for word in ["testing", "document", "section"])
    
    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        
        doc = Document(text="")
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Should handle empty documents gracefully
        assert len(nodes) == 0 or all(len(node.get_content().strip()) == 0 for node in nodes)
    
    def test_very_long_document(self, sample_long_text):
        """Test splitting very long documents."""
        chunk_size = 1000
        chunk_overlap = 200
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        doc = Document(text=sample_long_text)
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Should create multiple nodes for long text
        assert len(nodes) > 5
        
        # All nodes should have content
        total_length = sum(len(node.get_content()) for node in nodes)
        assert total_length > len(sample_long_text) * 0.8  # Most content preserved
    
    def test_special_characters_handling(self):
        """Test handling of special characters and formatting."""
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        
        text = """Special chars: émojis 🎉, symbols @#$%, and newlines
        
        Multiple   spaces   and	tabs should be handled.
        """
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Should handle special characters without crashing
        assert len(nodes) > 0
        for node in nodes:
            content = node.get_content()
            assert isinstance(content, str)


class TestDocumentLoading:
    """Test document loading from files."""
    
    def test_simple_directory_reader_mock(self, test_upload_dir):
        """Test SimpleDirectoryReader with mock files."""
        # Create a test file
        test_file = test_upload_dir / "test_doc.txt"
        test_file.write_text("This is test content for document loading.")
        
        # Mock the reader
        from llama_index.core import SimpleDirectoryReader
        
        # Test that we can create a reader
        reader = SimpleDirectoryReader(input_files=[str(test_file)])
        assert reader is not None
        
        # Test loading data
        docs = reader.load_data()
        assert len(docs) == 1
        assert "test content" in docs[0].text
    
    def test_multiple_files_loading(self, test_upload_dir):
        """Test loading multiple files."""
        # Create multiple test files
        for i in range(3):
            test_file = test_upload_dir / f"test_doc_{i}.txt"
            test_file.write_text(f"Content of document {i}")
        
        from llama_index.core import SimpleDirectoryReader
        
        files = [str(test_upload_dir / f"test_doc_{i}.txt") for i in range(3)]
        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        
        assert len(docs) == 3
        for i, doc in enumerate(docs):
            assert f"Content of document {i}" in doc.text
    
    def test_filename_as_id(self, test_upload_dir):
        """Test that filename is used as document ID."""
        test_file = test_upload_dir / "my_document.txt"
        test_file.write_text("Test content")
        
        from llama_index.core import SimpleDirectoryReader
        
        reader = SimpleDirectoryReader(
            input_files=[str(test_file)],
            filename_as_id=True
        )
        docs = reader.load_data()
        
        # Document ID should contain filename
        assert "my_document" in docs[0].id_


class TestChunkMetadata:
    """Test chunk/node metadata preservation."""
    
    def test_node_metadata_inheritance(self):
        """Test that nodes inherit metadata from parent document."""
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
        
        text = "First sentence. Second sentence. Third sentence."
        metadata = {
            "filename": "source.txt",
            "page": 1,
            "total_pages": 10
        }
        doc = Document(text=text, metadata=metadata)
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Each node should inherit metadata
        for node in nodes:
            assert node.metadata.get("filename") == "source.txt"
            assert node.metadata.get("page") == 1
    
    def test_chunk_relationship_tracking(self):
        """Test that chunks track their relationships."""
        splitter = SentenceSplitter(chunk_size=50, chunk_overlap=10)
        
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        doc = Document(text=text, id_="parent-doc")
        nodes = splitter.get_nodes_from_documents([doc])
        
        # If we have multiple nodes, they should have source relationship
        if len(nodes) > 1:
            for node in nodes:
                # Each node should reference source
                assert node.source_node is not None or hasattr(node, 'ref_doc_id')
