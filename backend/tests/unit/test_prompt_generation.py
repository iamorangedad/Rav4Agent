"""Unit tests for prompt generation functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any


class TestChatMessageFormatting:
    """Test chat message formatting."""
    
    def test_user_message_formatting(self):
        """Test formatting of user messages."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        message = ChatMessage(
            role=MessageRole.USER,
            content="What is machine learning?"
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "What is machine learning?"
    
    def test_assistant_message_formatting(self):
        """Test formatting of assistant messages."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Machine learning is a subset of AI."
        )
        
        assert message.role == MessageRole.ASSISTANT
        assert "Machine learning" in message.content
    
    def test_system_message_formatting(self):
        """Test formatting of system messages."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        system_prompt = "You are a helpful AI assistant."
        message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt
        )
        
        assert message.role == MessageRole.SYSTEM
        assert message.content == system_prompt
    
    def test_message_with_special_characters(self):
        """Test messages containing special characters."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        content = "Test with émojis 🎉, symbols @#$%, and \"quotes\""
        message = ChatMessage(
            role=MessageRole.USER,
            content=content
        )
        
        assert message.content == content
        assert "🎉" in message.content
    
    def test_message_with_multiline_content(self):
        """Test messages with multiline content."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        content = """Line 1
        Line 2
        Line 3"""
        message = ChatMessage(
            role=MessageRole.USER,
            content=content
        )
        
        assert "Line 1" in message.content
        assert "Line 2" in message.content
        assert "Line 3" in message.content
    
    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        message = ChatMessage(
            role=MessageRole.USER,
            content=""
        )
        
        assert message.content == ""


class TestContextInjection:
    """Test context injection into prompts."""
    
    def test_basic_context_injection(self):
        """Test basic context injection into prompt template."""
        template = """Context information:
{context}

Question: {question}
Answer:"""
        
        context = "This is the retrieved context."
        question = "What is this about?"
        
        prompt = template.format(context=context, question=question)
        
        assert context in prompt
        assert question in prompt
        assert "Context information:" in prompt
    
    def test_context_with_multiple_documents(self):
        """Test injecting context from multiple documents."""
        template = """Context:
{context}

Question: {question}"""
        
        contexts = [
            "Document 1: Information about AI.",
            "Document 2: Details on machine learning.",
            "Document 3: Deep learning overview."
        ]
        context_str = "\n\n".join(contexts)
        question = "What is AI?"
        
        prompt = template.format(context=context_str, question=question)
        
        assert "Document 1" in prompt
        assert "Document 2" in prompt
        assert "Document 3" in prompt
        assert question in prompt
    
    def test_context_truncation(self):
        """Test context truncation when too long."""
        max_context_length = 1000
        
        template = "Context: {context}\nQuestion: {question}"
        
        # Create very long context
        long_context = "Word " * 10000
        question = "What is this?"
        
        # Simulate truncation
        if len(long_context) > max_context_length:
            truncated_context = long_context[:max_context_length] + "..."
        else:
            truncated_context = long_context
        
        prompt = template.format(context=truncated_context, question=question)
        
        assert len(truncated_context) <= max_context_length + 3
        assert "..." in prompt
    
    def test_context_with_source_citations(self):
        """Test context with source citation information."""
        template = """Context information (with sources):
{context}

Question: {question}"""
        
        context_with_sources = """[Source: doc1.pdf, Page 1]
First piece of information.

[Source: doc2.pdf, Page 3]
Second piece of information."""
        
        question = "What are the key points?"
        prompt = template.format(context=context_with_sources, question=question)
        
        assert "Source: doc1.pdf" in prompt
        assert "Source: doc2.pdf" in prompt
        assert question in prompt
    
    def test_context_relevance_ranking(self):
        """Test that context is ordered by relevance."""
        contexts = [
            {"content": "Most relevant information", "score": 0.95},
            {"content": "Somewhat relevant", "score": 0.75},
            {"content": "Least relevant", "score": 0.50}
        ]
        
        # Sort by relevance score
        sorted_contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)
        
        assert sorted_contexts[0]["score"] == 0.95
        assert sorted_contexts[1]["score"] == 0.75
        assert sorted_contexts[2]["score"] == 0.50
    
    def test_empty_context_handling(self):
        """Test handling of empty context."""
        template = """Context: {context}

Question: {question}"""
        
        context = ""
        question = "What is this?"
        
        prompt = template.format(context=context, question=question)
        
        assert "Context:" in prompt
        assert question in prompt


class TestConversationHistory:
    """Test conversation history handling."""
    
    def test_basic_conversation_history(self):
        """Test basic conversation history tracking."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        history = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?")
        ]
        
        assert len(history) == 3
        assert history[0].role == MessageRole.USER
        assert history[1].role == MessageRole.ASSISTANT
        assert history[2].role == MessageRole.USER
    
    def test_history_truncation(self):
        """Test conversation history truncation for token limits."""
        max_messages = 10
        
        # Create long history
        history = []
        for i in range(50):
            history.append({"role": "user", "content": f"Message {i}"})
            history.append({"role": "assistant", "content": f"Response {i}"})
        
        # Truncate to last N messages
        truncated = history[-max_messages:]
        
        assert len(truncated) == max_messages
        assert truncated[0]["content"] == "Message 45"
        assert truncated[-1]["content"] == "Response 49"
    
    def test_history_message_formatting(self):
        """Test formatting conversation history for prompt."""
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "What can I do with it?"}
        ]
        
        formatted = ""
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role_label}: {msg['content']}\n"
        
        assert "User: What is Python?" in formatted
        assert "Assistant: Python is a programming language." in formatted
        assert "User: What can I do with it?" in formatted
    
    def test_history_with_system_prompt(self):
        """Test history that includes system message."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi!"),
            ChatMessage(role=MessageRole.USER, content="Help me")
        ]
        
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        conversation = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        assert len(system_messages) == 1
        assert len(conversation) == 3
        assert system_messages[0].content == "You are a helpful assistant."
    
    def test_empty_history_handling(self):
        """Test handling of empty conversation history."""
        history = []
        
        # Format empty history
        formatted = ""
        for msg in history:
            formatted += f"{msg['role']}: {msg['content']}\n"
        
        assert formatted == ""
        assert len(history) == 0
    
    def test_history_summary_generation(self):
        """Test generating summary from conversation history."""
        history = [
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
            {"role": "user", "content": "What about machine learning?"},
            {"role": "assistant", "content": "ML is a subset of AI..."},
            {"role": "user", "content": "Thanks!"}
        ]
        
        # Generate simple summary
        topics = []
        for msg in history:
            if msg["role"] == "user":
                content = msg["content"].lower()
                if "ai" in content:
                    topics.append("AI")
                if "machine learning" in content:
                    topics.append("Machine Learning")
        
        assert "AI" in topics
        assert "Machine Learning" in topics


class TestSystemPromptGeneration:
    """Test system prompt generation."""
    
    def test_basic_system_prompt(self):
        """Test generation of basic system prompt."""
        system_prompt = """You are a helpful AI assistant specialized in answering questions based on provided documents.
Answer the user's question using only the information from the provided context.
If the answer is not in the context, say "I don't have enough information to answer this question."""
        
        assert "helpful AI assistant" in system_prompt
        assert "provided documents" in system_prompt or "provided context" in system_prompt
        assert "don't have enough information" in system_prompt
    
    def test_system_prompt_with_persona(self):
        """Test system prompt with specific persona."""
        persona = "technical expert"
        domain = "software engineering"
        
        system_prompt = f"""You are a {persona} specializing in {domain}.
Provide detailed, accurate answers based on the provided context.
Use technical terminology appropriate for the field.
If unsure, acknowledge limitations rather than making assumptions."""
        
        assert persona in system_prompt
        assert domain in system_prompt
        assert "technical terminology" in system_prompt
    
    def test_system_prompt_with_constraints(self):
        """Test system prompt with response constraints."""
        max_length = 500
        style = "concise"
        
        system_prompt = f"""You are an AI assistant. Follow these rules:
1. Keep responses under {max_length} characters
2. Be {style} and to the point
3. Only use information from the provided context
4. Cite sources when possible"""
        
        assert str(max_length) in system_prompt
        assert style in system_prompt
        assert "Cite sources" in system_prompt
    
    def test_system_prompt_with_capabilities(self):
        """Test system prompt listing capabilities."""
        capabilities = [
            "Answering questions from documents",
            "Summarizing content",
            "Explaining technical concepts"
        ]
        
        capabilities_str = "\n".join(f"- {cap}" for cap in capabilities)
        system_prompt = f"""You are a document-based AI assistant.

Your capabilities:
{capabilities_str}

Guidelines:
- Only use information from the provided context
- Be accurate and helpful
- Acknowledge if you cannot answer a question"""
        
        for cap in capabilities:
            assert cap in system_prompt
        assert "Guidelines:" in system_prompt
    
    def test_system_prompt_for_rag(self):
        """Test system prompt specifically for RAG (Retrieval Augmented Generation)."""
        system_prompt = """You are a Retrieval-Augmented Generation (RAG) assistant.
Your task is to answer questions based on the retrieved context provided below.

Rules:
1. Use ONLY the information from the provided context
2. Do not use outside knowledge or make assumptions
3. If the context doesn't contain the answer, say so clearly
4. Cite the source documents when providing information
5. Be concise but complete

Context will be provided between [CONTEXT] and [/CONTEXT] tags."""
        
        assert "Retrieval-Augmented Generation" in system_prompt
        assert "ONLY the information" in system_prompt
        assert "[CONTEXT]" in system_prompt
    
    def test_dynamic_system_prompt_generation(self):
        """Test dynamically generating system prompt based on settings."""
        settings = {
            "language": "English",
            "tone": "professional",
            "citations": True,
            "max_response_length": "medium"
        }
        
        prompt_parts = [
            "You are an AI assistant.",
            f"Respond in {settings['language']}.",
            f"Use a {settings['tone']} tone."
        ]
        
        if settings["citations"]:
            prompt_parts.append("Always cite your sources.")
        
        if settings["max_response_length"] == "short":
            prompt_parts.append("Keep responses brief.")
        elif settings["max_response_length"] == "medium":
            prompt_parts.append("Provide balanced, medium-length responses.")
        
        system_prompt = " ".join(prompt_parts)
        
        assert settings["language"] in system_prompt
        assert settings["tone"] in system_prompt
        assert "cite your sources" in system_prompt
        assert "medium-length" in system_prompt


class TestPromptAssembly:
    """Test complete prompt assembly."""
    
    def test_complete_rag_prompt_assembly(self):
        """Test assembling complete RAG prompt with all components."""
        from llama_index.core.llms import ChatMessage, MessageRole
        
        # System prompt
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful document assistant."
        )
        
        # Context
        context = """[Document 1] Machine learning is a subset of AI.
[Document 2] Deep learning uses neural networks."""
        
        # History
        history = [
            ChatMessage(role=MessageRole.USER, content="What is AI?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="AI is artificial intelligence.")
        ]
        
        # Current question
        question = ChatMessage(
            role=MessageRole.USER,
            content="Tell me about machine learning"
        )
        
        # Assemble full prompt
        messages = [system_msg] + history + [question]
        
        assert len(messages) == 4
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[-1].role == MessageRole.USER
        assert messages[-1].content == "Tell me about machine learning"
    
    def test_prompt_with_context_and_history(self):
        """Test prompt including both context and conversation history."""
        template = """System: {system_prompt}

Context:
{context}

Conversation History:
{history}

User: {question}
Assistant:"""
        
        system_prompt = "You are a helpful assistant."
        context = "The sky is blue."
        history = "User: Hello\nAssistant: Hi!"
        question = "What color is the sky?"
        
        full_prompt = template.format(
            system_prompt=system_prompt,
            context=context,
            history=history,
            question=question
        )
        
        assert system_prompt in full_prompt
        assert context in full_prompt
        assert "Conversation History:" in full_prompt
        assert "User: Hello" in full_prompt
        assert question in full_prompt
    
    def test_prompt_token_estimation(self):
        """Test estimating token count for prompt."""
        prompt = "This is a test prompt with about fifteen words in total."
        
        # Rough estimation: ~1.3 tokens per word
        words = len(prompt.split())
        estimated_tokens = int(words * 1.3)
        
        assert words == 11
        assert estimated_tokens > words  # Should have more tokens than words
    
    def test_prompt_with_metadata_injection(self):
        """Test prompt with metadata injected into context."""
        context_items = [
            {"content": "Information about Python.", "source": "doc1.pdf", "page": 1},
            {"content": "Information about Java.", "source": "doc2.pdf", "page": 5}
        ]
        
        formatted_context = ""
        for item in context_items:
            formatted_context += f"[{item['source']}, Page {item['page']}]\n{item['content']}\n\n"
        
        prompt_template = """Based on the following context:
{context}

Answer this question: {question}"""
        
        prompt = prompt_template.format(
            context=formatted_context,
            question="What programming languages are mentioned?"
        )
        
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt
        assert "Page 1" in prompt
        assert "Page 5" in prompt
        assert "Python" in prompt
        assert "Java" in prompt


class TestPromptEdgeCases:
    """Test edge cases in prompt generation."""
    
    def test_very_long_question_handling(self):
        """Test handling of very long user questions."""
        max_length = 500
        long_question = "word " * 200  # Very long question
        
        # Truncate if necessary
        if len(long_question) > max_length:
            truncated = long_question[:max_length] + "..."
        else:
            truncated = long_question
        
        assert len(truncated) <= max_length + 3
    
    def test_special_characters_in_question(self):
        """Test handling special characters in questions."""
        questions = [
            "What is 2+2?",
            "Explain \"machine learning\"",
            "Cost is $50, right?",
            "Use the & operator",
            "What's the <difference> between {A} and [B]?"
        ]
        
        for question in questions:
            # Verify question can be formatted into template
            prompt = f"Question: {question}\nAnswer:"
            assert question in prompt
            assert isinstance(prompt, str)
    
    def test_unicode_content_handling(self):
        """Test handling of unicode content in prompts."""
        content = """
        English: Hello
        Chinese: 你好
        Japanese: こんにちは
        Arabic: مرحبا
        Emoji: 🌍🎉👋
        """
        
        prompt = f"Content: {content}\nQuestion: What languages are shown?"
        
        assert "你好" in prompt
        assert "こんにちは" in prompt
        assert "🌍" in prompt
    
    def test_empty_question_handling(self):
        """Test handling of empty questions."""
        question = ""
        
        prompt = f"User: {question}\nAssistant:"
        
        assert "User:" in prompt
        assert "Assistant:" in prompt
    
    def test_prompt_with_code_blocks(self):
        """Test handling code blocks in context or questions."""
        code_context = """
        Here's a Python example:
        ```python
        def hello():
            print("Hello, World!")
        ```
        """
        
        prompt = f"Context: {code_context}\nQuestion: Explain this code"
        
        assert "```python" in prompt
        assert "def hello():" in prompt
        assert "print" in prompt
