"""Pydantic models for API requests and responses."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    model: Optional[str] = Field(None, description="LLM model name to use")
    embedding_model: Optional[str] = Field(None, description="Embedding model name to use")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")


class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    size: str = Field(..., description="Model size")
    parameter_size: str = Field(..., description="Parameter count")
    quantization: str = Field(..., description="Quantization type")
    requires_gpu: bool = Field(..., description="Whether GPU is required")


class RecommendedModel(BaseModel):
    """Recommended model information."""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    context_window: str = Field(..., description="Context window size")
    gpu_required: bool = Field(..., description="Whether GPU is required")


class EmbeddingModelInfo(BaseModel):
    """Embedding model information."""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    dimensions: int = Field(..., description="Embedding dimensions")
    gpu_required: bool = Field(..., description="Whether GPU is required")


class UploadResponse(BaseModel):
    """Response model for file upload."""
    message: str = Field(..., description="Success message")
    filename: str = Field(..., description="Uploaded filename")
    size: int = Field(..., description="File size in bytes")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[str] = Field(..., description="List of document filenames")


class DeleteResponse(BaseModel):
    """Response model for delete operation."""
    message: str = Field(..., description="Success message")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    vector_store: str = Field(..., description="Vector store type")
    ollama_url: str = Field(..., description="Ollama service URL")
    documents_count: int = Field(..., description="Number of uploaded documents")


class ModelsListResponse(BaseModel):
    """Response model for models list."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    count: int = Field(..., description="Total count of models")


class RecommendedModelsResponse(BaseModel):
    """Response model for recommended models."""
    recommended_models: List[RecommendedModel] = Field(..., description="Recommended LLM models")
    embedding_models: List[EmbeddingModelInfo] = Field(..., description="Recommended embedding models")
