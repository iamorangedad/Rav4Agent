"""Application configuration management."""
import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://10.0.0.55:11434", alias="OLLAMA_BASE_URL")
    default_model_name: str = Field(default="llama3.2", alias="MODEL_NAME")
    default_embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="simple", alias="VECTOR_STORE_TYPE")
    
    # ChromaDB Configuration
    use_chroma: bool = Field(default=False, alias="USE_CHROMA")
    chroma_host: str = Field(default="chroma", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="documents", alias="CHROMA_COLLECTION")
    
    # File Storage Configuration
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    chroma_dir: str = Field(default="chroma_db", alias="CHROMA_DIR")
    static_dir: str = Field(default="static", alias="STATIC_DIR")
    
    # CORS Configuration
    allowed_origins: str = Field(default="", alias="ALLOWED_ORIGINS")
    
    # Application Configuration
    app_name: str = Field(default="Document Chat API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable."""
        if not self.allowed_origins:
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def effective_vector_store_type(self) -> str:
        """Get effective vector store type (handles legacy USE_CHROMA flag)."""
        if self.use_chroma and self.vector_store_type == "simple":
            return "chroma"
        return self.vector_store_type
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
