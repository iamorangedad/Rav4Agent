from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import httpx
import re
import time
import threading
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
import shutil


class AppConfig:
    """Application configuration from environment variables"""
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://10.0.0.55:11434")
        self.default_model_name = os.getenv("MODEL_NAME", "llama3.2")
        self.default_embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.use_chroma = os.getenv("USE_CHROMA", "false").lower() == "true"
        self.chroma_host = os.getenv("CHROMA_HOST", "chroma")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))


config = AppConfig()

app = FastAPI(
    title="Document Chat API", description="Document Chat System with LlamaIndex"
)

# CORS 配置：从环境变量读取允许的域名，默认为安全值
cors_origins = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

Settings.llm = Ollama(model=config.default_model_name, base_url=config.ollama_base_url)
# Try different parameter names for OllamaEmbedding
try:
    Settings.embed_model = OllamaEmbedding(
        model_name=config.default_embedding_model, base_url=config.ollama_base_url
    )
except TypeError:
    Settings.embed_model = OllamaEmbedding(
        model=config.default_embedding_model, base_url=config.ollama_base_url
    )


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


class ModelInfo(BaseModel):
    name: str
    size: str
    parameter_size: str
    quantization: str
    requires_gpu: bool


conversation_history = {}
conversation_timestamps = {}  # 记录会话最后访问时间


def validate_filename(filename: str) -> bool:
    """验证文件名是否安全，防止路径遍历攻击"""
    if not filename:
        return False
    # 检查非法字符和路径遍历尝试
    if re.search(r'[\\/:*?"<>|]', filename):
        return False
    # 检查是否包含 .. 路径遍历
    if '..' in filename:
        return False
    # 只允许字母数字、中文、空格、点和下划线
    if not re.match(r'^[\w\-. \u4e00-\u9fa5]+$', filename):
        return False
    return True


def cleanup_expired_conversations():
    """清理超过24小时未访问的会话"""
    current_time = time.time()
    expired = []
    for conv_id, last_access in list(conversation_timestamps.items()):
        if current_time - last_access > 86400:  # 24小时
            expired.append(conv_id)
    
    for conv_id in expired:
        if conv_id in conversation_history:
            del conversation_history[conv_id]
        if conv_id in conversation_timestamps:
            del conversation_timestamps[conv_id]
    
    if expired:
        print(f"Cleaned up {len(expired)} expired conversations")


def start_cleanup_thread():
    """启动定期清理线程"""
    def cleanup_loop():
        while True:
            time.sleep(3600)  # 每小时检查一次
            cleanup_expired_conversations()
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    print("Started conversation cleanup thread")


def get_vector_store(persist_dir: str):
    if config.use_chroma:
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore

            chroma_client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
            chroma_collection = chroma_client.get_or_create_collection("documents")
            return ChromaVectorStore(chroma_collection=chroma_collection)
        except Exception as e:
            print(f"Chroma connection failed, using in-memory: {e}")
            return SimpleVectorStore()
    else:
        return SimpleVectorStore()


def get_docstore():
    if config.use_chroma:
        return SimpleDocumentStore()
    return SimpleDocumentStore()


def get_index_store():
    if config.use_chroma:
        return SimpleIndexStore()
    return SimpleIndexStore()


def get_ollama_models():
    """Fetch available models from Ollama"""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{config.ollama_base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    name = model.get("name", "")
                    size = model.get("size", 0)
                    size_str = format_size(size)
                    model_info = {
                        "name": name,
                        "size": size_str,
                        "parameter_size": get_param_size(name),
                        "quantization": get_quantization(name),
                        "requires_gpu": requires_gpu(name),
                    }
                    models.append(model_info)
                return models
            return []
    except Exception as e:
        print(f"Failed to fetch models from Ollama: {e}")
        return get_default_models()


def format_size(size_bytes):
    """Format byte size to human readable"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def get_param_size(model_name):
    """Extract parameter size from model name"""
    if "70b" in model_name or "70B" in model_name:
        return "70B"
    elif "34b" in model_name or "34B" in model_name:
        return "34B"
    elif "13b" in model_name or "13B" in model_name:
        return "13B"
    elif "8b" in model_name or "8B" in model_name:
        return "8B"
    elif "7b" in model_name or "7B" in model_name:
        return "7B"
    elif "3b" in model_name or "3B" in model_name:
        return "3B"
    elif "1b" in model_name or "1B" in model_name:
        return "1B"
    else:
        return "Unknown"


def get_quantization(model_name):
    """Extract quantization from model name"""
    if "-q4_0" in model_name:
        return "Q4_0"
    elif "-q4_1" in model_name:
        return "Q4_1"
    elif "-q5_0" in model_name:
        return "Q5_0"
    elif "-q5_1" in model_name:
        return "Q5_1"
    elif "-q8_0" in model_name:
        return "Q8_0"
    elif "-f16" in model_name:
        return "F16"
    elif "-raw" in model_name:
        return "Raw"
    else:
        return "Default"


def requires_gpu(model_name):
    """Check if model typically requires GPU"""
    large_models = ["70b", "70B", "34b", "34B", "13b", "13B"]
    return any(param in model_name for param in large_models)


def get_default_models():
    """Return default model list if Ollama is not available"""
    return [
        {
            "name": config.default_model_name,
            "size": "~2GB",
            "parameter_size": "3B",
            "quantization": "Default",
            "requires_gpu": False,
        },
        {
            "name": config.default_embedding_model,
            "size": "~150MB",
            "parameter_size": "137M",
            "quantization": "Default",
            "requires_gpu": False,
        },
        {
            "name": "llama3.1:8b",
            "size": "~4.7GB",
            "parameter_size": "8B",
            "quantization": "Default",
            "requires_gpu": False,
        },
        {
            "name": "mistral",
            "size": "~4.1GB",
            "parameter_size": "7B",
            "quantization": "Default",
            "requires_gpu": False,
        },
        {
            "name": "qwen2:7b",
            "size": "~4.4GB",
            "parameter_size": "7B",
            "quantization": "Default",
            "requires_gpu": False,
        },
    ]


@app.on_event("startup")
async def startup_event():
    if config.use_chroma:
        print(f"Using Chroma vector database at {config.chroma_host}:{config.chroma_port}")
    else:
        print("Using in-memory vector database")
    print(f"Ollama base URL: {config.ollama_base_url}")
    # 启动会话清理线程
    start_cleanup_thread()


@app.get("/models", summary="Get available Ollama models")
async def get_models():
    """Get list of available models from Ollama"""
    models = get_ollama_models()
    return {"models": models, "count": len(models)}


@app.get("/models/recommended", summary="Get recommended models for document chat")
async def get_recommended_models():
    """Get recommended models for document chat"""
    return {
        "recommended_models": [
            {
                "name": config.default_model_name,
                "description": "Default model configured for this deployment",
                "context_window": "128K",
                "gpu_required": False,
            },
            {
                "name": "llama3.1:8b",
                "description": "Good balance of performance and resource usage",
                "context_window": "128K",
                "gpu_required": False,
            },
            {
                "name": "mistral",
                "description": "Excellent reasoning capabilities",
                "context_window": "32K",
                "gpu_required": False,
            },
            {
                "name": "qwen2:7b",
                "description": "Strong multilingual support",
                "context_window": "128K",
                "gpu_required": False,
            },
        ],
        "embedding_models": [
            {
                "name": config.default_embedding_model,
                "description": "Default embedding model configured for this deployment",
                "dimensions": 768,
                "gpu_required": False,
            },
            {
                "name": "mxbai-embed-large",
                "description": "Large embeddings for better retrieval",
                "dimensions": 1024,
                "gpu_required": False,
            },
        ],
    }


@app.post("/upload", summary="Upload document")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": f"File {file.filename} uploaded successfully",
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", summary="Get uploaded documents list")
async def get_documents():
    try:
        files = os.listdir(UPLOAD_DIR)
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="Send chat message")
async def chat(request: ChatRequest):
    try:
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())

        documents_dir = UPLOAD_DIR
        if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
            return ChatResponse(
                response="Please upload documents first, then I can answer your questions.",
                conversation_id=request.conversation_id,
            )

        if request.conversation_id not in conversation_history:
            try:
                model_name = request.model or config.default_model_name
                embed_model_name = request.embedding_model or config.default_embedding_model

                Settings.llm = Ollama(model=model_name, base_url=config.ollama_base_url)
                # Try different parameter names for OllamaEmbedding
                try:
                    Settings.embed_model = OllamaEmbedding(
                        model_name=embed_model_name, base_url=config.ollama_base_url
                    )
                except TypeError:
                    Settings.embed_model = OllamaEmbedding(
                        model=embed_model_name, base_url=config.ollama_base_url
                    )

                vector_store = get_vector_store(CHROMA_DIR)
                docstore = get_docstore()
                index_store = get_index_store()

                storage_context = {
                    "vector_store": vector_store,
                    "docstore": docstore,
                    "index_store": index_store,
                }

                documents = SimpleDirectoryReader(documents_dir).load_data()
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True,
                )
                conversation_history[request.conversation_id] = index.as_query_engine()
            except Exception as e:
                return ChatResponse(
                    response=f"Error loading documents: {str(e)}",
                    conversation_id=request.conversation_id,
                )

        query_engine = conversation_history[request.conversation_id]
        response = query_engine.query(request.message)
        
        # 更新会话访问时间
        conversation_timestamps[request.conversation_id] = time.time()

        return ChatResponse(
            response=str(response), conversation_id=request.conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}", summary="Delete document")
async def delete_document(filename: str):
    try:
        # 验证文件名安全性
        if not validate_filename(filename):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(UPLOAD_DIR, filename)
        # 确保解析后的路径仍在 UPLOAD_DIR 内
        real_path = os.path.realpath(file_path)
        real_upload_dir = os.path.realpath(UPLOAD_DIR)
        if not real_path.startswith(real_upload_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            conversation_history.clear()
            conversation_timestamps.clear()
            return {"message": f"File {filename} deleted"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": "chroma" if config.use_chroma else "in-memory",
        "ollama_url": config.ollama_base_url,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
