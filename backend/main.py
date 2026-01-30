from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import httpx
import re
import time
import threading
import logging
import traceback
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

# Mount static files for frontend
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    """Serve the main frontend page"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # Fallback: return API info if frontend not built yet
    return {"message": "Doc-Chat API", "docs": "/docs", "health": "/health"}

def check_ollama_model_exists(model_name: str, base_url: str) -> bool:
    """Check if a model exists in Ollama"""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(model_name in m or m in model_name for m in models)
    except Exception as e:
        logger.warning(f"[ModelCheck] Failed to check model {model_name}: {e}")
    return False


def create_ollama_embedding(model_name: str, base_url: str):
    """Create OllamaEmbedding with automatic parameter compatibility handling"""
    # 先检查模型是否存在
    if not check_ollama_model_exists(model_name, base_url):
        error_msg = (
            f'Model "{model_name}" not found in Ollama. '
            f'Please run: ollama pull {model_name}'
        )
        logger.error(f"[Embedding] {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    try:
        return OllamaEmbedding(model_name=model_name, base_url=base_url)
    except TypeError:
        return OllamaEmbedding(model=model_name, base_url=base_url)


Settings.llm = Ollama(model=config.default_model_name, base_url=config.ollama_base_url)
Settings.embed_model = create_ollama_embedding(
    config.default_embedding_model, config.ollama_base_url
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
    logger.info("=" * 50)
    logger.info("[Startup] Doc-Chat API starting...")
    logger.info(f"[Startup] Ollama base URL: {config.ollama_base_url}")
    logger.info(f"[Startup] Default model: {config.default_model_name}")
    logger.info(f"[Startup] Embedding model: {config.default_embedding_model}")
    logger.info(f"[Startup] Vector store: {'chroma' if config.use_chroma else 'in-memory'}")
    if config.use_chroma:
        logger.info(f"[Startup] Chroma host: {config.chroma_host}:{config.chroma_port}")
    logger.info(f"[Startup] Upload directory: {UPLOAD_DIR}")
    logger.info(f"[Startup] Static directory: {STATIC_DIR}")
    logger.info("=" * 50)
    
    # 测试 Ollama 连接并检查必需模型
    try:
        models = get_ollama_models()
        logger.info(f"[Startup] Successfully connected to Ollama, found {len(models)} models")
        
        # 检查默认模型是否存在
        default_model_exists = check_ollama_model_exists(
            config.default_model_name, config.ollama_base_url
        )
        if not default_model_exists:
            logger.warning(
                f"[Startup] Default model '{config.default_model_name}' NOT FOUND! "
                f"Run: ollama pull {config.default_model_name}"
            )
        else:
            logger.info(f"[Startup] Default model '{config.default_model_name}' is ready")
        
        # 检查嵌入模型是否存在
        embed_model_exists = check_ollama_model_exists(
            config.default_embedding_model, config.ollama_base_url
        )
        if not embed_model_exists:
            logger.warning(
                f"[Startup] Embedding model '{config.default_embedding_model}' NOT FOUND! "
                f"Run: ollama pull {config.default_embedding_model}"
            )
        else:
            logger.info(f"[Startup] Embedding model '{config.default_embedding_model}' is ready")
            
    except Exception as e:
        logger.warning(f"[Startup] Could not connect to Ollama: {e}")
    
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
    logger.info(f"[Upload] Uploading file: {file.filename}")
    try:
        # 验证文件名
        if not validate_filename(file.filename):
            logger.warning(f"[Upload] Invalid filename rejected: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file_size = 0
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            file_size = os.path.getsize(file_path)

        logger.info(f"[Upload] File saved: {file.filename} ({file_size} bytes)")
        return {
            "message": f"File {file.filename} uploaded successfully",
            "filename": file.filename,
            "size": file_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] Error uploading {file.filename}: {str(e)}")
        logger.error(f"[Upload] Stack trace: {traceback.format_exc()}")
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
    start_time = time.time()
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    logger.info(f"[Chat] New request - conv_id: {conv_id}, model: {request.model or config.default_model_name}")
    
    try:
        documents_dir = UPLOAD_DIR
        if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
            logger.warning(f"[Chat] No documents uploaded for conv_id: {conv_id}")
            return ChatResponse(
                response="Please upload documents first, then I can answer your questions.",
                conversation_id=conv_id,
            )

        # 检查是否需要创建新的索引（首次请求或模型变更）
        if conv_id not in conversation_history:
            logger.info(f"[Chat] Creating new index for conv_id: {conv_id}")
            index_start = time.time()
            
            try:
                model_name = request.model or config.default_model_name
                embed_model_name = request.embedding_model or config.default_embedding_model
                
                logger.info(f"[Chat] Using model: {model_name}, embedding: {embed_model_name}")
                logger.info(f"[Chat] Ollama URL: {config.ollama_base_url}")

                # 设置 LLM
                llm_start = time.time()
                Settings.llm = Ollama(model=model_name, base_url=config.ollama_base_url)
                logger.info(f"[Chat] LLM initialized in {time.time() - llm_start:.2f}s")
                
                # 设置 Embedding
                embed_start = time.time()
                Settings.embed_model = create_ollama_embedding(
                    embed_model_name, config.ollama_base_url
                )
                logger.info(f"[Chat] Embedding model initialized in {time.time() - embed_start:.2f}s")

                # 获取存储组件
                vector_store = get_vector_store(CHROMA_DIR)
                docstore = get_docstore()
                index_store = get_index_store()
                logger.info(f"[Chat] Storage components ready")

                # 加载文档
                load_start = time.time()
                documents = SimpleDirectoryReader(documents_dir).load_data()
                logger.info(f"[Chat] Loaded {len(documents)} documents in {time.time() - load_start:.2f}s")

                # 创建索引（这是最耗时的步骤）
                index_create_start = time.time()
                logger.info(f"[Chat] Starting index creation...")
                index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True,
                )
                index_time = time.time() - index_create_start
                logger.info(f"[Chat] Index created in {index_time:.2f}s")

                # 创建查询引擎
                query_engine_start = time.time()
                conversation_history[conv_id] = index.as_query_engine()
                logger.info(f"[Chat] Query engine created in {time.time() - query_engine_start:.2f}s")
                
                total_index_time = time.time() - index_start
                logger.info(f"[Chat] Total index setup time: {total_index_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Error loading documents: {str(e)}"
                logger.error(f"[Chat] {error_msg}")
                logger.error(f"[Chat] Stack trace: {traceback.format_exc()}")
                return ChatResponse(
                    response=error_msg,
                    conversation_id=conv_id,
                )

        # 执行查询
        query_start = time.time()
        logger.info(f"[Chat] Executing query: {request.message[:50]}...")
        
        query_engine = conversation_history[conv_id]
        response = query_engine.query(request.message)
        query_time = time.time() - query_start
        
        logger.info(f"[Chat] Query completed in {query_time:.2f}s")
        
        # 更新会话访问时间
        conversation_timestamps[conv_id] = time.time()
        
        total_time = time.time() - start_time
        logger.info(f"[Chat] Total request time: {total_time:.2f}s - conv_id: {conv_id}")

        return ChatResponse(
            response=str(response), conversation_id=conv_id
        )
    except Exception as e:
        error_msg = f"Chat error: {str(e)}"
        logger.error(f"[Chat] {error_msg}")
        logger.error(f"[Chat] Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


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
    logger.debug("[Health] Health check requested")
    return {
        "status": "healthy",
        "vector_store": "chroma" if config.use_chroma else "in-memory",
        "ollama_url": config.ollama_base_url,
        "documents_count": len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
