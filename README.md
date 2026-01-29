# Smart Document Assistant - Deployment Guide

## Overview
A document chat system built with LlamaIndex + Ollama + Chroma, featuring a beautiful WebUI interface.

## Features
- Document upload (PDF, TXT, DOC, DOCX, MD)
- Intelligent Q&A based on document content
- Model selection via WebUI
- GPU-accelerated inference via external Ollama
- Persistent vector storage with Chroma

## Prerequisites
- Kubernetes cluster
- Configured Ingress Controller (e.g., Nginx Ingress)
- External Ollama service (http://10.0.0.55:11434)

## Deployment Steps

### 1. Build the Docker image
```bash
docker build -t doc-chat:latest .
```

### 2. Push to registry (if needed)
```bash
docker tag doc-chat:latest your-registry/doc-chat:latest
docker push your-registry/doc-chat:latest
```

### 3. Apply K8s configuration
```bash
kubectl apply -f deployment/deployment.yaml
```

### 4. Check deployment status
```bash
kubectl get pods -n doc-chat-system -l app=doc-chat
kubectl get pods -n doc-chat-system -l app=chroma
kubectl get svc -n doc-chat-system doc-chat chroma
kubectl get ingress -n doc-chat-system doc-chat
```

## API Documentation

### Upload Document
```
POST /upload
Content-Type: multipart/form-data

Parameter: file (supports multiple files)
```

### Get Document List
```
GET /documents
```

### Send Chat Message
```
POST /chat
Content-Type: application/json

{
  "message": "Your question",
  "conversation_id": "Session ID (optional)",
  "model": "llama3.2",
  "embedding_model": "nomic-embed-text"
}
```

### Get Available Models
```
GET /models
```

### Get Recommended Models
```
GET /models/recommended
```

### Delete Document
```
DELETE /documents/{filename}
```

### Health Check
```
GET /health
```

## Environment Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| OLLAMA_BASE_URL | Ollama service URL | http://10.0.0.55:11434 |
| MODEL_NAME | LLM model name | llama3.2 |
| EMBEDDING_MODEL | Embedding model name | nomic-embed-text |
| USE_CHROMA | Enable Chroma vector store | true |
| CHROMA_HOST | Chroma server host | chroma |
| CHROMA_PORT | Chroma server port | 8000 |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Web Browser                   │
│                  (React HTML)                   │
└─────────────────────┬───────────────────────────┘
                      │ HTTP API
                      ▼
┌─────────────────────────────────────────────────┐
│              Doc-Chat Backend                   │
│  ┌───────────────────────────────────────────┐  │
│  │  FastAPI Server (Port 8000)               │  │
│  │  - Document Upload & Management            │  │
│  │  - Chat API                               │  │
│  │  - Model Selection                        │  │
│  └───────────────────────────────────────────┘  │
│                      │                          │
│              ┌───────┴───────┐                  │
│              ▼               ▼                  │
│  ┌─────────────────┐ ┌─────────────────┐       │
│  │   External      │ │     Chroma      │       │
│  │   Ollama        │ │  (Vector Store) │       │
│  │  10.0.0.55:11434│ │  Port: 8000     │       │
│  └─────────────────┘ └─────────────────┘       │
└─────────────────────────────────────────────────┘
```

## Notes
1. Ollama service is external - ensure http://10.0.0.55:11434 is accessible
2. Document embeddings are persisted to Chroma PVC (10GB)
3. Uploaded files are stored in temporary volume
4. Chroma requires at least 512MB memory
5. Application requires at least 2GB memory
