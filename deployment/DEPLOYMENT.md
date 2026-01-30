# Kubernetes Deployment Guide

This guide covers deploying the Smart Document Assistant on Kubernetes with the new modular architecture.

## Architecture Overview

The application now supports multiple vector store backends through a pluggable architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Namespace: doc-chat-system         │   │
│  │                                                     │   │
│  │  ┌──────────────┐         ┌──────────────┐         │   │
│  │  │  doc-chat    │────────▶│   chroma     │         │   │
│  │  │  (Backend)   │         │(Vector Store)│         │   │
│  │  │  Port: 8000  │         │  Port: 8000  │         │   │
│  │  └──────────────┘         └──────────────┘         │   │
│  │         │                       │                  │   │
│  │         ▼                       ▼                  │   │
│  │  ┌──────────────┐         ┌──────────────┐         │   │
│  │  │ uploads-pvc  │         │ chroma-pvc   │         │   │
│  │  │   (5Gi)      │         │   (10Gi)     │         │   │
│  │  └──────────────┘         └──────────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│              External Ollama Service                       │
│              (http://10.0.0.55:11434)                      │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Scenarios

### Scenario 1: Simple Mode (No Chroma)
Use this for testing or when persistence is not required.

```yaml
# ConfigMap overrides
VECTOR_STORE_TYPE: "simple"
USE_CHROMA: "false"
```

**Pros:**
- No ChromaDB pod required
- Faster startup
- Lower resource usage

**Cons:**
- Vector embeddings lost on pod restart
- In-memory only

### Scenario 2: Chroma Mode (Recommended for Production)
Use this for persistent vector storage.

```yaml
# ConfigMap overrides
VECTOR_STORE_TYPE: "chroma"
USE_CHROMA: "true"
CHROMA_HOST: "chroma"
CHROMA_PORT: "8000"
```

**Pros:**
- Persistent embeddings survive restarts
- Supports large document collections
- Better query performance

**Cons:**
- Requires additional ChromaDB pod
- Higher resource usage

## Quick Deployment

### 1. Build Docker Image

```bash
# Build the image
docker build -t doc-chat:latest .

# Tag for registry (optional)
docker tag doc-chat:latest your-registry/doc-chat:v1.0.0
docker push your-registry/doc-chat:v1.0.0
```

### 2. Update Deployment Configuration

Edit `deployment/deployment.yaml` to customize:

#### a. Ollama Configuration
```yaml
data:
  OLLAMA_BASE_URL: "http://your-ollama-host:11434"
  MODEL_NAME: "llama3.2"  # or your preferred model
  EMBEDDING_MODEL: "nomic-embed-text"
```

#### b. Vector Store Selection
```yaml
data:
  # For Chroma (persistent)
  VECTOR_STORE_TYPE: "chroma"
  USE_CHROMA: "true"
  
  # For Simple mode (in-memory)
  # VECTOR_STORE_TYPE: "simple"
  # USE_CHROMA: "false"
```

#### c. CORS Settings
```yaml
data:
  # For development (allows all)
  ALLOWED_ORIGINS: ""
  
  # For production (specific origins)
  ALLOWED_ORIGINS: "https://yourdomain.com,https://app.yourdomain.com"
```

#### d. Resource Limits
```yaml
resources:
  requests:
    memory: "2Gi"    # Minimum recommended
    cpu: "1000m"
  limits:
    memory: "4Gi"    # Adjust based on document count
    cpu: "2000m"
```

### 3. Apply Deployment

```bash
kubectl apply -f deployment/deployment.yaml
```

### 4. Verify Deployment

```bash
# Check all resources
kubectl get all -n doc-chat-system

# Check pods
kubectl get pods -n doc-chat-system

# Check logs
kubectl logs -n doc-chat-system -l app=doc-chat --tail=50
kubectl logs -n doc-chat-system -l app=chroma --tail=20

# Check PVCs
kubectl get pvc -n doc-chat-system

# Test service
kubectl port-forward -n doc-chat-system svc/doc-chat 8080:80
curl http://localhost:8080/health
```

## Configuration Reference

### ConfigMap: doc-chat-config

| Key | Description | Default |
|-----|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama service URL | `http://10.0.0.55:11434` |
| `MODEL_NAME` | Default LLM model | `llama3.2` |
| `EMBEDDING_MODEL` | Default embedding model | `nomic-embed-text` |
| `VECTOR_STORE_TYPE` | Vector store type (`simple` or `chroma`) | `chroma` |
| `USE_CHROMA` | Legacy flag (overrides VECTOR_STORE_TYPE if `true`) | `true` |
| `CHROMA_HOST` | ChromaDB service hostname | `chroma` |
| `CHROMA_PORT` | ChromaDB service port | `8000` |
| `CHROMA_COLLECTION` | Chroma collection name | `documents` |
| `UPLOAD_DIR` | Document storage path | `/app/uploads` |
| `CHROMA_DIR` | Chroma persistence path | `/app/chroma_db` |
| `STATIC_DIR` | Static files path | `/app/static` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `` (empty = all) |
| `DEBUG` | Debug mode | `false` |

### Persistent Volume Claims

| PVC Name | Purpose | Size | Access Mode |
|----------|---------|------|-------------|
| `uploads-pvc` | Document storage | 5Gi | ReadWriteOnce |
| `chroma-pvc` | Vector embeddings | 10Gi | ReadWriteOnce |

## Advanced Configuration

### Using External ChromaDB

If you have an existing ChromaDB instance:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
data:
  VECTOR_STORE_TYPE: "chroma"
  CHROMA_HOST: "external-chroma.example.com"  # External host
  CHROMA_PORT: "8000"
```

Remove the Chroma deployment from the YAML if using external service.

### Node Affinity

Current deployment uses soft affinity (preferred but not required):

```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - "ubuntu"  # Change to your preferred node
```

To use hard affinity (required):

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - "ubuntu"
```

### Resource Tuning

**Small Deployment (Testing):**
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

**Large Deployment (Production):**
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## Troubleshooting

### Pod Status Issues

```bash
# Check pod status
kubectl get pods -n doc-chat-system

# Describe pod for events
kubectl describe pod -n doc-chat-system -l app=doc-chat

# View logs
kubectl logs -n doc-chat-system -l app=doc-chat
```

### Common Issues

#### 1. ImagePullBackOff
```bash
# Check if image exists locally (for IfNotPresent)
docker images | grep doc-chat

# Or push to registry and update image reference
kubectl set image deployment/doc-chat doc-chat=your-registry/doc-chat:v1.0.0 -n doc-chat-system
```

#### 2. CrashLoopBackOff
```bash
# Check logs for errors
kubectl logs -n doc-chat-system -l app=doc-chat --previous

# Common causes:
# - Ollama not accessible
# - Missing required models
# - Insufficient resources
```

#### 3. Chroma Connection Failed
The application will fallback to simple mode automatically. Check logs:
```bash
kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "chroma\|vector"
```

#### 4. PVC Pending
```bash
# Check storage class
kubectl get storageclass

# Check PVC events
kubectl describe pvc uploads-pvc -n doc-chat-system
```

### Health Checks

```bash
# Port forward to test locally
kubectl port-forward -n doc-chat-system svc/doc-chat 8080:80

# Test health endpoint
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "vector_store": "chroma",
#   "ollama_url": "http://10.0.0.55:11434",
#   "documents_count": 0
# }

# Test model list
curl http://localhost:8080/models
```

## Upgrading from Old Version

### Changes in New Architecture

1. **Environment Variables:**
   - Added `VECTOR_STORE_TYPE` (replaces `USE_CHROMA` as primary config)
   - Added `CHROMA_COLLECTION`, `APP_NAME`, `APP_VERSION`
   - Added storage path configs: `UPLOAD_DIR`, `CHROMA_DIR`, `STATIC_DIR`

2. **File Structure:**
   - New modular structure in `/app/` directory
   - `PYTHONPATH` set to `/app`

3. **Dockerfile Changes:**
   - Now copies `main.py` and `app/` separately
   - Production-grade CMD without `--reload`

### Migration Steps

1. Update ConfigMap with new variables
2. Rebuild Docker image with new Dockerfile
3. Redeploy with updated YAML
4. PVCs will retain existing data

## Scaling Considerations

### Horizontal Scaling

**Current Limitations:**
- Simple mode: Cannot scale horizontally (in-memory state)
- Chroma mode: Can scale backend, but ChromaDB is single-instance

**Recommended Architecture for High Availability:**
```
┌──────────────────────────────────────────┐
│           Load Balancer                  │
└──────────────┬───────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────┐          ┌─────────┐
│doc-chat-1│          │doc-chat-2│
└────┬────┘          └────┬────┘
     │                    │
     └────────┬───────────┘
              ▼
       ┌────────────┐
       │   Chroma   │
       │   (HA)     │
       └────────────┘
```

### Vertical Scaling

Adjust resource limits based on:
- Number of documents
- Document sizes
- Concurrent users
- Model sizes

**Memory Calculation:**
- Base: 1GB
- Per 1000 documents: ~500MB
- ChromaDB: 512MB base + vector storage

## Security Best Practices

### 1. Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: doc-chat-network-policy
  namespace: doc-chat-system
spec:
  podSelector:
    matchLabels:
      app: doc-chat
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: chroma
  - to:
    - ipBlock:
        cidr: 10.0.0.55/32  # Ollama host
```

### 2. Secrets Management
Use Kubernetes Secrets for sensitive data:
```bash
kubectl create secret generic doc-chat-secrets \
  --from-literal=ollama-api-key=your-key \
  -n doc-chat-system
```

### 3. RBAC
Create service accounts with minimal permissions.

## Monitoring

### Prometheus Metrics (Future Enhancement)

Add to deployment for metrics endpoint:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

### Log Aggregation

Configure logging format:
```yaml
env:
- name: LOG_LEVEL
  value: "INFO"  # DEBUG, INFO, WARNING, ERROR
- name: LOG_FORMAT
  value: "json"  # or "text"
```

## Cleanup

```bash
# Remove all resources
kubectl delete -f deployment/deployment.yaml

# Remove PVCs (WARNING: Data will be lost!)
kubectl delete pvc uploads-pvc chroma-pvc -n doc-chat-system

# Remove namespace
kubectl delete namespace doc-chat-system
```

---

For more information, see [README.md](../README.md) and [ARCHITECTURE.md](../backend/ARCHITECTURE.md).
