#!/bin/bash
# Build and Deploy Script for Smart Document Assistant
# Usage: ./deploy.sh [build|deploy|all|clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="doc-chat-system"
IMAGE_NAME="doc-chat"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    cd "$PROJECT_ROOT"
    
    if [ -n "$REGISTRY" ]; then
        IMAGE_FULL="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    else
        IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"
    fi
    
    print_status "Building image: ${IMAGE_FULL}"
    docker build -t "${IMAGE_FULL}" .
    
    if [ -n "$REGISTRY" ]; then
        print_status "Pushing image to registry..."
        docker push "${IMAGE_FULL}"
    fi
    
    print_status "Build completed successfully!"
}

# Deploy to Kubernetes
deploy_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    
    # Update image in deployment file if using registry
    if [ -n "$REGISTRY" ]; then
        IMAGE_FULL="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        print_status "Updating image reference to: ${IMAGE_FULL}"
        sed -i.bak "s|image: doc-chat:latest|image: ${IMAGE_FULL}|g" "${SCRIPT_DIR}/deployment.yaml"
        rm -f "${SCRIPT_DIR}/deployment.yaml.bak"
    fi
    
    # Apply deployment
    print_status "Applying Kubernetes manifests..."
    kubectl apply -f "${SCRIPT_DIR}/deployment.yaml"
    
    # Wait for deployment
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=120s deployment/doc-chat -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=120s deployment/chroma -n ${NAMESPACE} 2>/dev/null || print_warning "Chroma deployment not found or not ready"
    
    print_status "Deployment completed!"
    print_status "Check status with: kubectl get all -n ${NAMESPACE}"
}

# Check deployment status
check_status() {
    print_status "Checking deployment status..."
    
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n ${NAMESPACE}
    
    echo ""
    echo "=== Services ==="
    kubectl get svc -n ${NAMESPACE}
    
    echo ""
    echo "=== PVCs ==="
    kubectl get pvc -n ${NAMESPACE}
    
    echo ""
    echo "=== Recent Logs (doc-chat) ==="
    kubectl logs -n ${NAMESPACE} -l app=doc-chat --tail=20
}

# Clean up deployment
cleanup() {
    print_warning "This will delete all resources including PVCs (data will be lost)!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_status "Cleaning up deployment..."
        kubectl delete -f "${SCRIPT_DIR}/deployment.yaml"
        kubectl delete pvc uploads-pvc chroma-pvc -n ${NAMESPACE} 2>/dev/null || true
        print_status "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Port forward for local testing
port_forward() {
    print_status "Setting up port forwarding..."
    print_status "Access the app at: http://localhost:8080"
    kubectl port-forward -n ${NAMESPACE} svc/doc-chat 8080:80
}

# Show help
show_help() {
    echo "Smart Document Assistant - Build and Deploy Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build      Build Docker image"
    echo "  deploy     Deploy to Kubernetes"
    echo "  all        Build and deploy (default)"
    echo "  status     Check deployment status"
    echo "  forward    Port forward for local testing"
    echo "  clean      Clean up deployment and PVCs"
    echo "  help       Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  REGISTRY   Docker registry (e.g., docker.io/username)"
    echo "  IMAGE_TAG  Image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build locally"
    echo "  REGISTRY=docker.io/user $0 all        # Build and push to registry"
    echo "  $0 deploy                             # Deploy only"
    echo "  $0 status                             # Check status"
}

# Main
main() {
    COMMAND="${1:-all}"
    
    case "$COMMAND" in
        build)
            build_image
            ;;
        deploy)
            deploy_k8s
            ;;
        all)
            build_image
            deploy_k8s
            ;;
        status)
            check_status
            ;;
        forward)
            port_forward
            ;;
        clean)
            cleanup
            ;;
        help)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
