#!/bin/bash
# deploy-and-test.sh - Build, deploy and test with large PDF

set -e

echo "🚀 Large Document Processing Test & Deploy"
echo "=========================================="
echo ""

# Check if OM0R028U.pdf exists
if [ ! -f "OM0R028U.pdf" ]; then
    echo "❌ Test file OM0R028U.pdf not found!"
    exit 1
fi

echo "📄 Test file: OM0R028U.pdf ($(ls -lh OM0R028U.pdf | awk '{print $5}'))"
echo ""

# Step 1: Build new Docker image
echo "🔨 Step 1: Building Docker image..."
docker build -t doc-chat:v2.0-large-file-fix .

# Step 2: Save image for transfer (if needed on remote K8s)
echo "💾 Step 2: Saving image..."
docker save doc-chat:v2.0-large-file-fix | gzip > doc-chat-v2.0.tar.gz

echo ""
echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "1. Transfer to K8s node: scp doc-chat-v2.0.tar.gz <node>:/tmp/"
echo "2. Load on node: docker load < /tmp/doc-chat-v2.0.tar.gz"
echo "3. Update deployment: kubectl set image deployment/doc-chat doc-chat=doc-chat:v2.0-large-file-fix"
echo ""
echo "Or for local Docker:"
echo "  docker run -p 8000:8000 -v \$(pwd)/uploads:/app/uploads doc-chat:v2.0-large-file-fix"
