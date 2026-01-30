#!/bin/bash
# run-real-env-tests.sh - Run integration tests against real environment

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Real Environment Integration Tests${NC}"
echo "=========================================="
echo ""

# Check if test file exists
if [ ! -f "OM0R028U.pdf" ]; then
    echo -e "${RED}❌ Test file OM0R028U.pdf not found!${NC}"
    exit 1
fi

echo -e "📄 Test file: OM0R028U.pdf ($(ls -lh OM0R028U.pdf | awk '{print $5}'))"
echo ""

# Check Ollama connectivity
echo -e "${YELLOW}🔍 Checking Ollama connectivity...${NC}"
TEST_OLLAMA_URL="${TEST_OLLAMA_URL:-http://10.0.0.55:11434}"

if curl -s "${TEST_OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is accessible at ${TEST_OLLAMA_URL}${NC}"
else
    echo -e "${RED}❌ Cannot connect to Ollama at ${TEST_OLLAMA_URL}${NC}"
    echo "   Please ensure Ollama is running and accessible"
    exit 1
fi

echo ""

# Change to backend directory
cd backend

# Export environment variables
export TEST_REAL_ENV=true
export TEST_OLLAMA_URL="${TEST_OLLAMA_URL}"
export TEST_TIMEOUT=300  # 5 minutes

echo -e "${YELLOW}🧪 Running tests...${NC}"
echo ""

# Run specific test or all
if [ "$1" == "setup" ]; then
    echo "Running environment setup tests only..."
    ~/.local/bin/uv run pytest tests/integration/test_real_environment.py::TestRealEnvironmentSetup -v -s --tb=short
elif [ "$1" == "embedding" ]; then
    echo "Running embedding tests only..."
    ~/.local/bin/uv run pytest tests/integration/test_real_environment.py::TestRealDocumentProcessing::test_real_embedding_generation -v -s --tb=short
elif [ "$1" == "full" ]; then
    echo "Running full pipeline test (may take 10+ minutes)..."
    ~/.local/bin/uv run pytest tests/integration/test_real_environment.py::TestRealDocumentProcessing::test_full_large_document_pipeline -v -s --tb=long
elif [ "$1" == "service" ]; then
    echo "Running indexing service test..."
    ~/.local/bin/uv run pytest tests/integration/test_real_environment.py::TestRealIndexingService -v -s --tb=long
else
    # Run all integration tests
    echo "Running all real environment tests..."
    echo "   (Use './run-real-env-tests.sh setup' for quick setup check)"
    echo "   (Use './run-real-env-tests.sh full' for complete pipeline test)"
    echo ""
    ~/.local/bin/uv run pytest tests/integration/test_real_environment.py -v -s --tb=short
fi

echo ""
echo -e "${GREEN}✅ Tests completed!${NC}"
