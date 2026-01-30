#!/bin/bash
# test.sh - Unified test runner using uv

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Smart Document Assistant Test Runner${NC}"
echo ""

# Change to backend directory
cd "$(dirname "$0")/backend"

# Function to check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv is not installed${NC}"
        echo "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Function to setup test environment
setup_env() {
    echo -e "${YELLOW}📦 Setting up test environment...${NC}"
    
    # Create test environment if it doesn't exist
    if [ ! -d ".venv-test" ]; then
        echo "Creating test virtual environment..."
        uv venv .venv-test
    fi
    
    # Install dependencies using uv (ultra-fast)
    echo "Installing dependencies with uv..."
    uv pip install -e ".[test]" --python .venv-test
}

# Function to run unit tests
run_unit_tests() {
    echo -e "${YELLOW}🧪 Running unit tests...${NC}"
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "not slow"
}

# Function to run all tests including slow ones
run_all_tests() {
    echo -e "${YELLOW}🧪 Running all tests...${NC}"
    uv run --python .venv-test pytest tests/unit -v --tb=short
}

# Function to run tests with coverage
run_coverage() {
    echo -e "${YELLOW}📊 Running tests with coverage...${NC}"
    uv run --python .venv-test pytest tests/unit -v --tb=short \
        --cov=app --cov-report=term-missing --cov-report=html
    
    echo ""
    echo -e "${GREEN}✅ Coverage report generated in htmlcov/index.html${NC}"
}

# Function to run specific test category
run_category() {
    local category=$1
    echo -e "${YELLOW}🧪 Running ${category} tests...${NC}"
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "${category}"
}

# Function to run tests matching pattern
run_pattern() {
    local pattern=$1
    echo -e "${YELLOW}🧪 Running tests matching: ${pattern}${NC}"
    uv run --python .venv-test pytest tests/unit -v --tb=short -k "${pattern}"
}

# Main command handler
case "${1:-}" in
    setup)
        check_uv
        setup_env
        echo -e "${GREEN}✅ Test environment ready!${NC}"
        ;;
    unit|"")
        check_uv
        setup_env
        run_unit_tests
        ;;
    all)
        check_uv
        setup_env
        run_all_tests
        ;;
    coverage)
        check_uv
        setup_env
        run_coverage
        ;;
    document)
        check_uv
        setup_env
        run_category "document"
        ;;
    vector)
        check_uv
        setup_env
        run_category "vector"
        ;;
    embedding)
        check_uv
        setup_env
        run_category "embedding"
        ;;
    storage)
        check_uv
        setup_env
        run_category "storage"
        ;;
    retrieval)
        check_uv
        setup_env
        run_category "retrieval"
        ;;
    prompt)
        check_uv
        setup_env
        run_category "prompt"
        ;;
    pattern)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please provide a pattern${NC}"
            echo "Usage: $0 pattern <test_pattern>"
            exit 1
        fi
        check_uv
        setup_env
        run_pattern "$2"
        ;;
    clean)
        echo -e "${YELLOW}🧹 Cleaning test environment...${NC}"
        rm -rf .venv-test
        rm -rf htmlcov
        rm -rf .pytest_cache
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        echo -e "${GREEN}✅ Cleaned!${NC}"
        ;;
    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup           Setup test environment"
        echo "  unit            Run unit tests (default, excludes slow tests)"
        echo "  all             Run all tests including slow tests"
        echo "  coverage        Run tests with coverage report"
        echo "  document        Run document parsing tests"
        echo "  vector          Run vector operation tests"
        echo "  embedding       Run embedding tests"
        echo "  storage         Run storage tests"
        echo "  retrieval       Run retrieval tests"
        echo "  prompt          Run prompt generation tests"
        echo "  pattern <p>     Run tests matching pattern"
        echo "  clean           Clean test environment and cache"
        echo "  help            Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                     # Run unit tests"
        echo "  $0 coverage            # Run with coverage"
        echo "  $0 pattern split       # Run tests with 'split' in name"
        echo "  $0 document            # Run document-related tests"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"
