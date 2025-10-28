#!/bin/bash
# Setup script for CI/CD and local development

set -e

echo "=========================================="
echo "Recipe ML Project - CI/CD Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/5] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | sed 's/Python //')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

# Check if Python >= 3.9
if [ "$PYTHON_MAJOR" -gt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]); then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python $PYTHON_VERSION is too old. Need >= 3.9${NC}"
    exit 1
fi

# Install dependencies
echo -e "\n${YELLOW}[2/5] Installing dependencies...${NC}"
pip install -e ".[dev,test]"
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Setup Kaggle API (optional)
echo -e "\n${YELLOW}[3/5] Checking Kaggle API setup...${NC}"
if [ -f ~/.kaggle/kaggle.json ]; then
    echo -e "${GREEN}✓ Kaggle credentials found${NC}"
    chmod 600 ~/.kaggle/kaggle.json
else
    echo -e "${YELLOW}⚠ Kaggle credentials not found${NC}"
    echo "To download data automatically:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Create API token"
    echo "  3. Place kaggle.json in ~/.kaggle/"
    
    read -p "Do you want to setup Kaggle credentials now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p ~/.kaggle
        read -p "Enter your Kaggle username: " KAGGLE_USER
        read -p "Enter your Kaggle API key: " KAGGLE_KEY
        echo "{\"username\":\"$KAGGLE_USER\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
        echo -e "${GREEN}✓ Kaggle credentials saved${NC}"
    fi
fi

# Create necessary directories
echo -e "\n${YELLOW}[4/5] Creating directories...${NC}"
mkdir -p data/raw data/processed
mkdir -p outputs/models outputs/figures outputs/reports
mkdir -p outputs/sentiment_model
echo -e "${GREEN}✓ Directories created${NC}"

# Run tests to verify setup
echo -e "\n${YELLOW}[5/5] Running tests to verify setup...${NC}"
if pytest tests/ -v --tb=short -x; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "This is normal if you haven't downloaded data yet."
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download data: make download-data"
echo "  2. Run tests: make test"
echo "  3. Run pipeline: make pipeline"
echo ""
echo "For CI/CD setup:"
echo "  - See .github/CI_CD_GUIDE.md"
echo "  - Configure GitHub secrets for Kaggle API"
echo ""
echo "Documentation:"
echo "  - CI/CD Guide: .github/CI_CD_GUIDE.md"
echo "  - Tests Guide: tests/README.md"
echo ""
