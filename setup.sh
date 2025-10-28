#!/bin/bash
# MangeTaMain - Automated Setup Script
# This script automates the installation and model generation process

set -e  # Exit on any error

echo "================================================================"
echo "MangeTaMain - Automated Setup Script"
echo "================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
echo "Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 9 ]); then
    print_error "Python 3.9 or higher required. Found: $python_version"
    exit 1
fi
print_success "Python $python_version detected"
echo ""

# Check if data files exist
echo "Step 2: Checking data files..."
if [ ! -f "data/RAW_recipes.csv" ]; then
    print_error "data/RAW_recipes.csv not found!"
    echo "  Please download from: https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions"
    exit 1
fi
if [ ! -f "data/RAW_interactions.csv" ]; then
    print_error "data/RAW_interactions.csv not found!"
    echo "  Please download from: https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions"
    exit 1
fi
print_success "Data files found"
echo ""

# Install UV if not present
echo "Step 3: Installing UV package manager..."
if ! command -v uv &> /dev/null; then
    print_warning "UV not found. Installing..."
    pip3 install uv
    print_success "UV installed"
else
    print_success "UV already installed"
fi
echo ""

# Install dependencies
echo "Step 4: Installing dependencies..."
print_warning "This may take a few minutes..."
uv pip install -e .
print_success "Core dependencies installed"
echo ""

# Ask about sentiment analysis
echo "Step 5: Optional - Sentiment Analysis"
echo "Do you want to install sentiment analysis dependencies?"
echo "This includes PyTorch and Transformers (~2GB download)"
read -p "Install sentiment analysis? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Installing sentiment analysis dependencies..."
    uv pip install -e ".[sentiment]"
    print_success "Sentiment analysis dependencies installed"
    INSTALL_SENTIMENT=true
else
    print_warning "Skipping sentiment analysis installation"
    INSTALL_SENTIMENT=false
fi
echo ""

# Generate clustering models
echo "Step 6: Generating clustering models (REQUIRED)..."
print_warning "This will take 5-15 minutes..."
python3 run_clustering_models.py
if [ $? -eq 0 ]; then
    print_success "Clustering models generated successfully"
else
    print_error "Failed to generate clustering models"
    exit 1
fi
echo ""

# Train sentiment model if requested
if [ "$INSTALL_SENTIMENT" = true ]; then
    echo "Step 7: Training sentiment analysis model (OPTIONAL)..."
    echo "This will take 30 minutes (GPU) to 3 hours (CPU)"
    read -p "Train sentiment model now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Training sentiment model (this will take a while)..."
        python3 train_sentiment_model.py
        if [ $? -eq 0 ]; then
            print_success "Sentiment model trained successfully"
        else
            print_warning "Sentiment model training failed or was interrupted"
            print_warning "You can train it later by running: python3 train_sentiment_model.py"
        fi
    else
        print_warning "Skipping sentiment model training"
        print_warning "You can train it later by running: python3 train_sentiment_model.py"
    fi
fi
echo ""

# Verify setup
echo "Step 8: Verifying setup..."
if [ -d "outputs/models" ]; then
    model_count=$(ls -1 outputs/models/*.pkl 2>/dev/null | wc -l)
    if [ "$model_count" -ge 2 ]; then
        print_success "Found $model_count clustering model files"
    else
        print_warning "Expected at least 2 model files, found $model_count"
    fi
else
    print_error "outputs/models directory not found"
    exit 1
fi
echo ""

# Final instructions
echo "================================================================"
echo "Setup Complete!"
echo "================================================================"
echo ""
print_success "MangeTaMain is ready to run!"
echo ""
echo "To start the application, run:"
echo "  ${GREEN}streamlit run streamlit_app_final.py${NC}"
echo ""
echo "The app will open automatically in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "First-time load takes 2-3 minutes for model initialization."
echo ""
echo "Sample User IDs to try:"
echo "  - 424680 (high activity)"
echo "  - 52282 (medium activity)"
echo "  - 742802 (medium activity)"
echo ""
echo "For detailed usage instructions, see SETUP_GUIDE.md"
echo "================================================================"
