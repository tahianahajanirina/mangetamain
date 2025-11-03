# MangeTaMain - Intelligent Recipe Discovery Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**MangeTaMain** is a production-ready machine learning application that transforms 230,000+ recipes and 1 million+ user reviews into personalized, actionable cooking recommendations using collaborative filtering, content-based recommendation, and deep learning.

## Overview

MangeTaMain is an end-to-end ML system that combines multiple recommendation algorithms, predictive models, and natural language processing to deliver an intelligent recipe discovery experience. Built with scikit-learn, PyTorch, Transformers, and Streamlit, it provides:

- **Collaborative Filtering**: SVD-based user recommendations analyzing millions of interactions
- **Content-Based Recommendations**: Recipe similarity using clustering (HDBSCAN) and feature engineering
- **Time Prediction**: Multi-model regression for accurate cooking time estimates
- **Nutrition Classification**: Multi-label binary classifiers for health tagging
- **Sentiment Analysis**: Transformer-based review analysis (DistilBERT/RoBERTa)
- **RAG Chatbot**: Google Gemini-powered conversational recipe assistant
- **Interactive UI**: Streamlit web application with real-time predictions

### Key Features

- **Personalized Suggestions**: Get recipe recommendations tailored to your taste
- **Smart Search**: Find recipes similar to ones you already love
- **Time Estimation**: Know how long a recipe will take before you start
- **Nutrition Information**: Understand the nutritional profile of any recipe
- **Recipe Categories**: Browse recipes organized by cooking time, complexity, and health profile
- **User Reviews Analysis**: See what other home cooks think about recipes

## Table of Contents

- [Quick Start](#quick-start)
- [For Users](#for-users)
  - [How to Use](#how-to-use-mangetamain)
  - [Understanding Results](#understanding-your-results)
  - [Tips & Best Practices](#tips-for-best-results)
- [For Developers](#for-developers)
  - [Project Structure](#project-structure)
  - [Technology Stack](#technology-stack)
  - [Development Setup](#development-setup)
  - [Running Tests](#running-tests)
  - [CI/CD Pipeline](#cicd-pipeline)
- [Dataset Information](#dataset-information)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-organization]/mangetamain.git
cd mangetamain

# Install dependencies using make
make install

# Or manually with pip
pip install -r requirements.txt

# Download the dataset (requires Kaggle API credentials)
make download-data

# Or manually download from Kaggle and place in data/raw/
# https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions
```

### Run the Application

```bash
# Launch the Streamlit web app
streamlit run streamlit_app_final.py

# Or using make
make app
```

The application will open in your browser at `http://localhost:8501`

### Using Docker

```bash
# Build and run with Docker Compose
make docker-up

# Or manually
docker-compose up -d
```

---

## For Users

### Installation Requirements

Before you begin, make sure you have:
- Python 3.9 or higher installed on your computer
- Basic familiarity with running commands in a terminal

### Step 1: Download the Project

```bash
# Clone the project from GitHub
git clone https://github.com/[your-organization]/mangetamain.git
cd mangetamain
```

### Step 2: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Step 3: Prepare Your Data

1. Download the recipe dataset from [Kaggle - Food.com Recipes](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)
2. Place the following files in the `data/` folder:
   - `RAW_recipes.csv` - Contains all recipe information
   - `RAW_interactions.csv` - Contains user ratings and reviews

### Step 4: Launch the Application

```bash
# Start the web application
streamlit run streamlit_app_final.py
```

The application will open in your web browser automatically!

---

## How to Use MangeTaMain

### Discovering New Recipes

**Option 1: Get Personalized Suggestions**
1. Navigate to "Personalized Recommendations"
2. Enter your User ID (or try one of the sample IDs provided)
3. Choose how many recipe suggestions you want (5-20)
4. Apply filters if you have specific needs:
   - Maximum cooking time
   - High protein recipes
   - Low calorie options
5. Click "Get Recommendations"

**Option 2: Find Similar Recipes**
1. Go to "Recipe-Based Recommendations"
2. Search for a recipe you like (e.g., "chocolate cake", "pasta carbonara")
3. Select the recipe from the search results
4. Click "Find Similar Recipes" to discover comparable options

### Understanding Cooking Time

Want to know how long a recipe will take?

1. Navigate to "Cooking Time Prediction"
2. **Enter New Recipe**: Input your recipe details (steps, ingredients, cooking methods)
3. **Search Existing Recipe**: Find a recipe from our database
4. Get an instant time estimate to help with meal planning

### Checking Nutritional Information

1. Go to "Nutrition Information"
2. **Enter Values**: Input nutritional data for any recipe
3. **Search Recipe**: Find an existing recipe in the database
4. See instant categorization:
   - High/Low Calorie
   - High Protein
   - Low Fat
   - Low Sugar
   - Overall Health Score

### Exploring Recipe Categories

1. Visit "Recipe Categories"
2. **Classify Your Recipe**: Enter recipe characteristics to find its category
3. **View Overview**: Explore different recipe types:
   - Quick & Simple meals
   - Healthy & Balanced options
   - Elaborate dishes
   - Desserts & Treats
   - Complex recipes

### Reading User Reviews

1. Navigate to "Review Analysis"
2. **Write Your Own**: Get instant sentiment analysis on your review
3. **Analyze Existing**: Enter a Recipe ID to see what others think
4. View overall sentiment (Positive, Neutral, Negative)

### Exploring the Dataset

Visit "Dataset Insights" to:
- See overall statistics about our recipe collection
- Explore cooking time distributions
- Understand recipe complexity patterns
- View nutritional trends

---

## Understanding Your Results

### Recipe Cards

Each recipe recommendation shows:
- **Recipe Name**: The dish you'll be making
- **Time**: How long it takes to prepare
- **Steps**: Number of cooking steps
- **Ingredients**: How many ingredients you'll need
- **Calories**: Per serving
- **Categories**: Health tags and recipe type

### Recommendation Types

- **Personalized**: Based on your past preferences and similar users
- **Popular**: Highly-rated recipes loved by the community
- **Content-Based**: Similar to recipes you already enjoy

### Health Categories

- **High Protein**: Great for fitness and muscle building
- **Low Calorie**: Perfect for weight management
- **Low Fat**: Heart-healthy options
- **Low Sugar**: Diabetic-friendly or sugar-conscious choices
- **Healthy Recipe**: Overall nutritious and balanced

### Recipe Categories

- **Quick & Simple**: Fast meals for busy days (15-30 minutes)
- **Healthy & Balanced**: Nutritious everyday options
- **Elaborate Meals**: Special occasion cooking (60+ minutes)
- **Complex Dishes**: Advanced techniques for cooking enthusiasts

---

## Tips for Best Results

### Getting Better Recommendations

1. **Be Active**: The more recipes you rate, the better your suggestions become
2. **Use Filters**: Narrow down results based on your needs (time, nutrition)
3. **Try Both Modes**: User-based for variety, recipe-based for similar dishes
4. **Explore Categories**: Find recipes that match your cooking style

### Planning Your Meals

1. **Check Time Estimates**: Use the cooking time prediction before shopping
2. **Review Nutrition**: Ensure recipes fit your dietary goals
3. **Read Reviews**: See what challenges others faced
4. **Start Simple**: Try quick recipes first, then explore complex ones

### Finding the Right Recipe

1. **Search Broadly**: Use general terms like "chicken" or "pasta"
2. **Apply Filters**: Add time or nutrition constraints
3. **Compare Options**: Look at multiple suggestions before deciding
4. **Check Ingredients**: Make sure you have what you need

---

## Dataset Information

MangeTaMain is powered by comprehensive recipe data:

- **230,000+ Recipes**: From Food.com community
- **1,000,000+ Reviews**: Real user feedback and ratings
- **18 Years of Data**: Recipe trends from 2000-2018
- **Detailed Information**: Ingredients, steps, nutrition, cooking time

### What Each Recipe Includes

- Ingredient list and quantities
- Step-by-step cooking instructions
- Preparation and cooking time
- Nutritional information (calories, protein, fat, sugar, etc.)
- User ratings and reviews
- Recipe tags and categories

---

## Frequently Asked Questions

**Q: Do I need to create an account?**
A: Currently, you can explore recipes using existing User IDs from our dataset. Sample IDs are provided in the app.

**Q: How accurate are the time predictions?**
A: Our predictions are typically within 15 minutes of actual cooking time, based on recipe complexity and methods.

**Q: Can I add my own recipes?**
A: The current version works with existing recipes. Custom recipe support may be added in future updates.

**Q: What if I have dietary restrictions?**
A: Use the nutrition filters to find recipes that match your needs (low calorie, high protein, etc.).

**Q: How are recommendations generated?**
A: We analyze your preferences and compare them with similar users who enjoyed recipes you haven't tried yet.

**Q: Can I save my favorite recipes?**
A: This feature is planned for future releases. Currently, note down Recipe IDs you like.

---

## Troubleshooting

### Application Won't Start

1. Check that Python 3.8+ is installed: `python --version`
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure data files are in the correct location

### No Recommendations Showing

1. Try a different User ID
2. Remove strict filters (time, nutrition)
3. Increase the number of recommendations requested
4. Check that data files are loaded correctly

### Slow Performance

1. Reduce the number of recommendations (try 5-10 instead of 20)
2. Disable enrichment temporarily
3. Close other applications to free up memory

---

## For Developers

### Project Structure

```
mangetamain/
├── .devcontainer/              # VS Code Dev Container configuration
├── .github/
│   └── workflows/              # GitHub Actions CI/CD pipelines
│       ├── ci.yml              # Main CI pipeline (test, lint, coverage)
│       ├── ml-pipeline.yml     # ML training pipeline
│       └── docker.yml          # Docker build and publish
├── config/
│   └── config.py               # Central configuration (paths, hyperparameters, thresholds)
├── data/
│   ├── raw/                    # Raw CSV files from Kaggle
│   │   ├── RAW_recipes.csv     # 230K+ recipes
│   │   └── RAW_interactions.csv # 1M+ user reviews
│   └── processed/              # Cleaned and preprocessed data
├── outputs/
│   ├── figures/                # EDA visualizations
│   ├── models/                 # Trained ML models (.pkl files)
│   ├── reports/                # Analysis reports
│   └── sentiment_model/        # Sentiment analysis model checkpoints
├── src/
│   ├── chatbot/                # RAG chatbot (Gemini integration)
│   │   └── rag_chatbot.py
│   ├── eda/                    # Exploratory data analysis
│   │   └── visualization.py
│   ├── feature_engineering/    # Feature generation modules
│   │   ├── nutrition_features.py
│   │   ├── recipe_features.py
│   │   ├── time_features.py
│   │   └── user_features.py
│   ├── integration/            # Unified recommendation pipeline
│   │   ├── recommendation_pipeline.py
│   │   └── unified_data_loader.py
│   ├── modeling/               # ML models
│   │   ├── nutrition_classifier.py    # Multi-class classification
│   │   ├── nutrition_tagger.py        # Multi-label binary classification
│   │   ├── recipe_clustering.py       # HDBSCAN clustering
│   │   ├── time_predictor.py          # Regression models
│   │   └── user_clustering.py         # User segmentation
│   ├── preprocessing/          # Data loading and cleaning
│   │   └── data_loader.py
│   ├── recommendation/         # Recommendation algorithms
│   │   ├── svd_recommender.py         # Collaborative filtering
│   │   ├── recipe_recommender.py      # Content-based
│   │   └── data_processor.py
│   ├── sentiment_analysis/     # Review sentiment models
│   └── utils/                  # Utility functions
│       └── data_cache.py       # Memory-efficient data caching
├── tests/                      # Test suite (pytest)
│   ├── conftest.py             # Shared fixtures
│   ├── test_feature_engineering/
│   ├── test_modeling/
│   └── test_scripts/
├── scripts/                    # Automation scripts
│   ├── download_kaggle_data.py
│   ├── clean_data.py
│   ├── run_recipe_pipeline.py
│   ├── run_nutrition_pipeline.py
│   └── train_sentiment_model.py
├── streamlit_app_final.py      # Main Streamlit application (2700 lines)
├── main_nutrition_tagging.py   # Nutrition pipeline orchestrator
├── main_time_prediction.py     # Time prediction pipeline
├── train_sentiment_model.py    # Sentiment model training
├── Dockerfile                  # Docker image configuration
├── docker-compose.yml          # Multi-container orchestration
├── Makefile                    # Build automation commands
├── pyproject.toml              # Project metadata, dependencies, tool configs
├── requirements.txt            # Pinned dependencies (UV compiled)
└── README.md                   # This file
```

**Project Statistics:**
- **60 Python files** | **13,689 lines of code** | **5.9 MB total**
- **Main app**: 2,700 lines (streamlit_app_final.py)
- **30 source modules** | **13 test modules** | **4 CI/CD workflows**

### Technology Stack

#### Machine Learning & Data Science
- **scikit-learn 1.7.2** - ML algorithms (Random Forest, Gradient Boosting, SVD)
- **pandas 2.3.3** - Data manipulation (230K+ recipes, 1M+ reviews)
- **numpy 2.3.4** - Numerical computing
- **scipy 1.16.3** - Scientific computing and optimization
- **joblib 1.5.2** - Model serialization and parallel processing

#### Deep Learning & NLP
- **PyTorch 2.9.0** - Deep learning framework
- **transformers 4.57.1** - Pre-trained models (DistilBERT, RoBERTa)
- **huggingface-hub 0.36.0** - Model hub integration
- **nltk 3.9.2** - Natural language processing
- **textblob 0.19.0** - Sentiment analysis utilities
- **textstat 0.7.10** - Text complexity analysis

#### Web Framework
- **Streamlit 1.50.0** - Interactive web application
- **plotly 6.3.1** - Interactive visualizations
- **matplotlib 3.10.7** - Static plots
- **seaborn 0.13.2** - Statistical visualization

#### AI/LLM
- **google-generativeai 0.8.5** - Google Gemini API for RAG chatbot

#### Clustering
- **hdbscan 0.8.40** - Hierarchical density-based clustering

#### Development & Testing
- **pytest 7.3.0+** - Testing framework with fixtures
- **pytest-cov** - Coverage reporting
- **black 23.0.0** - Code formatting (88-char line length)
- **ruff 0.1.0+** - Fast Python linter
- **jupyter** - Interactive notebooks for experimentation

#### DevOps
- **Docker** - Containerization (multi-stage builds)
- **GitHub Actions** - CI/CD automation (4 workflows)
- **make** - Build automation (12+ commands)

### Development Setup

#### 1. Clone and Install

```bash
git clone https://github.com/[your-organization]/mangetamain.git
cd mangetamain

# Install all dependencies including dev tools
make install-dev

# Or manually
pip install -e ".[dev]"
```

#### 2. Configure Environment

```bash
# Set up Kaggle API credentials (for data download)
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"

# Or create ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
echo '{"username":"your-username","key":"your-api-key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

#### 3. Download Data

```bash
# Using make
make download-data

# Or manually
python scripts/download_kaggle_data.py
```

#### 4. Run Data Pipeline

```bash
# Clean and preprocess data
make clean-data

# Run full ML pipeline
make pipeline

# Or run individual pipelines
python main_time_prediction.py
python main_nutrition_tagging.py
python train_sentiment_model.py
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_modeling/test_time_predictor.py

# Run with verbose output
pytest -v

# Run only fast tests (skip slow/integration)
pytest -m "not slow"
```

**Test Coverage:** The project includes comprehensive unit tests covering:
- Data loading and preprocessing
- Feature engineering (nutrition, time, recipe, user)
- Model training and prediction
- Clustering algorithms
- Data cleaning utilities

**Test Fixtures:** 6 shared fixtures in `tests/conftest.py`:
- `sample_recipes_df` - 5 sample recipes
- `sample_interactions_df` - User interaction data
- `sample_recipes_with_features` - Pre-engineered features
- `temp_data_dir` - Temporary directory structure
- `temp_output_dir` - Temporary output structure

### Code Quality

```bash
# Format code with Black (88-char line length)
make format

# Lint with Ruff
make lint

# Run both
make format && make lint
```

**Code Style:**
- **Black** formatting (88-char lines, Python 3.9+)
- **Ruff** linting (fast, enforces best practices)
- Type hints encouraged (not enforced)
- Docstrings for public APIs (Google style)

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

#### 1. CI Workflow (`ci.yml`)
**Triggers:** Push/PR on main and develop branches
**Python Versions:** 3.9, 3.10, 3.11
**Steps:**
1. Install dependencies with caching
2. Lint with ruff and black
3. Run pytest with coverage
4. Upload coverage to Codecov
5. Generate HTML coverage reports

#### 2. ML Pipeline Workflow (`ml-pipeline.yml`)
**Triggers:** Manual dispatch, Sunday 2 AM cron, push to main
**Steps:**
1. Setup Kaggle credentials from secrets
2. Download dataset (optional)
3. Clean data
4. Run recipe pipeline (clustering, features)
5. Run nutrition pipeline (classification)
6. Run time prediction (regression)
7. Train sentiment model (transformers)
8. Archive model artifacts (30-day retention)

**Secrets Required:**
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

#### 3. Docker Workflow (`docker.yml`)
**Triggers:** Push to main, version tags (`v*`), PRs (test only)
**Steps:**
1. Build Docker image
2. Run container validation tests
3. Push to GitHub Container Registry
4. Tag with `latest`, version, and git SHA

### Available Make Commands

```bash
make install          # Install dependencies
make install-dev     # Install with dev tools
make test            # Run tests
make test-cov        # Run tests with coverage
make lint            # Lint code with ruff
make format          # Format code with black
make pipeline        # Run full ML pipeline
make docker-build    # Build Docker image
make docker-up       # Start Docker containers
make docker-down     # Stop Docker containers
make download-data   # Download Kaggle dataset
make clean-data      # Run data cleaning script
make app             # Run Streamlit application
make help            # Show all available commands
```

### Memory Optimization

The application includes advanced memory management for deployment on limited resources (e.g., Streamlit Cloud with 1GB RAM):

**DataCache System:**
- Singleton-pattern caching to load large CSVs only once
- Selective column loading (load only needed columns)
- Type optimization (reduce datatype sizes)
- Garbage collection and strategic unloading

**Streamlit Cloud Mode:**
- Aggressive memory mode when RAM < 2GB
- Selective dataframe unloading between pages
- Recipe data cached (used by 75% of pages)
- Sentiment model unloaded after use

See `src/utils/data_cache.py` and `test_memory_optimization.py` for implementation details.

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following our code style guidelines
4. **Add tests** for any new functionality
5. **Run the test suite** (`make test`) to ensure everything passes
6. **Format and lint** your code (`make format && make lint`)
7. **Commit your changes** (`git commit -m 'Add amazing feature'`)
8. **Push to your fork** (`git push origin feature/amazing-feature`)
9. **Open a Pull Request** with a clear description

### Contribution Guidelines

- Follow the existing code style (Black + Ruff)
- Write clear, descriptive commit messages
- Add unit tests for new features
- Update documentation as needed
- Keep PRs focused on a single feature/fix
- Ensure all tests pass before submitting

---

## Support and Feedback

Having issues or suggestions? Here's how to get help:

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/[your-organization]/mangetamain/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/[your-organization]/mangetamain/discussions)
- **Check README**: Most common questions are answered in this guide

---

## Roadmap

Planned features and improvements:

**Version 3.0 (Q2 2025)**
- User account creation and authentication
- Personal recipe collections and favorites
- Shopping list generation with ingredient aggregation
- Meal planning calendar with weekly views

**Version 3.5 (Q3 2025)**
- Social features (sharing, following, recipe discussions)
- Recipe rating and review submission
- Advanced filters (cuisine type, dietary restrictions)
- Recipe scaling (adjust servings)

**Version 4.0 (Q4 2025)**
- Mobile app (iOS/Android) with React Native
- Voice-activated cooking assistant
- Ingredient substitution suggestions
- Cooking mode with step-by-step timer

**Future Considerations**
- Integration with grocery delivery services
- Nutrition tracking and meal analytics
- Community recipe submissions
- Video cooking tutorials integration

---

## Acknowledgments

MangeTaMain is built on the shoulders of giants:

- **Dataset**: [Food.com Recipes and Interactions](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) by Shuyang Li (Kaggle)
- **Technology**: Built with Python, scikit-learn, PyTorch, Transformers, Streamlit, and many other open-source libraries
- **Inspiration**: Powered by the passion of home cooks and data science enthusiasts
- **Community**: Special thanks to all contributors and testers

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Dataset License**: The Food.com dataset is subject to Kaggle's terms of use. Please review the [dataset page](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) for specific licensing information.

---

## Citation

If you use MangeTaMain in your research or project, please cite:

```bibtex
@software{mangetamain2025,
  title = {MangeTaMain: Intelligent Recipe Discovery Platform},
  author = {[Your Name/Organization]},
  year = {2025},
  url = {https://github.com/[your-organization]/mangetamain},
  version = {2.0.0}
}
```

---

## Project Statistics

- **230,000+ recipes** from Food.com
- **1,000,000+ user reviews** spanning 18 years (2000-2018)
- **13,689 lines of Python code** across 60 files
- **30 source modules** | **13 test modules** | **4 CI/CD workflows**
- **2,700-line Streamlit application** for interactive exploration
- **6 machine learning models** (collaborative filtering, time prediction, nutrition classification, sentiment analysis, clustering)

---

**Version**: 2.0.0
**Last Updated**: 2025-10-31
**Status**: Production Ready

---

**Built with passion by data scientists and home cooks who believe that great food and great technology go hand in hand.**
