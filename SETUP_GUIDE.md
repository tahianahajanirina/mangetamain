# MangeTaMain - Complete Setup Guide for Running the Streamlit App

This guide will walk you through setting up and running the MangeTaMain recipe discovery platform on your machine.

---

## Prerequisites

Before starting, make sure you have:
- **Python 3.9 or higher** installed
- **Git** installed (to clone the project)
- At least **4GB of free RAM** (for model training)
- **5GB of free disk space** (for data and models)

Check your Python version:
```bash
python --version
# or
python3 --version
```

---

## Step 1: Get the Project

If you haven't already, navigate to the project directory:

```bash
cd /path/to/mangetamain
```

---

## Step 2: Install UV Package Manager

UV is a fast Python package installer. Install it:

### On macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### On Windows:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative: Install via pip
```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

---

## Step 3: Install Dependencies

### Option A: Using UV (Recommended - Fastest)

Install all dependencies from pyproject.toml:

```bash
# Install core dependencies
uv pip install -e .

# Install sentiment analysis dependencies (optional but recommended)
uv pip install -e ".[sentiment]"
```

### Option B: Using pip

If you prefer pip:

```bash
pip install -e .
pip install -e ".[sentiment]"
```

This will install:
- Core ML libraries (numpy, pandas, scikit-learn)
- Data processing tools (scipy, joblib)
- Visualization (matplotlib, seaborn, plotly)
- NLP tools (nltk, textblob)
- Streamlit web framework
- Deep learning (transformers, torch) for sentiment analysis
- Google Gemini API for chatbot

---

## Step 4: Verify Data Files

Make sure you have the required data files in the `data/` directory:

**Required files:**
- `data/RAW_recipes.csv` - Recipe information (230K+ recipes)
- `data/RAW_interactions.csv` - User ratings and reviews (1M+ interactions)

**Check if files exist:**
```bash
ls -lh data/RAW*.csv
```

If these files are missing, download them from:
[Kaggle - Food.com Recipes Dataset](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)

---

## Step 5: Generate Required Models

The Streamlit app requires several pre-trained models. You need to generate them **in this order**:

### 5.1 Generate Clustering Models (Required)

This creates user and recipe clustering models:

```bash
python run_clustering_models.py
```

**What this does:**
- Builds user behavior profiles
- Creates recipe feature matrices
- Trains K-means clustering models
- Saves models to `outputs/models/`
- Generates visualizations in `outputs/figures/`

**Time:** ~5-15 minutes depending on your machine

**Output files:**
- `outputs/models/user_clustering_*.pkl`
- `outputs/models/recipe_clustering_v3_*.pkl`
- `data/users_clustered.csv`
- `data/recipes_clustered_v3.csv`

---

### 5.2 Generate Sentiment Analysis Model (Optional but Recommended)

This creates the review sentiment analyzer:

```bash
python train_sentiment_model.py
```

**What this does:**
- Loads recipe reviews from interactions data
- Fine-tunes a RoBERTa model for sentiment analysis
- Trains on 50K sampled reviews
- Saves model to `outputs/models/sentiment_model_roberta/`

**Time:**
- With GPU: ~20-30 minutes
- Without GPU (CPU): ~2-3 hours

**Note:** If you skip this step, the sentiment analysis feature won't work in the app, but everything else will.

**Output files:**
- `outputs/models/sentiment_model_roberta/pytorch_model.bin`
- `outputs/models/sentiment_model_roberta/config.json`
- `outputs/models/sentiment_model_roberta/tokenizer files`

---

### 5.3 Other Models (Auto-generated)

The following models are generated automatically when the Streamlit app first runs:

- **Time Prediction Models** - Estimates cooking time
- **Nutrition Tagging Models** - Classifies recipes by health attributes
- **SVD Recommender** - Provides personalized recommendations

These are trained on-the-fly during the first app load (takes 2-3 minutes).

---

## Step 6: Run the Streamlit App

Now you're ready to launch the app:

```bash
streamlit run streamlit_app_final.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The app will automatically open in your default web browser!

---

## Step 7: First-Time App Usage

When the app loads for the first time:

1. **Wait for initialization** (~2-3 minutes)
   - The app loads recipe data
   - Trains recommendation models
   - Loads pre-trained models

2. **System status** will show in the sidebar:
   - Recipes: ✅
   - Time Info: ✅
   - Nutrition: ✅
   - Categories: ✅
   - Reviews: ✅

3. **Start exploring!**
   - Try the "Home & Getting Started" section
   - Use sample User IDs: 424680, 52282, 742802
   - Explore different features

---

## Using the Application

### Main Features:

#### 1. Find Recipes (Personalized Recommendations)
```
Navigation: "Find Recipes" → "Personalized for You"
- Enter User ID: 52282 (sample user)
- Number of suggestions: 10
- Enable filters if needed
- Click "Get My Recipe Suggestions"
```

#### 2. Similar Recipe Search
```
Navigation: "Find Recipes" → "Similar Recipes"
- Search: "chocolate cake"
- Select a recipe
- Click "Find Similar Recipes"
```

#### 3. Cooking Time Estimation
```
Navigation: "Cooking Time"
- Enter recipe details (steps, ingredients)
- Click "Estimate Time"
```

#### 4. Nutrition Information
```
Navigation: "Nutrition Info"
- Enter nutritional values
- Click "Analyze Nutrition"
```

#### 5. Recipe Categories
```
Navigation: "Recipe Categories"
- Enter recipe characteristics
- Click "Find Cluster"
```

#### 6. Review Analysis
```
Navigation: "Review Analysis"
- Write your own review OR
- Enter Recipe ID: 275022
- Click "Analyze Sentiment"
```

#### 7. AI Chatbot (Optional - Requires API Key)
```
Navigation: "Recipe Chatbot"
- Ask questions about recipes
- Get AI-powered recommendations
```

**To enable chatbot:**
Set your Gemini API key in `config/config.py` or:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
Get API key: https://makersuite.google.com/app/apikey

---

## Troubleshooting

### Problem: "Module not found" errors

**Solution:**
```bash
# Reinstall dependencies
uv pip install -e ".[sentiment]"
```

### Problem: "Data files not found"

**Solution:**
```bash
# Check if files exist
ls -lh data/RAW*.csv

# If missing, download from Kaggle and place in data/ folder
```

### Problem: App loads but no recommendations

**Solution:**
```bash
# Make sure clustering models are generated
python run_clustering_models.py

# Check if model files exist
ls -lh outputs/models/
```

### Problem: Sentiment analysis not working

**Solution:**
```bash
# Train the sentiment model
python train_sentiment_model.py

# This is optional - skip if taking too long
```

### Problem: Out of memory errors

**Solution:**
- Close other applications
- Reduce sample size in `train_sentiment_model.py` (line 94: change 50000 to 10000)
- Skip sentiment model training if needed

### Problem: Slow performance

**Solutions:**
- Reduce number of recommendations (use 5-10 instead of 20)
- Disable "Add extra details" option temporarily
- Use filters to narrow down results

---

## Quick Start Summary

Here's the absolute minimum to get started:

```bash
# 1. Install dependencies
uv pip install -e ".[sentiment]"

# 2. Generate clustering models (REQUIRED)
python run_clustering_models.py

# 3. Run the app
streamlit run streamlit_app_final.py
```

**Optional:** Train sentiment model for review analysis:
```bash
python train_sentiment_model.py
```

---

## File Structure Overview

After setup, your project should look like:

```
mangetamain/
├── data/
│   ├── RAW_recipes.csv              # [Required] Recipe data
│   ├── RAW_interactions.csv         # [Required] User reviews
│   ├── processed_recipes.csv        # [Auto-generated]
│   ├── users_clustered.csv          # [Generated by script]
│   └── recipes_clustered_v3.csv     # [Generated by script]
├── outputs/
│   ├── models/
│   │   ├── user_clustering_*.pkl    # [Generated by script]
│   │   ├── recipe_clustering_*.pkl  # [Generated by script]
│   │   └── sentiment_model_roberta/ # [Optional - generated by script]
│   └── figures/                      # [Auto-generated visualizations]
├── streamlit_app_final.py            # [Main app - run this]
├── run_clustering_models.py          # [Run to generate models]
├── train_sentiment_model.py          # [Optional - run for sentiment]
└── pyproject.toml                    # [Dependency definitions]
```

---

## Performance Tips

1. **First run is always slower** - Models are being trained
2. **Use caching** - Streamlit caches data and models automatically
3. **Start with small requests** - Try 5-10 recommendations first
4. **GPU recommended** - For sentiment model training (but not required)

---

## System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 4GB
- Storage: 5GB
- Python: 3.9+

**Recommended:**
- CPU: 8 cores
- RAM: 8GB
- Storage: 10GB
- GPU: Any CUDA-compatible (for sentiment training)
- Python: 3.10+

---

## Getting Help

If you encounter issues:

1. **Check error messages** - They usually point to the problem
2. **Verify data files** - Make sure RAW_*.csv files are present
3. **Check model files** - Ensure outputs/models/ has the .pkl files
4. **Review logs** - Check outputs/pipeline.log for details
5. **Restart the app** - Sometimes Streamlit needs a fresh start

---

## Next Steps

Once the app is running:

1. ✅ Try personalized recommendations with sample users
2. ✅ Explore different recipe categories
3. ✅ Test time and nutrition predictions
4. ✅ Analyze recipe reviews
5. ✅ Set up the AI chatbot (optional)

---

## Additional Notes

- **Data privacy**: All processing is local, no data is sent externally (except Gemini API if used)
- **Model persistence**: Models are saved and reused across sessions
- **Updates**: Pull latest changes with `git pull` before running
- **Customization**: Edit `config/config.py` to adjust model parameters

---

**Last Updated**: 2025-10-28

**Version**: 2.0.0

---

Happy cooking with MangeTaMain! 👨‍🍳👩‍🍳
