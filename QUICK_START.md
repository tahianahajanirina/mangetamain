# MangeTaMain - Quick Start Guide

## Super Fast Setup (5 Commands)

```bash
# 1. Install UV package manager
pip install uv

# 2. Install all dependencies
uv pip install -e ".[sentiment]"

# 3. Generate clustering models (REQUIRED - takes ~10 min)
python scripts/clean_data.py 
python scripts/run_recipe_pipeline.py

# 4. (Optional) Train sentiment model (~30 min with GPU, ~2-3 hours CPU)
# python train_sentiment_model.py

# 5. Run the app!
streamlit run streamlit_app_final.py
```

---

## Minimum Setup (Skip Sentiment Analysis)

If you want to start quickly without sentiment analysis:

```bash
uv pip install -e .
python scripts/clean_data.py 
python scripts/run_recipe_pipeline.py
streamlit run streamlit_app_final.py
```

---

## Required Files

Before running, make sure these exist:
```
data/RAW_recipes.csv
data/RAW_interactions.csv
```

Download from: https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions

---

## Expected Output After Setup

Your `outputs/models/` folder should contain:
```
user_clustering_YYYYMMDD_HHMMSS.pkl
recipe_clustering_v3_YYYYMMDD_HHMMSS.pkl
sentiment_model_roberta/ (if you ran train_sentiment_model.py)
```

---

## First Time Running the App

1. Open browser: http://localhost:8501
2. Wait 2-3 minutes for first-time initialization
3. Try sample User ID: **52282** or **424680**
4. Start exploring recipes!

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Module not found | Run: `uv pip install -e ".[sentiment]"` |
| No data files | Download from Kaggle, place in `data/` |
| No recommendations | Run: `python run_clustering_models.py` |
| Out of memory | Close other apps, reduce sample sizes |

---

## What Each Script Does

| Script | Purpose | Time | Required? |
|--------|---------|------|-----------|
| `run_clustering_models.py` | Creates user & recipe clusters | ~10 min | ✅ YES |
| `train_sentiment_model.py` | Trains review analyzer | ~30 min-3 hrs | ⚠️ Optional |
| `streamlit_app_final.py` | Runs the web app | instant | ✅ YES |

---

## Pro Tips

- ✅ Use GPU for sentiment training (way faster)
- ✅ Start with 5-10 recommendations first
- ✅ Check `outputs/pipeline.log` if errors occur
- ✅ Streamlit caches everything - restart if needed

---

**Need detailed help?** See `SETUP_GUIDE.md`
