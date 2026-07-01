"""Configuration file for recipe ML project.

This module contains all configuration parameters for data paths,
model hyperparameters, and feature engineering settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUTS_FIGURES = FIGURE_DIR  # Alias for compatibility
REPORT_DIR = OUTPUT_DIR / "reports"

DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Data files
RAW_DATA_FILE = DATA_RAW / "RAW_recipes.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED / "processed_recipes.csv"

# Task configurations
TASK_CONFIG = {
    "time_prediction": {
        "target": "minutes",
        "task_type": "regression",
        "description": "Predict recipe preparation time",
        "key_features": [
            "n_steps",
            "n_ingredients",
            "cooking_verb_count",
            "equipment_count",
            "avg_step_length",
            "steps_word_count",
        ],
    },
    "nutrition_tagging": {
        "target": "nutrition_tag",
        "task_type": "classification",
        "description": "Tag recipes based on nutritional value",
        "key_features": [
            "calories",
            "protein_pdv",
            "carbs_pdv",
            "total_fat_pdv",
            "sugar_pdv",
            "sodium_pdv",
            "saturated_fat_pdv",
        ],
    },
}

# Feature engineering configuration
FEATURE_CONFIG = {
    # Text processing
    "stopwords_lang": "english",
    "tfidf_max_features": 50,
    # Cooking verbs and equipment
    "cooking_verbs": {
        "bake",
        "boil",
        "fry",
        "grill",
        "roast",
        "saute",
        "simmer",
        "steam",
        "broil",
        "blend",
        "mix",
        "whisk",
        "chop",
        "dice",
        "mince",
        "slice",
        "marinate",
        "season",
        "toss",
    },
    "equipment_terms": {
        "oven",
        "stove",
        "microwave",
        "blender",
        "mixer",
        "processor",
        "pan",
        "pot",
        "skillet",
        "bowl",
        "dish",
        "sheet",
        "rack",
        "slow cooker",
        "instant pot",
    },
    # Dietary patterns
    "dietary_patterns": {
        "dietary_vegetarian": ["vegetarian", "veggie"],
        "dietary_vegan": ["vegan"],
        "dietary_gluten_free": ["gluten-free", "gluten free"],
        "dietary_dairy_free": ["dairy-free", "dairy free", "lactose-free"],
        "dietary_low_carb": ["low-carb", "low carb", "keto"],
        "dietary_low_fat": ["low-fat", "low fat"],
        "dietary_low_calorie": ["low-calorie", "low calorie"],
        "dietary_paleo": ["paleo"],
        "dietary_whole_grain": ["whole grain", "whole-grain"],
    },
    # Ingredient categories
    "ingredient_categories": {
        "has_dairy": ["milk", "cheese", "butter", "cream", "yogurt"],
        "has_meat": ["chicken", "beef", "pork", "turkey", "lamb", "bacon"],
        "has_fish": ["fish", "salmon", "tuna", "shrimp", "cod", "seafood"],
        "has_eggs": ["egg"],
        "has_vegetables": [
            "tomato",
            "onion",
            "garlic",
            "pepper",
            "spinach",
            "broccoli",
            "carrot",
        ],
        "has_fruit": ["apple", "banana", "lemon", "orange", "berry", "pineapple"],
        "has_grain": ["rice", "pasta", "flour", "bread", "quinoa"],
        "has_nuts": ["almond", "walnut", "pecan", "peanut", "cashew"],
        "has_chocolate": ["chocolate", "cocoa"],
    },
    # Nutrition columns
    "nutrition_cols": [
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
    ],
    # Time thresholds
    "time_thresholds": {"quick": 30, "moderate": 60},
    # Outlier handling
    "minutes_cap_percentile": 0.99,
    "unrealistic_time_threshold": 10080,  # 1 week in minutes
}

# EDA configuration
EDA_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8-darkgrid",
    "palette": "husl",
    "top_n_tags": 15,
    "top_n_ingredients": 25,
    "correlation_threshold": 0.7,
}

# Model configuration
MODEL_CONFIG = {
    "time_prediction": {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "models": {
            "linear": {"name": "LinearRegression"},
            "ridge": {"name": "Ridge", "alpha": 1.0},
            "random_forest": {
                "name": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
            "gradient_boosting": {
                "name": "GradientBoostingRegressor",
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42,
            },
        },
    },
    "nutrition_tagging": {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "nutrition_tags": {
            "high_calorie": {"threshold": 500, "direction": "above"},
            "low_calorie": {"threshold": 200, "direction": "below"},
            "high_protein": {"threshold": 30, "direction": "above"},
            "low_fat": {"threshold": 10, "direction": "below"},
            "low_sugar": {"threshold": 10, "direction": "below"},
        },
        "models": {
            "logistic": {"name": "LogisticRegression", "max_iter": 1000},
            "random_forest": {
                "name": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
            "gradient_boosting": {
                "name": "GradientBoostingClassifier",
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42,
            },
        },
    },
}

# Chatbot configuration
CHATBOT_CONFIG = {
    # To use the chatbot, add your Google Gemini API key here
    # Get your API key from: https://makersuite.google.com/app/apikey
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),  # Set via environment variable GEMINI_API_KEY
    "model": "gemini-2.0-flash-exp",
    "max_history": 10,
    "retrieval_top_k": 5,
}

# Memory management for Streamlit Cloud (1GB RAM limit)
MEMORY_CONFIG = {
    "aggressive_mode": True,  # True pour Streamlit Cloud, False pour local
    "unload_dataframes_after_page": True,  # Décharger DataFrames entre pages
    "keep_recipes_cached": True,  # Garder recipes (utilisé par 75% des pages)
    "unload_interactions_after_use": True,  # Décharger interactions (38% des pages)
    "unload_sentiment_model": True,  # Décharger modèle sentiment après utilisation
    "use_column_subset": True,  # Charger seulement colonnes nécessaires
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(OUTPUT_DIR / "pipeline.log"),
            "level": "DEBUG",
            "formatter": "standard",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
}
