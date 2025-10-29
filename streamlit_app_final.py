"""
MangeTaMain - Your Personal Recipe Discovery Platform
A smart cooking companion that helps you discover recipes you'll love!

Features:
1. Personalized Suggestions - Recommendations tailored to your taste
2. Cooking Time Estimation - Know how long recipes will take
3. Nutrition Information - Understand recipe health profiles
4. Recipe Categories - Browse by cooking time and complexity
5. Review Analysis - See what others think about recipes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
import time
import os
import json
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integration.recommendation_pipeline import IntegratedRecommendationPipeline
from src.modeling.time_predictor import TimePredictionModel
from src.modeling.nutrition_tagger import NutritionTaggerModel
from src.modeling.recipe_clustering import RecipeClusterer
from config.config import CHATBOT_CONFIG

# Page configuration
st.set_page_config(
    page_title="MangeTaMain - Recipe Discovery",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4ECDC4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E8F4F8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin-bottom: 1rem;
    }
    .recipe-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive { color: #2ecc71; font-weight: bold; }
    .sentiment-neutral { color: #f39c12; font-weight: bold; }
    .sentiment-negative { color: #e74c3c; font-weight: bold; }
    .tag-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 15px;
        background-color: #4ECDC4;
        color: white;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_pipeline():
    """Load and cache the recipe discovery system."""
    with st.spinner("Loading recipe database..."):
        try:
            pipeline = IntegratedRecommendationPipeline(
                recipes_path="data/raw/RAW_recipes.csv",
                interactions_path="data/raw/RAW_interactions.csv",
                models_dir="outputs/models",
                load_models=True
            )

            # Train recommendation system if not trained
            if not pipeline.svd_trained:
                st.info("Setting up your personalized recommendation system (this may take a few minutes)...")
                pipeline.train_svd_recommender(k=50)

            return pipeline
        except Exception as e:
            st.error(f"Error loading recipe database: {e}")
            logger.error(f"Pipeline loading error: {e}", exc_info=True)
            return None




def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<p class="main-header">🍳 MangeTaMain - Recipe Discovery</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>"
        "Your Smart Cooking Companion"
        "</p>",
        unsafe_allow_html=True
    )

    # Load pipeline
    pipeline = load_pipeline()

    if pipeline is None:
        st.error("❌ Unable to load recipe database. Please check that data files are in the correct location.")
        return

    # Sidebar - System Status & Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/chef-hat.png", width=80)
        st.markdown("### 🎛️ Quick Access")

        status = pipeline.get_pipeline_status()

        # Status indicators
        st.markdown("#### System Ready")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Recipes", "✅" if status['svd_recommender'] == 'trained' else "❌")
            st.metric("Time Info", "✅" if status['time_model'] == 'loaded' else "⚠️")
            st.metric("Nutrition", f"✅ Ready" if status['nutrition_models']['count'] > 0 else "⚠️")

        with col2:
            st.metric("Categories", "✅" if status['recipe_clusterer'] == 'loaded' else "⚠️")
            st.metric("Reviews", "✅")

        st.markdown("---")

        # Navigation
        st.markdown("### 📋 Menu")

        tab_selection = st.radio(
            "Where would you like to go?",
            [
                "🏠 Home",
                "🎯 Find Recipes",
                "🤖 Recipe Chatbot",
                "⏰ Cooking Time",
                "🥗 Nutrition Info",
                "📊 Recipe Categories",
                "💭 Review Analysis",
                "📈 Dataset Insights",
                "ℹ️ About & Help"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Version:** 2.0.0")
        st.markdown("**Recipes:** 230,000+")
        st.markdown("**Updated:** 2025-10-28")

    # Main content based on selected tab
    if tab_selection == "🏠 Home & Getting Started":
        show_home_page(pipeline)
    elif tab_selection == "🎯 Find Recipes":
        show_recommendations_page(pipeline)
    elif tab_selection == "🤖 Recipe Chatbot":
        show_chatbot_page(pipeline)
    elif tab_selection == "⏰ Cooking Time":
        show_time_prediction_inference(pipeline)
    elif tab_selection == "🥗 Nutrition Info":
        show_nutrition_tagging_inference(pipeline)
    elif tab_selection == "📊 Recipe Categories":
        show_clustering_page(pipeline)
    elif tab_selection == "💭 Review Analysis":
        show_sentiment_page(pipeline)
    elif tab_selection == "📈 Dataset Insights":
        show_analytics_page(pipeline)
    elif tab_selection == "ℹ️ About & Help":
        show_about_page()


def show_home_page(pipeline):
    """Home page with quick start guide."""

    st.markdown('<p class="sub-header">🏠 Welcome to MangeTaMain</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome!</h3>
        <p>MangeTaMain is your smart cooking companion that helps you discover recipes you'll love!
        With over 230,000 recipes and 1 million user reviews, we provide personalized suggestions,
        cooking time estimates, nutrition information, recipe categories, and review insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown("### 🚀 Quick Start Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 For First-Time Users
        1. **Start here**: Go to "Find Recipes"
        2. **Enter a User ID**: Try 52282 (sample user)
        3. **Get Suggestions**: Click the button
        4. **Apply filters**: Try time limits or health preferences

        #### ⏰ Estimate Cooking Time
        1. Navigate to "Cooking Time"
        2. Enter your recipe details (steps, ingredients)
        3. Get instant time estimates
        4. Plan your meals better
        """)

    with col2:
        st.markdown("""
        #### 🥗 Check Nutrition
        1. Go to "Nutrition Info"
        2. Enter nutritional values
        3. Get health category classifications
        4. Make informed choices

        #### 📊 Browse & Explore
        1. Explore "Recipe Categories" for types
        2. Check "Dataset Insights" for trends
        3. Use "Review Analysis" to see ratings
        """)

    # System Overview
    st.markdown("### 📊 System Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h4>Recipe Finder</h4>
            <p>Personalized suggestions just for you</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>⏰</h2>
            <h4>Time Estimator</h4>
            <p>Know how long recipes take</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🥗</h2>
            <h4>Nutrition Guide</h4>
            <p>Understand recipe health profiles</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>📊</h2>
            <h4>Recipe Browser</h4>
            <p>Organized by time and complexity</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="metric-card">
            <h2>💭</h2>
            <h4>Review Reader</h4>
            <p>See what others think</p>
        </div>
        """, unsafe_allow_html=True)

    # Dataset Stats
    st.markdown("### 📊 Dataset Statistics")

    try:
        all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Recipes", f"{len(all_recipes):,}")
        with col2:
            st.metric("Avg Cooking Time", f"{all_recipes['minutes'].median():.0f} min")
        with col3:
            st.metric("Avg Steps", f"{all_recipes['n_steps'].mean():.1f}")
        with col4:
            st.metric("Avg Ingredients", f"{all_recipes['n_ingredients'].mean():.1f}")
    except Exception as e:
        st.warning("Dataset statistics not available")


def get_similar_recipes(pipeline, base_recipe, all_recipes, top_n=10, enrich=True, filters=None):
    """
    Find similar recipes based on content (features) similarity.

    Args:
        pipeline: The recommendation pipeline
        base_recipe: The recipe to find similar ones for (pandas Series)
        all_recipes: DataFrame of all recipes
        top_n: Number of similar recipes to return
        enrich: Whether to enrich with ML predictions
        filters: Optional filters to apply

    Returns:
        DataFrame of similar recipes
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    # Extract features for similarity calculation
    feature_columns = ['minutes', 'n_steps', 'n_ingredients']

    # Add nutrition features if available
    if 'calories' in all_recipes.columns:
        feature_columns.extend(['calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv'])

    # Filter out recipes with missing features
    valid_recipes = all_recipes.dropna(subset=feature_columns)

    # Exclude the base recipe itself
    valid_recipes = valid_recipes[valid_recipes['id'] != base_recipe['id']]

    # Extract feature matrix
    X = valid_recipes[feature_columns].values

    # Extract base recipe features
    base_features = base_recipe[feature_columns].values.reshape(1, -1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    base_scaled = scaler.transform(base_features)

    # Calculate cosine similarity
    similarities = cosine_similarity(base_scaled, X_scaled)[0]

    # Add similarity scores to recipes
    valid_recipes = valid_recipes.copy()
    valid_recipes['similarity_score'] = similarities

    # Sort by similarity
    similar_recipes = valid_recipes.sort_values('similarity_score', ascending=False)

    # Apply filters if provided
    if filters:
        if 'max_time' in filters and filters['max_time']:
            similar_recipes = similar_recipes[similar_recipes['minutes'] <= filters['max_time']]

        # For nutrition filters, we need to enrich first or check if already available
        if enrich:
            # Enrich a larger set first before filtering
            similar_recipes_enriched = similar_recipes.head(top_n * 3).copy()

            # Add time predictions
            if pipeline.time_model:
                similar_recipes_enriched = pipeline._add_time_predictions(similar_recipes_enriched)

            # Add nutrition tags
            if pipeline.nutrition_models:
                similar_recipes_enriched = pipeline._add_nutrition_tags(similar_recipes_enriched)

            # Add cluster assignments
            if pipeline.recipe_clusterer:
                similar_recipes_enriched = pipeline._add_clusters(similar_recipes_enriched)

            # Apply nutrition filters
            if filters.get('require_high_protein'):
                similar_recipes_enriched = similar_recipes_enriched[
                    similar_recipes_enriched.get('high_protein', 0) == 1
                ]

            if filters.get('require_low_calorie'):
                similar_recipes_enriched = similar_recipes_enriched[
                    similar_recipes_enriched.get('low_calorie', 0) == 1
                ]

            return similar_recipes_enriched.head(top_n)

    # Get top N similar recipes
    result = similar_recipes.head(top_n * 2 if enrich else top_n)

    # Enrich with ML predictions if requested
    if enrich:
        # Add time predictions
        if pipeline.time_model:
            result = pipeline._add_time_predictions(result)

        # Add nutrition tags
        if pipeline.nutrition_models:
            result = pipeline._add_nutrition_tags(result)

        # Add cluster assignments
        if pipeline.recipe_clusterer:
            result = pipeline._add_clusters(result)

    return result.head(top_n)


def show_recommendations_page(pipeline):
    """Recipe finder page with detailed guide."""

    st.markdown('<p class="sub-header">🎯 Find Your Perfect Recipe</p>', unsafe_allow_html=True)

    # How to use section
    with st.expander("📖 How to Use This Feature", expanded=False):
        st.markdown("""
        ### How Recipe Suggestions Work

        **Two ways to discover recipes:**

        **1. Personalized for You:**
        - Analyzes what you and similar home cooks enjoy
        - Suggests recipes tailored to your taste
        - Works best with users who have rated recipes before

        **2. Similar Recipe Search:**
        - Find recipes similar to ones you already love
        - Based on cooking time, ingredients, and health profile
        - Just search for any recipe by name

        **Steps to get suggestions:**
        1. Choose your preferred method (Personalized or Similar Recipes)
        2. Enter User ID or search for a recipe by name
        3. Choose how many suggestions you want (5-20)
        4. Enable "Add extra details" for cooking time and nutrition info
        5. Apply filters if needed (cooking time, health preferences)
        6. Click the button to get your suggestions!

        **Understanding your results:**
        - **Personalized**: Based on your taste and rating history
        - **Popular**: Well-loved recipes from the community
        - **Similar Recipes**: Comparable to your selected recipe
        - **Extra Details**: Includes time estimates and nutrition categories
        """)

    # Tabs for different recommendation modes
    tab1, tab2 = st.tabs(["👤 Personalized for You", "🍽️ Similar Recipes"])

    with tab1:
        # User-based recommendations (existing functionality)
        st.markdown("### 👤 Get Recipes Picked Just for You")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Your Preferences")

            user_id = st.number_input(
                "User ID",
                min_value=1,
                value=52282,
                help="Enter a user ID from the dataset",
                key="user_id_input"
            )

            top_n = st.slider(
                "Number of Suggestions",
                min_value=5,
                max_value=20,
                value=10,
                help="How many recipes to suggest",
                key="user_top_n"
            )

            enrich = st.checkbox(
                "Add extra details",
                value=True,
                help="Include cooking time, nutrition info, and categories",
                key="user_enrich"
            )

            # Sample user IDs
            st.markdown("#### 💡 Sample User IDs")
            st.markdown("""
            **High Activity:**
            - 424680 (7,671 interactions)
            - 37449 (5,603 interactions)

            **Medium Activity:**
            - 742802 (19 interactions)
            - 55536 (19 interactions)

            **Low Activity:**
            - 400008 (4 interactions)
            - 1190462 (4 interactions)
            """)

        with col2:
            st.markdown("#### 🔍 Filters (Optional)")

            use_filters = st.checkbox("Enable filters", value=False, key="user_filters")

            filters = {}
            if use_filters:
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    max_time = st.number_input(
                        "Max cooking time (min)",
                        min_value=10,
                        max_value=300,
                        value=60,
                        help="Filter recipes by maximum cooking time",
                        key="user_max_time"
                    )
                    filters['max_time'] = max_time

                with col_b:
                    require_high_protein = st.checkbox(
                        "High Protein only",
                        help="Only show high-protein recipes",
                        key="user_high_protein"
                    )
                    filters['require_high_protein'] = require_high_protein

                with col_c:
                    require_low_calorie = st.checkbox(
                        "Low Calorie only",
                        help="Only show low-calorie recipes",
                        key="user_low_calorie"
                    )
                    filters['require_low_calorie'] = require_low_calorie

        # Get Recommendations
        st.markdown("---")

        if st.button("🔍 Get My Recipe Suggestions", type="primary", use_container_width=True, key="user_recommend_btn"):

            with st.spinner("🔄 Finding recipes you'll love..."):
                try:
                    start_time = time.time()

                    result = pipeline.get_recommendations(
                        user_id=user_id,
                        top_n=top_n,
                        enrich=enrich,
                        filters=filters if use_filters else None
                    )

                    elapsed_time = time.time() - start_time

                    # Display results
                    st.success(f"✅ Found your recipes in {elapsed_time:.2f}s")

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("User ID", result['user_id'])
                    with col2:
                        st.metric("Type", result['recommendation_type'])
                    with col3:
                        st.metric("Fallback", "Yes" if result['fallback_used'] else "No")
                    with col4:
                        st.metric("Results", len(result['recommendations']))

                    # Historical recipes
                    if len(result['historical_recipes']) > 0:
                        with st.expander("📜 Your Historical Favorites", expanded=False):
                            st.dataframe(
                                result['historical_recipes'][['name', 'rating']].head(10),
                                use_container_width=True
                            )

                    # Recommendations
                    st.markdown("### 🎯 Your Personalized Recommendations")

                    recommendations = result['recommendations']

                    if len(recommendations) == 0:
                        st.warning("⚠️ No recipes match your filters. Try adjusting the criteria.")
                    else:
                        for idx, recipe in recommendations.iterrows():
                            display_recipe_card(recipe, enrich)

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.error(f"Recommendation error: {e}", exc_info=True)

    with tab2:
        # Recipe-based recommendations (new functionality)
        st.markdown("### 🍽️ Get Recommendations Based on a Specific Recipe")
        st.markdown("Search for a recipe you like, and we'll find similar recipes for you!")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🔍 Search for a Recipe")

            recipe_search = st.text_input(
                "Recipe Name",
                placeholder="e.g., chocolate cake, pasta carbonara, chicken soup...",
                help="Search for a recipe by name",
                key="recipe_search_input"
            )

        with col2:
            st.markdown("#### ⚙️ Settings")

            recipe_top_n = st.slider(
                "Number of Similar Recipes",
                min_value=5,
                max_value=20,
                value=10,
                help="How many similar recipes to show",
                key="recipe_top_n"
            )

            recipe_enrich = st.checkbox(
                "Enrich with ML predictions",
                value=True,
                help="Add time predictions, nutrition tags, and cluster assignments",
                key="recipe_enrich"
            )

        # Search and display recipe matches
        if recipe_search:
            with st.spinner("🔍 Searching for recipes..."):
                all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)
                matches = all_recipes[all_recipes['name'].str.contains(recipe_search, case=False, na=False)].head(20)

                if len(matches) == 0:
                    st.warning("⚠️ No recipes found. Try different keywords.")
                else:
                    st.markdown(f"### 📋 Found {len(matches)} matching recipes")

                    # Let user select a recipe
                    recipe_names = {f"{row['name']} (ID: {row['id']}, {row['minutes']}min)": idx
                                  for idx, row in matches.iterrows()}
                    selected_recipe = st.selectbox(
                        "Select a recipe to find similar ones:",
                        list(recipe_names.keys()),
                        key="recipe_select"
                    )

                    if selected_recipe:
                        # Filters for recipe-based recommendations
                        st.markdown("#### 🔍 Filters (Optional)")

                        use_recipe_filters = st.checkbox("Enable filters", value=False, key="recipe_filters")

                        recipe_filters = {}
                        if use_recipe_filters:
                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                recipe_max_time = st.number_input(
                                    "Max cooking time (min)",
                                    min_value=10,
                                    max_value=300,
                                    value=60,
                                    help="Filter recipes by maximum cooking time",
                                    key="recipe_max_time"
                                )
                                recipe_filters['max_time'] = recipe_max_time

                            with col_b:
                                recipe_high_protein = st.checkbox(
                                    "High Protein only",
                                    help="Only show high-protein recipes",
                                    key="recipe_high_protein"
                                )
                                recipe_filters['require_high_protein'] = recipe_high_protein

                            with col_c:
                                recipe_low_calorie = st.checkbox(
                                    "Low Calorie only",
                                    help="Only show low-calorie recipes",
                                    key="recipe_low_calorie"
                                )
                                recipe_filters['require_low_calorie'] = recipe_low_calorie

                        st.markdown("---")

                        if st.button("🎯 Find Similar Recipes", type="primary", use_container_width=True, key="recipe_recommend_btn"):

                            idx = recipe_names[selected_recipe]
                            selected_recipe_data = matches.loc[idx]

                            with st.spinner("🔄 Finding similar recipes..."):
                                try:
                                    start_time = time.time()

                                    # Get similar recipes based on content
                                    similar_recipes = get_similar_recipes(
                                        pipeline,
                                        selected_recipe_data,
                                        all_recipes,
                                        top_n=recipe_top_n,
                                        enrich=recipe_enrich,
                                        filters=recipe_filters if use_recipe_filters else None
                                    )

                                    elapsed_time = time.time() - start_time

                                    # Display results
                                    st.success(f"✅ Similar recipes found in {elapsed_time:.2f}s")

                                    # Metrics
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric("Base Recipe", selected_recipe_data['name'][:30] + "...")
                                    with col2:
                                        st.metric("Recommendation Type", "Content-Based")
                                    with col3:
                                        st.metric("Results", len(similar_recipes))

                                    # Show the selected recipe first
                                    st.markdown("### 🎯 Your Selected Recipe")
                                    display_recipe_card(selected_recipe_data, recipe_enrich)

                                    # Show similar recipes
                                    st.markdown("### 🍽️ Similar Recipes You Might Like")

                                    if len(similar_recipes) == 0:
                                        st.warning("⚠️ No similar recipes match your filters. Try adjusting the criteria.")
                                    else:
                                        for idx, recipe in similar_recipes.iterrows():
                                            display_recipe_card(recipe, recipe_enrich)

                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    logger.error(f"Recipe similarity error: {e}", exc_info=True)


def show_time_prediction_inference(pipeline):
    """Cooking time estimation with recipe details."""

    st.markdown('<p class="sub-header">⏰ Estimate Cooking Time</p>', unsafe_allow_html=True)

    # How to use
    with st.expander("📖 How to Use This Feature", expanded=False):
        st.markdown("""
        ### Cooking Time Estimator

        **What it does:**
        Estimates how long a recipe will take based on its details.

        **Two ways to use:**
        1. **Enter Your Recipe**: Input recipe details manually to get a time estimate
        2. **Search Database**: Find recipes in our collection and see time estimates

        **How it works:** Analyzes recipe complexity, cooking methods, and ingredient count
        """)

    if pipeline.time_model is None:
        st.warning("⚠️ Time estimation not available.")
        return

    st.success("✅ Time estimator ready")

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["✏️ Enter Your Recipe", "🔍 Search Database"])

    with tab1:
        st.markdown("### ✏️ Tell Us About Your Recipe")
        st.markdown("Enter basic details about your recipe to get a time estimate:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Basic Info**")
            n_steps = st.number_input("Number of Steps", min_value=1, max_value=100, value=8, help="How many steps does your recipe have?")
            n_ingredients = st.number_input("Number of Ingredients", min_value=1, max_value=100, value=10, help="How many ingredients?")

            st.markdown("**Recipe Type**")
            recipe_type = st.selectbox(
                "Course Type",
                ["Main Dish", "Appetizer", "Dessert", "Beverage", "Other"],
                help="What type of recipe is this?"
            )
            is_easy = st.checkbox("Easy/Simple Recipe", value=True)

        with col2:
            st.markdown("**Cooking Methods**")
            has_baking = st.checkbox("Involves Baking/Oven", value=False)
            has_frying = st.checkbox("Involves Frying", value=False)
            has_slow_cooking = st.checkbox("Slow Cooking Method", value=False)
            has_waiting_time = st.checkbox("Has Waiting Time (rest/chill/marinate)", value=False)

        with col3:
            st.markdown("**Time Estimate**")
            explicit_time = st.number_input(
                "Expected Time (minutes)",
                min_value=0,
                value=0,
                help="If you know roughly how long this should take, enter it here. This greatly improves accuracy!"
            )

            if explicit_time == 0:
                st.info("💡 Tip: Providing an estimated time significantly improves prediction accuracy!")

        if st.button("⏰ Estimate Time", type="primary", key="predict_new"):
            with st.spinner("Predicting cooking time..."):
                # IMPROVED: Estimate tag-based features from user input
                # This addresses the 86% importance of tag_time_limit and has_time_tag

                try:
                    # Build step text based on user inputs to capture cooking methods
                    step_texts = []
                    for i in range(n_steps):
                        step = f"Step {i+1}"
                        # Add cooking method keywords to steps so feature engineering can detect them
                        if i == 0 and has_baking:
                            step += " bake in oven"
                        if i == 1 and has_frying:
                            step += " fry"
                        if i == 2 and has_slow_cooking:
                            step += " slow cook"
                        if i == 3 and has_waiting_time:
                            step += " let rest"
                        if explicit_time > 0 and i == 0:
                            step += f" for {explicit_time} minutes"
                        step_texts.append(step)

                    # Build ingredient list
                    ingredient_texts = [f"Ingredient {i+1}" for i in range(n_ingredients)]

                    # Build tags based on user input - CRITICAL for model accuracy!
                    tags = []

                    # 1. Add difficulty tag
                    if is_easy:
                        tags.append('easy')

                    # 2. Add course type tags
                    if recipe_type == "Appetizer":
                        tags.append('appetizer')
                    elif recipe_type == "Dessert":
                        tags.append('dessert')
                    elif recipe_type == "Main Dish":
                        tags.append('main-dish')
                    elif recipe_type == "Beverage":
                        tags.append('beverage')

                    # 3. CRITICAL: Estimate time tag (86% of model importance!)
                    estimated_time = explicit_time
                    if estimated_time == 0:
                        # Estimate time based on recipe complexity
                        # Simple heuristics based on training data patterns
                        base_time = 15 + (n_steps * 5) + (n_ingredients * 2)

                        # Adjust for cooking methods
                        if has_slow_cooking:
                            base_time = max(base_time, 240)  # At least 4 hours
                        if has_baking:
                            base_time = max(base_time, 40)  # At least 40 min
                        if has_waiting_time:
                            base_time += 30  # Add waiting time

                        estimated_time = base_time

                    # Add time limit tag in the format the model expects
                    # Use common time brackets from training data
                    if estimated_time <= 15:
                        tags.append('15-minutes-or-less')
                    elif estimated_time <= 30:
                        tags.append('30-minutes-or-less')
                    elif estimated_time <= 60:
                        tags.append('60-minutes-or-less')
                    elif estimated_time <= 120:
                        tags.append('2-hours-or-less')
                    elif estimated_time <= 240:
                        tags.append('4-hours-or-less')
                    else:
                        tags.append('8-hours-or-less')

                    # Create minimal recipe DataFrame with fields expected by the pipeline
                    recipe_df = pd.DataFrame([{
                        'id': 999999,  # dummy ID
                        'name': f'User Recipe: {n_steps} steps, {n_ingredients} ingredients',
                        'minutes': 0,  # will be predicted
                        'contributor_id': 0,
                        'submitted': '2024-01-01',
                        'tags': tags,  # Now includes estimated time tags!
                        'nutrition': [0, 0, 0, 0, 0, 0, 0],  # dummy nutrition
                        'n_steps': n_steps,
                        'steps': step_texts,
                        'description': '',
                        'ingredients': ingredient_texts,
                        'n_ingredients': n_ingredients
                    }])

                    # Use the SAME pipeline method as for existing recipes
                    # This ensures feature engineering is done correctly
                    enriched_df = pipeline._add_time_predictions(recipe_df)
                    prediction = enriched_df['predicted_time'].iloc[0]

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Result")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("🔮 Predicted Time", f"{prediction:.0f} minutes")

                    with col2:
                        hours = int(prediction // 60)
                        mins = int(prediction % 60)
                        st.metric("⏰ Time Format", f"{hours}h {mins}m" if hours > 0 else f"{mins}m")

                    with col3:
                        if prediction < 30:
                            difficulty = "Quick"
                            color = "🟢"
                        elif prediction < 60:
                            difficulty = "Moderate"
                            color = "🟡"
                        else:
                            difficulty = "Time-Intensive"
                            color = "🔴"
                        st.metric("⏱️ Category", f"{color} {difficulty}")

                    # Show recipe summary
                    st.markdown("### 📝 Recipe Summary")
                    st.markdown(f"""
                    - **Steps:** {n_steps}
                    - **Ingredients:** {n_ingredients}
                    - **Cooking Methods:** {', '.join([m for m, v in [('Baking', has_baking), ('Frying', has_frying), ('Slow Cooking', has_slow_cooking)] if v]) or 'Standard'}
                    - **Complexity:** {"Simple" if is_easy else "Moderate"}
                    """)

                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    logger.error(f"Time prediction error: {e}", exc_info=True)

    with tab2:
        st.markdown("### 🔍 Find an Existing Recipe")

        search_term = st.text_input(
            "Search recipe name:",
            placeholder="e.g., chocolate cake, pasta, chicken...",
            key="search_existing"
        )

        if search_term:
            with st.spinner("Searching recipes..."):
                all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)
                matches = all_recipes[all_recipes['name'].str.contains(search_term, case=False, na=False)].head(10)

                if len(matches) == 0:
                    st.warning("No recipes found. Try different keywords.")
                else:
                    st.markdown(f"### 📋 Found {len(matches)} recipes")

                    # Let user select a recipe
                    recipe_names = {f"{row['name']} (ID: {row['id']})": idx for idx, row in matches.iterrows()}
                    selected = st.selectbox("Select a recipe:", list(recipe_names.keys()))

                    if selected and st.button("⏱️ Predict Cooking Time", type="primary", key="predict_existing"):
                        idx = recipe_names[selected]
                        recipe = matches.loc[idx]

                        with st.spinner("Predicting cooking time..."):
                            # Prepare recipe for prediction
                            recipe_df = pd.DataFrame([recipe])
                            enriched = pipeline._add_time_predictions(recipe_df)

                            st.markdown("### 📊 Prediction Result")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                actual_time = recipe.get('minutes', 'N/A')
                                st.metric("⏰ Actual Time", f"{actual_time:.0f} min" if pd.notna(actual_time) else "N/A")

                            with col2:
                                pred_time = enriched['predicted_time'].iloc[0]
                                st.metric("🔮 Predicted Time", f"{pred_time:.0f} min")

                            with col3:
                                if pd.notna(actual_time):
                                    error = abs(pred_time - actual_time)
                                    st.metric("📏 Error", f"{error:.0f} min")

                            # Recipe details
                            st.markdown("### 🍽️ Recipe Details")
                            display_recipe_card(enriched.iloc[0], show_predictions=True)


def show_nutrition_tagging_inference(pipeline):
    """Nutrition information and categorization."""

    st.markdown('<p class="sub-header">🥗 Nutrition Information</p>', unsafe_allow_html=True)

    # How to use
    with st.expander("📖 How to Use This Feature", expanded=False):
        st.markdown("""
        ### Nutrition Categorizer

        **What it does:**
        Tells you the health profile of any recipe.

        **6 Health Categories:**
        - High Calorie, Low Calorie, High Protein, Low Fat, Low Sugar, Healthy Recipe

        **Two ways to use:**
        1. **Enter Nutrition Values**: Input nutritional information manually
        2. **Search Database**: Find recipes and see their health profiles

        **How it works:** Analyzes calories, macronutrients, and nutritional balance
        """)

    if not pipeline.nutrition_models:
        st.warning("⚠️ Nutrition information not available.")
        return

    st.success(f"✅ Nutrition categorizer ready")

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["✏️ Enter Nutrition Values", "🔍 Search Database"])

    with tab1:
        st.markdown("### ✏️ Enter Nutritional Information")
        st.markdown("Enter the nutritional values for your recipe to get predicted nutrition tags:")

        col1, col2 = st.columns(2)

        with col1:
            calories = st.number_input("Calories", min_value=0, max_value=5000, value=300, help="Total calories per serving")
            protein = st.number_input("Protein (g)", min_value=0.0, max_value=500.0, value=15.0, help="Protein in grams")
            total_fat = st.number_input("Total Fat (g)", min_value=0.0, max_value=500.0, value=10.0, help="Total fat in grams")
            saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, max_value=200.0, value=3.0, help="Saturated fat in grams")

        with col2:
            carbs = st.number_input("Carbohydrates (g)", min_value=0.0, max_value=500.0, value=40.0, help="Total carbs in grams")
            sugar = st.number_input("Sugar (g)", min_value=0.0, max_value=200.0, value=5.0, help="Sugar in grams")
            sodium = st.number_input("Sodium (mg)", min_value=0, max_value=10000, value=400, help="Sodium in milligrams")
            n_ingredients = st.number_input("Number of Ingredients", min_value=1, max_value=100, value=10, help="For context")

        if st.button("🏷️ Analyze Nutrition", type="primary", key="predict_nutrition_new"):
            with st.spinner("Predicting nutrition tags..."):
                try:
                    # Create feature dict based on user input
                    # Convert to PDV (Percent Daily Value) format
                    nutrition_data = {
                        'calories': calories,
                        'protein_pdv': protein,  # Simplified - treating as PDV directly
                        'total_fat_pdv': total_fat,
                        'saturated_fat_pdv': saturated_fat,
                        'carbs_pdv': carbs,
                        'sugar_pdv': sugar,
                        'sodium_pdv': sodium / 24,  # Convert mg to approximate PDV (2400mg = 100%)
                        'n_ingredients': n_ingredients,
                    }

                    # Calculate derived features
                    nutrition_data['calories_per_ingredient'] = calories / max(n_ingredients, 1)
                    nutrition_data['protein_to_carb_ratio'] = protein / max(carbs, 1)
                    nutrition_data['protein_to_fat_ratio'] = protein / max(total_fat, 1)
                    nutrition_data['saturated_fat_pct'] = (saturated_fat / max(total_fat, 1)) * 100

                    # Macronutrient calories
                    protein_cals = protein * 4
                    carb_cals = carbs * 4
                    fat_cals = total_fat * 9
                    macro_total = max(protein_cals + carb_cals + fat_cals, 1)

                    nutrition_data['protein_pct'] = (protein_cals / macro_total) * 100
                    nutrition_data['carbs_pct'] = (carb_cals / macro_total) * 100
                    nutrition_data['fat_pct'] = (fat_cals / macro_total) * 100
                    nutrition_data['nutritional_density'] = protein / max(calories, 1)

                    # Health scores
                    nutrition_data['healthiness_score'] = max(0, protein * 0.3 - sugar * 0.25 - saturated_fat * 0.25 - (sodium/24) * 0.2)
                    nutrition_data['macro_balance_score'] = 100 - (abs(nutrition_data['protein_pct'] - 30) +
                                                                   abs(nutrition_data['fat_pct'] - 30) +
                                                                   abs(nutrition_data['carbs_pct'] - 40)) / 3

                    # Fill in other features with defaults
                    feature_defaults = {
                        'has_protein_ingredients': 1 if protein > 15 else 0,
                        'has_dairy': 0,
                        'has_vegetables': 0,
                        'has_fruits': 0,
                        'has_grains': 0,
                        'has_meat': 1 if protein > 20 else 0,
                        'is_vegetarian': 0,
                        'is_vegan': 0,
                        'is_low_carb': 1 if carbs < 20 else 0,
                        'is_gluten_free': 0,
                        'nutrition_pc1': 0,
                        'nutrition_pc2': 0,
                        'nutrition_pc3': 0,
                    }
                    nutrition_data.update(feature_defaults)

                    # Create DataFrame
                    recipe_df = pd.DataFrame([nutrition_data])

                    # Predict tags using the nutrition models
                    result_df = recipe_df.copy()
                    for tag_name, model in pipeline.nutrition_models.items():
                        try:
                            model_features = model.feature_names
                            # Ensure all features exist
                            for feat in model_features:
                                if feat not in result_df.columns:
                                    result_df[feat] = 0

                            X = result_df[model_features]
                            prediction = model.predict(X, scale_features=True)[0]
                            result_df[tag_name] = prediction
                        except Exception as e:
                            logger.warning(f"Could not predict {tag_name}: {e}")
                            result_df[tag_name] = 0

                    st.markdown("---")
                    st.markdown("### 📊 Nutritional Information")

                    # Display input values
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔥 Calories", f"{calories}")
                    with col2:
                        st.metric("💪 Protein", f"{protein}g")
                    with col3:
                        st.metric("🥑 Fat", f"{total_fat}g")
                    with col4:
                        st.metric("🍬 Sugar", f"{sugar}g")

                    # Show predicted tags
                    st.markdown("### 🏷️ Predicted Nutrition Tags")

                    tag_labels = {
                        'high_calorie': ('🔴 High Calorie', 'More than 500 calories'),
                        'low_calorie': ('🟢 Low Calorie', 'Less than 200 calories'),
                        'high_protein': ('💪 High Protein', 'More than 30g protein PDV'),
                        'low_fat': ('✅ Low Fat', 'Less than 10g fat PDV'),
                        'low_sugar': ('🍬 Low Sugar', 'Less than 10g sugar PDV'),
                        'healthy_recipe': ('🥗 Healthy Recipe', 'Overall health score')
                    }

                    cols = st.columns(3)
                    col_idx = 0

                    for tag, (label, description) in tag_labels.items():
                        if tag in result_df.columns:
                            with cols[col_idx % 3]:
                                is_tagged = result_df[tag].iloc[0] == 1
                                if is_tagged:
                                    st.success(f"**{label}**")
                                    st.caption(description)
                                else:
                                    st.info(f"**Not {label.split()[1]}**")
                            col_idx += 1

                    # Show macronutrient breakdown
                    st.markdown("### 📊 Macronutrient Breakdown")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Protein %", f"{nutrition_data['protein_pct']:.1f}%")
                    with col2:
                        st.metric("Carbs %", f"{nutrition_data['carbs_pct']:.1f}%")
                    with col3:
                        st.metric("Fat %", f"{nutrition_data['fat_pct']:.1f}%")

                    # Health scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Healthiness Score", f"{nutrition_data['healthiness_score']:.1f}")
                    with col2:
                        st.metric("Macro Balance Score", f"{nutrition_data['macro_balance_score']:.1f}")

                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    logger.error(f"Nutrition prediction error: {e}", exc_info=True)

    with tab2:
        st.markdown("### 🔍 Find an Existing Recipe")

        search_term = st.text_input(
            "Search recipe name:",
            placeholder="e.g., salad, burger, smoothie...",
            key="nutrition_search"
        )

        if search_term:
            with st.spinner("Searching recipes..."):
                all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)
                matches = all_recipes[all_recipes['name'].str.contains(search_term, case=False, na=False)].head(10)

                if len(matches) == 0:
                    st.warning("No recipes found. Try different keywords.")
                else:
                    st.markdown(f"### 📋 Found {len(matches)} recipes")

                    # Let user select a recipe
                    recipe_names = {f"{row['name']} (ID: {row['id']})": idx for idx, row in matches.iterrows()}
                    selected = st.selectbox("Select a recipe:", list(recipe_names.keys()), key="nutrition_select")

                    if selected and st.button("🏷️ Predict Nutrition Tags", type="primary", key="predict_nutrition_existing"):
                        idx = recipe_names[selected]
                        recipe = matches.loc[idx]

                        with st.spinner("Predicting nutrition tags..."):
                            # Prepare recipe for prediction
                            recipe_df = pd.DataFrame([recipe])
                            enriched = pipeline._add_nutrition_tags(recipe_df)

                            st.markdown("### 📊 Nutrition Tags")

                            # Show nutritional values
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                calories = recipe.get('calories', 'N/A')
                                cal_display = f"{float(calories):.0f}" if pd.notna(calories) and isinstance(calories, (int, float)) else "N/A"
                                st.metric("🔥 Calories", cal_display)

                            with col2:
                                protein = recipe.get('protein', 'N/A')
                                prot_display = f"{float(protein):.0f}g" if pd.notna(protein) and isinstance(protein, (int, float)) else "N/A"
                                st.metric("💪 Protein", prot_display)

                            with col3:
                                fat = recipe.get('total_fat', 'N/A')
                                fat_display = f"{float(fat):.0f}g" if pd.notna(fat) and isinstance(fat, (int, float)) else "N/A"
                                st.metric("🥑 Fat", fat_display)

                            with col4:
                                sugar = recipe.get('sugar', 'N/A')
                                sugar_display = f"{float(sugar):.0f}g" if pd.notna(sugar) and isinstance(sugar, (int, float)) else "N/A"
                                st.metric("🍬 Sugar", sugar_display)

                            # Show predicted tags
                            st.markdown("### 🏷️ Predicted Nutrition Tags")

                            tag_labels = {
                                'high_calorie': ('🔴 High Calorie', 'More than 600 calories'),
                                'low_calorie': ('🟢 Low Calorie', 'Less than 200 calories'),
                                'high_protein': ('💪 High Protein', 'More than 25g protein'),
                                'low_fat': ('✅ Low Fat', 'Less than 10g fat'),
                                'low_sugar': ('🍬 Low Sugar', 'Less than 10g sugar'),
                                'healthy_recipe': ('🥗 Healthy Recipe', 'Overall health score')
                            }

                            cols = st.columns(3)
                            col_idx = 0

                            for tag, (label, description) in tag_labels.items():
                                if tag in enriched.columns:
                                    with cols[col_idx % 3]:
                                        is_tagged = enriched[tag].iloc[0] == 1
                                        if is_tagged:
                                            st.success(f"**{label}**")
                                            st.caption(description)
                                        else:
                                            st.info(f"**Not {label.split()[1:][0] if len(label.split()) > 1 else label}**")
                                    col_idx += 1

                            # Recipe details
                            st.markdown("### 🍽️ Recipe Details")
                            display_recipe_card(enriched.iloc[0], show_predictions=True)


def show_clustering_page(pipeline):
    """Recipe categorization and browsing page."""

    st.markdown('<p class="sub-header">📊 Recipe Categories</p>', unsafe_allow_html=True)

    # How to use
    with st.expander("📖 How to Use This Feature", expanded=False):
        st.markdown("""
        ### Recipe Categories

        **What it does:**
        Organizes recipes into categories based on cooking time, complexity, and health profile.

        **What we consider:**
        - Cooking time
        - Recipe complexity (steps and ingredients)
        - Cooking efficiency
        - Nutritional profile

        **Two ways to use:**
        1. **Find Your Recipe's Category**: Enter recipe details to see which category it fits
        2. **Browse Categories**: Explore different recipe types

        **How it helps:**
        Find recipes that match your available time, skill level, and health goals
        """)

    if pipeline.recipe_clusterer is None:
        st.warning("⚠️ Recipe categories not available.")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["✏️ Find My Category", "📈 Browse All Categories", "🥗 Nutrition Classification"])

    with tab1:
        st.markdown("### ✏️ Find Your Recipe's Cluster")
        st.markdown("Enter recipe characteristics to find which category it belongs to:")

        col1, col2 = st.columns(2)

        with col1:
            minutes = st.number_input("Cooking Time (minutes)", min_value=1, max_value=500, value=30, help="Total cooking time")
            n_steps = st.number_input("Number of Steps", min_value=1, max_value=100, value=8, help="How many steps?")
            n_ingredients = st.number_input("Number of Ingredients", min_value=1, max_value=100, value=10, help="How many ingredients?")

        with col2:
            calories = st.number_input("Calories (optional)", min_value=0, max_value=5000, value=300, help="For health category")
            protein = st.number_input("Protein (g, optional)", min_value=0.0, max_value=500.0, value=15.0)
            fat = st.number_input("Fat (g, optional)", min_value=0.0, max_value=500.0, value=10.0)

        if st.button("📂 Find Cluster", type="primary", key="predict_cluster"):
            with st.spinner("Finding cluster..."):
                try:
                    # Calculate features for clustering
                    import numpy as np

                    # Features used by clustering model
                    log_minutes = np.log1p(minutes)
                    time_complexity = n_steps * n_ingredients
                    efficiency = minutes / max(n_steps, 1)

                    # Simple health category calculation
                    # Based on calories and protein-to-fat ratio
                    if calories > 0:
                        if protein / max(fat, 1) > 2.0 and calories < 400:
                            health_category = 3  # Healthy
                        elif protein / max(fat, 1) > 1.5:
                            health_category = 2  # Moderate
                        elif calories > 600:
                            health_category = 1  # Indulgent
                        else:
                            health_category = 2  # Average
                    else:
                        health_category = 2

                    # Create feature dict
                    recipe_features = pd.DataFrame([{
                        'log_minutes': log_minutes,
                        'time_complexity': time_complexity,
                        'efficiency': efficiency,
                        'health_category': health_category
                    }])

                    # Predict cluster
                    result = pipeline.recipe_clusterer.predict(recipe_features)
                    cluster_id = result['cluster'].iloc[0]
                    cluster_name = result['cluster_name'].iloc[0]

                    st.markdown("---")
                    st.markdown("### 📂 Cluster Assignment")

                    # Display result
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Cluster ID", cluster_id)
                    with col2:
                        st.metric("Cluster Name", cluster_name)
                    with col3:
                        st.metric("Complexity Score", f"{time_complexity:.0f}")

                    # Show characteristics
                    st.markdown("### 📊 Recipe Characteristics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏰ Time", f"{minutes} min")
                    with col2:
                        st.metric("📝 Steps", n_steps)
                    with col3:
                        st.metric("🥕 Ingredients", n_ingredients)
                    with col4:
                        health_labels = {1: "🔴 Indulgent", 2: "🟡 Moderate", 3: "🟢 Healthy"}
                        st.metric("Health", health_labels.get(health_category, "N/A"))

                    # Cluster description
                    st.markdown("### 📝 Cluster Description")

                    cluster_descriptions = {
                        'Rapides': "Quick recipes that are fast to prepare, perfect for busy weeknights.",
                        'Lentes': "Longer recipes that require time and patience, often for special occasions.",
                        'Elaborees': "Elaborate recipes with multiple steps and techniques.",
                        'Moderees': "Moderate complexity recipes, balanced in time and effort.",
                        'Tres Saines': "Very healthy recipes with excellent nutritional profiles.",
                        'Saines': "Healthy recipes with good nutritional balance.",
                        'Gourmandes': "Indulgent, rich recipes meant for special treats.",
                        'Simples': "Simple recipes with few steps and ingredients.",
                        'Tres Complexes': "Very complex recipes requiring advanced techniques.",
                        'Complexes': "Complex recipes that challenge cooking skills.",
                        'Efficaces': "Efficient recipes that maximize output for time invested.",
                        'Equilibrees': "Well-balanced recipes in all aspects.",
                        'Standard': "Standard recipes without specific characteristics."
                    }

                    # Find description based on cluster name
                    description = "A recipe category with specific characteristics."
                    for key in cluster_descriptions:
                        if key in cluster_name:
                            description = cluster_descriptions[key]
                            break

                    st.info(f"**{cluster_name}**: {description}")

                    # Similar recipes info
                    st.markdown("### 🔍 What This Means")
                    st.markdown(f"""
                    Your recipe belongs to the **{cluster_name}** cluster, which means it shares characteristics with similar recipes in terms of:
                    - Preparation time and complexity
                    - Cooking efficiency
                    - Health profile

                    Recipes in this cluster typically have:
                    - Cooking time around **{minutes} minutes**
                    - Complexity level: **{time_complexity:.0f}** points
                    - Health category: **{health_labels.get(health_category, 'N/A')}**
                    """)

                except Exception as e:
                    st.error(f"❌ Clustering error: {e}")
                    logger.error(f"Clustering error: {e}", exc_info=True)

    with tab2:
        st.markdown("### 📊 Cluster Distribution Overview")

        # Load recipes with clusters
        try:
            # Show cluster information
            st.markdown("""
            **Recipe clusters group similar recipes by:**
            - Cooking time and complexity
            - Nutritional profile
            - Preparation efficiency
            - Health characteristics

            This helps you find recipes that match your preferences for:
            - Quick meals vs. elaborate dishes
            - Healthy vs. indulgent options
            - Simple vs. complex preparations
            """)

            # Sample visualization
            st.markdown("### 📈 Typical Cluster Distribution")

            # Create sample cluster data
            cluster_info = {
                'Cluster': ['Quick & Simple', 'Healthy & Balanced', 'Elaborate Meals', 'Desserts & Treats', 'Complex Dishes'],
                'Recipes': [45000, 38000, 32000, 28000, 25000],
                'Avg Time (min)': [25, 35, 65, 40, 80],
                'Avg Steps': [8, 10, 15, 12, 18]
            }

            cluster_df = pd.DataFrame(cluster_info)

            # Bar chart
            fig = px.bar(
                cluster_df,
                x='Cluster',
                y='Recipes',
                title="Recipe Distribution Across Clusters",
                color='Avg Time (min)',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Table view
            st.dataframe(cluster_df, use_container_width=True)

            # Cluster names from the model
            if hasattr(pipeline.recipe_clusterer, 'cluster_names'):
                st.markdown("### 🏷️ Actual Cluster Names")
                cluster_names_dict = pipeline.recipe_clusterer.cluster_names
                cluster_names_df = pd.DataFrame([
                    {"Cluster ID": k, "Cluster Name": v}
                    for k, v in cluster_names_dict.items()
                ])
                st.dataframe(cluster_names_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading cluster data: {e}")

    with tab3:
        st.markdown("### 🥗 Nutrition Classification")
        st.markdown("Classify recipes into 4 nutritional categories using supervised machine learning.")

        # Load nutrition classifier model
        @st.cache_resource
        def load_nutrition_classifier():
            """Load the nutrition classification model."""
            try:
                model_dir = Path('outputs/models')
                
                # Find latest nutrition classifier model
                metadata_files = list(model_dir.glob('nutrition_classifier_*_metadata.json'))
                
                if not metadata_files:
                    return None, None
                
                latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
                
                with open(latest, 'r') as f:
                    metadata = json.load(f)
                
                model_name = metadata['model_name']
                
                # Load model and scaler
                model = joblib.load(model_dir / f"{model_name}_model.pkl")
                scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")
                
                return model, scaler, metadata
            except Exception as e:
                logger.error(f"Error loading nutrition classifier: {e}")
                return None, None, None

        model, scaler, metadata = load_nutrition_classifier()

        if model is None:
            st.warning("⚠️ Nutrition classification model not available.")
            st.info("""
            To use this feature, you need to train the nutrition classifier first.
            
            Run the following command:
            ```bash
            python scripts/run_nutrition_pipeline.py --model-type random_forest
            ```
            """)
        else:
            st.success(f"✅ Nutrition classifier loaded successfully!")
            
            # Display model info
            with st.expander("📊 Model Information", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Model Type", metadata.get('model_type', 'N/A').replace('_', ' ').title())
                with col2:
                    st.metric("Features", metadata.get('n_features', 'N/A'))
                with col3:
                    st.metric("CV F1-Score", f"{metadata.get('cv_f1_mean', 0):.4f}")
                with col4:
                    st.metric("Categories", len(metadata.get('class_names', [])))
                
                st.markdown("**Category Labels:**")
                for i, class_name in enumerate(metadata.get('class_names', [])):
                    st.markdown(f"- **{i}**: {class_name}")

            st.markdown("---")

            # Input form for nutrition classification
            st.markdown("### 📝 Enter Recipe Nutrition Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Macronutrients**")
                calories = st.number_input("Calories", min_value=0, max_value=5000, value=300, key="nutr_cal")
                protein = st.number_input("Protein (g)", min_value=0.0, max_value=500.0, value=15.0, key="nutr_prot")
                total_fat = st.number_input("Total Fat (g)", min_value=0.0, max_value=500.0, value=10.0, key="nutr_fat")
                saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, max_value=200.0, value=3.0, key="nutr_sat")
            
            with col2:
                st.markdown("**Carbs & Sugar**")
                carbohydrates = st.number_input("Carbohydrates (g)", min_value=0.0, max_value=500.0, value=40.0, key="nutr_carb")
                sugar = st.number_input("Sugar (g)", min_value=0.0, max_value=200.0, value=5.0, key="nutr_sugar")
                sodium = st.number_input("Sodium (mg)", min_value=0, max_value=10000, value=400, key="nutr_sodium")
            
            with col3:
                st.markdown("**Recipe Details**")
                nutr_steps = st.number_input("Number of Steps", min_value=1, max_value=100, value=8, key="nutr_steps")
                nutr_ingredients = st.number_input("Number of Ingredients", min_value=1, max_value=100, value=10, key="nutr_ingr")
                nutr_minutes = st.number_input("Cooking Time (min)", min_value=1, max_value=1000, value=30, key="nutr_min")

            if st.button("🎯 Classify Nutrition Category", type="primary", use_container_width=True, key="classify_nutrition"):
                with st.spinner("Classifying recipe..."):
                    try:
                        # Calculate all required features
                        # PDV percentages (based on FDA daily values)
                        calories_pdv = (calories / 2000) * 100
                        fat_pdv = (total_fat / 78) * 100
                        saturated_fat_pdv = (saturated_fat / 20) * 100
                        sodium_pdv = (sodium / 2300) * 100
                        carbs_pdv = (carbohydrates / 275) * 100
                        sugar_pdv = (sugar / 50) * 100
                        protein_pdv = (protein / 50) * 100
                        
                        # Ratios
                        protein_density = protein / max(calories, 1) * 100
                        sugar_ratio = sugar / max(carbohydrates, 1) * 100
                        sodium_density = sodium / max(calories, 1)
                        saturated_fat_ratio = saturated_fat / max(total_fat, 1) * 100
                        
                        # Macro balance (ideal: 30% protein, 30% fat, 40% carbs)
                        protein_cal = protein * 4
                        carb_cal = carbohydrates * 4
                        fat_cal = total_fat * 9
                        total_cal = max(protein_cal + carb_cal + fat_cal, 1)
                        
                        protein_pct = (protein_cal / total_cal) * 100
                        carbs_pct = (carb_cal / total_cal) * 100
                        fat_pct = (fat_cal / total_cal) * 100
                        
                        macro_balance = 100 - (abs(protein_pct - 30) + abs(fat_pct - 30) + abs(carbs_pct - 40)) / 3
                        
                        # Health score (custom formula)
                        health_score = max(0, 
                            protein_pdv * 0.3 - 
                            sugar_pdv * 0.25 - 
                            saturated_fat_pdv * 0.25 - 
                            sodium_pdv * 0.2
                        )
                        
                        # Binary features
                        is_low_calorie = 1 if calories < 200 else 0
                        is_high_protein = 1 if protein_pdv > 30 else 0
                        is_low_fat = 1 if fat_pdv < 10 else 0
                        is_low_sodium = 1 if sodium_pdv < 20 else 0
                        is_low_sugar = 1 if sugar_pdv < 10 else 0
                        
                        # Complexity and interactions
                        complexity_score = (nutr_steps * nutr_ingredients) / max(nutr_minutes, 1)
                        calorie_protein_interaction = calories * protein_pdv
                        fat_sodium_interaction = fat_pdv * sodium_pdv
                        
                        # Create feature vector matching training data
                        features = {
                            'calories': calories,
                            'total_fat': total_fat,
                            'saturated_fat': saturated_fat,
                            'sodium': sodium,
                            'carbohydrates': carbohydrates,
                            'sugar': sugar,
                            'protein': protein,
                            'calories_pdv': calories_pdv,
                            'fat_pdv': fat_pdv,
                            'saturated_fat_pdv': saturated_fat_pdv,
                            'sodium_pdv': sodium_pdv,
                            'carbs_pdv': carbs_pdv,
                            'sugar_pdv': sugar_pdv,
                            'protein_pdv': protein_pdv,
                            'protein_density': protein_density,
                            'sugar_ratio': sugar_ratio,
                            'sodium_density': sodium_density,
                            'saturated_fat_ratio': saturated_fat_ratio,
                            'macro_balance': macro_balance,
                            'health_score': health_score,
                            'is_low_calorie': is_low_calorie,
                            'is_high_protein': is_high_protein,
                            'is_low_fat': is_low_fat,
                            'is_low_sodium': is_low_sodium,
                            'is_low_sugar': is_low_sugar,
                            'complexity_score': complexity_score,
                            'calorie_protein_interaction': calorie_protein_interaction,
                            'fat_sodium_interaction': fat_sodium_interaction,
                            'n_steps': nutr_steps,
                            'n_ingredients': nutr_ingredients,
                            'minutes': nutr_minutes
                        }
                        
                        # Create DataFrame with features in correct order
                        X_new = pd.DataFrame([features])[metadata['feature_names']]
                        
                        # Scale and predict
                        X_scaled = scaler.transform(X_new)
                        prediction = model.predict(X_scaled)[0]
                        probabilities = model.predict_proba(X_scaled)[0]
                        
                        # Get class name
                        class_names = metadata['class_names']
                        predicted_class = class_names[prediction]
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Classification Result")
                        
                        # Main result
                        result_colors = {
                            'Very Healthy': '🟢',
                            'Healthy': '🟡',
                            'Moderate': '🟠',
                            'Indulgent': '🔴'
                        }
                        
                        color_emoji = result_colors.get(predicted_class, '⚪')
                        
                        st.markdown(f"""
                        <div style='background-color: #E8F4F8; padding: 2rem; border-radius: 10px; border-left: 5px solid #4ECDC4; text-align: center;'>
                            <h1 style='color: #4ECDC4; font-size: 3rem; margin: 0;'>{color_emoji} {predicted_class}</h1>
                            <p style='font-size: 1.2rem; color: #666; margin-top: 0.5rem;'>Nutrition Category</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Confidence scores
                        st.markdown("### 📊 Confidence Scores")
                        
                        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(prob)
                            with col2:
                                st.markdown(f"**{class_name}**: {prob*100:.1f}%")
                        
                        # Nutritional breakdown
                        st.markdown("### 📈 Nutritional Breakdown")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**PDV Percentages**")
                            st.metric("Calories", f"{calories_pdv:.1f}%")
                            st.metric("Protein", f"{protein_pdv:.1f}%")
                            st.metric("Fat", f"{fat_pdv:.1f}%")
                        
                        with col2:
                            st.markdown("**Ratios**")
                            st.metric("Protein Density", f"{protein_density:.2f}")
                            st.metric("Sugar Ratio", f"{sugar_ratio:.1f}%")
                            st.metric("Sat. Fat Ratio", f"{saturated_fat_ratio:.1f}%")
                        
                        with col3:
                            st.markdown("**Scores**")
                            st.metric("Health Score", f"{health_score:.1f}")
                            st.metric("Macro Balance", f"{macro_balance:.1f}")
                            st.metric("Complexity", f"{complexity_score:.2f}")
                        
                        # Macronutrient distribution
                        st.markdown("### 🥧 Macronutrient Distribution")
                        
                        macro_data = pd.DataFrame({
                            'Nutrient': ['Protein', 'Carbs', 'Fat'],
                            'Percentage': [protein_pct, carbs_pct, fat_pct]
                        })
                        
                        fig = px.pie(
                            macro_data,
                            values='Percentage',
                            names='Nutrient',
                            title='Calorie Distribution by Macronutrient',
                            color_discrete_sequence=['#4ECDC4', '#FF6B6B', '#FFD93D']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("### 💡 Interpretation")
                        
                        interpretations = {
                            'Very Healthy': """
                            This recipe is classified as **Very Healthy** 🟢
                            
                            Characteristics:
                            - Excellent nutritional balance
                            - Low in unhealthy components (sugar, saturated fat, sodium)
                            - High in beneficial nutrients (protein, fiber)
                            - Ideal for regular consumption
                            """,
                            'Healthy': """
                            This recipe is classified as **Healthy** 🟡
                            
                            Characteristics:
                            - Good nutritional profile
                            - Moderate amounts of all nutrients
                            - Suitable for regular meals
                            - Generally balanced diet-friendly
                            """,
                            'Moderate': """
                            This recipe is classified as **Moderate** 🟠
                            
                            Characteristics:
                            - Average nutritional value
                            - May have some higher amounts of calories, fat, or sodium
                            - Enjoy in moderation
                            - Balance with healthier meals
                            """,
                            'Indulgent': """
                            This recipe is classified as **Indulgent** 🔴
                            
                            Characteristics:
                            - Higher in calories, fat, sugar, or sodium
                            - Meant for occasional enjoyment
                            - Special occasions or treats
                            - Balance with lighter meals throughout the day
                            """
                        }
                        
                        st.info(interpretations.get(predicted_class, ""))
                        
                        # Feature importance (if available)
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("### 🔍 Top Contributing Features")
                            
                            feature_importance = pd.DataFrame({
                                'Feature': metadata['feature_names'],
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig = px.bar(
                                feature_importance,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 10 Most Important Features for Classification',
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Classification error: {e}")
                        logger.error(f"Nutrition classification error: {e}", exc_info=True)
                        import traceback
                        st.code(traceback.format_exc())


def show_sentiment_page(pipeline):
    """Review analysis page."""

    st.markdown('<p class="sub-header">💭 Recipe Review Analysis</p>', unsafe_allow_html=True)

    # Check if sentiment analysis is enabled
    from config.config import SENTIMENT_CONFIG
    
    if not SENTIMENT_CONFIG.get("enabled", True):
        st.warning("⚠️ Sentiment Analysis is currently disabled to save memory.")
        st.info("""
        To enable sentiment analysis:
        1. Open `config/config.py`
        2. Set `SENTIMENT_CONFIG["enabled"] = True`
        3. Restart the Streamlit app
        
        **Note**: Enabling sentiment analysis will increase memory usage by ~500MB.
        """)
        return

    # How to use
    with st.expander("📖 How to Use This Feature", expanded=False):
        st.markdown("""
        ### Review Analyzer

        **What it does:**
        Tells you whether reviews are positive, neutral, or negative.

        **How it works:**
        - Reads and analyzes recipe reviews
        - Specialized in understanding food-related feedback
        - Shows confidence levels for each analysis

        **Two ways to use:**

        **Option 1: Write Your Own Review**
        1. Write your review in the text area
        2. Click "Predict Sentiment"
        3. See the model's prediction instantly

        **Option 2: Analyze Existing Reviews**
        1. Enter a Recipe ID
        2. Choose how many reviews to analyze (5-50)
        3. Click "Analyze Existing Reviews"
        4. View individual review sentiments and overall distribution

        **Understanding results:**
        - **Positive**: Happy, satisfied reviews
        - **Neutral**: Mixed or informational reviews
        - **Negative**: Disappointed or critical reviews
        - **Confidence**: How certain the model is (0-100%)
        """)

    # Section 1: Write Your Own Review
    st.markdown("### ✍️ Write Your Own Review")

    user_review = st.text_area(
        "Enter your review:",
        placeholder="Write your thoughts about a recipe here... (e.g., 'This dish was absolutely delicious! The flavors were amazing and it was easy to make.')",
        height=150,
        help="Write a review and the model will predict its sentiment"
    )

    @st.cache_resource(show_spinner="🔄 Loading sentiment analysis model...")
    def get_sentiment_analyzer():
        """Lazy load sentiment analyzer only when needed."""
        from src.sentiment_analysis import SentimentAnalyzer
        return SentimentAnalyzer

    if st.button("🔮 Predict Sentiment", type="primary", use_container_width=True):
        if user_review and len(user_review.strip()) > 0:
            with st.spinner("🔄 Analyzing your review..."):
                try:
                    SentimentAnalyzer = get_sentiment_analyzer()

                    # Get prediction
                    label, confidence = SentimentAnalyzer.predict_sentiment(user_review)

                    # Display result with styling
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Result")

                    # Sentiment emoji and color
                    sentiment_info = {
                        "positive": ("😊", "#2ecc71", "Positive"),
                        "neutral": ("😐", "#f39c12", "Neutral"),
                        "negative": ("😞", "#e74c3c", "Negative")
                    }

                    emoji, color, sentiment_label = sentiment_info.get(label.lower(), ("❓", "#95a5a6", label))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f'<div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">'
                            f'<h1 style="color: white; margin: 0;">{emoji}</h1>'
                            f'<h2 style="color: white; margin: 10px 0 0 0;">{sentiment_label.upper()}</h2>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.metric("Confidence Score", f"{confidence:.1%}")
                        st.progress(confidence)

                    # Display the review
                    st.markdown("### 📝 Your Review")
                    st.info(user_review)

                except Exception as e:
                    st.error(f"❌ Error analyzing sentiment: {e}")
                    logger.error(f"Custom sentiment analysis error: {e}", exc_info=True)
        else:
            st.warning("⚠️ Please write a review before predicting sentiment.")

    # Separator
    st.markdown("---")

    # Section 2: Analyze Existing Reviews
    st.markdown("### 🔍 Analyze Existing Reviews")

    col1, col2 = st.columns([1, 1])

    with col1:
        recipe_id = st.number_input(
            "Recipe ID",
            min_value=1,
            value=275022,
            help="Enter a recipe ID to analyze its reviews"
        )

    with col2:
        num_reviews = st.slider(
            "Number of reviews",
            min_value=5,
            max_value=50,
            value=10,
            help="How many reviews to analyze"
        )

    if st.button("🔍 Analyze Existing Reviews", type="primary", use_container_width=True, key="analyze_existing"):

        with st.spinner("🔄 Analyzing review sentiments..."):
            try:
                sentiment_result = pipeline.analyze_recipe_sentiment(recipe_id, limit=num_reviews)

                if sentiment_result['review_count'] == 0:
                    st.warning(f"⚠️ No reviews found for recipe {recipe_id}")
                else:
                    st.success(f"✅ Analyzed {sentiment_result['review_count']} reviews")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Reviews", sentiment_result['review_count'])
                    with col2:
                        st.metric("😊 Positive", sentiment_result['summary']['positive'])
                    with col3:
                        st.metric("😐 Neutral", sentiment_result['summary']['neutral'])
                    with col4:
                        st.metric("😞 Negative", sentiment_result['summary']['negative'])

                    # Pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=['Positive', 'Neutral', 'Negative'],
                        values=[
                            sentiment_result['summary']['positive'],
                            sentiment_result['summary']['neutral'],
                            sentiment_result['summary']['negative']
                        ],
                        marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c']),
                        hole=0.3
                    )])

                    fig.update_layout(title="Sentiment Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                    # Individual reviews
                    st.markdown("### 📝 Individual Reviews")

                    for i, review in enumerate(sentiment_result['sentiments'], 1):
                        sentiment_emoji = {
                            "positive": "😊",
                            "neutral": "😐",
                            "negative": "😞"
                        }[review['sentiment']]

                        with st.expander(
                            f"Review {i}: {sentiment_emoji} {review['sentiment'].upper()} "
                            f"(Rating: {review['rating']}/5, Confidence: {review['confidence']:.1%})"
                        ):
                            st.write(f"**Review:** {review['review']}")
                            st.progress(review['confidence'])

            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Sentiment analysis error: {e}", exc_info=True)


def show_chatbot_page(pipeline):
    """AI-powered recipe chatbot using RAG with Gemini."""

    st.markdown('<p class="sub-header">🤖 Recipe Chatbot - Ask Me Anything!</p>', unsafe_allow_html=True)

    # How to use
    with st.expander("📖 How to Use the Chatbot", expanded=False):
        st.markdown("""
        ### AI Recipe Assistant

        **What it does:**
        Chat with an AI assistant that knows about all 230,000+ recipes in our database!

        **How it works:**
        - Uses Google's Gemini Flash 2.5 model
        - Searches relevant recipes based on your question
        - Provides accurate, context-aware answers

        **Example questions:**
        - "What are some quick vegetarian recipes under 30 minutes?"
        - "Show me healthy low-calorie desserts"
        - "What recipes use chicken and broccoli?"
        - "Find me baking recipes with chocolate"

        **Tips:**
        - Be specific in your questions
        - Ask about ingredients, cooking times, or dietary preferences
        - The chatbot remembers your conversation context
        """)

    # Check if API key is configured
    api_key = os.environ.get("GEMINI_API_KEY") or CHATBOT_CONFIG.get("gemini_api_key")

    if not api_key or api_key.strip() == "":
        st.warning("⚠️ Gemini API Key not configured")
        st.markdown("""
        ### 🔑 How to Set Up the API Key

        **Option 1: Environment Variable (Recommended)**
        ```bash
        export GEMINI_API_KEY="your-api-key-here"
        ```

        **Option 2: Config File**
        Edit `config/config.py` and add your API key:
        ```python
        CHATBOT_CONFIG = {
            "gemini_api_key": "your-api-key-here",
            ...
        }
        ```

        **Get your API key:**
        1. Visit: https://makersuite.google.com/app/apikey
        2. Sign in with your Google account
        3. Create a new API key
        4. Copy and paste it using one of the methods above

        After adding your API key, restart the Streamlit app.
        """)
        return

    # Initialize chatbot
    try:
        if 'chatbot' not in st.session_state:
            with st.spinner("🔄 Initializing AI chatbot..."):
                from src.chatbot.rag_chatbot import RecipeRAGChatbot

                # Load recipes
                all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)

                # Create chatbot instance
                st.session_state.chatbot = RecipeRAGChatbot(api_key, all_recipes)
                st.session_state.chat_history = []

        chatbot = st.session_state.chatbot

        st.success(f"✅ Chatbot ready! Loaded {len(chatbot.recipes_df):,} recipes")

        # Display dataset stats
        col1, col2, col3, col4 = st.columns(4)
        stats = chatbot.get_recipe_stats()

        with col1:
            st.metric("Total Recipes", f"{stats['total_recipes']:,}")
        with col2:
            st.metric("Avg Cook Time", f"{stats['avg_time']:.0f} min")
        with col3:
            st.metric("Avg Steps", f"{stats['avg_steps']:.1f}")
        with col4:
            st.metric("Avg Ingredients", f"{stats['avg_ingredients']:.1f}")

        st.markdown("---")

        # Chat interface
        st.markdown("### 💬 Chat with the Recipe Assistant")

        # Display chat history
        if st.session_state.chat_history:
            for i, exchange in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**You:** {exchange['user']}")
                    st.markdown(f"**Assistant:** {exchange['assistant']}")
                    st.markdown("---")

        # User input
        user_input = st.text_area(
            "Your question:",
            placeholder="Ask me anything about recipes! (e.g., 'What are some quick dinner recipes?')",
            height=100,
            key="chat_input"
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            send_button = st.button("📤 Send", type="primary", use_container_width=True)

        with col2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        if send_button and user_input.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get response from chatbot
                    response = chatbot.chat(user_input, st.session_state.chat_history)

                    # Add to history
                    st.session_state.chat_history.append({
                        'user': user_input,
                        'assistant': response
                    })

                    # Limit history size
                    max_history = CHATBOT_CONFIG.get('max_history', 10)
                    if len(st.session_state.chat_history) > max_history:
                        st.session_state.chat_history = st.session_state.chat_history[-max_history:]

                    # Rerun to display new message
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Chatbot error: {e}", exc_info=True)

        # Quick example questions
        st.markdown("### 💡 Try These Example Questions")

        example_questions = [
            "What are some quick recipes under 20 minutes?",
            "Show me healthy high-protein recipes",
            "Find recipes with chicken and vegetables",
            "What are popular dessert recipes?",
            "Give me easy recipes for beginners"
        ]

        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            response = chatbot.chat(question, st.session_state.chat_history)
                            st.session_state.chat_history.append({
                                'user': question,
                                'assistant': response
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        logger.error(f"Chatbot initialization error: {e}", exc_info=True)
        st.markdown("""
        **Troubleshooting:**
        - Make sure `google-generativeai` package is installed
        - Verify your API key is correct
        - Check your internet connection
        - See logs for more details
        """)


def show_analytics_page(pipeline):
    """Dataset insights and visualizations."""

    st.markdown('<p class="sub-header">📈 Recipe Collection Insights</p>', unsafe_allow_html=True)

    with st.spinner("📊 Loading dataset analytics..."):
        try:
            all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)

            # Basic statistics
            st.markdown("### 📊 Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Recipes", f"{len(all_recipes):,}")
            with col2:
                st.metric("Median Time", f"{all_recipes['minutes'].median():.0f} min")
            with col3:
                st.metric("Avg Steps", f"{all_recipes['n_steps'].mean():.1f}")
            with col4:
                st.metric("Avg Ingredients", f"{all_recipes['n_ingredients'].mean():.1f}")

            # Visualizations
            st.markdown("### 📊 Data Visualizations")

            tab1, tab2, tab3, tab4 = st.tabs([
                "⏰ Time Distribution",
                "📈 Complexity",
                "🥗 Nutrition",
                "📊 Correlations"
            ])

            with tab1:
                # Time distribution
                fig = px.histogram(
                    all_recipes[all_recipes['minutes'] < 200],
                    x='minutes',
                    nbins=50,
                    title="Distribution of Cooking Times",
                    labels={'minutes': 'Cooking Time (minutes)', 'count': 'Number of Recipes'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Complexity scatter
                sample = all_recipes.sample(min(5000, len(all_recipes)), random_state=42)
                fig = px.scatter(
                    sample,
                    x='n_ingredients',
                    y='n_steps',
                    color='minutes',
                    title="Recipe Complexity: Steps vs Ingredients",
                    labels={
                        'n_ingredients': 'Number of Ingredients',
                        'n_steps': 'Number of Steps',
                        'minutes': 'Time (min)'
                    },
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                # Nutrition box plots
                if 'calories' in all_recipes.columns:
                    fig = px.box(
                        all_recipes[all_recipes['calories'] < 1000],
                        y='calories',
                        title="Calorie Distribution",
                        labels={'calories': 'Calories'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                # Correlations
                numeric_cols = ['minutes', 'n_steps', 'n_ingredients', 'calories']
                if all(col in all_recipes.columns for col in numeric_cols):
                    corr = all_recipes[numeric_cols].corr()

                    fig = px.imshow(
                        corr,
                        title="Feature Correlations",
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu',
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading analytics: {e}")


def show_about_page():
    """About page with helpful information."""

    st.markdown('<p class="sub-header">ℹ️ About MangeTaMain</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 What is This?",
        "🎯 Features",
        "💡 How to Use",
        "❓ Help & FAQ"
    ])

    with tab1:
        st.markdown("""
        ## 🎯 What is MangeTaMain?

        MangeTaMain is your **smart cooking companion** that helps you discover recipes you'll love!
        We analyze over 230,000 recipes and 1 million user reviews to provide personalized suggestions
        and helpful cooking information.

        ### 🌟 What We Offer

        - ✅ **Personalized Suggestions** based on your taste
        - ✅ **Smart Filtering** by cooking time and health needs
        - ✅ **Time Estimates** to plan your meals
        - ✅ **Nutrition Information** for healthy choices
        - ✅ **Review Analysis** to see what others think
        - ✅ **Recipe Categories** organized by time and complexity
        - ✅ **Helpful Insights** from our recipe collection

        ### 📊 Our Recipe Collection

        - **Recipes:** 230,000+ recipes from Food.com
        - **Reviews:** 1,000,000+ user ratings and feedback
        - **Details:** Ingredients, steps, nutrition, cooking time
        - **History:** 18 years of recipe data (2000-2018)

        ### 👨‍💻 About This Project

        MangeTaMain was created to make cooking easier and more enjoyable by helping you
        find the perfect recipes for any occasion.
        """)

    with tab2:
        st.markdown("""
        ## 🎯 Our Features

        ### 1. 🎯 Recipe Finder

        **What it does:** Suggests recipes you'll love

        **How it works:**
        - Analyzes what you and similar users enjoy
        - Learns your taste preferences over time
        - Falls back to popular recipes for new users
        - Works best after rating a few recipes

        **Speed:** Instant suggestions (less than 1 second)

        ---

        ### 2. ⏰ Cooking Time Estimator

        **What it does:** Estimates how long recipes take

        **What it considers:**
        - Number of steps and ingredients
        - Cooking methods (baking, frying, grilling, etc.)
        - Equipment needed
        - Recipe complexity

        **Accuracy:** Typically within 15 minutes of actual time

        ---

        ### 3. 🥗 Nutrition Categorizer

        **What it does:** Identifies recipe health profiles

        **Categories:**
        - High/Low Calorie
        - High Protein
        - Low Fat
        - Low Sugar
        - Overall Healthy

        **Accuracy:** Very reliable (80-95% accurate)

        ---

        ### 4. 📊 Recipe Categories

        **What it does:** Organizes recipes by type

        **How it groups recipes:**
        - By cooking time (quick vs. slow)
        - By complexity (simple vs. elaborate)
        - By cooking efficiency
        - By health profile

        **Result:** Easy browsing of similar recipe types

        ---

        ### 5. 💭 Review Analyzer

        **What it does:** Understands recipe reviews

        **What it tells you:**
        - Whether reviews are positive, neutral, or negative
        - How confident the analysis is
        - Overall sentiment for each recipe

        **Accuracy:** Very reliable (over 90% accurate)
        """)

    with tab3:
        st.markdown("""
        ## 💡 How to Use Each Feature

        ### 🎯 Personalized Recommendations

        1. Navigate to "Personalized Recommendations"
        2. Enter your User ID (or try sample IDs)
        3. Choose number of recommendations (5-20)
        4. Enable enrichment for ML predictions
        5. Optionally apply filters
        6. Click "Get Recommendations"

        **Best for:** Finding new recipes based on your taste

        ---

        ### ⏰ Time Prediction

        1. Go to "Time Prediction Models"
        2. Enter recipe characteristics
        3. Click "Predict Cooking Time"
        4. Compare predictions from 4 models
        5. Use for meal planning

        **Best for:** Estimating cooking time for meal planning

        ---

        ### 🥗 Nutrition Tagging

        1. Navigate to "Nutrition Tagging"
        2. Enter nutritional values
        3. Select tags to predict
        4. Choose models to compare
        5. Click "Predict Nutrition Tags"

        **Best for:** Understanding recipe nutrition profile

        ---

        ### 📊 Recipe Clustering

        1. Visit "Recipe Clustering"
        2. Explore cluster distribution
        3. Understand recipe groupings
        4. Find similar recipe categories

        **Best for:** Discovering recipe types and categories

        ---

        ### 💭 Sentiment Analysis

        1. Go to "Sentiment Analysis"
        2. Enter a Recipe ID
        3. Choose number of reviews
        4. Click "Analyze Sentiment"
        5. View sentiment distribution

        **Best for:** Gauging recipe quality from reviews

        ---

        ### 📈 Dataset Analytics

        1. Visit "Dataset Analytics"
        2. Explore various visualizations
        3. Understand dataset patterns
        4. View correlations and distributions

        **Best for:** Understanding the dataset and trends
        """)

    with tab4:
        st.markdown("""
        ## 🔧 Technical Details

        ### Technology Stack

        **Backend:**
        - Python 3.8+
        - scikit-learn (ML models)
        - pandas & numpy (data processing)
        - scipy (sparse matrices)

        **NLP:**
        - transformers (Hugging Face)
        - torch (PyTorch)
        - Pre-trained BERT models

        **Frontend:**
        - Streamlit (UI framework)
        - Plotly (visualizations)
        - Custom CSS styling

        ### Model Performance

        **SVD Recommender:**
        - Training time: ~2-3 minutes
        - Inference time: <1 second per user
        - Memory: ~500MB for full dataset

        **Time Prediction:**
        - Best model: Random Forest
        - R² score: ~0.70
        - MAE: ~15 minutes

        **Nutrition Tagging:**
        - Best overall: Random Forest
        - Average F1: ~0.85
        - Inference: <100ms per recipe

        **Clustering:**
        - K-Means with k=5-10
        - Silhouette score: ~0.45
        - PCA explained variance: >80%

        **Sentiment Analysis:**
        - Accuracy: >90%
        - Inference: ~200ms per review

        ### Data Pipeline

        1. **Data Loading:** CSV files → pandas DataFrames
        2. **Preprocessing:** Cleaning, parsing, type conversion
        3. **Feature Engineering:** 40+ derived features
        4. **Model Training:** Offline training with cross-validation
        5. **Model Serving:** Cached models for fast inference
        6. **Result Enrichment:** Combine multiple model outputs

        ### Caching & Optimization

        - `@st.cache_resource`: Model caching
        - `@st.cache_data`: Data caching
        - Lazy loading for large datasets
        - Efficient sparse matrix operations

        ### Version Information

        - **Version:** 2.0.0
        - **Last Updated:** 2025-10-27
        - **Models:** 5 ML systems, 27 total models
        - **Features:** 44 time features, 30+ nutrition features
        """)


def display_recipe_card(recipe, show_predictions=True):
    """Display a comprehensive recipe card."""

    with st.container():
        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)

        # Recipe name
        st.markdown(f"### 🍽️ {recipe['name']}")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            actual_time = recipe.get('minutes', 'N/A')
            st.metric("⏰ Actual Time", f"{actual_time} min" if actual_time != 'N/A' else 'N/A')

        with col2:
            if show_predictions and 'predicted_time' in recipe and pd.notna(recipe['predicted_time']):
                st.metric("🔮 Predicted", f"{recipe['predicted_time']:.0f} min")
            else:
                st.metric("📝 Steps", recipe.get('n_steps', 'N/A'))

        with col3:
            st.metric("🥕 Ingredients", recipe.get('n_ingredients', 'N/A'))

        with col4:
            calories = recipe.get('calories', 'N/A')
            cal_display = f"{float(calories):.0f}" if calories != 'N/A' and isinstance(calories, (int, float)) else 'N/A'
            st.metric("🔥 Calories", cal_display)

        # Description
        if 'description' in recipe and pd.notna(recipe['description']):
            with st.expander("📝 Description"):
                st.write(recipe['description'])

        # Nutrition tags
        if show_predictions:
            tag_columns = ['high_calorie', 'low_calorie', 'high_protein', 'low_fat', 'low_sugar', 'healthy_recipe']
            nutrition_tags = []

            for tag in tag_columns:
                if tag in recipe and recipe[tag] == 1:
                    nutrition_tags.append(tag.replace('_', ' ').title())

            if nutrition_tags:
                tags_html = " ".join([f'<span class="tag-badge">{tag}</span>' for tag in nutrition_tags])
                st.markdown(f"**Nutrition Tags:** {tags_html}", unsafe_allow_html=True)

            # Cluster info
            if 'cluster_name' in recipe and pd.notna(recipe['cluster_name']):
                st.markdown(f"**Category:** 📂 {recipe['cluster_name']}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")


if __name__ == "__main__":
    main()
