"""
MangeTaMain - Simple Cooking Recipe Finder
Find recipes based on what you want to cook!
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integration.recommendation_pipeline import IntegratedRecommendationPipeline

# Page configuration
st.set_page_config(
    page_title="MangeTaMain - Find Your Recipe",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recipe-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recipe-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .tag {
        display: inline-block;
        background-color: #4ECDC4;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_system():
    """Load the recipe system."""
    try:
        pipeline = IntegratedRecommendationPipeline(
            recipes_path="data/RAW_recipes.csv",
            interactions_path="data/RAW_interactions.csv",
            models_dir="outputs/models",
            load_models=True
        )

        if not pipeline.svd_trained:
            pipeline.train_svd_recommender(k=50)

        return pipeline
    except Exception as e:
        st.error(f"Error loading recipes: {e}")
        return None


def main():
    """Main app."""

    # Header
    st.markdown('<p class="main-title">🍳 MangeTaMain</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find the perfect recipe in seconds!</p>', unsafe_allow_html=True)

    # Load system
    pipeline = load_system()
    if pipeline is None:
        st.error("Cannot load recipes. Please check data files.")
        return

    # Main tabs - Simple navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Find Recipes", "⭐ Get Recommendations", "💬 Check Reviews"])

    with tab1:
        show_find_recipes(pipeline)

    with tab2:
        show_recommendations(pipeline)

    with tab3:
        show_reviews(pipeline)


def show_find_recipes(pipeline):
    """Simple recipe search."""

    st.header("🔍 Find Recipes")
    st.write("Tell us what you're looking for, and we'll find the perfect recipes!")

    # User inputs in columns
    col1, col2 = st.columns(2)

    with col1:
        # What do you want to cook?
        search_term = st.text_input(
            "🍽️ What do you want to cook?",
            placeholder="e.g., chocolate cake, pasta, chicken curry...",
            help="Type any dish name or ingredient"
        )

        # How much time?
        time_filter = st.selectbox(
            "⏰ How much time do you have?",
            ["Any time", "Quick (under 30 min)", "Moderate (30-60 min)", "I have time (60+ min)"]
        )

        # Dietary preference
        diet_pref = st.selectbox(
            "🥗 Any dietary preferences?",
            ["No preference", "Low calorie", "High protein", "Healthy options", "Low sugar"]
        )

    with col2:
        # Complexity
        complexity = st.selectbox(
            "👨‍🍳 Cooking skill level?",
            ["Any level", "Simple (few ingredients)", "Moderate", "I'm a pro!"]
        )

        # Number of results
        num_results = st.slider(
            "📊 How many recipes to show?",
            min_value=5,
            max_value=30,
            value=10
        )

    # Search button
    if st.button("🔍 Find My Recipes!", type="primary", use_container_width=True):

        with st.spinner("Searching for recipes..."):
            # Load all recipes
            all_recipes = pipeline.data_loader.load_all_recipes(use_cache=True)

            # Apply filters
            filtered = all_recipes.copy()

            # Search term filter
            if search_term:
                filtered = filtered[
                    filtered['name'].str.contains(search_term, case=False, na=False)
                ]

            # Time filter
            if time_filter == "Quick (under 30 min)":
                filtered = filtered[filtered['minutes'] <= 30]
            elif time_filter == "Moderate (30-60 min)":
                filtered = filtered[(filtered['minutes'] > 30) & (filtered['minutes'] <= 60)]
            elif time_filter == "I have time (60+ min)":
                filtered = filtered[filtered['minutes'] > 60]

            # Complexity filter
            if complexity == "Simple (few ingredients)":
                filtered = filtered[filtered['n_ingredients'] <= 8]
            elif complexity == "Moderate":
                filtered = filtered[(filtered['n_ingredients'] > 8) & (filtered['n_ingredients'] <= 12)]

            # Calorie filter for dietary preferences
            if diet_pref == "Low calorie" and 'calories' in filtered.columns:
                filtered = filtered[filtered['calories'] <= 400]
            elif diet_pref == "Low sugar" and 'sugar' in filtered.columns:
                filtered = filtered[filtered['sugar'] <= 20]

            # Show results
            if len(filtered) == 0:
                st.warning("😕 No recipes found. Try different criteria!")
            else:
                st.success(f"✅ Found {len(filtered)} recipes!")

                # Show top results
                results = filtered.head(num_results)

                for idx, recipe in results.iterrows():
                    display_simple_recipe(recipe)


def show_recommendations(pipeline):
    """Get personalized recommendations."""

    st.header("⭐ Get Personalized Recommendations")
    st.write("Answer a few questions and we'll recommend recipes you'll love!")

    # Create form for user preferences
    with st.form("preferences_form"):
        st.subheader("Tell us your preferences:")

        col1, col2 = st.columns(2)

        with col1:
            # User's past experience (optional)
            user_id = st.number_input(
                "👤 Your User ID (optional)",
                min_value=0,
                value=0,
                help="If you have a user ID from past visits, enter it. Otherwise, leave at 0."
            )

            # What they like
            preferred_time = st.selectbox(
                "⏰ Preferred cooking time?",
                ["Under 30 minutes", "30-45 minutes", "45-60 minutes", "Over 60 minutes", "No preference"]
            )

            dietary = st.multiselect(
                "🥗 Dietary needs? (select all that apply)",
                ["Low Calorie", "High Protein", "Low Fat", "Low Sugar", "Healthy"]
            )

        with col2:
            num_recipes = st.slider(
                "📊 How many recommendations?",
                min_value=5,
                max_value=20,
                value=10
            )

            skill = st.radio(
                "👨‍🍳 Your cooking experience?",
                ["Beginner", "Intermediate", "Advanced"]
            )

            show_predictions = st.checkbox(
                "📊 Show AI predictions (time, nutrition tags)",
                value=True
            )

        submit = st.form_submit_button("✨ Get My Recommendations!", use_container_width=True)

    if submit:
        with st.spinner("🤖 AI is finding perfect recipes for you..."):
            try:
                # Build filters based on preferences
                filters = {}

                # Time filter
                if preferred_time == "Under 30 minutes":
                    filters['max_time'] = 30
                elif preferred_time == "30-45 minutes":
                    filters['max_time'] = 45
                elif preferred_time == "45-60 minutes":
                    filters['max_time'] = 60

                # Dietary filters
                if "High Protein" in dietary:
                    filters['require_high_protein'] = True
                if "Low Calorie" in dietary:
                    filters['require_low_calorie'] = True

                # Get recommendations
                # If user_id is 0, use a random popular user for demo
                actual_user_id = user_id if user_id > 0 else 52282

                result = pipeline.get_recommendations(
                    user_id=actual_user_id,
                    top_n=num_recipes,
                    enrich=show_predictions,
                    filters=filters if filters else None
                )

                # Show results
                st.success(f"✅ Found {len(result['recommendations'])} recipes perfect for you!")

                if user_id == 0:
                    st.info("💡 Since you didn't provide a User ID, we showed popular recipes. Create an account to get truly personalized recommendations!")

                recommendations = result['recommendations']

                if len(recommendations) == 0:
                    st.warning("😕 No recipes match all your criteria. Try relaxing some filters.")
                else:
                    for idx, recipe in recommendations.iterrows():
                        display_detailed_recipe(recipe, show_predictions)

            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
                logger.error(f"Recommendation error: {e}", exc_info=True)


def show_reviews(pipeline):
    """Check recipe reviews."""

    st.header("💬 Check Recipe Reviews")
    st.write("See what others are saying about a recipe before you cook it!")

    col1, col2 = st.columns([2, 1])

    with col1:
        recipe_id = st.number_input(
            "📝 Enter Recipe ID",
            min_value=1,
            value=275022,
            help="You can find recipe IDs from the search results"
        )

    with col2:
        num_reviews = st.slider("Reviews to check", 5, 30, 10)

    if st.button("📊 Analyze Reviews", type="primary"):

        with st.spinner("Reading reviews..."):
            try:
                result = pipeline.analyze_recipe_sentiment(recipe_id, limit=num_reviews)

                if result['review_count'] == 0:
                    st.warning("No reviews found for this recipe.")
                else:
                    st.success(f"Analyzed {result['review_count']} reviews")

                    # Simple metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total", result['review_count'])
                    with col2:
                        st.metric("😊 Positive", result['summary']['positive'])
                    with col3:
                        st.metric("😐 Neutral", result['summary']['neutral'])
                    with col4:
                        st.metric("😞 Negative", result['summary']['negative'])

                    # Simple percentage
                    positive_pct = (result['summary']['positive'] / result['review_count']) * 100

                    if positive_pct >= 70:
                        st.success(f"✅ {positive_pct:.0f}% positive reviews - This recipe is highly rated!")
                    elif positive_pct >= 50:
                        st.info(f"👍 {positive_pct:.0f}% positive reviews - Most people liked it")
                    else:
                        st.warning(f"⚠️ Only {positive_pct:.0f}% positive reviews - You might want to try another recipe")

                    # Show some reviews
                    with st.expander("📖 Read Some Reviews"):
                        for i, review in enumerate(result['sentiments'][:5], 1):
                            sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}[review['sentiment']]
                            st.write(f"{sentiment_emoji} **{review['sentiment'].title()}** (Rating: {review['rating']}/5)")
                            st.write(f"_{review['review']}_")
                            st.write("---")

            except Exception as e:
                st.error(f"Error analyzing reviews: {e}")


def display_simple_recipe(recipe):
    """Display a simple recipe card."""

    st.markdown('<div class="recipe-box">', unsafe_allow_html=True)

    # Title
    st.markdown(f'<p class="recipe-title">🍽️ {recipe["name"]}</p>', unsafe_allow_html=True)

    # Quick info in one line
    info_parts = []

    if 'minutes' in recipe and pd.notna(recipe['minutes']):
        time_val = int(recipe['minutes'])
        if time_val < 30:
            info_parts.append(f"⚡ {time_val} min (Quick!)")
        else:
            info_parts.append(f"⏰ {time_val} min")

    if 'n_ingredients' in recipe and pd.notna(recipe['n_ingredients']):
        ing_val = int(recipe['n_ingredients'])
        info_parts.append(f"🛒 {ing_val} ingredients")

    if 'n_steps' in recipe and pd.notna(recipe['n_steps']):
        steps_val = int(recipe['n_steps'])
        info_parts.append(f"📋 {steps_val} steps")

    if 'calories' in recipe and pd.notna(recipe['calories']):
        cal_val = int(recipe['calories'])
        if cal_val < 300:
            info_parts.append(f"🟢 {cal_val} cal (Light)")
        elif cal_val < 600:
            info_parts.append(f"🟡 {cal_val} cal")
        else:
            info_parts.append(f"🔴 {cal_val} cal")

    st.write(" | ".join(info_parts))

    # Description if available
    if 'description' in recipe and pd.notna(recipe['description']) and str(recipe['description']).strip():
        with st.expander("📖 Description"):
            st.write(recipe['description'])

    st.markdown('</div>', unsafe_allow_html=True)


def display_detailed_recipe(recipe, show_predictions=True):
    """Display a detailed recipe card with predictions."""

    st.markdown('<div class="recipe-box">', unsafe_allow_html=True)

    # Title
    st.markdown(f'<p class="recipe-title">🍽️ {recipe["name"]}</p>', unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        time_val = recipe.get('predicted_time' if show_predictions and 'predicted_time' in recipe else 'minutes', 'N/A')
        if pd.notna(time_val) and time_val != 'N/A':
            time_str = f"{int(time_val)} min"
            time_emoji = "⚡" if time_val < 30 else "⏰"
        else:
            time_str = "N/A"
            time_emoji = "⏰"
        st.metric(f"{time_emoji} Time", time_str)

    with col2:
        steps = recipe.get('n_steps', 'N/A')
        if pd.notna(steps):
            st.metric("📋 Steps", int(steps))
        else:
            st.metric("📋 Steps", "N/A")

    with col3:
        ingredients = recipe.get('n_ingredients', 'N/A')
        if pd.notna(ingredients):
            st.metric("🛒 Ingredients", int(ingredients))
        else:
            st.metric("🛒 Ingredients", "N/A")

    with col4:
        calories = recipe.get('calories', 'N/A')
        if pd.notna(calories) and calories != 'N/A':
            cal_emoji = "🟢" if calories < 400 else "🟡" if calories < 700 else "🔴"
            st.metric(f"{cal_emoji} Calories", f"{int(calories)}")
        else:
            st.metric("🔥 Calories", "N/A")

    # Tags if predictions enabled
    if show_predictions:
        tags = []
        if 'high_protein' in recipe and recipe['high_protein'] == 1:
            tags.append('<span class="tag">💪 High Protein</span>')
        if 'low_calorie' in recipe and recipe['low_calorie'] == 1:
            tags.append('<span class="tag">🟢 Low Calorie</span>')
        if 'healthy_recipe' in recipe and recipe['healthy_recipe'] == 1:
            tags.append('<span class="tag">🥗 Healthy</span>')
        if 'low_sugar' in recipe and recipe['low_sugar'] == 1:
            tags.append('<span class="tag">🍬 Low Sugar</span>')

        if tags:
            st.markdown(' '.join(tags), unsafe_allow_html=True)

    # Description
    if 'description' in recipe and pd.notna(recipe['description']) and str(recipe['description']).strip():
        with st.expander("📖 Description"):
            st.write(recipe['description'])

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
