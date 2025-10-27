"""
Streamlit application for SVD Recommendation System - User Analysis Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import time
import sys
import os

# Add the Recommendation_recipe directory to the path to import our modules
current_dir = os.path.dirname(__file__)
recommendation_recipe_path = os.path.join(current_dir, '..', 'Recommendation_recipe')
sys.path.append(recommendation_recipe_path)

from data_processor import DataProcessor
from svd_recommender import SVDRecommender

# Configure Streamlit page
st.set_page_config(
    page_title="SVD Recommendation - User Analysis",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def find_data_files():
    """Find the CSV data files in various possible locations."""
    current_dir = Path(__file__).parent
    
    # Possible locations for the CSV files
    possible_locations = [
        # Same directory as script
        current_dir,
        # Project root (Mange_ta_main)
        current_dir.parent.parent,
        # Ms_Data_Projet directory  
        current_dir.parent.parent.parent,
        # Explicit path to Ms_Data_Projet
        Path("C:/Users/antoi/Ms_Data_Projet"),
    ]
    
    for location in possible_locations:
        interactions_path = location / "RAW_interactions.csv"
        recipes_path = location / "RAW_recipes.csv"
        
        if interactions_path.exists() and recipes_path.exists():
            return str(interactions_path), str(recipes_path)
    
    return None, None

def load_model():
    """Load and train the SVD recommendation model."""
    if st.session_state.recommender is None:
        with st.spinner("Loading and training SVD model..."):
            # Find data files
            interactions_path, recipes_path = find_data_files()
            
            if interactions_path is None or recipes_path is None:
                st.error("❌ Could not find CSV data files!")
                st.info("🔍 Please ensure RAW_interactions.csv and RAW_recipes.csv are available")
                return None
            
            st.success(f"✅ Found data files:")
            st.text(f"📊 Interactions: {interactions_path}")
            st.text(f"🍽️ Recipes: {recipes_path}")
            
            # Initialize and train model
            recommender = SVDRecommender()
            start_time = time.time()
            recommender.fit(interactions_path, recipes_path, k=50)
            training_time = time.time() - start_time
            
            st.session_state.recommender = recommender
            st.session_state.training_time = training_time
            st.session_state.data_loaded = True
            
            st.success(f"🎯 Model trained successfully in {training_time:.2f} seconds!")
    
    return st.session_state.recommender

def analyze_user_recommendations(recommender, user_id, top_n=10):
    """Analyze recommendations for a specific user."""
    try:
        start_time = time.time()
        
        # Get recommendations (this returns a dict with 'historical', 'svd_recommendations', etc.)
        rec_result = recommender.recommend_for_user(user_id, top_n=top_n)
        
        # Extract the data
        historical_data = rec_result.get('historical', pd.DataFrame())
        recommendations = rec_result.get('svd_recommendations', pd.DataFrame())
        fallback_used = rec_result.get('fallback_used', True)
        recommendation_type = rec_result.get('recommendation_type', 'global')
        
        prediction_time = time.time() - start_time
        
        return {
            "user_id": user_id,
            "historical_data": historical_data,
            "recommendations": recommendations,
            "fallback_used": fallback_used,
            "recommendation_type": recommendation_type,
            "prediction_time": prediction_time,
            "full_result": rec_result  # Include full result for additional info
        }
    except Exception as e:
        st.error(f"Error analyzing user {user_id}: {str(e)}")
        return None

def show_user_analysis(recommender):
    """Display user-specific analysis."""
    st.header("👤 User Analysis")
    
    # User selection section
    st.subheader("🔍 User Selection")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_id = st.number_input("Enter User ID:", min_value=1, value=52282, step=1)
    
    with col2:
        top_n = st.slider("Number of recommendations:", 1, 20, 10)
    
    with col3:
        st.write("")  # Spacing
        analyze_button = st.button("🎯 Analyze User", type="primary")
    
    # Quick user info (lightweight)
    if user_id:
        try:
            interaction_count = recommender.data_processor.get_user_interaction_count(user_id)
            st.info(f"User {user_id} has {interaction_count} historical interactions")
            
            # Determine recommendation type without full analysis
            min_threshold = 5
            rec_type = "Personalized SVD" if interaction_count >= min_threshold else "Global Popular"
            fallback = interaction_count < min_threshold
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Historical Interactions", interaction_count)
            with col2:
                st.metric("Recommendation Type", rec_type)
            with col3:
                st.metric("Uses Fallback", "Yes" if fallback else "No")
                
        except:
            st.warning(f"User {user_id} not found in the dataset")
    
    # Full analysis only when button is clicked
    if analyze_button:
        with st.spinner(f"Analyzing user {user_id}..."):
            analysis = analyze_user_recommendations(recommender, user_id, top_n)
            
            if analysis:
                st.success("Analysis completed!")
                
                # Performance metrics
                st.subheader("⚡ Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("User ID", analysis["user_id"])
                with col2:
                    st.metric("Recommendation Type", analysis["recommendation_type"])
                with col3:
                    st.metric("Fallback Used", "Yes" if analysis["fallback_used"] else "No")
                with col4:
                    st.metric("Prediction Time", f"{analysis['prediction_time']:.3f}s")
                
                # Results comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📚 Historical Preferences")
                    if not analysis["historical_data"].empty:
                        st.dataframe(analysis["historical_data"], use_container_width=True)
                        
                        # Quick stats
                        if 'rating' in analysis["historical_data"].columns:
                            avg_rating = analysis["historical_data"]['rating'].mean()
                            st.metric("Average Historical Rating", f"{avg_rating:.2f}")
                    else:
                        st.info("No historical data available for this user.")
                
                with col2:
                    st.subheader("🎯 New Recommendations")
                    if not analysis["recommendations"].empty:
                        st.dataframe(analysis["recommendations"], use_container_width=True)
                        st.metric("Recommendations Generated", len(analysis["recommendations"]))
                    else:
                        st.info("No recommendations generated.")
    
    # Sample user suggestions
    st.subheader("💡 Sample Users to Try")
    
    # Real users with different activity levels (discovered from data analysis)
    st.markdown("""
    **High Activity Users (20+ interactions):**
    - **424680** (7,671 interactions)
    - **37449** (5,603 interactions)   
    - **383346** (4,628 interactions)
    
    **Medium Activity Users (5-19 interactions):**
    - **742802** (19 interactions)
    - **383877** (19 interactions)
    - **55536** (19 interactions)
    
    **Low Activity Users (1-4 interactions):**
    - **400008** (4 interactions)
    - **1190462** (4 interactions)
    - **327976** (4 interactions)
    
    """)
    
    
    # Add a function to find more valid user IDs
    if st.button("🔍 Find More Valid User IDs", help="Click to discover additional users with interaction history"):
        with st.spinner("Searching for users with interaction history..."):
            try:
                # Sample a subset of users to check
                sample_size = 100
                user_categories = recommender.data_processor.user_cat.cat.categories
                sample_users = np.random.choice(user_categories, size=min(sample_size, len(user_categories)), replace=False)
                
                valid_users = []
                for user in sample_users:
                    try:
                        count = recommender.data_processor.get_user_interaction_count(user)
                        if count > 0:
                            valid_users.append((user, count))
                    except:
                        continue
                
                # Sort by interaction count
                valid_users.sort(key=lambda x: x[1], reverse=True)
                
                if valid_users:
                    st.success(f"Found {len(valid_users)} additional users with interaction history:")
                    
                    # Categorize users
                    high_activity = [u for u in valid_users if u[1] >= 20]
                    medium_activity = [u for u in valid_users if 5 <= u[1] < 20]
                    low_activity = [u for u in valid_users if 1 <= u[1] < 5]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Additional High Activity:**")
                        for user, count in high_activity[:3]:
                            st.text(f"ID {user}: {count} interactions")
                    
                    with col2:
                        st.markdown("**Additional Medium Activity:**")
                        for user, count in medium_activity[:3]:
                            st.text(f"ID {user}: {count} interactions")
                    
                    with col3:
                        st.markdown("**Additional Low Activity:**")
                        for user, count in low_activity[:3]:
                            st.text(f"ID {user}: {count} interactions")
                else:
                    st.warning("No additional users with interaction history found in the sample.")
                    
            except Exception as e:
                st.error(f"Error finding valid users: {str(e)}")
    
    # Quick stats about user distribution
    st.subheader("📊 User Activity Distribution")
    
    # Sample a subset of users for performance
    sample_size = min(1000, len(recommender.data_processor.user_cat.cat.categories))
    sample_users = np.random.choice(
        recommender.data_processor.user_cat.cat.categories, 
        size=sample_size, 
        replace=False
    )
    
    # Get interaction counts for sample
    interaction_counts = []
    for user in sample_users[:100]:  # Limit to 100 for performance
        try:
            count = recommender.data_processor.get_user_interaction_count(user)
            interaction_counts.append(count)
        except:
            continue
    
    if interaction_counts:
        import plotly.express as px
        fig = px.histogram(
            x=interaction_counts,
            title="User Activity Distribution (Sample of 100 users)",
            labels={"x": "Number of Interactions", "y": "Number of Users"},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Interactions", min(interaction_counts))
        with col2:
            st.metric("Max Interactions", max(interaction_counts))
        with col3:
            st.metric("Avg Interactions", f"{np.mean(interaction_counts):.1f}")
        with col4:
            st.metric("Median Interactions", f"{np.median(interaction_counts):.1f}")

def main():
    """Main application function."""
    st.title("🎯 SVD Recommendation System - User Analysis")
    st.markdown("Analyze individual user recommendations and compare with historical preferences")
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("🚀 Model Control")
        
        if st.button("Load Model", type="primary"):
            load_model()
        
        if st.session_state.data_loaded:
            st.success("✅ Model loaded successfully!")
            st.metric("Training Time", f"{st.session_state.training_time:.2f}s")
        else:
            st.info("Please load the model first using the button above.")
    
    # Main content
    if st.session_state.recommender is not None:
        show_user_analysis(st.session_state.recommender)
    else:
        st.info("🔄 Please load the model first using the sidebar button.")

if __name__ == "__main__":
    main()