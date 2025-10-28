"""
Integrated Recommendation Pipeline
Combines all ML models into a unified recommendation system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.recommendation.svd_recommender import SVDRecommender
from src.modeling.time_predictor import TimePredictionModel
from src.modeling.nutrition_tagger import NutritionTaggerModel
from src.modeling.recipe_clustering import RecipeClusterer
from src.sentiment_analysis import SentimentAnalyzer
from .unified_data_loader import UnifiedRecipeDataLoader

logger = logging.getLogger(__name__)


class IntegratedRecommendationPipeline:
    """
    Complete recommendation pipeline that integrates all models.

    Features:
    - SVD-based collaborative filtering
    - Time prediction
    - Nutrition tagging
    - Recipe clustering
    - Sentiment analysis
    """

    def __init__(
        self,
        recipes_path: str = "data/RAW_recipes.csv",
        interactions_path: str = "data/RAW_interactions.csv",
        models_dir: str = "outputs/models",
        load_models: bool = True
    ):
        """
        Initialize the integrated pipeline.

        Args:
            recipes_path: Path to recipes CSV
            interactions_path: Path to interactions CSV
            models_dir: Directory containing trained models
            load_models: Whether to load pre-trained models
        """
        self.recipes_path = Path(recipes_path)
        self.interactions_path = Path(interactions_path)
        self.models_dir = Path(models_dir)

        # Initialize data loader
        logger.info("Initializing data loader...")
        self.data_loader = UnifiedRecipeDataLoader(recipes_path)

        # Initialize SVD recommender
        logger.info("Initializing SVD recommender...")
        self.svd_recommender = SVDRecommender()
        self.svd_trained = False

        # Initialize other models (will be loaded if available)
        self.time_model = None
        self.nutrition_models = {}
        self.recipe_clusterer = None

        if load_models:
            self._load_models()

    def _load_models(self):
        """Load pre-trained models if they exist."""

        # Load time prediction model
        time_model_path = self.models_dir / "time_random_forest.pkl"
        if time_model_path.exists():
            try:
                self.time_model = TimePredictionModel.load_model(time_model_path)
                logger.info("✓ Time prediction model loaded")
            except Exception as e:
                logger.warning(f"Could not load time model: {e}")
        else:
            logger.info("⚠ Time prediction model not found (will skip time predictions)")

        # Load nutrition tagging models
        nutrition_tags = ['high_calorie', 'low_calorie', 'high_protein', 'low_fat', 'low_sugar', 'healthy_recipe']
        for tag in nutrition_tags:
            model_path = self.models_dir / f"nutrition_{tag}_random_forest.pkl"
            if model_path.exists():
                try:
                    self.nutrition_models[tag] = NutritionTaggerModel.load_model(model_path)
                    logger.info(f"✓ Nutrition model loaded: {tag}")
                except Exception as e:
                    logger.warning(f"Could not load nutrition model {tag}: {e}")

        if not self.nutrition_models:
            logger.info("⚠ No nutrition models found (will skip nutrition tagging)")

        # Load recipe clusterer
        # Look for most recent clustering model
        cluster_models = list(self.models_dir.glob("recipe_clustering_v3_*_kmeans.pkl"))
        if cluster_models:
            try:
                # Get the most recent model
                latest_model = sorted(cluster_models)[-1]
                model_name = latest_model.stem.replace("_kmeans", "")
                self.recipe_clusterer = RecipeClusterer.load_model(model_name, str(self.models_dir))
                logger.info("✓ Recipe clustering model loaded")
            except Exception as e:
                logger.warning(f"Could not load clustering model: {e}")
        else:
            logger.info("⚠ Recipe clustering model not found (will skip clustering)")

    def train_svd_recommender(self, k: int = 50):
        """
        Train the SVD recommender system.

        Args:
            k: Number of latent factors for SVD
        """
        logger.info(f"Training SVD recommender with k={k}...")

        if not self.interactions_path.exists():
            raise FileNotFoundError(f"Interactions file not found: {self.interactions_path}")

        if not self.recipes_path.exists():
            raise FileNotFoundError(f"Recipes file not found: {self.recipes_path}")

        self.svd_recommender.fit(
            str(self.interactions_path),
            str(self.recipes_path),
            k=k
        )

        self.svd_trained = True
        logger.info("✓ SVD recommender trained successfully")

    def get_recommendations(
        self,
        user_id: int,
        top_n: int = 10,
        enrich: bool = True,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Get personalized recipe recommendations with enrichment.

        Args:
            user_id: User ID to get recommendations for
            top_n: Number of recommendations to return
            enrich: Whether to enrich with model predictions
            filters: Optional filters (e.g., max_time, dietary_preferences)

        Returns:
            Dictionary with recommendations and metadata
        """
        if not self.svd_trained:
            raise RuntimeError("SVD recommender not trained. Call train_svd_recommender() first.")

        logger.info(f"Getting recommendations for user {user_id}...")

        # Step 1: Get base recommendations from SVD
        svd_result = self.svd_recommender.recommend_for_user(
            user_real_id=user_id,
            top_n=top_n * 2,  # Get more to allow for filtering
            min_historical_threshold=5
        )

        result = {
            'user_id': svd_result['user_id'],
            'recommendation_type': svd_result['recommendation_type'],
            'fallback_used': svd_result['fallback_used'],
            'historical_recipes': svd_result['historical'],
            'recommendations': svd_result['svd_recommendations']
        }

        # If no enrichment requested, return base recommendations
        if not enrich:
            result['recommendations'] = result['recommendations'].head(top_n)
            return result

        # Step 2: Load full recipe data for recommended recipes
        recipe_ids = result['recommendations']['id'].tolist()
        recipes = self.data_loader.load_recipes_by_ids(recipe_ids)

        # Step 3: Enrich with model predictions
        if len(recipes) > 0:
            recipes = self._enrich_recipes(recipes)

            # Step 4: Apply filters if provided
            if filters:
                recipes = self._apply_filters(recipes, filters)

            # Step 5: Merge with SVD recommendations to preserve order
            recipes = recipes.merge(
                result['recommendations'][['id']].reset_index().rename(columns={'index': 'svd_rank'}),
                on='id',
                how='left'
            ).sort_values('svd_rank')

            # Limit to top_n
            recipes = recipes.head(top_n)

            result['recommendations'] = recipes

        logger.info(f"✓ Returned {len(result['recommendations'])} enriched recommendations")

        return result

    def _enrich_recipes(self, recipes: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich recipes with all model predictions.

        Args:
            recipes: DataFrame with recipe data

        Returns:
            Enriched DataFrame
        """
        recipes = recipes.copy()

        # Add time predictions
        if self.time_model is not None:
            recipes = self._add_time_predictions(recipes)

        # Add nutrition tags
        if self.nutrition_models:
            recipes = self._add_nutrition_tags(recipes)

        # Add clusters
        if self.recipe_clusterer is not None:
            recipes = self._add_clusters(recipes)

        return recipes

    def _add_time_predictions(self, recipes: pd.DataFrame) -> pd.DataFrame:
        """Add predicted cooking time."""
        try:
            logger.info("Adding time predictions...")

            # Prepare features for time prediction
            recipes_with_features = self.data_loader.prepare_for_time_prediction(recipes)

            # Get feature names
            feature_names = self.time_model.feature_names

            # Check if all features are available
            available_features = [f for f in feature_names if f in recipes_with_features.columns]

            if len(available_features) < len(feature_names):
                logger.warning(f"Missing features for time prediction: {set(feature_names) - set(available_features)}")

            # Make predictions
            X = recipes_with_features[available_features]
            predictions = self.time_model.predict(X, scale_features=True)

            recipes['predicted_time'] = predictions

            logger.info(f"✓ Time predictions added (avg: {predictions.mean():.1f} min)")

        except Exception as e:
            logger.error(f"Error adding time predictions: {e}")
            recipes['predicted_time'] = recipes.get('minutes', np.nan)

        return recipes

    def _add_nutrition_tags(self, recipes: pd.DataFrame) -> pd.DataFrame:
        """Add nutrition tags."""
        try:
            logger.info("Adding nutrition tags...")

            # Prepare features for nutrition tagging
            recipes_with_features = self.data_loader.prepare_for_nutrition_tagging(recipes)

            # Predict each nutrition tag
            for tag_name, model in self.nutrition_models.items():
                feature_names = model.feature_names

                # Check available features
                available_features = [f for f in feature_names if f in recipes_with_features.columns]

                if len(available_features) < len(feature_names):
                    logger.warning(f"Missing features for {tag_name}: {set(feature_names) - set(available_features)}")
                    continue

                # Make predictions
                X = recipes_with_features[available_features]
                predictions = model.predict(X, scale_features=True)

                recipes[tag_name] = predictions

            logger.info(f"✓ Nutrition tags added: {list(self.nutrition_models.keys())}")

        except Exception as e:
            logger.error(f"Error adding nutrition tags: {e}")

        return recipes

    def _add_clusters(self, recipes: pd.DataFrame) -> pd.DataFrame:
        """Add recipe cluster assignments."""
        try:
            logger.info("Adding cluster assignments...")

            # For clustering, we need pre-computed features
            # If features are not available, skip clustering
            required_features = ['log_minutes', 'time_complexity', 'efficiency', 'health_category']

            if not all(f in recipes.columns for f in required_features):
                logger.warning("Required features for clustering not available. Skipping clustering.")
                return recipes

            # Predict clusters
            recipes_clustered = self.recipe_clusterer.predict(recipes)

            recipes['cluster'] = recipes_clustered['cluster']
            recipes['cluster_name'] = recipes_clustered['cluster_name']

            logger.info(f"✓ Clusters added")

        except Exception as e:
            logger.error(f"Error adding clusters: {e}")

        return recipes

    def _apply_filters(self, recipes: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Apply filters to recommendations.

        Args:
            recipes: DataFrame with recipes
            filters: Dictionary of filters

        Returns:
            Filtered DataFrame
        """
        recipes = recipes.copy()

        # Time filter
        if 'max_time' in filters and 'predicted_time' in recipes.columns:
            max_time = filters['max_time']
            recipes = recipes[recipes['predicted_time'] <= max_time]
            logger.info(f"Filtered by time <= {max_time} minutes: {len(recipes)} recipes remain")

        # Nutrition filters
        if 'require_high_protein' in filters and filters['require_high_protein']:
            if 'high_protein' in recipes.columns:
                recipes = recipes[recipes['high_protein'] == 1]
                logger.info(f"Filtered by high protein: {len(recipes)} recipes remain")

        if 'require_low_calorie' in filters and filters['require_low_calorie']:
            if 'low_calorie' in recipes.columns:
                recipes = recipes[recipes['low_calorie'] == 1]
                logger.info(f"Filtered by low calorie: {len(recipes)} recipes remain")

        # Cluster filter
        if 'cluster_name' in filters and 'cluster_name' in recipes.columns:
            cluster = filters['cluster_name']
            recipes = recipes[recipes['cluster_name'] == cluster]
            logger.info(f"Filtered by cluster '{cluster}': {len(recipes)} recipes remain")

        return recipes

    def analyze_recipe_sentiment(self, recipe_id: int, limit: int = 10) -> Dict:
        """
        Analyze sentiment of reviews for a specific recipe.

        Args:
            recipe_id: Recipe ID
            limit: Maximum number of reviews to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for recipe {recipe_id}...")

        # Load interactions
        interactions = pd.read_csv(self.interactions_path)

        # Filter reviews for this recipe
        recipe_reviews = interactions[interactions['recipe_id'] == recipe_id].copy()

        if len(recipe_reviews) == 0:
            return {
                'recipe_id': recipe_id,
                'review_count': 0,
                'sentiments': [],
                'summary': {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                }
            }

        # Sample reviews if too many
        if len(recipe_reviews) > limit:
            recipe_reviews = recipe_reviews.sample(n=limit, random_state=42)

        # Analyze each review
        sentiments = []
        for _, row in recipe_reviews.iterrows():
            review_text = row['review']
            if pd.notna(review_text) and len(str(review_text).strip()) > 0:
                try:
                    label, confidence = SentimentAnalyzer.predict_sentiment(str(review_text))
                    sentiments.append({
                        'review': review_text[:200],  # Truncate for display
                        'rating': row['rating'],
                        'sentiment': label,
                        'confidence': confidence
                    })
                except Exception as e:
                    logger.warning(f"Error analyzing review: {e}")

        # Summarize sentiments
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        for s in sentiments:
            sentiment_counts[s['sentiment']] += 1

        return {
            'recipe_id': recipe_id,
            'review_count': len(sentiments),
            'sentiments': sentiments,
            'summary': sentiment_counts,
            'avg_rating': recipe_reviews['rating'].mean()
        }

    def get_pipeline_status(self) -> Dict:
        """
        Get status of all pipeline components.

        Returns:
            Dictionary with component statuses
        """
        return {
            'svd_recommender': 'trained' if self.svd_trained else 'not trained',
            'time_model': 'loaded' if self.time_model is not None else 'not available',
            'nutrition_models': {
                'loaded': list(self.nutrition_models.keys()),
                'count': len(self.nutrition_models)
            },
            'recipe_clusterer': 'loaded' if self.recipe_clusterer is not None else 'not available',
            'sentiment_analyzer': 'available (pre-trained)'
        }
