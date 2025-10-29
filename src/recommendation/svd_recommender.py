"""
SVD Recommender Module.
Handles SVD computation and recommendation generation using matrix factorization.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from src.recommendation.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SVDRecommender:
    """
    SVD-based recommendation system using matrix factorization.
    Handles SVD computation and generates personalized recommendations.
    """

    def __init__(self):
        """Initialize the SVD recommender."""
        self.data_processor = DataProcessor()
        self.U = None  # User factors
        self.S = None  # Singular values
        self.Vt = None  # Item factors (transposed)
        self.is_fitted = False

    def load_and_prepare_data(self, interactions_path, recipes_path):
        """
        Load and prepare data for SVD computation.
        Args:
            interactions_path (str or Path): path to interactions file
            recipes_path (str or Path): path to recipes file
        """
        self.data_processor.load_data(interactions_path, recipes_path)
        self.data_processor.create_sparse_matrix()

    def compute_svd(self, k=50):
        """
        Compute SVD decomposition on the sparse matrix.
        Args:
            k (int): number of latent factors/dimensions
        """
        if not self.data_processor.is_ready():
            raise ValueError("Data must be loaded and prepared first")

        try:
            self.U, self.S, self.Vt = svds(self.data_processor.sparse_matrix, k=k)
            logging.info(
                f"SVD computed with k={k}. Dimensions U: {self.U.shape}, S: {self.S.shape}, Vt: {self.Vt.shape}"
            )
            self.is_fitted = True
        except Exception as e:
            logging.error(f"Error during SVD computation: {e}")
            raise

    def predict_user_ratings(self, user_idx):
        """
        Predict ratings for all items for a given user.
        Args:
            user_idx (int): user index in the matrix
        Returns:
            np.array: predicted ratings for all items
        """
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted first")

        # Compute user predictions
        user_mean = self.data_processor.sparse_matrix[user_idx].mean()
        user_factors = self.U[user_idx, :]
        scores = user_factors * self.S
        user_pred_centered = scores @ self.Vt
        user_pred = user_pred_centered + user_mean

        return user_pred

    def generate_svd_recommendations(self, user_idx, user_id, top_n=10):
        """
        Generate SVD-based recommendations for a user.
        Args:
            user_idx (int): user index in the matrix
            user_id: real user ID
            top_n (int): number of recommendations
        Returns:
            pd.DataFrame: DataFrame with recommendations
        """
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted first")

        try:
            # Get user predictions
            user_pred = self.predict_user_ratings(user_idx)

            # Remove recipes already rated by this user
            user_rated_recipe_ids = self.data_processor.get_user_rated_recipes(user_id)
            recipe_id_list = self.data_processor.get_recipe_ids_list()

            # Indices of unrated recipes
            non_rated_indices = [
                i
                for i, rid in enumerate(recipe_id_list)
                if rid not in user_rated_recipe_ids
            ]

            if not non_rated_indices:
                logging.warning(f"No new recipes to recommend for user {user_id}")
                return self.data_processor.get_recipe_names([])

            # Predictions on unrated recipes
            user_pred_non_rated = user_pred[non_rated_indices]
            top_rec_idx = np.argsort(user_pred_non_rated)[-top_n:]
            top_rec_global_idx = [non_rated_indices[idx] for idx in top_rec_idx]
            top_rec_ids = [recipe_id_list[idx] for idx in top_rec_global_idx]
            top_rec_names = self.data_processor.get_recipe_names(top_rec_ids)

            # Remove detailed logging to avoid duplicates - results shown in data_processor
            return top_rec_names.reset_index(drop=True)

        except Exception as e:
            logging.error(f"Error generating SVD recommendations: {e}")
            raise

    def fit(self, interactions_path, recipes_path, k=50):
        """
        Complete training pipeline for the recommendation model.
        Args:
            interactions_path (str or Path): path to interactions file
            recipes_path (str or Path): path to recipes file
            k (int): number of SVD dimensions
        """
        logging.info(f"Training SVD recommendation model with k={k}")
        self.load_and_prepare_data(interactions_path, recipes_path)
        self.compute_svd(k=k)
        logging.info("Model training completed successfully")

    def recommend_for_user(self, user_real_id, top_n=10, min_historical_threshold=5):
        """
        Generate recommendations for a given user.
        Uses global popular recipes if user has insufficient historical data.
        Args:
            user_real_id (str/int): real user ID
            top_n (int): number of recommendations
            min_historical_threshold (int): minimum historical interactions to use SVD
        Returns:
            dict: Dictionary with 'user_id', 'historical', 'svd_recommendations', and 'fallback_used'
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        logging.info(f"Generating {top_n} recommendations for user {user_real_id}")

        # Try to find user
        try:
            user_idx, confirmed_user_id = self.data_processor.get_user_index(
                user_real_id
            )
            user_found = True
        except ValueError:
            logging.warning(
                f"User {user_real_id} not found, will use global recommendations"
            )
            confirmed_user_id = user_real_id
            user_found = False

        # Check user's historical data
        if user_found:
            interaction_count = self.data_processor.get_user_interaction_count(
                confirmed_user_id
            )
            logging.info(
                f"User {confirmed_user_id} has {interaction_count} historical interactions"
            )

            if interaction_count >= min_historical_threshold:
                # Sufficient data: use personalized recommendations
                logging.info("Using personalized SVD recommendations")

                # Historical recommendations
                historical = self.data_processor.get_user_historical_recipes(
                    confirmed_user_id, top_n
                )

                # SVD recommendations
                svd_recommendations = self.generate_svd_recommendations(
                    user_idx, confirmed_user_id, top_n
                )

                return {
                    "user_id": confirmed_user_id,
                    "historical": historical,
                    "svd_recommendations": svd_recommendations,
                    "fallback_used": False,
                    "recommendation_type": "personalized",
                }
            else:
                # Insufficient data: use global popular recipes
                logging.warning(
                    f"User has only {interaction_count} interactions (< {min_historical_threshold})"
                )
                logging.info("Using global popular recommendations as fallback")

                # Get user's limited historical data
                historical = self.data_processor.get_user_historical_recipes(
                    confirmed_user_id, top_n
                )

                # Use global popular recipes as recommendations
                global_popular = self.data_processor.get_global_popular_recipes(
                    top_n, min_ratings=20
                )

                return {
                    "user_id": confirmed_user_id,
                    "historical": historical,
                    "svd_recommendations": global_popular,
                    "fallback_used": True,
                    "recommendation_type": "global_popular",
                }
        else:
            # User not found: use global recommendations
            logging.info("User not found, using global popular recommendations")

            # No historical data
            historical = pd.DataFrame(columns=["id", "name", "rating"])

            # Use global popular recipes
            global_popular = self.data_processor.get_global_popular_recipes(
                top_n, min_ratings=20
            )

            return {
                "user_id": confirmed_user_id,
                "historical": historical,
                "svd_recommendations": global_popular,
                "fallback_used": True,
                "recommendation_type": "global_popular",
            }


def main():
    """
    Main function that runs the recommendation system with fallback mechanism.
    """
    # Path to current script directory
    script_dir = Path(__file__).parent.resolve()
    # Absolute paths for data files
    raw_interactions_path = script_dir.parent / "RAW_interactions.csv"
    recipes_path = script_dir.parent / "RAW_recipes.csv"

    # Create and train the recommendation system
    logging.info("Starting recommendation system demo")
    recommender = SVDRecommender()
    recommender.fit(raw_interactions_path, recipes_path, k=10)  # Smaller k for demo

    # Test different types of users
    logging.info("=" * 60)
    logging.info("TESTING RECOMMENDATION SYSTEM WITH FALLBACK")
    logging.info("=" * 60)

    # Test 1: Rich user
    logging.info("1. Testing user with rich history:")
    result1 = recommender.recommend_for_user(
        52282, top_n=10, min_historical_threshold=5
    )
    logging.info(
        f"   Result: {result1['recommendation_type']}, Fallback: {result1['fallback_used']}"
    )

    # Test 2: User with limited history
    logging.info("2. Testing user with limited history:")
    result2 = recommender.recommend_for_user(
        52282, top_n=10, min_historical_threshold=1000
    )  # High threshold
    logging.info(
        f"   Result: {result2['recommendation_type']}, Fallback: {result2['fallback_used']}"
    )

    # Test 3: Non-existent user
    logging.info("3. Testing non-existent user:")
    result3 = recommender.recommend_for_user(999999999, top_n=10)
    logging.info(
        f"   Result: {result3['recommendation_type']}, Fallback: {result3['fallback_used']}"
    )

    logging.info("=" * 60)
    logging.info("Demo completed successfully")


if __name__ == "__main__":
    main()
