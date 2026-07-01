"""
Data Processing Module for Recommendation System.
Handles data loading, validation, encoding, and sparse matrix creation.
"""

import logging
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix

from src.utils.data_cache import DataCache

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles all data processing operations for the recommendation system.
    Responsible for loading, validating, encoding, and preparing data for SVD.
    """

    def __init__(self):
        """Initialize the data processor."""
        self.df_interactions = None
        self.df_recipes = None
        self.user_cat = None
        self.recipe_cat = None
        self.sparse_matrix = None

    def load_csv_data(self, filepath, description="dataset"):
        """
        Load a CSV file with error handling.

        NOTE: This method is deprecated. Use load_data() which uses DataCache instead.

        Args:
            filepath (str or Path): path to the CSV file
            description (str): descriptive name for logs/errors
        Returns:
            pd.DataFrame
        """
        filepath = Path(filepath)
        try:
            # Direct CSV read (not recommended, use DataCache instead)
            df = pd.read_csv(filepath)
            logger.info(
                f"{description} loaded successfully ({filepath}): {df.shape[0]} rows"
            )
            return df
        except FileNotFoundError:
            logger.error(f"File {description} not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading {description} ({filepath}): {e}")
            raise

    def load_data(self, interactions_path, recipes_path):
        """
        Load interaction and recipe data.
        Args:
            interactions_path (str or Path): path to interactions file
            recipes_path (str or Path): path to recipes file
        """
        logger.info("Loading data...")
        # Use global cache to avoid redundant loading
        self.df_interactions = DataCache.get_interactions(
            path=str(interactions_path), optimize_dtypes=True
        )
        self.df_recipes = DataCache.get_recipes(
            path=str(recipes_path), optimize_dtypes=True
        )

        # Display basic information
        logger.info(f"Unique interactions: {len(self.df_interactions)}")
        logger.info(f"Unique users: {self.df_interactions['user_id'].nunique()}")
        logger.info(f"Unique recipes: {self.df_interactions['recipe_id'].nunique()}")
        logger.info(f"Total available recipes: {len(self.df_recipes)}")

    def create_sparse_matrix(
        self, user_col="user_id", recipe_col="recipe_id", rating_col="rating"
    ):
        """
        Create sparse user-recipe matrix from interactions data.
        Args:
            user_col (str): column name for user IDs
            recipe_col (str): column name for recipe IDs
            rating_col (str): column name for ratings
        """
        if self.df_interactions is None:
            raise ValueError("Interaction data must be loaded first")

        logger.info("Creating sparse matrix...")

        # Validate required columns exist
        required_cols = [user_col, recipe_col, rating_col]
        for col in required_cols:
            if col not in self.df_interactions.columns:
                raise ValueError(f"Missing column: {col}")

        # Encode IDs as categories
        self.user_cat = self.df_interactions[user_col].astype("category")
        self.recipe_cat = self.df_interactions[recipe_col].astype("category")
        user_codes = self.user_cat.cat.codes
        recipe_codes = self.recipe_cat.cat.codes

        # Get ratings
        ratings = self.df_interactions[rating_col].values

        # Build sparse matrix
        n_users = self.user_cat.cat.categories.size
        n_recipes = self.recipe_cat.cat.categories.size
        self.sparse_matrix = csr_matrix(
            (ratings, (user_codes, recipe_codes)), shape=(n_users, n_recipes)
        )

        logger.info(
            f"Sparse matrix created: {n_users} users, {n_recipes} recipes, {self.sparse_matrix.nnz} ratings."
        )

    def get_user_index(self, user_real_id):
        """
        Get user index in the sparse matrix from real user ID.
        Args:
            user_real_id (str/int): real user ID from dataset
        Returns:
            user_idx (int): index in the matrix
            user_real_id: confirmed real ID
        """
        if self.user_cat is None:
            raise ValueError("User categories must be created first")

        user_id_list = self.user_cat.cat.categories.tolist()

        # Convert to same type for comparison
        if isinstance(user_real_id, str):
            user_id_list = [str(uid) for uid in user_id_list]
        elif isinstance(user_real_id, int):
            user_id_list = [int(uid) for uid in user_id_list]

        try:
            user_idx = user_id_list.index(user_real_id)
            logger.info(f"User found: ID {user_real_id}, index {user_idx}")
            return user_idx, user_real_id
        except ValueError:
            raise ValueError(f"User {user_real_id} not found in data")

    def get_user_by_index(self, user_idx=0):
        """
        Get user ID by index in the matrix.
        Args:
            user_idx (int): user index to select
        Returns:
            selected_user_id (str/int): real user ID
        """
        if self.user_cat is None:
            raise ValueError("User categories must be created first")

        user_id_list = self.user_cat.cat.categories.tolist()
        if not (0 <= user_idx < len(user_id_list)):
            logger.error(
                f"User index {user_idx} out of bounds (0 to {len(user_id_list)-1})"
            )
            raise IndexError("User index out of bounds")
        selected_user_id = user_id_list[user_idx]
        logger.info(f"User selected: idx {user_idx}, id {selected_user_id}")
        return selected_user_id

    def get_user_historical_recipes(self, user_id, top_n=10):
        """
        Get user's top historical recipes sorted by rating.
        Args:
            user_id: user ID
            top_n (int): number of recipes to retrieve
        Returns:
            pd.DataFrame: DataFrame with recipes sorted by rating
        """
        user_ratings = self.df_interactions[self.df_interactions["user_id"] == user_id]
        if user_ratings.empty:
            logger.warning(f"No historical data found for user {user_id}")
            return pd.DataFrame(columns=["id", "name", "rating"])

        # Sort by rating descending and get top N
        top_ratings = user_ratings.nlargest(top_n, "rating")
        result = self.df_recipes[self.df_recipes["id"].isin(top_ratings["recipe_id"])][
            ["id", "name"]
        ]
        result = result.merge(
            top_ratings[["recipe_id", "rating"]],
            left_on="id",
            right_on="recipe_id",
            how="left",
        )
        result = result.drop("recipe_id", axis=1).sort_values("rating", ascending=False)

        # Remove detailed logging to avoid duplicates in output
        return result.reset_index(drop=True)

    def get_global_popular_recipes(self, top_n=10, min_ratings=50):
        """
        Get globally popular recipes based on average rating and number of ratings.
        Args:
            top_n (int): number of recipes to retrieve
            min_ratings (int): minimum number of ratings required
        Returns:
            pd.DataFrame: DataFrame with popular recipes
        """
        # Calculate recipe statistics
        recipe_stats = (
            self.df_interactions.groupby("recipe_id")
            .agg({"rating": ["mean", "count"]})
            .reset_index()
        )

        # Flatten column names
        recipe_stats.columns = ["recipe_id", "avg_rating", "rating_count"]

        # Filter by minimum ratings and sort by average rating
        popular_recipes = recipe_stats[recipe_stats["rating_count"] >= min_ratings]
        popular_recipes = popular_recipes.sort_values(
            ["avg_rating", "rating_count"], ascending=[False, False]
        )

        # Get top N
        top_popular = popular_recipes.head(top_n)

        # Merge with recipe names
        result = self.df_recipes[self.df_recipes["id"].isin(top_popular["recipe_id"])][
            ["id", "name"]
        ]
        result = result.merge(
            top_popular[["recipe_id", "avg_rating", "rating_count"]],
            left_on="id",
            right_on="recipe_id",
            how="left",
        )
        result = result.drop("recipe_id", axis=1)
        result = result.sort_values("avg_rating", ascending=False)

        # Remove detailed logging to avoid duplicates in output
        return result.reset_index(drop=True)

    def get_user_interaction_count(self, user_id):
        """
        Get number of interactions for a specific user.
        Args:
            user_id: user ID
        Returns:
            int: number of interactions
        """
        user_ratings = self.df_interactions[self.df_interactions["user_id"] == user_id]
        return len(user_ratings)

    def get_recipe_names(self, recipe_ids):
        """
        Get recipe names from recipe IDs.
        Args:
            recipe_ids (list): list of recipe IDs
        Returns:
            pd.DataFrame: DataFrame with recipe IDs and names
        """
        return self.df_recipes[self.df_recipes["id"].isin(recipe_ids)][["id", "name"]]

    def get_user_rated_recipes(self, user_id):
        """
        Get set of recipe IDs that user has already rated.
        Args:
            user_id: user ID
        Returns:
            set: set of recipe IDs
        """
        user_ratings = self.df_interactions[self.df_interactions["user_id"] == user_id]
        return set(user_ratings["recipe_id"])

    def get_recipe_ids_list(self):
        """
        Get list of all recipe IDs in the dataset.
        Returns:
            list: list of recipe IDs
        """
        if self.recipe_cat is None:
            raise ValueError("Recipe categories must be created first")
        return self.recipe_cat.cat.categories.astype(int).tolist()

    def is_ready(self):
        """
        Check if data processor is ready (data loaded and matrix created).
        Returns:
            bool: True if ready, False otherwise
        """
        return (
            self.df_interactions is not None
            and self.df_recipes is not None
            and self.sparse_matrix is not None
        )
