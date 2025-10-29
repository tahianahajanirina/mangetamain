"""Tests for recipe feature engineering."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.recipe_features import RecipeFeatureBuilder


class TestRecipeFeatureBuilder:
    """Test cases for RecipeFeatureBuilder."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        builder = RecipeFeatureBuilder()
        assert builder.min_rating > 0
        assert builder.min_rating <= 5.0

    def test_init_custom_rating(self):
        """Test initialization with custom min_rating."""
        builder = RecipeFeatureBuilder(min_rating=4.0)
        assert builder.min_rating == 4.0

    def test_calculate_popularity_metrics(
        self, sample_recipes_df, sample_interactions_df
    ):
        """Test popularity metrics calculation."""
        builder = RecipeFeatureBuilder()

        result = builder.calculate_popularity_metrics(
            sample_recipes_df, sample_interactions_df
        )

        assert "n_interactions" in result.columns
        assert "avg_rating" in result.columns
        assert pd.api.types.is_numeric_dtype(result["n_interactions"])
        assert pd.api.types.is_numeric_dtype(result["avg_rating"])

    def test_calculate_health_category(self):
        """Test health category calculation."""
        builder = RecipeFeatureBuilder()

        # Create test recipe row with ALL required columns
        recipe = pd.Series(
            {
                "calories": 250,
                "total_fat": 10,
                "sugar": 15,
                "sodium": 500,
                "protein": 20,
                "saturated_fat": 5,  # Added missing column
            }
        )

        result = builder.calculate_health_category(recipe)

        assert isinstance(result, (int, np.integer))
        assert 0 <= result <= 4

    def test_build_features_full_pipeline(
        self, temp_data_dir, sample_recipes_df, sample_interactions_df
    ):
        """Test complete feature building pipeline."""
        # Add required columns to sample data - MATCH the 5 rows in sample_recipes_df
        df_recipes = sample_recipes_df.copy()
        df_recipes["calories"] = [200, 300, 400, 150, 500]
        df_recipes["total_fat"] = [10, 15, 20, 5, 25]
        df_recipes["sugar"] = [5, 10, 15, 3, 18]
        df_recipes["sodium"] = [500, 600, 700, 200, 800]
        df_recipes["protein"] = [20, 25, 30, 8, 35]
        df_recipes["saturated_fat"] = [3, 5, 7, 2, 8]  # Added missing column

        # Save test data
        recipes_path = temp_data_dir["processed"] / "recipes_clean.csv"
        interactions_path = temp_data_dir["processed"] / "interactions_clean.csv"

        df_recipes.to_csv(recipes_path, index=False)
        sample_interactions_df.to_csv(interactions_path, index=False)

        builder = RecipeFeatureBuilder(min_rating=3.0)
        result = builder.build_features(str(recipes_path), str(interactions_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "id" in result.columns
        assert "popularity_score" in result.columns
        assert "log_minutes" in result.columns
        assert "time_complexity" in result.columns
        assert "efficiency" in result.columns
        assert "health_category" in result.columns

    def test_load_data(self, temp_data_dir, sample_recipes_df, sample_interactions_df):
        """Test data loading."""
        recipes_path = temp_data_dir["raw"] / "recipes.csv"
        interactions_path = temp_data_dir["raw"] / "interactions.csv"

        sample_recipes_df.to_csv(recipes_path, index=False)
        sample_interactions_df.to_csv(interactions_path, index=False)

        builder = RecipeFeatureBuilder()
        recipes, interactions = builder.load_data(
            str(recipes_path), str(interactions_path)
        )

        assert isinstance(recipes, pd.DataFrame)
        assert isinstance(interactions, pd.DataFrame)
        assert len(recipes) == len(sample_recipes_df)
        assert len(interactions) == len(sample_interactions_df)
