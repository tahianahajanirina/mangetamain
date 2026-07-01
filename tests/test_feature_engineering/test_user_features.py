"""Tests for user feature engineering."""

import pandas as pd
import pytest

from src.feature_engineering.user_features import UserFeatureBuilder


class TestUserFeatureBuilder:
    """Test cases for UserFeatureBuilder."""

    def test_init_default(self):
        """Test default initialization."""
        builder = UserFeatureBuilder()
        assert builder.min_interactions == 5
        assert builder.users_profiles is None

    def test_init_custom(self):
        """Test custom initialization."""
        builder = UserFeatureBuilder(min_interactions=10)
        assert builder.min_interactions == 10

    def test_load_data(self, temp_data_dir, sample_recipes_df, sample_interactions_df):
        """Test data loading."""
        recipes_path = temp_data_dir["raw"] / "recipes.csv"
        interactions_path = temp_data_dir["raw"] / "interactions.csv"
        sample_recipes_df.to_csv(recipes_path, index=False)
        sample_interactions_df.to_csv(interactions_path, index=False)

        builder = UserFeatureBuilder()
        recipes, interactions = builder.load_data(
            str(recipes_path), str(interactions_path)
        )

        assert isinstance(recipes, pd.DataFrame)
        assert isinstance(interactions, pd.DataFrame)

    def test_merge_data(self, sample_recipes_df, sample_interactions_df):
        """Test data merging."""
        builder = UserFeatureBuilder()
        merged = builder.merge_data(sample_recipes_df, sample_interactions_df)

        assert isinstance(merged, pd.DataFrame)
        assert "user_id" in merged.columns
        assert "recipe_id" in merged.columns

    def test_build_behavioral_features(
        self, sample_recipes_df, sample_interactions_df
    ):
        """Test behavioral feature building."""
        builder = UserFeatureBuilder()
        merged = builder.merge_data(sample_recipes_df, sample_interactions_df)
        behavioral = builder.build_behavioral_features(merged)

        assert "n_interactions" in behavioral.columns
        assert "avg_rating" in behavioral.columns
        assert len(behavioral) > 0

    def test_build_temporal_features(
        self, sample_recipes_df, sample_interactions_df
    ):
        """Test temporal feature building."""
        builder = UserFeatureBuilder()
        merged = builder.merge_data(sample_recipes_df, sample_interactions_df)
        temporal = builder.build_temporal_features(merged)

        assert "avg_minutes" in temporal.columns
        assert "std_minutes" in temporal.columns
        assert len(temporal) > 0

    def test_get_feature_stats_before_build(self):
        """Test that stats raise error before building features."""
        builder = UserFeatureBuilder()
        with pytest.raises(ValueError, match="Features non construites"):
            builder.get_feature_stats()

    def test_save_features_before_build(self):
        """Test that save raises error before building features."""
        builder = UserFeatureBuilder()
        with pytest.raises(ValueError, match="Features non construites"):
            builder.save_features("/tmp/test.csv")
