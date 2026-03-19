"""Tests for DataCache utility module."""

import pandas as pd
import pytest

from src.utils.data_cache import DataCache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before and after each test."""
    DataCache.clear_cache()
    yield
    DataCache.clear_cache()


class TestDataCache:
    """Test cases for DataCache class."""

    def test_get_recipes_loads_from_file(self, temp_data_dir, sample_recipes_df):
        """Test loading recipes from a CSV file."""
        csv_path = temp_data_dir["raw"] / "recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        df = DataCache.get_recipes(path=str(csv_path), optimize_dtypes=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_recipes_df)

    def test_get_recipes_uses_cache(self, temp_data_dir, sample_recipes_df):
        """Test that repeated calls use cached data."""
        csv_path = temp_data_dir["raw"] / "recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        df1 = DataCache.get_recipes(path=str(csv_path), optimize_dtypes=False)
        df2 = DataCache.get_recipes(path=str(csv_path), optimize_dtypes=False)

        assert df1 is df2  # Same object (cached)

    def test_get_recipes_force_reload(self, temp_data_dir, sample_recipes_df):
        """Test force reload bypasses cache."""
        csv_path = temp_data_dir["raw"] / "recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        df1 = DataCache.get_recipes(path=str(csv_path), optimize_dtypes=False)
        df2 = DataCache.get_recipes(
            path=str(csv_path), optimize_dtypes=False, force_reload=True
        )

        assert df1 is not df2  # Different objects after force reload

    def test_get_recipes_with_columns(self, temp_data_dir, sample_recipes_df):
        """Test loading specific columns."""
        csv_path = temp_data_dir["raw"] / "recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        df = DataCache.get_recipes(
            path=str(csv_path), optimize_dtypes=False, columns=["id", "name"]
        )

        assert list(df.columns) == ["id", "name"]

    def test_get_interactions_loads_from_file(
        self, temp_data_dir, sample_interactions_df
    ):
        """Test loading interactions from a CSV file."""
        csv_path = temp_data_dir["raw"] / "interactions.csv"
        sample_interactions_df.to_csv(csv_path, index=False)

        df = DataCache.get_interactions(path=str(csv_path), optimize_dtypes=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_interactions_df)

    def test_get_interactions_uses_cache(self, temp_data_dir, sample_interactions_df):
        """Test that repeated calls use cached data."""
        csv_path = temp_data_dir["raw"] / "interactions.csv"
        sample_interactions_df.to_csv(csv_path, index=False)

        df1 = DataCache.get_interactions(path=str(csv_path), optimize_dtypes=False)
        df2 = DataCache.get_interactions(path=str(csv_path), optimize_dtypes=False)

        assert df1 is df2

    def test_clear_cache(self, temp_data_dir, sample_recipes_df):
        """Test clearing cache."""
        csv_path = temp_data_dir["raw"] / "recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        DataCache.get_recipes(path=str(csv_path), optimize_dtypes=False)
        assert DataCache.is_cached("recipes")

        DataCache.clear_cache()
        assert not DataCache.is_cached("recipes")
        assert not DataCache.is_cached("interactions")

    def test_clear_cache_selective(
        self, temp_data_dir, sample_recipes_df, sample_interactions_df
    ):
        """Test selective cache clearing."""
        recipes_path = temp_data_dir["raw"] / "recipes.csv"
        interactions_path = temp_data_dir["raw"] / "interactions.csv"
        sample_recipes_df.to_csv(recipes_path, index=False)
        sample_interactions_df.to_csv(interactions_path, index=False)

        DataCache.get_recipes(path=str(recipes_path), optimize_dtypes=False)
        DataCache.get_interactions(path=str(interactions_path), optimize_dtypes=False)

        DataCache.clear_cache(clear_recipes=True, clear_interactions=False)
        assert not DataCache.is_cached("recipes")
        assert DataCache.is_cached("interactions")

    def test_get_cache_info(self, temp_data_dir, sample_recipes_df):
        """Test cache info retrieval."""
        csv_path = temp_data_dir["raw"] / "recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        DataCache.get_recipes(path=str(csv_path), optimize_dtypes=False)

        info = DataCache.get_cache_info()
        assert info["recipes_cached"] is True
        assert info["interactions_cached"] is False
        assert info["total_memory_mb"] > 0

    def test_is_cached_unknown_type(self):
        """Test is_cached with unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown data type"):
            DataCache.is_cached("unknown")

    def test_get_recipes_processed_path(self, temp_data_dir, sample_recipes_df):
        """Test loading from processed path uses correct dtype handling."""
        csv_path = temp_data_dir["processed"] / "recipes_clean.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        df = DataCache.get_recipes(path=str(csv_path), optimize_dtypes=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_recipes_df)
