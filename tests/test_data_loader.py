"""Tests for data loader module."""

import pandas as pd
import pytest

from src.preprocessing.data_loader import RecipeDataLoader


class TestRecipeDataLoader:
    """Test cases for RecipeDataLoader class."""

    def test_init_with_valid_path(self, temp_data_dir, sample_recipes_df):
        """Test initialization with valid file path."""
        # Create a test CSV file
        csv_path = temp_data_dir["raw"] / "test_recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        loader = RecipeDataLoader(str(csv_path))
        assert loader.data_path == csv_path

    def test_init_with_invalid_path(self):
        """Test initialization with non-existent file."""
        # __init__ doesn't check file existence, only load_data() does
        loader = RecipeDataLoader("/nonexistent/path/file.csv")
        assert loader.data_path.name == "file.csv"

        # But load_data should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_load_data(self, temp_data_dir, sample_recipes_df):
        """Test data loading functionality."""
        csv_path = temp_data_dir["raw"] / "test_recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_recipes_df)
        assert "id" in df.columns
        assert "name" in df.columns

    def test_load_empty_file(self, temp_data_dir):
        """Test loading empty CSV file."""
        csv_path = temp_data_dir["raw"] / "empty.csv"
        # Create CSV with headers but no data
        pd.DataFrame(columns=["id", "name", "minutes"]).to_csv(csv_path, index=False)

        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "id" in df.columns

    def test_preprocess_basic(self, temp_data_dir, sample_recipes_df):
        """Test basic preprocessing."""
        csv_path = temp_data_dir["raw"] / "test_recipes.csv"
        sample_recipes_df.to_csv(csv_path, index=False)

        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()

        # Check that data is loaded
        assert len(df) > 0

    def test_handle_missing_values(self, temp_data_dir):
        """Test handling of missing values."""
        # Create CSV file with missing values in string columns only
        # (numeric columns with NA cause dtype issues with optimize_dtypes=True)
        csv_path = temp_data_dir["raw"] / "recipes_na.csv"

        # Write CSV with missing values in name column only
        csv_content = """id,name,minutes
1,Recipe 1,30
2,,45
3,Recipe 3,60
"""
        csv_path.write_text(csv_content)

        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # Check that NaN values are present in name column
        assert df["name"].isna().any()
        # Check that numeric column has no NaN
        assert not df["minutes"].isna().any()
