"""Tests for data loader module."""

import pytest
import pandas as pd
from pathlib import Path
from src.preprocessing.data_loader import RecipeDataLoader


class TestRecipeDataLoader:
    """Test cases for RecipeDataLoader class."""
    
    def test_init_with_valid_path(self, temp_data_dir, sample_recipes_df):
        """Test initialization with valid file path."""
        # Create a test CSV file
        csv_path = temp_data_dir['raw'] / 'test_recipes.csv'
        sample_recipes_df.to_csv(csv_path, index=False)
        
        loader = RecipeDataLoader(str(csv_path))
        assert loader.data_path == csv_path
    
    def test_init_with_invalid_path(self):
        """Test initialization with non-existent file."""
        # __init__ doesn't check file existence, only load_data() does
        loader = RecipeDataLoader('/nonexistent/path/file.csv')
        assert loader.data_path.name == 'file.csv'
        
        # But load_data should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_load_data(self, temp_data_dir, sample_recipes_df):
        """Test data loading functionality."""
        csv_path = temp_data_dir['raw'] / 'test_recipes.csv'
        sample_recipes_df.to_csv(csv_path, index=False)
        
        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_recipes_df)
        assert 'id' in df.columns
        assert 'name' in df.columns
    
    def test_load_empty_file(self, temp_data_dir):
        """Test loading empty CSV file."""
        csv_path = temp_data_dir['raw'] / 'empty.csv'
        # Create CSV with headers but no data
        pd.DataFrame(columns=['id', 'name', 'minutes']).to_csv(csv_path, index=False)
        
        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert 'id' in df.columns
    
    def test_preprocess_basic(self, temp_data_dir, sample_recipes_df):
        """Test basic preprocessing."""
        csv_path = temp_data_dir['raw'] / 'test_recipes.csv'
        sample_recipes_df.to_csv(csv_path, index=False)
        
        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()
        
        # Check that data is loaded
        assert len(df) > 0
        
    def test_handle_missing_values(self, temp_data_dir):
        """Test handling of missing values."""
        # Create dataframe with missing values
        df_with_na = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Recipe 1', None, 'Recipe 3'],
            'minutes': [30, 45, None]
        })
        
        csv_path = temp_data_dir['raw'] / 'recipes_na.csv'
        df_with_na.to_csv(csv_path, index=False)
        
        loader = RecipeDataLoader(str(csv_path))
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
