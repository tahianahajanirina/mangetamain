"""Tests for clean_data script."""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from clean_data import safe_eval, clean_recipes, clean_interactions


class TestSafeEval:
    """Test safe_eval function."""
    
    def test_safe_eval_string(self):
        """Test evaluating string representation of list."""
        result = safe_eval("['a', 'b', 'c']")
        assert result == ['a', 'b', 'c']
    
    def test_safe_eval_list(self):
        """Test passing actual list."""
        result = safe_eval(['a', 'b', 'c'])
        assert result == ['a', 'b', 'c']
    
    def test_safe_eval_nan(self):
        """Test handling NaN values."""
        result = safe_eval(pd.NA)
        assert result == []
    
    def test_safe_eval_invalid(self):
        """Test handling invalid input."""
        result = safe_eval("invalid{syntax")
        assert result == []


class TestCleanRecipes:
    """Test clean_recipes function."""
    
    def test_clean_recipes_basic(self, temp_data_dir, sample_recipes_df):
        """Test basic recipe cleaning."""
        # Save raw data
        raw_path = temp_data_dir['raw'] / 'RAW_recipes.csv'
        sample_recipes_df.to_csv(raw_path, index=False)
        
        output_path = temp_data_dir['processed'] / 'recipes_clean.csv'
        
        result = clean_recipes(raw_path, output_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert output_path.exists()
    
    def test_clean_recipes_remove_duplicates(self, temp_data_dir):
        """Test duplicate removal."""
        df_with_dupes = pd.DataFrame({
            'id': [1, 1, 2, 3],
            'name': ['Recipe 1', 'Recipe 1', 'Recipe 2', 'Recipe 3'],
            'minutes': [30, 30, 45, 20]
        })
        
        raw_path = temp_data_dir['raw'] / 'RAW_recipes.csv'
        df_with_dupes.to_csv(raw_path, index=False)
        
        output_path = temp_data_dir['processed'] / 'recipes_clean.csv'
        
        result = clean_recipes(raw_path, output_path)
        
        # Should remove duplicates
        assert len(result) == 3
        assert result['id'].is_unique


class TestCleanInteractions:
    """Test clean_interactions function."""
    
    def test_clean_interactions_basic(self, temp_data_dir, sample_interactions_df):
        """Test basic interaction cleaning."""
        raw_path = temp_data_dir['raw'] / 'RAW_interactions.csv'
        sample_interactions_df.to_csv(raw_path, index=False)
        
        output_path = temp_data_dir['processed'] / 'interactions_clean.csv'
        
        result = clean_interactions(raw_path, output_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert output_path.exists()
