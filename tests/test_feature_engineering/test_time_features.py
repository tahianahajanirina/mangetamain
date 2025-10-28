"""Tests for time feature engineering."""

import pytest
import pandas as pd
from src.feature_engineering.time_features import TimeFeatureEngineer


class TestTimeFeatureEngineer:
    """Test cases for TimeFeatureEngineer."""
    
    def test_init(self):
        """Test initialization."""
        engineer = TimeFeatureEngineer()
        assert engineer is not None
    
    def test_engineer_features(self, sample_recipes_df):
        """Test feature engineering for time prediction."""
        engineer = TimeFeatureEngineer()
        result = engineer.engineer_features(sample_recipes_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_time_categories(self, sample_recipes_df):
        """Test time categorization."""
        engineer = TimeFeatureEngineer()
        df = sample_recipes_df.copy()
        
        # Should categorize recipes by time
        if hasattr(engineer, '_categorize_time'):
            result = engineer._categorize_time(df)
            assert 'time_category' in result.columns
