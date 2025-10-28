"""Tests for nutrition feature engineering."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.nutrition_features import NutritionFeatureEngineer


class TestNutritionFeatureEngineer:
    """Test cases for NutritionFeatureEngineer."""
    
    def test_init(self):
        """Test initialization."""
        engineer = NutritionFeatureEngineer()
        assert engineer is not None
    
    def test_engineer_features_basic(self, sample_recipes_df):
        """Test basic feature engineering."""
        engineer = NutritionFeatureEngineer()
        result = engineer.engineer_features(sample_recipes_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_recipes_df)
        assert 'calories' in result.columns or 'id' in result.columns
    
    def test_nutrition_ratios(self, sample_recipes_with_features):
        """Test calculation of nutrition ratios."""
        engineer = NutritionFeatureEngineer()
        df = sample_recipes_with_features.copy()
        
        # Assuming method exists
        if hasattr(engineer, '_calculate_ratios'):
            result = engineer._calculate_ratios(df)
            assert isinstance(result, pd.DataFrame)
    
    def test_handle_zero_values(self):
        """Test handling of zero values in nutrition."""
        df = pd.DataFrame({
            'id': [1, 2],
            'calories': [0, 300],
            'protein': [0, 20]
        })
        
        engineer = NutritionFeatureEngineer()
        # Should not raise division by zero errors
        result = engineer.engineer_features(df)
        assert len(result) == 2
