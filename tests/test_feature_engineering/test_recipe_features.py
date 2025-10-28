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
    
    def test_parse_nutrition_valid(self, sample_recipes_df):
        """Test parsing nutrition information."""
        builder = RecipeFeatureBuilder()
        df = sample_recipes_df.copy()
        
        result = builder._parse_nutrition(df)
        
        assert 'calories' in result.columns
        assert 'total_fat' in result.columns
        assert 'sugar' in result.columns
        assert 'sodium' in result.columns
        assert 'protein' in result.columns
        assert 'saturated_fat' in result.columns
        assert 'carbohydrates' in result.columns
        
        # Check that values are numeric
        assert pd.api.types.is_numeric_dtype(result['calories'])
        assert result['calories'].notna().all()
    
    def test_parse_nutrition_missing_values(self):
        """Test parsing nutrition with missing values."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'nutrition': ['[100, 10, 5, 2, 8, 15, 3]', None, '[150, 12, 6, 3, 10, 18, 4]']
        })
        
        builder = RecipeFeatureBuilder()
        result = builder._parse_nutrition(df)
        
        assert len(result) == 3
        assert result['calories'].isna().sum() == 1
    
    def test_extract_tags_features(self, sample_recipes_df):
        """Test extraction of tag-based features."""
        builder = RecipeFeatureBuilder()
        df = sample_recipes_df.copy()
        
        result = builder._extract_tags_features(df)
        
        # Check boolean columns exist
        assert 'is_dessert' in result.columns
        assert 'is_healthy' in result.columns
        assert 'is_quick' in result.columns
        
        # Check they are boolean/binary
        assert result['is_dessert'].isin([0, 1, True, False]).all()
        
    def test_calculate_complexity_score(self, sample_recipes_df):
        """Test complexity score calculation."""
        builder = RecipeFeatureBuilder()
        df = sample_recipes_df.copy()
        
        result = builder._calculate_complexity_score(df)
        
        assert 'complexity_score' in result.columns
        assert pd.api.types.is_numeric_dtype(result['complexity_score'])
        assert (result['complexity_score'] >= 0).all()
    
    def test_build_features_full_pipeline(self, temp_data_dir, sample_recipes_df, sample_interactions_df):
        """Test complete feature building pipeline."""
        # Save test data
        recipes_path = temp_data_dir['processed'] / 'recipes_clean.csv'
        interactions_path = temp_data_dir['processed'] / 'interactions_clean.csv'
        
        sample_recipes_df.to_csv(recipes_path, index=False)
        sample_interactions_df.to_csv(interactions_path, index=False)
        
        builder = RecipeFeatureBuilder(min_rating=3.0)
        result = builder.build_features(
            str(recipes_path),
            str(interactions_path)
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'id' in result.columns
        
    def test_handle_invalid_nutrition_format(self):
        """Test handling of invalid nutrition format."""
        df = pd.DataFrame({
            'id': [1, 2],
            'nutrition': ['invalid', '[100, 10, 5]']  # invalid formats
        })
        
        builder = RecipeFeatureBuilder()
        result = builder._parse_nutrition(df)
        
        # Should handle gracefully
        assert len(result) == 2
