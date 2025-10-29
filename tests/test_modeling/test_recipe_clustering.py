"""Tests for recipe clustering model."""

import pytest
import pandas as pd
import numpy as np
from src.modeling.recipe_clustering import RecipeClusterer


class TestRecipeClusterer:
    """Test cases for RecipeClusterer."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        clusterer = RecipeClusterer()
        assert clusterer is not None
        assert clusterer.pca_variance == 0.90
    
    def test_init_with_k(self):
        """Test initialization with specific k."""
        clusterer = RecipeClusterer(n_clusters=5)
        assert clusterer.n_clusters == 5
    
    def test_load_features(self, temp_data_dir, sample_recipes_with_features):
        """Test loading features from file."""
        # Add required columns for clustering
        df = sample_recipes_with_features.copy()
        df['log_minutes'] = np.log1p(df['minutes'])
        df['time_complexity'] = df['n_steps'] * df['n_ingredients']
        df['efficiency'] = df['avg_rating'] / (1 + df['log_minutes'])
        df['health_category'] = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
        df['popularity_score'] = df['avg_rating'] * df['num_ratings']
        
        features_path = temp_data_dir['processed'] / 'recipe_features.csv'
        df.to_csv(features_path, index=False)
        
        clusterer = RecipeClusterer(n_clusters=3)
        result = clusterer.load_features(str(features_path))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert clusterer.recipes is not None
    
    def test_prepare_features(self, temp_data_dir, sample_recipes_with_features):
        """Test feature preparation (scaling and PCA)."""
        df = sample_recipes_with_features.copy()
        df['log_minutes'] = np.log1p(df['minutes'])
        df['time_complexity'] = df['n_steps'] * df['n_ingredients']
        df['efficiency'] = df['avg_rating'] / (1 + df['log_minutes'])
        df['health_category'] = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
        df['popularity_score'] = df['avg_rating'] * df['num_ratings']
        
        clusterer = RecipeClusterer(n_clusters=3)
        clusterer.recipes = df
        
        features_scaled, features_pca = clusterer.prepare_features()
        
        assert features_scaled is not None
        assert features_pca is not None
        assert len(features_scaled) == len(df)
        assert len(features_pca) == len(df)
    
    def test_fit_with_features_file(self, temp_data_dir, sample_recipes_with_features):
        """Test complete fit pipeline with file path."""
        df = sample_recipes_with_features.copy()
        df['log_minutes'] = np.log1p(df['minutes'])
        df['time_complexity'] = df['n_steps'] * df['n_ingredients']
        df['efficiency'] = df['avg_rating'] / (1 + df['log_minutes'])
        df['health_category'] = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
        df['popularity_score'] = df['avg_rating'] * df['num_ratings']
        
        features_path = temp_data_dir['processed'] / 'recipe_features.csv'
        df.to_csv(features_path, index=False)
        
        clusterer = RecipeClusterer(n_clusters=3)
        clusterer.fit(str(features_path))
        
        assert clusterer.kmeans is not None
        assert 'cluster' in clusterer.recipes.columns
        assert clusterer.recipes['cluster'].nunique() == 3
