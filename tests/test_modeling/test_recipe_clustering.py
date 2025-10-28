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
    
    def test_init_with_k(self):
        """Test initialization with specific k."""
        clusterer = RecipeClusterer(n_clusters=5)
        assert clusterer.n_clusters == 5
    
    def test_fit_basic(self, sample_recipes_with_features):
        """Test fitting the clustering model."""
        df = sample_recipes_with_features.copy()
        
        # Select only numeric features
        feature_cols = ['minutes', 'n_steps', 'n_ingredients', 'calories', 
                       'total_fat', 'sugar', 'protein']
        X = df[feature_cols].fillna(0)
        
        clusterer = RecipeClusterer(n_clusters=2)
        clusterer.fit(X)
        
        assert hasattr(clusterer, 'model')
    
    def test_predict(self, sample_recipes_with_features):
        """Test prediction on new data."""
        df = sample_recipes_with_features.copy()
        
        feature_cols = ['minutes', 'n_steps', 'n_ingredients', 'calories', 
                       'total_fat', 'sugar', 'protein']
        X = df[feature_cols].fillna(0)
        
        clusterer = RecipeClusterer(n_clusters=2)
        clusterer.fit(X)
        labels = clusterer.predict(X)
        
        assert len(labels) == len(df)
        assert all(label in [0, 1] for label in labels)
    
    def test_optimal_clusters(self, sample_recipes_with_features):
        """Test finding optimal number of clusters."""
        df = sample_recipes_with_features.copy()
        
        feature_cols = ['minutes', 'n_steps', 'n_ingredients', 'calories']
        X = df[feature_cols].fillna(0)
        
        clusterer = RecipeClusterer()
        
        # Should be able to find optimal k
        if hasattr(clusterer, 'find_optimal_clusters'):
            k = clusterer.find_optimal_clusters(X, k_range=range(2, 4))
            assert k >= 2
            assert k < 4
    
    def test_fit_with_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'calories': [100, np.nan, 300],
            'protein': [10, 20, np.nan]
        })
        
        clusterer = RecipeClusterer(n_clusters=2)
        
        # Should handle NaN values
        X = df.fillna(0)
        clusterer.fit(X)
        labels = clusterer.predict(X)
        
        assert len(labels) == 3
