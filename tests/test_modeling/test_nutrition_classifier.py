"""Tests for nutrition classifier model."""

import pytest
import pandas as pd
import numpy as np
from src.modeling.nutrition_classifier import NutritionClassifier


class TestNutritionClassifier:
    """Test cases for NutritionClassifier."""
    
    def test_init_random_forest(self):
        """Test initialization with random forest."""
        classifier = NutritionClassifier(model_type='random_forest')
        assert classifier.model_type == 'random_forest'
    
    def test_init_gradient_boosting(self):
        """Test initialization with gradient boosting."""
        classifier = NutritionClassifier(model_type='gradient_boosting')
        assert classifier.model_type == 'gradient_boosting'
    
    def test_train_basic(self, sample_recipes_with_features):
        """Test basic training."""
        df = sample_recipes_with_features.copy()
        
        feature_cols = ['calories', 'total_fat', 'sugar', 'protein']
        X = df[feature_cols]
        y = df['is_healthy']
        
        classifier = NutritionClassifier(model_type='random_forest')
        classifier.train(X, y)
        
        assert hasattr(classifier, 'model')
    
    def test_predict(self, sample_recipes_with_features):
        """Test prediction."""
        df = sample_recipes_with_features.copy()
        
        feature_cols = ['calories', 'total_fat', 'sugar', 'protein']
        X = df[feature_cols]
        y = df['is_healthy']
        
        classifier = NutritionClassifier(model_type='random_forest')
        classifier.train(X, y)
        predictions = classifier.predict(X)
        
        assert len(predictions) == len(df)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_evaluate(self, sample_recipes_with_features):
        """Test model evaluation."""
        df = sample_recipes_with_features.copy()
        
        feature_cols = ['calories', 'total_fat', 'sugar', 'protein']
        X = df[feature_cols]
        y = df['is_healthy']
        
        classifier = NutritionClassifier(model_type='random_forest')
        classifier.train(X, y)
        
        if hasattr(classifier, 'evaluate'):
            metrics = classifier.evaluate(X, y)
            assert isinstance(metrics, dict)
