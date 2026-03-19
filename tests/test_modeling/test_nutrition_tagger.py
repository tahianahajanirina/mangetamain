"""Tests for nutrition tagger model."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.nutrition_tagger import NutritionTaggerModel, train_and_evaluate_nutrition_model


class TestNutritionTaggerModel:
    """Test cases for NutritionTaggerModel."""

    def test_init_random_forest(self):
        """Test initialization with random forest."""
        model = NutritionTaggerModel(model_type="random_forest")
        assert model.model_type == "random_forest"
        assert model.model is not None

    def test_init_logistic(self):
        """Test initialization with logistic regression."""
        model = NutritionTaggerModel(model_type="logistic", max_iter=500)
        assert model.model_type == "logistic"

    def test_init_gradient_boosting(self):
        """Test initialization with gradient boosting."""
        model = NutritionTaggerModel(model_type="gradient_boosting")
        assert model.model_type == "gradient_boosting"

    def test_init_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            NutritionTaggerModel(model_type="invalid")

    def test_prepare_features(self, sample_recipes_with_features):
        """Test feature preparation."""
        model = NutritionTaggerModel()
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["calories", "total_fat", "sugar"], "is_healthy")

        assert len(X) == len(df)
        assert len(y) == len(df)
        assert model.feature_names == ["calories", "total_fat", "sugar"]
        assert model.target_name == "is_healthy"

    def test_prepare_features_missing_columns(self, sample_recipes_with_features):
        """Test feature preparation with missing columns."""
        model = NutritionTaggerModel()
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(
            df, ["calories", "nonexistent_col"], "is_healthy"
        )

        assert "nonexistent_col" not in X.columns
        assert "calories" in model.feature_names

    def test_train_and_predict(self, sample_recipes_with_features):
        """Test training and prediction flow."""
        model = NutritionTaggerModel(model_type="random_forest", n_estimators=10, random_state=42)
        df = sample_recipes_with_features.copy()

        feature_cols = ["calories", "total_fat", "sugar", "protein"]
        X, y = model.prepare_features(df, feature_cols, "is_healthy")

        model.train(X, y, scale_features=True)
        predictions = model.predict(X, scale_features=True)

        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_evaluate(self, sample_recipes_with_features):
        """Test model evaluation."""
        model = NutritionTaggerModel(model_type="random_forest", n_estimators=10, random_state=42)
        df = sample_recipes_with_features.copy()

        feature_cols = ["calories", "total_fat", "sugar", "protein"]
        X, y = model.prepare_features(df, feature_cols, "is_healthy")

        model.train(X, y)
        metrics = model.evaluate(X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_predict_proba(self, sample_recipes_with_features):
        """Test probability predictions."""
        model = NutritionTaggerModel(model_type="random_forest", n_estimators=10, random_state=42)
        df = sample_recipes_with_features.copy()

        feature_cols = ["calories", "total_fat", "sugar", "protein"]
        X, y = model.prepare_features(df, feature_cols, "is_healthy")

        model.train(X, y)
        probas = model.predict_proba(X)

        assert probas.shape[0] == len(X)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_save_and_load_model(self, sample_recipes_with_features, tmp_path):
        """Test model save and load."""
        model = NutritionTaggerModel(model_type="random_forest", n_estimators=10, random_state=42)
        df = sample_recipes_with_features.copy()

        feature_cols = ["calories", "total_fat", "sugar", "protein"]
        X, y = model.prepare_features(df, feature_cols, "is_healthy")
        model.train(X, y)

        filepath = tmp_path / "test_model.pkl"
        model.save_model(filepath)

        loaded = NutritionTaggerModel.load_model(filepath)

        assert loaded.model_type == "random_forest"
        assert loaded.feature_names == feature_cols
        assert loaded.target_name == "is_healthy"

    def test_get_feature_importance(self, sample_recipes_with_features):
        """Test feature importance extraction."""
        model = NutritionTaggerModel(model_type="random_forest", n_estimators=10, random_state=42)
        df = sample_recipes_with_features.copy()

        feature_cols = ["calories", "total_fat", "sugar", "protein"]
        X, y = model.prepare_features(df, feature_cols, "is_healthy")
        model.train(X, y)

        importance = model.get_feature_importance(top_n=2)

        assert len(importance) == 2
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_train_and_evaluate_helper(self, sample_recipes_with_features):
        """Test the helper function."""
        df = sample_recipes_with_features.copy()
        feature_cols = ["calories", "total_fat", "sugar", "protein"]

        model, metrics = train_and_evaluate_nutrition_model(
            df, feature_cols, "is_healthy",
            model_type="random_forest",
            n_estimators=10, random_state=42,
        )

        assert model is not None
        assert "accuracy" in metrics
