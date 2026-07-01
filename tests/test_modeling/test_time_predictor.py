"""Tests for time predictor model."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.time_predictor import TimePredictionModel, train_and_evaluate_time_model


class TestTimePredictionModel:
    """Test cases for TimePredictionModel."""

    def test_init_linear(self):
        """Test initialization with linear regression."""
        model = TimePredictionModel(model_type="linear")
        assert model.model_type == "linear"

    def test_init_ridge(self):
        """Test initialization with ridge regression."""
        model = TimePredictionModel(model_type="ridge", alpha=0.5)
        assert model.model_type == "ridge"

    def test_init_random_forest(self):
        """Test initialization with random forest."""
        model = TimePredictionModel(
            model_type="random_forest", n_estimators=10, random_state=42
        )
        assert model.model_type == "random_forest"

    def test_init_gradient_boosting(self):
        """Test initialization with gradient boosting."""
        model = TimePredictionModel(
            model_type="gradient_boosting", n_estimators=10, random_state=42
        )
        assert model.model_type == "gradient_boosting"

    def test_init_invalid_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            TimePredictionModel(model_type="invalid_model")

    def test_prepare_features(self, sample_recipes_with_features):
        """Test feature preparation."""
        model = TimePredictionModel()
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients", "calories"])

        assert len(X) == len(df)
        assert len(y) == len(df)
        assert model.feature_names == ["n_steps", "n_ingredients", "calories"]

    def test_prepare_features_missing_columns(self, sample_recipes_with_features):
        """Test feature preparation with missing columns."""
        model = TimePredictionModel()
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "nonexistent_column"])

        assert "nonexistent_column" not in X.columns
        assert "n_steps" in model.feature_names

    def test_train_and_predict(self, sample_recipes_with_features):
        """Test training and prediction."""
        model = TimePredictionModel(
            model_type="random_forest", n_estimators=10, random_state=42
        )
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients", "calories"])
        model.train(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert all(isinstance(p, (int, float, np.floating)) for p in predictions)

    def test_evaluate(self, sample_recipes_with_features):
        """Test model evaluation."""
        model = TimePredictionModel(
            model_type="random_forest", n_estimators=10, random_state=42
        )
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients", "calories"])
        model.train(X, y)
        metrics = model.evaluate(X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_cross_validate(self, sample_recipes_with_features):
        """Test cross-validation."""
        model = TimePredictionModel(
            model_type="random_forest", n_estimators=10, random_state=42
        )
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients", "calories"])
        cv_metrics = model.cross_validate(X, y, cv=2)

        assert "cv_mae_mean" in cv_metrics
        assert "cv_mae_std" in cv_metrics
        assert cv_metrics["cv_mae_mean"] >= 0

    def test_get_feature_importance(self, sample_recipes_with_features):
        """Test feature importance retrieval."""
        model = TimePredictionModel(
            model_type="random_forest", n_estimators=10, random_state=42
        )
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients", "calories"])
        model.train(X, y)
        importance = model.get_feature_importance(top_n=2)

        assert len(importance) == 2
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_get_feature_importance_linear_raises(self, sample_recipes_with_features):
        """Test that linear model raises on feature importance."""
        model = TimePredictionModel(model_type="linear")
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients"])
        model.train(X, y)

        with pytest.raises(ValueError, match="doesn't support feature importance"):
            model.get_feature_importance()

    def test_save_and_load(self, sample_recipes_with_features, tmp_path):
        """Test model save and load."""
        model = TimePredictionModel(
            model_type="random_forest", n_estimators=10, random_state=42
        )
        df = sample_recipes_with_features.copy()

        X, y = model.prepare_features(df, ["n_steps", "n_ingredients", "calories"])
        model.train(X, y)
        model.evaluate(X, y)

        filepath = tmp_path / "model.pkl"
        model.save_model(filepath)

        loaded = TimePredictionModel.load_model(filepath)

        assert loaded.model_type == "random_forest"
        assert loaded.feature_names == ["n_steps", "n_ingredients", "calories"]
        assert "mae" in loaded.metrics

    def test_train_and_evaluate_helper(self, sample_recipes_with_features):
        """Test the helper function."""
        df = sample_recipes_with_features.copy()
        feature_cols = ["n_steps", "n_ingredients", "calories"]

        model, metrics = train_and_evaluate_time_model(
            df, feature_cols,
            model_type="random_forest",
            n_estimators=10,
            test_size=0.3,
            random_state=42,
        )

        assert model is not None
        assert "mae" in metrics
        assert "r2" in metrics
