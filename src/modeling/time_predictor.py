"""Model training and evaluation for recipe time prediction.

This module handles model training, evaluation, and prediction
for the recipe preparation time prediction task.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TimePredictionModel:
    """Model for predicting recipe preparation time.

    Attributes:
        model: Trained model instance.
        scaler: Feature scaler.
        feature_names: List of feature names used for training.
        metrics: Dictionary of evaluation metrics.
    """

    def __init__(self, model_type: str = "random_forest", **model_params):
        """Initialize the time prediction model.

        Args:
            model_type: Type of model to use.
            **model_params: Additional model parameters.
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, model_params)
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.metrics: Dict[str, float] = {}

    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance based on type.

        Args:
            model_type: Type of model.
            params: Model parameters.

        Returns:
            Model instance.

        Raises:
            ValueError: If model type is not supported.
        """
        models = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
        }

        if model_type not in models:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Choose from {list(models.keys())}"
            )

        return models[model_type](**params)

    def prepare_features(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "minutes"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.

        Args:
            df: Input DataFrame.
            feature_cols: List of feature column names.
            target_col: Target column name.

        Returns:
            Tuple of (features, target).
        """
        logger.info(f"Preparing {len(feature_cols)} features for modeling")

        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]

        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        X = df[available_features].copy()
        y = df[target_col].copy()

        # Handle missing values
        X = X.fillna(0)

        self.feature_names = available_features
        logger.info(f"Using {len(self.feature_names)} features")

        return X, y

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, scale_features: bool = True
    ):
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training target.
            scale_features: Whether to scale features.
        """
        logger.info(f"Training {self.model_type} model")

        if scale_features:
            X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def predict(self, X: pd.DataFrame, scale_features: bool = True) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features for prediction.
            scale_features: Whether to scale features.

        Returns:
            Array of predictions.
        """
        if scale_features:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, scale_features: bool = True
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test features.
            y_test: Test target.
            scale_features: Whether to scale features.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("Evaluating model")

        y_pred = self.predict(X_test, scale_features=scale_features)

        self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
        }

        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5, scale_features: bool = True
    ) -> Dict[str, float]:
        """Perform cross-validation.

        Args:
            X: Features.
            y: Target.
            cv: Number of cross-validation folds.
            scale_features: Whether to scale features.

        Returns:
            Dictionary of cross-validation scores.
        """
        logger.info(f"Performing {cv}-fold cross-validation")

        if scale_features:
            X = self.scaler.fit_transform(X)

        scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )

        cv_metrics = {"cv_mae_mean": -scores.mean(), "cv_mae_std": scores.std()}

        logger.info(f"CV metrics: {cv_metrics}")
        return cv_metrics

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with feature names and importance scores.

        Raises:
            ValueError: If model doesn't support feature importance.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError(f"{self.model_type} doesn't support feature importance")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save_model(self, filepath: Path):
        """Save model to file.

        Args:
            filepath: Path to save the model.
        """
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "model_type": self.model_type,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> "TimePredictionModel":
        """Load model from file.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded model instance.
        """
        logger.info(f"Loading model from {filepath}")
        model_data = joblib.load(filepath)

        instance = cls(model_type=model_data["model_type"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.metrics = model_data["metrics"]

        return instance


def train_and_evaluate_time_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "minutes",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    **model_params,
) -> Tuple[TimePredictionModel, Dict[str, float]]:
    """Train and evaluate a time prediction model.

    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.
        model_type: Type of model to use.
        test_size: Test set size.
        random_state: Random seed (for train-test split and models that support it).
        **model_params: Additional model parameters.

    Returns:
        Tuple of (trained model, metrics).
    """
    logger.info(f"Training {model_type} model for time prediction")

    # Prepare model parameters
    model_params_copy = model_params.copy()

    # Models that support random_state parameter
    models_with_random_state = {"ridge", "random_forest", "gradient_boosting"}

    # Only add random_state if the model supports it and it's not already specified
    if (
        model_type in models_with_random_state
        and "random_state" not in model_params_copy
    ):
        model_params_copy["random_state"] = random_state

    # Initialize model
    model = TimePredictionModel(model_type=model_type, **model_params_copy)

    # Prepare features
    X, y = model.prepare_features(df, feature_cols, target_col)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model.train(X_train, y_train)

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)

    logger.info(
        f"Model training completed. R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f}"
    )

    return model, metrics
