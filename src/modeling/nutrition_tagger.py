"""Model training and evaluation for nutritional value tagging.

This module handles model training, evaluation, and prediction
for the nutritional value estimation and tagging task.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class NutritionTaggerModel:
    """Model for tagging recipes based on nutritional values.

    This model can predict various nutritional tags such as:
    - high_calorie / low_calorie
    - high_protein
    - low_fat
    - low_sugar

    Attributes:
        model: Trained model instance.
        scaler: Feature scaler.
        feature_names: List of feature names used for training.
        metrics: Dictionary of evaluation metrics.
        target_name: Name of the target variable.
    """

    def __init__(self, model_type: str = "random_forest", **model_params):
        """Initialize the nutrition tagger model.

        Args:
            model_type: Type of model to use.
            **model_params: Additional model parameters.
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, model_params)
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.metrics: Dict[str, float] = {}
        self.target_name: Optional[str] = None

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
            "logistic": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier
        }

        if model_type not in models:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Choose from {list(models.keys())}"
            )

        return models[model_type](**params)

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.

        Args:
            df: Input DataFrame.
            feature_cols: List of feature column names.
            target_col: Target column name.

        Returns:
            Tuple of (features, target).
        """
        logger.info(f"Preparing {len(feature_cols)} features for {target_col}")

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
        self.target_name = target_col

        logger.info(
            f"Using {len(self.feature_names)} features for {target_col}. "
            f"Class distribution: {y.value_counts().to_dict()}"
        )

        return X, y

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_features: bool = True
    ):
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training target.
            scale_features: Whether to scale features.
        """
        logger.info(f"Training {self.model_type} model for {self.target_name}")

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

    def predict_proba(
        self,
        X: pd.DataFrame,
        scale_features: bool = True
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features for prediction.
            scale_features: Whether to scale features.

        Returns:
            Array of class probabilities.
        """
        if scale_features:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test features.
            y_test: Test target.
            scale_features: Whether to scale features.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model for {self.target_name}")

        y_pred = self.predict(X_test, scale_features=scale_features)

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        logger.info(f"Evaluation metrics: {self.metrics}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

        return self.metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scale_features: bool = True
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
        logger.info(f"Performing {cv}-fold cross-validation for {self.target_name}")

        if scale_features:
            X = self.scaler.fit_transform(X)

        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='f1_weighted'
        )

        cv_metrics = {
            "cv_f1_mean": scores.mean(),
            "cv_f1_std": scores.std()
        }

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
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError(
                f"{self.model_type} doesn't support feature importance"
            )

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save_model(self, filepath: Path):
        """Save model to file.

        Args:
            filepath: Path to save the model.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_type': self.model_type,
            'target_name': self.target_name
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> 'NutritionTaggerModel':
        """Load model from file.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded model instance.
        """
        logger.info(f"Loading model from {filepath}")
        model_data = joblib.load(filepath)

        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']
        instance.target_name = model_data['target_name']

        return instance


def train_and_evaluate_nutrition_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    **model_params
) -> Tuple[NutritionTaggerModel, Dict[str, float]]:
    """Train and evaluate a nutrition tagging model.

    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name (e.g., 'high_protein', 'low_calorie').
        model_type: Type of model to use.
        test_size: Test set size.
        random_state: Random seed.
        **model_params: Additional model parameters.

    Returns:
        Tuple of (trained model, metrics).
    """
    logger.info(f"Training {model_type} model for {target_col}")

    # Initialize model
    model = NutritionTaggerModel(model_type=model_type, **model_params)

    # Prepare features
    X, y = model.prepare_features(df, feature_cols, target_col)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    model.train(X_train, y_train)

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)

    logger.info(
        f"Model training completed. F1: {metrics['f1']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    return model, metrics
