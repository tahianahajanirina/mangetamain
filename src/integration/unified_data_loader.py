"""
Unified Recipe Data Loader
Handles data loading and preprocessing for all models in the integrated pipeline.
"""

import ast
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from config.config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


class UnifiedRecipeDataLoader:
    """
    Unified data loader that prepares recipe data for all models.

    Handles:
    - Loading recipes from CSV
    - String to list conversions
    - Nutrition extraction
    - Basic preprocessing
    """

    def __init__(self, recipes_path: str = "data/RAW_recipes.csv"):
        """
        Initialize the data loader.

        Args:
            recipes_path: Path to RAW_recipes.csv
        """
        self.recipes_path = Path(recipes_path)
        self._recipes_cache = None

    def load_all_recipes(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load all recipes from CSV.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with all recipes
        """
        if use_cache and self._recipes_cache is not None:
            logger.info("Using cached recipe data")
            return self._recipes_cache.copy()

        logger.info(f"Loading recipes from {self.recipes_path}")
        df = pd.read_csv(self.recipes_path)

        # Basic preprocessing
        df = self._preprocess_basic(df)

        # Cache the data
        self._recipes_cache = df.copy()

        logger.info(f"Loaded {len(df)} recipes")
        return df

    def load_recipes_by_ids(self, recipe_ids: List[int]) -> pd.DataFrame:
        """
        Load specific recipes by their IDs.

        Args:
            recipe_ids: List of recipe IDs to load

        Returns:
            DataFrame with requested recipes
        """
        # Load all recipes (uses cache if available)
        all_recipes = self.load_all_recipes(use_cache=True)

        # Filter by IDs
        recipes = all_recipes[all_recipes["id"].isin(recipe_ids)].copy()

        logger.info(f"Loaded {len(recipes)} recipes out of {len(recipe_ids)} requested")
        return recipes

    def _preprocess_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing: convert strings to lists, extract nutrition.

        Args:
            df: Raw dataframe

        Returns:
            Preprocessed dataframe
        """
        df = df.copy()

        # Convert string columns to lists
        list_columns = ["ingredients", "steps", "tags", "nutrition"]

        for col in list_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].apply(self._safe_literal_eval)
                except Exception as e:
                    logger.warning(f"Error converting {col} to list: {e}")

        # Extract nutrition values if nutrition column exists
        if "nutrition" in df.columns:
            df = self._extract_nutrition(df)

        # Add n_ingredients and n_steps if not present
        if "n_ingredients" not in df.columns and "ingredients" in df.columns:
            df["n_ingredients"] = df["ingredients"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

        if "n_steps" not in df.columns and "steps" in df.columns:
            df["n_steps"] = df["steps"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

        # Fill missing values
        df = self._fill_missing_values(df)

        return df

    @staticmethod
    def _safe_literal_eval(val):
        """Safely evaluate string to list."""
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return []

    def _extract_nutrition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract nutrition values from nutrition column.

        Nutrition format: [calories, total_fat, sugar, sodium, protein, saturated_fat, carbs]

        Args:
            df: DataFrame with nutrition column

        Returns:
            DataFrame with extracted nutrition columns
        """
        df = df.copy()

        nutrition_cols = [
            "calories",
            "total_fat_pdv",
            "sugar_pdv",
            "sodium_pdv",
            "protein_pdv",
            "saturated_fat_pdv",
            "carbs_pdv",
        ]

        # Check if nutrition values are already extracted
        if all(col in df.columns for col in nutrition_cols):
            return df

        if "nutrition" not in df.columns:
            logger.warning("No nutrition column found")
            return df

        try:
            # Extract nutrition values
            nutrition_df = pd.DataFrame(
                df["nutrition"].tolist(), columns=nutrition_cols, index=df.index
            )

            # Add to main dataframe
            for col in nutrition_cols:
                df[col] = nutrition_df[col]

            logger.info("Nutrition values extracted successfully")

        except Exception as e:
            logger.error(f"Error extracting nutrition: {e}")
            # Fill with NaN if extraction fails
            for col in nutrition_cols:
                if col not in df.columns:
                    df[col] = np.nan

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with appropriate defaults.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with filled values
        """
        df = df.copy()

        # Fill numeric nutrition columns with median
        nutrition_cols = [
            "calories",
            "total_fat_pdv",
            "sugar_pdv",
            "sodium_pdv",
            "protein_pdv",
            "saturated_fat_pdv",
            "carbs_pdv",
        ]

        for col in nutrition_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # Fill list columns with empty lists
        list_columns = ["ingredients", "steps", "tags"]
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        # Fill text columns with empty strings
        text_columns = ["name", "description"]
        for col in text_columns:
            if col in df.columns:
                df[col].fillna("", inplace=True)

        # Fill numeric columns
        if "minutes" in df.columns:
            df["minutes"].fillna(df["minutes"].median(), inplace=True)

        if "n_steps" in df.columns:
            df["n_steps"].fillna(0, inplace=True)

        if "n_ingredients" in df.columns:
            df["n_ingredients"].fillna(0, inplace=True)

        return df

    def prepare_for_time_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for time prediction model.

        Requires: name, steps, ingredients, tags, n_steps, n_ingredients

        Args:
            df: DataFrame with basic preprocessing done

        Returns:
            DataFrame with time prediction features
        """
        from src.feature_engineering.time_features import TimeFeatureEngineer

        df = df.copy()

        # Initialize time feature engineer
        time_fe = TimeFeatureEngineer(
            cooking_verbs=FEATURE_CONFIG["cooking_verbs"],
            equipment_terms=FEATURE_CONFIG["equipment_terms"],
        )

        # Engineer features
        df = time_fe.engineer_features(df)

        logger.info("Time prediction features engineered")
        return df

    def prepare_for_nutrition_tagging(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for nutrition tagging model.

        Requires: nutrition columns, ingredients, tags

        Args:
            df: DataFrame with basic preprocessing done

        Returns:
            DataFrame with nutrition features
        """
        from src.feature_engineering.nutrition_features import NutritionFeatureEngineer

        df = df.copy()

        # Initialize nutrition feature engineer
        nutrition_fe = NutritionFeatureEngineer(
            ingredient_categories=FEATURE_CONFIG["ingredient_categories"],
            dietary_patterns=FEATURE_CONFIG["dietary_patterns"],
            nutrition_cols=FEATURE_CONFIG["nutrition_cols"],
        )

        # Engineer features
        df = nutrition_fe.engineer_features(df, include_pca=False)

        logger.info("Nutrition features engineered")
        return df

    def get_recipe_display_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get display-friendly recipe information.

        Args:
            df: DataFrame with recipes

        Returns:
            DataFrame with display columns
        """
        display_cols = [
            "id",
            "name",
            "minutes",
            "n_steps",
            "n_ingredients",
            "description",
            "calories",
        ]

        # Select available columns
        available_cols = [col for col in display_cols if col in df.columns]

        result = df[available_cols].copy()

        # Format description (truncate if too long)
        if "description" in result.columns:
            result["description"] = result["description"].apply(
                lambda x: x[:150] + "..." if len(x) > 150 else x
            )

        return result

    def get_feature_names_for_model(self, model_type: str) -> List[str]:
        """
        Get feature names required for a specific model type.

        Args:
            model_type: 'time_prediction' or 'nutrition_tagging'

        Returns:
            List of feature names
        """
        if model_type == "time_prediction":
            from src.feature_engineering.time_features import TimeFeatureEngineer

            time_fe = TimeFeatureEngineer(
                cooking_verbs=set(),  # Empty sets for feature name retrieval
                equipment_terms=set(),
            )
            return time_fe.get_feature_names()

        elif model_type == "nutrition_tagging":
            from src.feature_engineering.nutrition_features import (
                NutritionFeatureEngineer,
            )

            nutrition_fe = NutritionFeatureEngineer(
                ingredient_categories={},
                dietary_patterns={},
                nutrition_cols=FEATURE_CONFIG["nutrition_cols"],
            )
            return nutrition_fe.get_feature_names(include_pca=False)

        else:
            raise ValueError(f"Unknown model type: {model_type}")
