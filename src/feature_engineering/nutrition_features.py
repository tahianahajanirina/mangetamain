"""Feature engineering for nutritional value estimation task.

This module contains feature engineering functions specifically
designed for estimating and tagging nutritional values.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class NutritionFeatureEngineer:
    """Engineer features for nutritional value estimation and tagging.

    This class creates features that are most relevant for estimating
    nutritional values and tagging recipes, focusing on:
    - Nutritional ratios and relationships
    - Ingredient categories
    - Dietary patterns
    - Health scores
    """

    def __init__(
        self,
        ingredient_categories: Dict[str, List[str]],
        dietary_patterns: Dict[str, List[str]],
        nutrition_cols: List[str],
    ):
        """Initialize the feature engineer.

        Args:
            ingredient_categories: Dictionary mapping category names to ingredient keywords.
            dietary_patterns: Dictionary mapping dietary pattern names to tag keywords.
            nutrition_cols: List of nutrition column names.
        """
        self.ingredient_categories = ingredient_categories
        self.dietary_patterns = dietary_patterns
        self.nutrition_cols = nutrition_cols

    def create_nutrition_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create nutritional ratio features.

        Args:
            df: Input DataFrame with nutrition columns.

        Returns:
            DataFrame with nutritional ratio features.
        """
        logger.info("Creating nutritional ratio features")

        # Calories per ingredient
        df["calories_per_ingredient"] = df["calories"] / df["n_ingredients"].replace(
            0, 1
        )

        # Macronutrient ratios
        df["protein_to_carb_ratio"] = df["protein_pdv"] / df["carbs_pdv"].replace(0, 1)

        df["protein_to_fat_ratio"] = df["protein_pdv"] / df["total_fat_pdv"].replace(
            0, 1
        )

        # Saturated fat percentage
        df["saturated_fat_pct"] = (
            df["saturated_fat_pdv"] / df["total_fat_pdv"].replace(0, 1) * 100
        )

        # Macronutrient calorie contributions
        df["protein_calories"] = df["protein_pdv"] * 4
        df["carb_calories"] = df["carbs_pdv"] * 4
        df["fat_calories"] = df["total_fat_pdv"] * 9

        macro_total = (
            df["protein_calories"] + df["carb_calories"] + df["fat_calories"]
        ).replace(0, 1)

        df["protein_pct"] = df["protein_calories"] / macro_total * 100
        df["carbs_pct"] = df["carb_calories"] / macro_total * 100
        df["fat_pct"] = df["fat_calories"] / macro_total * 100

        # Nutritional density (protein per calorie)
        df["nutritional_density"] = df["protein_pdv"] / df["calories"].replace(0, 1)

        logger.info("Nutritional ratio features created")
        return df

    def create_health_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create health-related scoring features.

        Args:
            df: Input DataFrame with nutrition columns.

        Returns:
            DataFrame with health score features.
        """
        logger.info("Creating health scores")

        # Healthiness score (weighted combination)
        df["healthiness_score"] = (
            df["protein_pdv"] * 0.3
            - df["sugar_pdv"] * 0.25
            - df["saturated_fat_pdv"] * 0.25
            - df["sodium_pdv"] * 0.2
        ).clip(lower=0)

        # Macronutrient balance score
        # Ideal balanced macros: 30% protein, 30% fat, 40% carbs
        df["macro_balance_score"] = (
            100
            - (
                abs(df["protein_pct"] - 30)
                + abs(df["fat_pct"] - 30)
                + abs(df["carbs_pct"] - 40)
            )
            / 3
        )

        logger.info("Health scores created")
        return df

    def create_ingredient_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ingredient category features.

        Args:
            df: Input DataFrame with 'ingredients' column.

        Returns:
            DataFrame with ingredient category flags.
        """
        logger.info("Creating ingredient category features")

        def ingredient_flag(ingredients: List[str], keywords: List[str]) -> int:
            """Check if any ingredient matches keywords."""
            ingredients_lower = [ing.lower() for ing in ingredients]
            return int(
                any(
                    any(keyword in ing for keyword in keywords)
                    for ing in ingredients_lower
                )
            )

        for feature, keywords in self.ingredient_categories.items():
            df[feature] = df["ingredients"].apply(
                lambda ings: ingredient_flag(ings, keywords)
            )

        logger.info(
            f"Created {len(self.ingredient_categories)} ingredient category features"
        )
        return df

    def create_dietary_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dietary pattern features based on tags.

        Args:
            df: Input DataFrame with 'tags' column.

        Returns:
            DataFrame with dietary pattern flags.
        """
        logger.info("Creating dietary pattern features")

        def contains_keyword(tags: List[str], keywords: List[str]) -> int:
            """Check if tags contain any of the keywords."""
            tags_lower = [t.lower() for t in tags]
            return int(
                any(any(keyword in tag for keyword in keywords) for tag in tags_lower)
            )

        for feature_name, keywords in self.dietary_patterns.items():
            df[feature_name] = df["tags"].apply(
                lambda tags: contains_keyword(tags, keywords)
            )

        logger.info(f"Created {len(self.dietary_patterns)} dietary pattern features")
        return df

    def create_nutrition_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create nutritional value tags for classification.

        Tags recipes as:
        - high_calorie / low_calorie
        - high_protein
        - low_fat
        - low_sugar
        - low_sodium

        Args:
            df: Input DataFrame with nutrition columns.

        Returns:
            DataFrame with nutrition tag columns.
        """
        logger.info("Creating nutrition tags")

        # Calorie tags
        df["high_calorie"] = (df["calories"] > 500).astype(int)
        df["low_calorie"] = (df["calories"] < 200).astype(int)

        # Protein tag
        df["high_protein"] = (df["protein_pdv"] > 30).astype(int)

        # Fat tags
        df["low_fat"] = (df["total_fat_pdv"] < 10).astype(int)
        df["high_fat"] = (df["total_fat_pdv"] > 40).astype(int)

        # Sugar tags
        df["low_sugar"] = (df["sugar_pdv"] < 10).astype(int)
        df["high_sugar"] = (df["sugar_pdv"] > 50).astype(int)

        # Sodium tags
        df["low_sodium"] = (df["sodium_pdv"] < 10).astype(int)
        df["high_sodium"] = (df["sodium_pdv"] > 40).astype(int)

        # Combined healthy tag
        df["healthy_recipe"] = (
            (df["low_calorie"] | (df["calories"] < 400))
            & (df["low_fat"] | (df["total_fat_pdv"] < 20))
            & (df["low_sugar"] | (df["sugar_pdv"] < 20))
            & (df["low_sodium"] | (df["sodium_pdv"] < 20))
        ).astype(int)

        logger.info("Nutrition tags created")
        return df

    def create_pca_features(
        self, df: pd.DataFrame, n_components: int = 3
    ) -> pd.DataFrame:
        """Create PCA features from nutrition columns.

        Args:
            df: Input DataFrame with nutrition columns.
            n_components: Number of principal components to create.

        Returns:
            DataFrame with PCA features.
        """
        logger.info(f"Creating {n_components} PCA features from nutrition data")

        from sklearn.decomposition import PCA

        # Apply PCA to nutritional features
        pca = PCA(n_components=n_components)
        nutrition_data = df[self.nutrition_cols].fillna(0)
        nutrition_pca = pca.fit_transform(nutrition_data)

        for i in range(n_components):
            df[f"nutrition_pc{i+1}"] = nutrition_pca[:, i]

        logger.info(
            f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}"
        )

        return df

    def engineer_features(
        self, df: pd.DataFrame, include_pca: bool = True
    ) -> pd.DataFrame:
        """Run full feature engineering pipeline for nutrition estimation.

        Args:
            df: Input DataFrame.
            include_pca: Whether to include PCA features.

        Returns:
            DataFrame with all nutrition estimation features.
        """
        logger.info("Starting nutrition feature engineering")

        df = self.create_nutrition_ratios(df)
        df = self.create_health_scores(df)
        df = self.create_ingredient_category_features(df)
        df = self.create_dietary_pattern_features(df)
        df = self.create_nutrition_tags(df)

        if include_pca:
            df = self.create_pca_features(df)

        # Drop temporary columns
        df = df.drop(
            columns=["protein_calories", "carb_calories", "fat_calories"],
            errors="ignore",
        )

        logger.info("Nutrition feature engineering completed")
        return df

    def get_feature_names(self, include_pca: bool = True) -> list:
        """Get list of all engineered feature names.

        Args:
            include_pca: Whether to include PCA feature names.

        Returns:
            List of feature names created by this engineer.
        """
        features = [
            "calories_per_ingredient",
            "protein_to_carb_ratio",
            "protein_to_fat_ratio",
            "saturated_fat_pct",
            "protein_pct",
            "carbs_pct",
            "fat_pct",
            "nutritional_density",
            "healthiness_score",
            "macro_balance_score",
            "high_calorie",
            "low_calorie",
            "high_protein",
            "low_fat",
            "high_fat",
            "low_sugar",
            "high_sugar",
            "low_sodium",
            "high_sodium",
            "healthy_recipe",
        ]

        features.extend(list(self.ingredient_categories.keys()))
        features.extend(list(self.dietary_patterns.keys()))

        if include_pca:
            features.extend(["nutrition_pc1", "nutrition_pc2", "nutrition_pc3"])

        return features
