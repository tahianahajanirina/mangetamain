"""Tests for nutrition classification feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.nutrition_classification_features import (
    NutritionClassificationFeatures,
)


@pytest.fixture
def nutrition_classifier_fe():
    """Create a NutritionClassificationFeatures instance."""
    return NutritionClassificationFeatures()


@pytest.fixture
def nutrition_df():
    """Create sample nutrition data for testing."""
    return pd.DataFrame(
        {
            "id": range(1, 11),
            "name": [f"Recipe {i}" for i in range(1, 11)],
            "calories": [150, 300, 500, 200, 800, 100, 400, 600, 250, 350],
            "total_fat": [5, 15, 30, 8, 40, 3, 20, 35, 10, 18],
            "saturated_fat": [2, 5, 12, 3, 15, 1, 8, 14, 4, 7],
            "sodium": [200, 500, 1200, 300, 2000, 150, 800, 1500, 400, 600],
            "carbohydrates": [20, 35, 50, 25, 60, 15, 40, 55, 30, 38],
            "sugar": [5, 10, 25, 8, 40, 3, 15, 30, 7, 12],
            "protein": [10, 20, 15, 25, 10, 8, 30, 12, 18, 22],
            "n_steps": [3, 5, 8, 4, 10, 2, 7, 9, 4, 6],
            "n_ingredients": [4, 6, 10, 5, 12, 3, 8, 11, 5, 7],
            "minutes": [15, 30, 60, 20, 90, 10, 45, 75, 25, 35],
        }
    )


class TestNutritionClassificationFeatures:
    """Test cases for NutritionClassificationFeatures."""

    def test_init(self, nutrition_classifier_fe):
        """Test initialization and thresholds."""
        assert nutrition_classifier_fe.daily_values is not None
        assert nutrition_classifier_fe.health_thresholds is not None
        assert "very_healthy" in nutrition_classifier_fe.health_thresholds
        assert "healthy" in nutrition_classifier_fe.health_thresholds
        assert "moderate" in nutrition_classifier_fe.health_thresholds

    def test_calculate_pdv_percentages(self, nutrition_classifier_fe, nutrition_df):
        """Test PDV percentage calculation."""
        result = nutrition_classifier_fe.calculate_pdv_percentages(nutrition_df)

        assert "calories_pdv" in result.columns
        assert "fat_pdv" in result.columns
        assert "protein_pdv" in result.columns
        assert "sugar_pdv" in result.columns
        assert "sodium_pdv" in result.columns
        # PDV should be positive
        assert (result["calories_pdv"] >= 0).all()

    def test_calculate_nutritional_ratios(self, nutrition_classifier_fe, nutrition_df):
        """Test nutritional ratio calculation."""
        df = nutrition_classifier_fe.calculate_pdv_percentages(nutrition_df)
        result = nutrition_classifier_fe.calculate_nutritional_ratios(df)

        assert "protein_density" in result.columns
        assert "sugar_ratio" in result.columns
        assert "sodium_density" in result.columns
        assert "saturated_fat_ratio" in result.columns
        assert "macro_balance" in result.columns

    def test_calculate_health_score(self, nutrition_classifier_fe, nutrition_df):
        """Test health score calculation."""
        df = nutrition_classifier_fe.calculate_pdv_percentages(nutrition_df)
        df = nutrition_classifier_fe.calculate_nutritional_ratios(df)
        result = nutrition_classifier_fe.calculate_health_score(df)

        assert "health_score" in result.columns
        assert (result["health_score"] >= 0).all()
        assert (result["health_score"] <= 100).all()

    def test_categorize_nutrition(self, nutrition_classifier_fe, nutrition_df):
        """Test nutrition categorization into 4 classes."""
        df = nutrition_classifier_fe.calculate_pdv_percentages(nutrition_df)
        df = nutrition_classifier_fe.calculate_nutritional_ratios(df)
        df = nutrition_classifier_fe.calculate_health_score(df)
        result = nutrition_classifier_fe.categorize_nutrition(df)

        assert "nutrition_category" in result.columns
        assert result["nutrition_category"].isin([0, 1, 2, 3]).all()

    def test_create_additional_features(self, nutrition_classifier_fe, nutrition_df):
        """Test additional feature creation."""
        df = nutrition_classifier_fe.calculate_pdv_percentages(nutrition_df)
        result = nutrition_classifier_fe.create_additional_features(df)

        assert "is_low_calorie" in result.columns
        assert "is_high_protein" in result.columns
        assert "complexity_score" in result.columns
        assert "calorie_protein_interaction" in result.columns

    def test_generate_statistics(self, nutrition_classifier_fe, nutrition_df):
        """Test statistics generation."""
        df = nutrition_classifier_fe.calculate_pdv_percentages(nutrition_df)
        df = nutrition_classifier_fe.calculate_nutritional_ratios(df)
        df = nutrition_classifier_fe.calculate_health_score(df)
        df = nutrition_classifier_fe.categorize_nutrition(df)

        stats = nutrition_classifier_fe._generate_statistics(df)

        assert "n_recipes" in stats
        assert "health_score_stats" in stats
        assert "category_distribution" in stats

    def test_zero_calorie_handling(self, nutrition_classifier_fe):
        """Test that zero calories don't cause division errors."""
        df = pd.DataFrame(
            {
                "calories": [0, 100],
                "total_fat": [0, 10],
                "saturated_fat": [0, 5],
                "sodium": [0, 200],
                "carbohydrates": [0, 25],
                "sugar": [0, 8],
                "protein": [0, 15],
            }
        )

        df = nutrition_classifier_fe.calculate_pdv_percentages(df)
        result = nutrition_classifier_fe.calculate_nutritional_ratios(df)

        assert not result["protein_density"].isna().any()
        assert not result["sugar_ratio"].isna().any()
