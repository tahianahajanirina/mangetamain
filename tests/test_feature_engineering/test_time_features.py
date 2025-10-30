"""Tests for time feature engineering."""

import pytest

from src.feature_engineering.time_features import TimeFeatureEngineer


@pytest.fixture
def time_engineer():
    """Create a time feature engineer with test configuration."""
    cooking_verbs = {
        "bake",
        "boil",
        "fry",
        "grill",
        "roast",
        "steam",
        "simmer",
        "sauté",
    }
    equipment_terms = {"oven", "stove", "microwave", "grill", "pan", "pot", "blender"}

    return TimeFeatureEngineer(cooking_verbs, equipment_terms)


class TestTimeFeatureEngineer:
    """Test cases for TimeFeatureEngineer."""

    def test_init(self, time_engineer):
        """Test initialization."""
        assert time_engineer is not None
        assert time_engineer.cooking_verbs is not None
        assert time_engineer.equipment_terms is not None

    def test_engineer_features(self, sample_recipes_df, time_engineer):
        """Test feature engineering for time prediction."""
        # Test that the main method exists
        assert hasattr(time_engineer, "engineer_features")
        assert hasattr(time_engineer, "create_step_features")

    def test_time_categories(self, sample_recipes_df, time_engineer):
        """Test time feature creation."""
        sample_recipes_df.copy()
        # Use the existing 'minutes' column

        # Test extract_time_from_text method
        result = time_engineer.extract_time_from_text("Bake for 30 minutes")
        assert isinstance(result, (int, float))
        assert result >= 0
