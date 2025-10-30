"""Tests for clean_data script."""

import sys
from pathlib import Path

import pandas as pd

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from clean_data import clean_interactions, clean_recipes, safe_eval


class TestSafeEval:
    """Test safe_eval function."""

    def test_safe_eval_string(self):
        """Test evaluating string representation of list."""
        result = safe_eval("['a', 'b', 'c']")
        assert result == ["a", "b", "c"]

    def test_safe_eval_list(self):
        """Test passing actual list."""
        result = safe_eval(["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == "a"

    def test_safe_eval_nan(self):
        """Test handling NaN values."""
        result = safe_eval(pd.NA)
        assert result == []

    def test_safe_eval_invalid(self):
        """Test handling invalid input."""
        result = safe_eval("invalid{syntax")
        assert result == []


class TestCleanRecipes:
    """Test clean_recipes function."""

    def test_clean_recipes_basic(self, temp_data_dir, sample_recipes_df):
        """Test basic recipe cleaning."""
        # Save raw data
        raw_path = temp_data_dir["raw"] / "RAW_recipes.csv"
        sample_recipes_df.to_csv(raw_path, index=False)

        output_path = temp_data_dir["processed"] / "recipes_clean.csv"

        result = clean_recipes(raw_path, output_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert output_path.exists()

    def test_clean_recipes_remove_duplicates(self, temp_data_dir):
        """Test duplicate removal."""
        df_with_dupes = pd.DataFrame(
            {
                "id": [1, 1, 2, 3],
                "name": ["Recipe 1", "Recipe 1", "Recipe 2", "Recipe 3"],
                "minutes": [30, 30, 45, 20],
                "n_steps": [5, 5, 7, 3],
                "n_ingredients": [6, 6, 10, 4],
                "tags": ["['tag1']", "['tag1']", "['tag2']", "['tag3']"],
                "nutrition": [
                    "[100, 10, 5, 2, 8, 15, 3]",
                    "[100, 10, 5, 2, 8, 15, 3]",
                    "[150, 12, 6, 3, 10, 18, 4]",
                    "[120, 5, 2, 1, 3, 20, 2]",
                ],
                "steps": ["['step1']", "['step1']", "['step2']", "['step3']"],
                "description": ["Desc 1", "Desc 1", "Desc 2", "Desc 3"],
                "ingredients": ["['ing1']", "['ing1']", "['ing2']", "['ing3']"],
            }
        )

        raw_path = temp_data_dir["raw"] / "RAW_recipes.csv"
        df_with_dupes.to_csv(raw_path, index=False)

        output_path = temp_data_dir["processed"] / "recipes_clean.csv"

        result = clean_recipes(raw_path, output_path)

        # Should remove duplicates based on id
        assert len(result) < len(df_with_dupes)
        assert result["id"].is_unique


class TestCleanInteractions:
    """Test clean_interactions function."""

    def test_clean_interactions_basic(self, temp_data_dir, sample_interactions_df):
        """Test basic interaction cleaning."""
        raw_path = temp_data_dir["raw"] / "RAW_interactions.csv"
        sample_interactions_df.to_csv(raw_path, index=False)

        output_path = temp_data_dir["processed"] / "interactions_clean.csv"

        result = clean_interactions(raw_path, output_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert output_path.exists()
