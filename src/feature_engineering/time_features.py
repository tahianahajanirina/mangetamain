"""Feature engineering for recipe time prediction task.

This module contains feature engineering functions specifically
designed for predicting recipe preparation time.
"""

import logging
import re
from typing import Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeFeatureEngineer:
    """Engineer features for recipe time prediction.

    This class creates features that are most relevant for predicting
    how long a recipe will take to prepare, focusing on:
    - Step complexity and count
    - Ingredient count
    - Cooking methods and equipment
    - Recipe structure
    """

    def __init__(self, cooking_verbs: Set[str], equipment_terms: Set[str]):
        """Initialize the feature engineer.

        Args:
            cooking_verbs: Set of cooking action verbs.
            equipment_terms: Set of cooking equipment terms.
        """
        self.cooking_verbs = cooking_verbs
        self.equipment_terms = equipment_terms

    def extract_time_from_text(self, text: str) -> float:
        """Extract time in minutes from text.

        Args:
            text: Text containing time references.

        Returns:
            Total time in minutes.
        """
        total_minutes = 0.0

        # Pattern for "X minutes" or "X mins"
        minutes_pattern = r"(\d+(?:\.\d+)?)\s*(?:minute|min)s?(?!\s*degrees)"
        for match in re.finditer(minutes_pattern, text, re.IGNORECASE):
            total_minutes += float(match.group(1))

        # Pattern for "X hours" or "X hrs"
        hours_pattern = r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?"
        for match in re.finditer(hours_pattern, text, re.IGNORECASE):
            total_minutes += float(match.group(1)) * 60

        # Pattern for "X-Y minutes" (take average)
        range_pattern = r"(\d+)\s*-\s*(\d+)\s*(?:minute|min)s?(?!\s*degrees)"
        for match in re.finditer(range_pattern, text, re.IGNORECASE):
            avg = (float(match.group(1)) + float(match.group(2))) / 2
            total_minutes += avg

        # Pattern for "X-Y hours" (take average)
        range_hours_pattern = r"(\d+)\s*-\s*(\d+)\s*(?:hour|hr)s?"
        for match in re.finditer(range_hours_pattern, text, re.IGNORECASE):
            avg = (float(match.group(1)) + float(match.group(2))) / 2
            total_minutes += avg * 60

        # Special cases
        if "overnight" in text.lower():
            total_minutes += 480  # 8 hours

        return total_minutes

    def create_step_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on recipe steps.

        Args:
            df: Input DataFrame with 'steps' column.

        Returns:
            DataFrame with new step-based features.
        """
        logger.info("Creating step features for time prediction")

        # Number of steps
        df["n_steps"] = df["steps"].apply(len)

        # Create steps text for analysis
        df["steps_text"] = df["steps"].apply(lambda steps: " ".join(steps).lower())

        # Extract explicit time mentions from steps
        df["total_explicit_time"] = df["steps_text"].apply(self.extract_time_from_text)
        df["has_explicit_time"] = (df["total_explicit_time"] > 0).astype(int)
        df["time_mentions_count"] = df["steps_text"].apply(
            lambda text: len(
                re.findall(r"\d+\s*(?:minute|min|hour|hr)s?", text, re.IGNORECASE)
            )
        )

        # Count cooking verbs
        df["cooking_verb_count"] = df["steps_text"].apply(
            lambda text: sum(verb in text for verb in self.cooking_verbs)
        )

        # Count equipment mentions
        df["equipment_count"] = df["steps_text"].apply(
            lambda text: sum(term in text for term in self.equipment_terms)
        )

        # Average step length (complexity indicator)
        df["avg_step_length"] = df["steps"].apply(
            lambda steps: np.mean([len(step) for step in steps]) if steps else 0
        )

        # Total words in steps
        df["steps_word_count"] = df["steps_text"].str.split().str.len()

        # Complexity indicators
        df["has_multiple_stages"] = (
            df["steps_text"]
            .str.contains("meanwhile|while|during", case=False)
            .astype(int)
        )

        df["has_waiting_time"] = (
            df["steps_text"]
            .str.contains(
                "rest|chill|refrigerate|freeze|marinate|cool|set|rise", case=False
            )
            .astype(int)
        )

        df["has_preheating"] = (
            df["steps_text"].str.contains("preheat", case=False).astype(int)
        )

        # Cooking method features
        df["has_baking"] = (
            df["steps_text"].str.contains(r"\bbake\b|\boven\b", case=False).astype(int)
        )

        df["has_simmering"] = (
            df["steps_text"]
            .str.contains(r"\bsimmer\b|\bboil\b", case=False)
            .astype(int)
        )

        df["has_frying"] = (
            df["steps_text"]
            .str.contains(r"\bfry\b|\bfrying\b|\bdeep.?fry\b", case=False)
            .astype(int)
        )

        df["has_grilling"] = (
            df["steps_text"]
            .str.contains(r"\bgrill\b|\bbroil\b|\bbarbecue\b|\bbbq\b", case=False)
            .astype(int)
        )

        df["has_slow_cooking"] = (
            df["steps_text"]
            .str.contains(r"slow.?cooker|crock.?pot|slow.?cook", case=False)
            .astype(int)
        )

        # Temperature features
        df["has_temperature"] = (
            df["steps_text"]
            .str.contains(r"\d+\s*degrees|°[fc]|\d+°", case=False)
            .astype(int)
        )

        # Extract max temperature if present
        def extract_max_temp(text):
            temps = re.findall(r"(\d+)\s*(?:degrees|°)", text, re.IGNORECASE)
            return max([int(t) for t in temps]) if temps else 0

        df["max_temperature"] = df["steps_text"].apply(extract_max_temp)

        logger.info("Step features created")
        return df

    def create_ingredient_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on ingredients.

        Args:
            df: Input DataFrame with 'ingredients' column.

        Returns:
            DataFrame with new ingredient-based features.
        """
        logger.info("Creating ingredient features for time prediction")

        # Number of ingredients
        df["n_ingredients"] = df["ingredients"].apply(len)

        # Average ingredient name length (complexity indicator)
        df["avg_ingredient_length"] = df["ingredients"].apply(
            lambda ings: np.mean([len(ing) for ing in ings]) if ings else 0
        )

        logger.info("Ingredient features created")
        return df

    def create_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create recipe complexity features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with complexity features.
        """
        logger.info("Creating complexity features")

        # Steps per ingredient ratio
        df["steps_per_ingredient"] = df["n_steps"] / df["n_ingredients"].replace(0, 1)

        # Combined complexity score
        from sklearn.preprocessing import MinMaxScaler

        complexity_features = df[["n_steps", "n_ingredients"]].values
        complexity_scaled = MinMaxScaler().fit_transform(complexity_features)
        df["step_ingredient_complexity"] = complexity_scaled.mean(axis=1)

        logger.info("Complexity features created")
        return df

    def create_tag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from recipe tags.

        Args:
            df: Input DataFrame with 'tags' column.

        Returns:
            DataFrame with tag-based features.
        """
        logger.info("Creating tag features")

        # Convert tags to lowercase strings
        df["tags_text"] = df["tags"].apply(
            lambda tags: (
                " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
            )
        )

        # Extract time limit from tags like "60-minutes-or-less", "4-hours-or-less"
        def extract_tag_time_limit(tags_text):
            # Pattern for "X-minutes-or-less"
            minutes_match = re.search(
                r"(\d+)[-\s]*minutes?[-\s]*or[-\s]*less", tags_text
            )
            if minutes_match:
                return float(minutes_match.group(1))

            # Pattern for "X-hours-or-less"
            hours_match = re.search(r"(\d+)[-\s]*hours?[-\s]*or[-\s]*less", tags_text)
            if hours_match:
                return float(hours_match.group(1)) * 60

            return 0.0

        df["tag_time_limit"] = df["tags_text"].apply(extract_tag_time_limit)
        df["has_time_tag"] = (df["tag_time_limit"] > 0).astype(int)

        # Course type features (affects typical time)
        df["is_appetizer"] = (
            df["tags_text"].str.contains("appetizer|snack", case=False).astype(int)
        )
        df["is_dessert"] = (
            df["tags_text"].str.contains("dessert|sweet", case=False).astype(int)
        )
        df["is_main_dish"] = (
            df["tags_text"]
            .str.contains("main-dish|main-ingredient", case=False)
            .astype(int)
        )
        df["is_beverage"] = (
            df["tags_text"].str.contains("beverage|drink", case=False).astype(int)
        )

        # Difficulty indicators
        df["is_easy"] = (
            df["tags_text"].str.contains("easy|beginner|simple", case=False).astype(int)
        )

        logger.info("Tag features created")
        return df

    def create_name_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from recipe name.

        Args:
            df: Input DataFrame with 'name' column.

        Returns:
            DataFrame with name-based features.
        """
        logger.info("Creating name features")

        df["name"] = df["name"].fillna("")
        df["name_lower"] = df["name"].str.lower()

        # Name length
        df["name_length"] = df["name"].str.len()
        df["name_word_count"] = df["name"].str.split().str.len()

        # Quick/fast indicators
        df["name_has_quick"] = (
            df["name_lower"]
            .str.contains("quick|fast|minute|rapid|easy", case=False)
            .astype(int)
        )

        # Slow cooking indicators
        df["name_has_slow"] = (
            df["name_lower"]
            .str.contains("slow|crockpot|overnight", case=False)
            .astype(int)
        )

        logger.info("Name features created")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline for time prediction.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with all time prediction features.
        """
        logger.info("Starting time prediction feature engineering")

        df = self.create_step_features(df)
        df = self.create_ingredient_features(df)
        df = self.create_complexity_features(df)
        df = self.create_tag_features(df)
        df = self.create_name_features(df)

        # Drop helper columns
        df = df.drop(columns=["steps_text", "tags_text", "name_lower"], errors="ignore")

        logger.info("Time prediction feature engineering completed")
        return df

    def get_feature_names(self) -> list:
        """Get list of all engineered feature names.

        Returns:
            List of feature names created by this engineer.
        """
        return [
            # Basic counts
            "n_steps",
            "n_ingredients",
            "steps_word_count",
            # Explicit time features (MOST IMPORTANT)
            "total_explicit_time",
            "has_explicit_time",
            "time_mentions_count",
            "tag_time_limit",
            "has_time_tag",
            # Cooking methods
            "cooking_verb_count",
            "equipment_count",
            "has_baking",
            "has_simmering",
            "has_frying",
            "has_grilling",
            "has_slow_cooking",
            # Temperature
            "has_temperature",
            "max_temperature",
            # Complexity
            "avg_step_length",
            "avg_ingredient_length",
            "steps_per_ingredient",
            "step_ingredient_complexity",
            # Time indicators
            "has_multiple_stages",
            "has_waiting_time",
            "has_preheating",
            # Course and difficulty
            "is_appetizer",
            "is_dessert",
            "is_main_dish",
            "is_beverage",
            "is_easy",
            # Name features
            "name_length",
            "name_word_count",
            "name_has_quick",
            "name_has_slow",
        ]
