"""Data loading and preprocessing module.

This module handles loading raw recipe data and performing
initial preprocessing steps including type conversions and
basic cleaning.
"""

import ast
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.data_cache import DataCache

logger = logging.getLogger(__name__)


class RecipeDataLoader:
    """Loads and preprocesses raw recipe data.

    This class handles reading the raw recipe CSV file and performing
    initial preprocessing including type conversions, JSON parsing,
    and basic data validation.

    Attributes:
        data_path: Path to the raw recipe CSV file.
        df: Loaded pandas DataFrame.
    """

    def __init__(self, data_path: Union[str, Path]):
        """Initialize the data loader.

        Args:
            data_path: Path to the raw recipe CSV file.
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load the raw recipe data from CSV.

        Returns:
            Loaded pandas DataFrame.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")
        # Use global cache to avoid redundant loading
        self.df = DataCache.get_recipes(path=str(self.data_path), optimize_dtypes=True)
        logger.info(
            f"Loaded {len(self.df)} recipes with {len(self.df.columns)} columns"
        )

        return self.df

    @staticmethod
    def safe_eval(x: Union[str, List]) -> List:
        """Safely evaluate string representations of lists.

        Args:
            x: String representation of a list or an actual list.

        Returns:
            Evaluated list or empty list if evaluation fails.
        """
        try:
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    def convert_types(self) -> pd.DataFrame:
        """Convert data types for key columns.

        Converts:
        - submitted to datetime
        - tags, steps, ingredients, nutrition from string to list

        Returns:
            DataFrame with converted types.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Converting data types")

        # Convert submitted to datetime
        self.df["submitted"] = pd.to_datetime(self.df["submitted"])

        # Parse JSON/list formats
        list_columns = ["tags", "steps", "ingredients", "nutrition"]
        for col in list_columns:
            self.df[col] = self.df[col].apply(self.safe_eval)

        logger.info("Type conversions completed")
        return self.df

    def expand_nutrition(self) -> pd.DataFrame:
        """Expand nutrition column into separate columns.

        The nutrition column contains a list of values:
        [calories, total_fat_pdv, sugar_pdv, sodium_pdv,
         protein_pdv, saturated_fat_pdv, carbs_pdv]

        Returns:
            DataFrame with expanded nutrition columns.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Expanding nutrition column")

        nutrition_cols = [
            "calories",
            "total_fat_pdv",
            "sugar_pdv",
            "sodium_pdv",
            "protein_pdv",
            "saturated_fat_pdv",
            "carbs_pdv",
        ]

        for i, col in enumerate(nutrition_cols):
            self.df[col] = self.df["nutrition"].apply(
                lambda x: x[i] if len(x) > i else np.nan
            )

        logger.info(f"Expanded {len(nutrition_cols)} nutrition columns")
        return self.df

    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Strategy:
        - Fill missing descriptions with empty string
        - Fill missing names with empty string
        - Impute missing nutrition values with median

        Returns:
            DataFrame with missing values handled.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Handling missing values")

        # Fill text columns
        self.df["description"] = self.df["description"].fillna("")
        self.df["name"] = self.df["name"].fillna("")

        # Impute nutrition columns with median
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
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median {median_val:.2f}")

        return self.df

    def handle_outliers(self) -> pd.DataFrame:
        """Handle outliers in the dataset.

        Creates flags for outliers and caps extreme values:
        - Cap minutes at 99th percentile
        - Flag unrealistic cooking times (> 1 week)

        Returns:
            DataFrame with outliers handled.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Handling outliers")

        # Cap extreme values in minutes
        cap_value = self.df["minutes"].quantile(0.99)
        self.df["minutes_original"] = self.df["minutes"]
        self.df["minutes"] = self.df["minutes"].clip(upper=cap_value)

        # Flag unrealistic times (> 1 week)
        self.df["unrealistic_time"] = (self.df["minutes_original"] > 10080).astype(int)

        logger.info(f"Capped minutes at {cap_value:.0f}")
        logger.info(f"Flagged {self.df['unrealistic_time'].sum()} unrealistic times")

        return self.df

    def preprocess(self) -> pd.DataFrame:
        """Run full preprocessing pipeline.

        Executes all preprocessing steps in sequence:
        1. Load data
        2. Convert types
        3. Expand nutrition
        4. Handle missing values
        5. Handle outliers

        Returns:
            Fully preprocessed DataFrame.
        """
        logger.info("Starting preprocessing pipeline")

        self.load_data()
        self.convert_types()
        self.expand_nutrition()
        self.handle_missing_values()
        self.handle_outliers()

        logger.info("Preprocessing pipeline completed")
        return self.df

    def get_summary(self) -> dict:
        """Get summary statistics of the loaded data.

        Returns:
            Dictionary containing summary statistics.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return {
            "n_recipes": len(self.df),
            "n_columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "dtypes": self.df.dtypes.to_dict(),
        }
