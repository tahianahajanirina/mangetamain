"""
Global Data Cache Manager for CSV Files

This module provides a centralized cache for large CSV files to avoid
redundant loading and reduce memory usage across the application.

Usage:
    from src.utils.data_cache import DataCache

    # Load recipes (cached automatically)
    recipes = DataCache.get_recipes()

    # Load interactions (cached automatically)
    interactions = DataCache.get_interactions()

    # Clear cache if needed
    DataCache.clear_cache()
"""

import gc
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    Singleton-style cache for CSV data files.

    Ensures each CSV is loaded only once and reused across the application.
    """

    # Cache storage (class variables = shared across all instances)
    _recipes_cache: Optional[pd.DataFrame] = None
    _interactions_cache: Optional[pd.DataFrame] = None
    _cache_metadata: dict = {}

    # Default paths (can be overridden)
    DEFAULT_RECIPES_PATH = "data/raw/RAW_recipes.csv"
    DEFAULT_INTERACTIONS_PATH = "data/raw/RAW_interactions.csv"

    @classmethod
    def get_recipes(
        cls,
        path: Optional[str] = None,
        force_reload: bool = False,
        optimize_dtypes: bool = True,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Get recipes DataFrame from cache or load it.

        Args:
            path: Path to recipes CSV file (uses default if None)
            force_reload: Force reload even if cached
            optimize_dtypes: Use optimized data types to reduce memory
            columns: Specific columns to load (None = all columns) ✅ NOUVEAU

        Returns:
            DataFrame with recipes data
        """
        path = path or cls.DEFAULT_RECIPES_PATH

        # Check if already cached
        if not force_reload and cls._recipes_cache is not None:
            cache_path = cls._cache_metadata.get("recipes_path")
            if cache_path == path:
                logger.info(
                    f"✅ Using cached recipes data ({len(cls._recipes_cache)} recipes)"
                )
                # Si colonnes spécifiques demandées, retourner seulement ces colonnes
                if columns:
                    return cls._recipes_cache[columns]
                return cls._recipes_cache

        # Load from file
        logger.info(f"Loading recipes from {path}...")

        # Déterminer si c'est un fichier preprocessed (structure différente)
        is_processed = "processed" in str(path).lower()

        if optimize_dtypes and not is_processed:
            # ✅ Pour fichiers RAW: Optimized dtypes to reduce memory by ~50%
            dtypes = {
                "id": "int32",
                "minutes": "int32",
                "contributor_id": "int32",
                "n_steps": "int16",
                "n_ingredients": "int16",
                "calories": "float32",
                "total_fat_pdv": "float32",
                "sugar_pdv": "float32",
                "sodium_pdv": "float32",
                "protein_pdv": "float32",
                "saturated_fat_pdv": "float32",
                "carbs_pdv": "float32",
            }

            cls._recipes_cache = pd.read_csv(
                path, usecols=columns, dtype=dtypes, low_memory=False
            )
        else:
            # ✅ Pour fichiers PROCESSED ou sans optimisation: laisser pandas inférer les types
            cls._recipes_cache = pd.read_csv(path, usecols=columns, low_memory=False)

            # Si optimize_dtypes est True, optimiser après chargement
            if optimize_dtypes and is_processed:
                # Conversion sécurisée des types après chargement
                for col in cls._recipes_cache.columns:
                    if col in ["id", "minutes", "contributor_id"]:
                        cls._recipes_cache[col] = cls._recipes_cache[col].astype(
                            "int32", errors="ignore"
                        )
                    elif col in ["n_steps", "n_ingredients"]:
                        cls._recipes_cache[col] = cls._recipes_cache[col].astype(
                            "int16", errors="ignore"
                        )
                    elif col in [
                        "calories",
                        "total_fat_pdv",
                        "sugar_pdv",
                        "sodium_pdv",
                        "protein_pdv",
                        "saturated_fat_pdv",
                        "carbs_pdv",
                        "total_fat",
                        "sugar",
                        "sodium",
                        "protein",
                        "saturated_fat",
                        "carbohydrates",
                    ]:
                        cls._recipes_cache[col] = cls._recipes_cache[col].astype(
                            "float32", errors="ignore"
                        )

        # Store metadata
        cls._cache_metadata["recipes_path"] = path
        cls._cache_metadata["recipes_rows"] = len(cls._recipes_cache)
        cls._cache_metadata["recipes_memory_mb"] = (
            cls._recipes_cache.memory_usage(deep=True).sum() / 1024**2
        )

        logger.info(
            f"✓ Loaded {len(cls._recipes_cache)} recipes "
            f"({cls._cache_metadata['recipes_memory_mb']:.1f} MB in memory)"
        )

        return cls._recipes_cache

    @classmethod
    def get_interactions(
        cls,
        path: Optional[str] = None,
        force_reload: bool = False,
        optimize_dtypes: bool = True,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Get interactions DataFrame from cache or load it.

        Args:
            path: Path to interactions CSV file (uses default if None)
            force_reload: Force reload even if cached
            optimize_dtypes: Use optimized data types to reduce memory
            columns: Specific columns to load (None = all columns) ✅ NOUVEAU

        Returns:
            DataFrame with interactions data
        """
        path = path or cls.DEFAULT_INTERACTIONS_PATH

        # Check if already cached
        if not force_reload and cls._interactions_cache is not None:
            cache_path = cls._cache_metadata.get("interactions_path")
            if cache_path == path:
                logger.info(
                    f"✅ Using cached interactions data ({len(cls._interactions_cache)} interactions)"
                )
                # Si colonnes spécifiques demandées, retourner seulement ces colonnes
                if columns:
                    return cls._interactions_cache[columns]
                return cls._interactions_cache

        # Load from file
        logger.info(f"Loading interactions from {path}...")

        # Déterminer si c'est un fichier preprocessed
        is_processed = "processed" in str(path).lower() or "clean" in str(path).lower()

        if optimize_dtypes and not is_processed:
            # ✅ Pour fichiers RAW: Optimized dtypes to reduce memory
            dtypes = {
                "user_id": "int32",
                "recipe_id": "int32",
                "rating": "int8",  # Ratings are typically 0-5
            }

            cls._interactions_cache = pd.read_csv(
                path, usecols=columns, dtype=dtypes, low_memory=False
            )
        else:
            # ✅ Pour fichiers PROCESSED ou sans optimisation: laisser pandas inférer
            cls._interactions_cache = pd.read_csv(
                path, usecols=columns, low_memory=False
            )

            # Si optimize_dtypes est True, optimiser après chargement
            if optimize_dtypes and is_processed:
                for col in cls._interactions_cache.columns:
                    if col in ["user_id", "recipe_id"]:
                        cls._interactions_cache[col] = cls._interactions_cache[
                            col
                        ].astype("int32", errors="ignore")
                    elif col == "rating":
                        cls._interactions_cache[col] = cls._interactions_cache[
                            col
                        ].astype("int8", errors="ignore")

        # Store metadata
        cls._cache_metadata["interactions_path"] = path
        cls._cache_metadata["interactions_rows"] = len(cls._interactions_cache)
        cls._cache_metadata["interactions_memory_mb"] = (
            cls._interactions_cache.memory_usage(deep=True).sum() / 1024**2
        )

        logger.info(
            f"✓ Loaded {len(cls._interactions_cache)} interactions "
            f"({cls._cache_metadata['interactions_memory_mb']:.1f} MB in memory)"
        )

        return cls._interactions_cache

    @classmethod
    def get_cache_info(cls) -> dict:
        """
        Get information about cached data.

        Returns:
            Dictionary with cache status and memory usage
        """
        info = {
            "recipes_cached": cls._recipes_cache is not None,
            "interactions_cached": cls._interactions_cache is not None,
            "total_memory_mb": 0,
        }

        if cls._recipes_cache is not None:
            info["recipes_rows"] = cls._cache_metadata.get("recipes_rows", 0)
            info["recipes_memory_mb"] = cls._cache_metadata.get("recipes_memory_mb", 0)
            info["total_memory_mb"] += info["recipes_memory_mb"]

        if cls._interactions_cache is not None:
            info["interactions_rows"] = cls._cache_metadata.get("interactions_rows", 0)
            info["interactions_memory_mb"] = cls._cache_metadata.get(
                "interactions_memory_mb", 0
            )
            info["total_memory_mb"] += info["interactions_memory_mb"]

        return info

    @classmethod
    def clear_cache(cls, clear_recipes: bool = True, clear_interactions: bool = True):
        """
        Clear cached data to free memory.

        Args:
            clear_recipes: Clear recipes cache
            clear_interactions: Clear interactions cache
        """
        if clear_recipes and cls._recipes_cache is not None:
            memory_freed = cls._cache_metadata.get("recipes_memory_mb", 0)
            del cls._recipes_cache
            cls._recipes_cache = None
            logger.info(f"Cleared recipes cache (~{memory_freed:.1f} MB freed)")

        if clear_interactions and cls._interactions_cache is not None:
            memory_freed = cls._cache_metadata.get("interactions_memory_mb", 0)
            del cls._interactions_cache
            cls._interactions_cache = None
            logger.info(f"Cleared interactions cache (~{memory_freed:.1f} MB freed)")

        # Force garbage collection
        gc.collect()

        # Clear metadata
        if clear_recipes:
            cls._cache_metadata.pop("recipes_path", None)
            cls._cache_metadata.pop("recipes_rows", None)
            cls._cache_metadata.pop("recipes_memory_mb", None)

        if clear_interactions:
            cls._cache_metadata.pop("interactions_path", None)
            cls._cache_metadata.pop("interactions_rows", None)
            cls._cache_metadata.pop("interactions_memory_mb", None)

    @classmethod
    def is_cached(cls, data_type: str) -> bool:
        """
        Check if a specific data type is cached.

        Args:
            data_type: 'recipes' or 'interactions'

        Returns:
            True if cached, False otherwise
        """
        if data_type == "recipes":
            return cls._recipes_cache is not None
        elif data_type == "interactions":
            return cls._interactions_cache is not None
        else:
            raise ValueError(f"Unknown data type: {data_type}")
