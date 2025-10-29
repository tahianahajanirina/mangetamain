"""
Feature Engineering pour les Recettes (Clustering )

Ce module construit les features recettes optimisées pour le clustering

"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.data_cache import DataCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeFeatureBuilder:
    """Construit les features pour le clustering recettes"""

    def __init__(self, min_rating: float = 3.5):
        """
        Args:
            min_rating: Rating minimum pour filtrer les recettes de qualité
        """
        self.min_rating = min_rating
        self.recipes_features = None

    def load_data(self, recipes_path: str, interactions_path: str) -> tuple:
        """Charge les données nettoyées"""
        logger.info("Chargement des données...")
        # Use global cache to avoid redundant loading
        recipes = DataCache.get_recipes(path=recipes_path, optimize_dtypes=True)
        interactions = DataCache.get_interactions(path=interactions_path, optimize_dtypes=True)

        logger.info(f"Recettes: {len(recipes):,}")
        logger.info(f"Interactions: {len(interactions):,}")

        return recipes, interactions

    def calculate_popularity_metrics(
        self, recipes: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Calcule les métriques de popularité (interactions et ratings)"""
        logger.info("Calcul des métriques de popularité...")

        # Agréger par recette
        interactions_agg = (
            interactions.groupby("recipe_id")
            .agg({"user_id": "count", "rating": "mean"})
            .rename(columns={"user_id": "n_interactions", "rating": "avg_rating"})
            .reset_index()
        )

        # Merger avec recettes
        recipes_merged = recipes.merge(
            interactions_agg, left_on="id", right_on="recipe_id", how="left"
        )

        # Remplir les NaN
        recipes_merged["n_interactions"] = recipes_merged["n_interactions"].fillna(0)
        recipes_merged["avg_rating"] = recipes_merged["avg_rating"].fillna(0)

        return recipes_merged

    def calculate_health_category(self, row: pd.Series) -> int:
        """Calcule un score de santé (0-4) basé sur les valeurs nutritionnelles"""
        score = 0

        # Critères santé
        if row["sugar"] < 30:
            score += 1
        if row["sodium"] < 30:
            score += 1
        if row["protein"] > 20:
            score += 1
        if row["saturated_fat"] < 20:
            score += 1

        return score

    def build_features(self, recipes_path: str, interactions_path: str) -> pd.DataFrame:
        """Pipeline complet de construction des features recettes"""
        logger.info("=" * 80)
        logger.info("CONSTRUCTION DES FEATURES RECETTES ")
        logger.info("=" * 80)

        # 1. Charger les données
        recipes, interactions = self.load_data(recipes_path, interactions_path)

        # 2. Calculer popularité
        recipes_merged = self.calculate_popularity_metrics(recipes, interactions)

        # 3. Filtrer recettes de qualité
        logger.info(f"Filtrage recettes (rating >= {self.min_rating})...")
        initial_count = len(recipes_merged)
        recipes_filtered = recipes_merged[
            recipes_merged["avg_rating"] >= self.min_rating
        ].copy()
        logger.info(
            f"Recettes conservées: {len(recipes_filtered):,} / {initial_count:,}"
        )

        # 4. Construire les 5 features
        logger.info("Création des features optimisées...")

        # Feature 1: popularity_score (log interactions * rating)
        recipes_filtered["log_interactions"] = np.log1p(
            recipes_filtered["n_interactions"]
        )
        recipes_filtered["popularity_score"] = (
            recipes_filtered["log_interactions"] * recipes_filtered["avg_rating"]
        )

        # Feature 2: log_minutes (transformation log du temps)
        recipes_filtered["log_minutes"] = np.log1p(recipes_filtered["minutes"])

        # Feature 3: time_complexity (steps * ingredients)
        recipes_filtered["time_complexity"] = (
            recipes_filtered["n_steps"] * recipes_filtered["n_ingredients"]
        )

        # Feature 4: efficiency (popularité / temps)
        recipes_filtered["efficiency"] = recipes_filtered["popularity_score"] / (
            1 + recipes_filtered["log_minutes"]
        )

        # Feature 5: health_category (score santé 0-4)
        recipes_filtered["health_category"] = recipes_filtered.apply(
            self.calculate_health_category, axis=1
        )

        # 5. Sélectionner les colonnes finales
        features_cols = [
            "id",
            "name",
            "popularity_score",
            "log_minutes",
            "time_complexity",
            "efficiency",
            "health_category",
        ]

        recipes_features = recipes_filtered[features_cols].copy()

        self.recipes_features = recipes_features

        logger.info("=" * 80)
        logger.info("FEATURES RECETTES  CRÉÉES")
        logger.info(f"  Recettes: {len(recipes_features):,}")
        logger.info(f"  Features: {len(features_cols) - 2}")  # -2 pour id et name
        logger.info("=" * 80)

        return recipes_features

    def save_features(self, output_path: str) -> None:
        """Sauvegarde les features dans un fichier CSV"""
        if self.recipes_features is None:
            raise ValueError(
                "Features non construites. Appelez build_features() d'abord."
            )

        logger.info(f"Sauvegarde: {output_path}")
        self.recipes_features.to_csv(output_path, index=False)

        # Afficher les stats
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"  Shape: {self.recipes_features.shape}")
        logger.info(f"  Taille: {file_size:.2f} MB")
        logger.info("✓ Sauvegarde terminée")

    def get_feature_stats(self) -> Dict:
        """Retourne les statistiques des features"""
        if self.recipes_features is None:
            raise ValueError("Features non construites.")

        stats = {
            "n_recipes": len(self.recipes_features),
            "n_features": len(self.recipes_features.columns) - 2,
            "columns": list(self.recipes_features.columns),
        }

        # Stats sur les features (MODIFIÉ: health_category au lieu de popularity_score)
        feature_cols = [
            "log_minutes",
            "time_complexity",
            "efficiency",
            "health_category",
        ]

        for col in feature_cols:
            stats[f"{col}_mean"] = self.recipes_features[col].mean()
            stats[f"{col}_std"] = self.recipes_features[col].std()

        return stats


def build_recipe_features(
    recipes_path: str,
    interactions_path: str,
    output_path: str = None,
    min_rating: float = 3.5,
) -> pd.DataFrame:
    """
    Helper function pour construire les features recettes .

    Args:
        recipes_path: Chemin vers recipes_clean.csv
        interactions_path: Chemin vers interactions_clean.csv
        output_path: Chemin de sortie (optionnel)
        min_rating: Rating minimum (défaut: 3.5)

    Returns:
        DataFrame avec les features construites
    """
    builder = RecipeFeatureBuilder(min_rating=min_rating)
    recipes_features = builder.build_features(recipes_path, interactions_path)

    if output_path:
        builder.save_features(output_path)

    return recipes_features


def main():
    """Fonction principale pour exécution en ligne de commande"""
    from config.config import DATA_PROCESSED

    # Chemins
    recipes_path = DATA_PROCESSED / "recipes_clean.csv"
    interactions_path = DATA_PROCESSED / "interactions_clean.csv"
    output_path = DATA_PROCESSED / "recipes_features_.csv"

    # Construire les features
    recipes_features = build_recipe_features(
        recipes_path=str(recipes_path),
        interactions_path=str(interactions_path),
        output_path=str(output_path),
        min_rating=3.5,
    )

    print(f"\n✓ Features créées: {len(recipes_features):,} recettes")


if __name__ == "__main__":
    main()
