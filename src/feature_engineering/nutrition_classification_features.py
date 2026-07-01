"""
Feature Engineering pour Classification Nutritionnelle des Recettes

Ce module calcule des features nutritionnelles avancées pour classer
les recettes en catégories de santé.

Auteur: Équipe Data Science
Date: 2025-10-27
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.data_cache import DataCache

logger = logging.getLogger(__name__)


class NutritionClassificationFeatures:
    """
    Classe pour créer des features nutritionnelles avancées.

    Stratégie:
    1. Normalisation des valeurs nutritionnelles (% Daily Value)
    2. Calcul de ratios nutritionnels (protéines/calories, etc.)
    3. Score de densité nutritionnelle
    4. Catégorisation en 4 classes nutritionnelles
    """

    def __init__(self):
        """Initialise les seuils nutritionnels basés sur les recommandations FDA."""
        # Valeurs quotidiennes recommandées (FDA 2000 cal diet)
        self.daily_values = {
            "calories": 2000,
            "total_fat": 78,  # grammes
            "saturated_fat": 20,  # grammes
            "sodium": 2300,  # mg
            "carbohydrates": 275,  # grammes
            "sugar": 50,  # grammes
            "protein": 50,  # grammes
        }

        # Seuils pour classification (basés sur les PDV%)
        self.health_thresholds = {
            "very_healthy": {
                "calories_max": 400,
                "fat_pdv_max": 15,
                "sodium_pdv_max": 15,
                "sugar_pdv_max": 15,
                "protein_pdv_min": 10,
            },
            "healthy": {
                "calories_max": 600,
                "fat_pdv_max": 30,
                "sodium_pdv_max": 25,
                "sugar_pdv_max": 30,
            },
            "moderate": {
                "calories_max": 800,
                "fat_pdv_max": 50,
                "sodium_pdv_max": 40,
                "sugar_pdv_max": 50,
            },
            # Au-dessus = indulgent
        }

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Charge les données de recettes nettoyées.

        Args:
            filepath: Chemin vers recipes_clean.csv

        Returns:
            DataFrame avec les données chargées
        """
        logger.info(f"Chargement des données depuis {filepath}...")
        # Use global cache to avoid redundant loading
        df = DataCache.get_recipes(path=str(filepath), optimize_dtypes=True)
        logger.info(f"  ✓ {len(df):,} recettes chargées")
        logger.info(f"  ✓ {df.shape[1]} colonnes")
        return df

    def calculate_pdv_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les pourcentages de valeurs quotidiennes (PDV%).

        Args:
            df: DataFrame avec colonnes nutritionnelles

        Returns:
            DataFrame avec nouvelles colonnes PDV%
        """
        logger.info("Calcul des pourcentages de valeurs quotidiennes...")

        df = df.copy()

        # Calcul des PDV% pour chaque nutriment
        df["calories_pdv"] = (df["calories"] / self.daily_values["calories"]) * 100
        df["fat_pdv"] = (df["total_fat"] / self.daily_values["total_fat"]) * 100
        df["saturated_fat_pdv"] = (
            df["saturated_fat"] / self.daily_values["saturated_fat"]
        ) * 100
        df["sodium_pdv"] = (df["sodium"] / self.daily_values["sodium"]) * 100
        df["carbs_pdv"] = (
            df["carbohydrates"] / self.daily_values["carbohydrates"]
        ) * 100
        df["sugar_pdv"] = (df["sugar"] / self.daily_values["sugar"]) * 100
        df["protein_pdv"] = (df["protein"] / self.daily_values["protein"]) * 100

        logger.info("  ✓ 7 colonnes PDV% créées")
        return df

    def calculate_nutritional_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des ratios nutritionnels significatifs.

        Ratios importants:
        - Protéines/Calories: Indicateur de densité protéique
        - Sucre/Calories: Indicateur de sucres ajoutés
        - Sodium/Calories: Indicateur de sel
        - Graisses saturées/Graisses totales: Qualité des lipides

        Args:
            df: DataFrame avec valeurs nutritionnelles

        Returns:
            DataFrame avec nouvelles colonnes de ratios
        """
        logger.info("Calcul des ratios nutritionnels...")

        df = df.copy()

        # Ratio Protéines/Calories (g/100kcal) - plus élevé = meilleur
        df["protein_density"] = np.where(
            df["calories"] > 0, (df["protein"] * 100) / df["calories"], 0
        )

        # Ratio Sucre/Calories (%) - plus bas = meilleur
        df["sugar_ratio"] = np.where(
            df["calories"] > 0,
            (df["sugar"] * 4 * 100) / df["calories"],  # 4 cal/g sucre
            0,
        )

        # Ratio Sodium/Calories (mg/100kcal) - plus bas = meilleur
        df["sodium_density"] = np.where(
            df["calories"] > 0, (df["sodium"] * 100) / df["calories"], 0
        )

        # Ratio Graisses saturées/Graisses totales (%)
        df["saturated_fat_ratio"] = np.where(
            df["total_fat"] > 0, (df["saturated_fat"] / df["total_fat"]) * 100, 0
        )

        # Équilibre macro-nutriments (écart-type des PDV%)
        macros = df[["fat_pdv", "carbs_pdv", "protein_pdv"]].values
        df["macro_balance"] = np.std(macros, axis=1)

        logger.info("  ✓ 5 ratios nutritionnels créés")
        return df

    def calculate_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un score de santé composite (0-100).

        Formule:
        - Points positifs: protéines, fibres (implicite)
        - Points négatifs: calories excessives, graisses saturées, sodium, sucre

        Args:
            df: DataFrame avec PDV% et ratios

        Returns:
            DataFrame avec colonne health_score
        """
        logger.info("Calcul du score de santé composite...")

        df = df.copy()

        # Initialiser le score à 100
        score = pd.Series(100.0, index=df.index)

        # Pénalités pour excès (0-30 points chacun)
        score -= np.clip(df["calories_pdv"] - 20, 0, 30)  # Pénalité si > 20% DV
        score -= np.clip(df["fat_pdv"] - 20, 0, 25)
        score -= np.clip(df["saturated_fat_pdv"] - 15, 0, 20)
        score -= np.clip(df["sodium_pdv"] - 15, 0, 20)
        score -= np.clip(df["sugar_pdv"] - 15, 0, 25)

        # Bonus pour protéines (0-15 points)
        score += np.clip(df["protein_pdv"], 0, 15)

        # Bonus pour haute densité protéique (0-10 points)
        score += np.clip(df["protein_density"] / 2, 0, 10)

        # Normaliser entre 0-100
        df["health_score"] = np.clip(score, 0, 100)

        logger.info(f"  ✓ Score moyen: {df['health_score'].mean():.1f}/100")
        return df

    def categorize_nutrition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Catégorise les recettes en 4 classes nutritionnelles.

        Classes:
        0 - Very Healthy: Faible calories, peu de gras/sodium/sucre, riche en protéines
        1 - Healthy: Équilibré avec quelques excès mineurs
        2 - Moderate: Calories/nutriments moyens à élevés
        3 - Indulgent: Riches en calories, graisses, sucre ou sodium

        Args:
            df: DataFrame avec toutes les features

        Returns:
            DataFrame avec colonne nutrition_category (0-3)
        """
        logger.info("Catégorisation nutritionnelle...")

        df = df.copy()

        # Initialiser à 3 (indulgent) par défaut
        df["nutrition_category"] = 3

        # Critères pour "Very Healthy" (0)
        very_healthy_mask = (
            (df["calories"] <= self.health_thresholds["very_healthy"]["calories_max"])
            & (df["fat_pdv"] <= self.health_thresholds["very_healthy"]["fat_pdv_max"])
            & (
                df["sodium_pdv"]
                <= self.health_thresholds["very_healthy"]["sodium_pdv_max"]
            )
            & (
                df["sugar_pdv"]
                <= self.health_thresholds["very_healthy"]["sugar_pdv_max"]
            )
            & (
                df["protein_pdv"]
                >= self.health_thresholds["very_healthy"]["protein_pdv_min"]
            )
        )
        df.loc[very_healthy_mask, "nutrition_category"] = 0

        # Critères pour "Healthy" (1)
        healthy_mask = (
            (df["nutrition_category"] != 0)
            & (df["calories"] <= self.health_thresholds["healthy"]["calories_max"])
            & (df["fat_pdv"] <= self.health_thresholds["healthy"]["fat_pdv_max"])
            & (df["sodium_pdv"] <= self.health_thresholds["healthy"]["sodium_pdv_max"])
            & (df["sugar_pdv"] <= self.health_thresholds["healthy"]["sugar_pdv_max"])
        )
        df.loc[healthy_mask, "nutrition_category"] = 1

        # Critères pour "Moderate" (2)
        moderate_mask = (
            (df["nutrition_category"] != 0)
            & (df["nutrition_category"] != 1)
            & (df["calories"] <= self.health_thresholds["moderate"]["calories_max"])
            & (df["fat_pdv"] <= self.health_thresholds["moderate"]["fat_pdv_max"])
            & (df["sodium_pdv"] <= self.health_thresholds["moderate"]["sodium_pdv_max"])
            & (df["sugar_pdv"] <= self.health_thresholds["moderate"]["sugar_pdv_max"])
        )
        df.loc[moderate_mask, "nutrition_category"] = 2

        # Distribution des catégories
        category_counts = df["nutrition_category"].value_counts().sort_index()
        logger.info("  Distribution des catégories:")
        categories = ["Very Healthy", "Healthy", "Moderate", "Indulgent"]
        for cat_id, count in category_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"    {cat_id} - {categories[cat_id]}: {count:,} ({pct:.1f}%)")

        return df

    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features supplémentaires pour le modèle.

        Args:
            df: DataFrame avec features existantes

        Returns:
            DataFrame avec features additionnelles
        """
        logger.info("Création de features additionnelles...")

        df = df.copy()

        # Indicateurs binaires pour valeurs extrêmes
        df["is_low_calorie"] = (df["calories"] < 200).astype(int)
        df["is_high_protein"] = (df["protein_pdv"] > 30).astype(int)
        df["is_low_fat"] = (df["fat_pdv"] < 10).astype(int)
        df["is_low_sodium"] = (df["sodium_pdv"] < 10).astype(int)
        df["is_low_sugar"] = (df["sugar_pdv"] < 10).astype(int)

        # Feature de complexité (corrélée avec temps de préparation)
        df["complexity_score"] = np.log1p(df["n_steps"]) + np.log1p(df["n_ingredients"])

        # Interactions
        df["calorie_protein_interaction"] = df["calories"] * df["protein_pdv"]
        df["fat_sodium_interaction"] = df["fat_pdv"] * df["sodium_pdv"]

        logger.info("  ✓ 9 features additionnelles créées")
        return df

    def build_features(
        self, input_path: str, output_path: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Pipeline complet de feature engineering.

        Args:
            input_path: Chemin vers recipes_clean.csv
            output_path: Chemin de sortie pour recipes_nutrition_features.csv

        Returns:
            Tuple (DataFrame avec features, dictionnaire de statistiques)
        """
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING - CLASSIFICATION NUTRITIONNELLE")
        logger.info("=" * 80)

        # 1. Chargement
        df = self.load_data(input_path)
        initial_count = len(df)

        # 2. Filtrage des valeurs aberrantes (si nécessaire)
        # Garder uniquement recettes avec données nutritionnelles complètes
        nutrition_cols = [
            "calories",
            "total_fat",
            "sugar",
            "sodium",
            "protein",
            "saturated_fat",
            "carbohydrates",
        ]
        df = df.dropna(subset=nutrition_cols)

        # Filtrer outliers extrêmes (99.9 percentile)
        for col in nutrition_cols:
            threshold = df[col].quantile(0.999)
            df = df[df[col] <= threshold]

        logger.info(f"Filtrage: {initial_count:,} → {len(df):,} recettes")

        # 3. Calcul des PDV%
        df = self.calculate_pdv_percentages(df)

        # 4. Ratios nutritionnels
        df = self.calculate_nutritional_ratios(df)

        # 5. Score de santé
        df = self.calculate_health_score(df)

        # 6. Catégorisation (TARGET)
        df = self.categorize_nutrition(df)

        # 7. Features additionnelles
        df = self.create_additional_features(df)

        # 8. Sélection des colonnes pour export
        feature_cols = [
            # Identifiants
            "id",
            "name",
            # Nutrition brute
            "calories",
            "total_fat",
            "saturated_fat",
            "sodium",
            "carbohydrates",
            "sugar",
            "protein",
            # PDV%
            "calories_pdv",
            "fat_pdv",
            "saturated_fat_pdv",
            "sodium_pdv",
            "carbs_pdv",
            "sugar_pdv",
            "protein_pdv",
            # Ratios
            "protein_density",
            "sugar_ratio",
            "sodium_density",
            "saturated_fat_ratio",
            "macro_balance",
            # Score et catégorie
            "health_score",
            "nutrition_category",
            # Features additionnelles
            "is_low_calorie",
            "is_high_protein",
            "is_low_fat",
            "is_low_sodium",
            "is_low_sugar",
            "complexity_score",
            "calorie_protein_interaction",
            "fat_sodium_interaction",
            # Autres utiles
            "n_steps",
            "n_ingredients",
            "minutes",
        ]

        df_export = df[feature_cols].copy()

        # 9. Export
        logger.info(f"\nExport vers {output_path}...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_export.to_csv(output_path, index=False)

        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Fichier créé: {file_size:.2f} MB")
        logger.info(f"  ✓ {len(df_export):,} recettes")
        logger.info(f"  ✓ {len(feature_cols)} colonnes")

        # 10. Statistiques
        stats = self._generate_statistics(df_export)

        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING TERMINÉ")
        logger.info("=" * 80)

        return df_export, stats

    def _generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Génère des statistiques descriptives."""
        stats = {
            "n_recipes": len(df),
            "n_features": df.shape[1],
            "category_distribution": df["nutrition_category"].value_counts().to_dict(),
            "health_score_stats": {
                "mean": float(df["health_score"].mean()),
                "std": float(df["health_score"].std()),
                "min": float(df["health_score"].min()),
                "max": float(df["health_score"].max()),
            },
            "feature_correlations": {
                "health_score_calories": float(
                    df[["health_score", "calories"]].corr().iloc[0, 1]
                ),
                "health_score_protein": float(
                    df[["health_score", "protein_pdv"]].corr().iloc[0, 1]
                ),
                "health_score_sugar": float(
                    df[["health_score", "sugar_pdv"]].corr().iloc[0, 1]
                ),
            },
        }
        return stats


def main():
    """Point d'entrée principal."""
    engineer = NutritionClassificationFeatures()

    input_path = "data/processed/recipes_clean.csv"
    output_path = "data/processed/recipes_nutrition_features.csv"

    df, stats = engineer.build_features(input_path, output_path)

    logger.info("=" * 80)
    logger.info("STATISTIQUES FINALES")
    logger.info("=" * 80)
    logger.info(f"Recettes totales: {stats['n_recipes']:,}")
    logger.info(f"Features créées: {stats['n_features']}")
    logger.info(
        f"Health Score: {stats['health_score_stats']['mean']:.1f} +/- {stats['health_score_stats']['std']:.1f}"
    )
    logger.info("Corrélations avec Health Score:")
    logger.info(f"  - Calories: {stats['feature_correlations']['health_score_calories']:.3f}")
    logger.info(f"  - Protéines: {stats['feature_correlations']['health_score_protein']:.3f}")
    logger.info(f"  - Sucre: {stats['feature_correlations']['health_score_sugar']:.3f}")


if __name__ == "__main__":
    main()
