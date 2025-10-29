#!/usr/bin/env python3
"""
Script de nettoyage des données brutes

Ce script remplace le notebook 02_data_cleaning.ipynb
et peut être exécuté en ligne de commande.

Usage:
    python scripts/clean_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import DATA_PROCESSED, DATA_RAW


def safe_eval(x):
    """Évalue en toute sécurité les chaînes représentant des listes"""
    # Check if it's already a list first
    if isinstance(x, list):
        return x
    # Then check for NaN (works with scalars only)
    if pd.isna(x):
        return []
    try:
        import ast

        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def clean_recipes(recipes_raw_path: Path, output_path: Path) -> pd.DataFrame:

    # 1. Chargement
    print("\n[1/7] Chargement des données brutes...")
    recipes = pd.read_csv(recipes_raw_path)
    print(f"  Recettes brutes: {len(recipes):,}")

    # 2. Doublons
    print("\n[2/7] Suppression des doublons...")
    recipes_before = len(recipes)
    recipes = recipes.drop_duplicates(subset=["id"])
    print(f"  Doublons supprimés: {recipes_before - len(recipes):,}")

    # 3. Valeurs manquantes
    print("\n[3/7] Gestion des valeurs manquantes...")
    # Supprimer lignes avec valeurs critiques manquantes
    critical_cols = ["name", "minutes", "n_steps", "n_ingredients"]
    recipes = recipes.dropna(subset=critical_cols)
    print(f"  Recettes après nettoyage: {len(recipes):,}")

    # 4. Outliers
    print("\n[4/7] Gestion des outliers...")
    # Temps: garder entre 1 min et 24h (1440 min)
    recipes = recipes[(recipes["minutes"] >= 1) & (recipes["minutes"] <= 1440)]
    # Calories: garder entre 0 et 5000
    if "calories" in recipes.columns:
        recipes = recipes[(recipes["calories"] >= 0) & (recipes["calories"] <= 5000)]
    print(f"  Recettes après outliers: {len(recipes):,}")

    # 5. Parser colonnes JSON
    print("\n[5/7] Parsing des colonnes JSON...")
    for col in ["tags", "steps", "ingredients"]:
        if col in recipes.columns:
            recipes[f"{col}_parsed"] = recipes[col].apply(safe_eval)
            recipes[f"n_{col}"] = recipes[f"{col}_parsed"].apply(len)

    # 6. Extraire nutrition
    print("\n[6/7] Extraction des valeurs nutritionnelles...")
    if "nutrition" in recipes.columns:
        nutrition_data = recipes["nutrition"].apply(safe_eval)

        # Colonnes nutrition: [calories, total_fat, sugar, sodium, protein, saturated_fat, carbohydrates]
        nutrition_cols = [
            "calories",
            "total_fat",
            "sugar",
            "sodium",
            "protein",
            "saturated_fat",
            "carbohydrates",
        ]

        for i, col in enumerate(nutrition_cols):
            recipes[col] = nutrition_data.apply(
                lambda x: x[i] if len(x) > i else np.nan
            )

    # 7. Export
    print("\n[7/7] Export des données nettoyées...")
    recipes.to_csv(output_path, index=False)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Fichier: {output_path}")
    print(f"  Taille: {file_size:.2f} MB")
    print(f"  Colonnes: {len(recipes.columns)}")
    print(f"  Recettes: {len(recipes):,}")

    return recipes


def clean_interactions(interactions_raw_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Nettoie les interactions brutes.

    Étapes:
    1. Charger données brutes
    2. Supprimer doublons
    3. Valider ratings (0-5)
    4. Supprimer valeurs manquantes
    5. Exporter données propres
    """
    print("\n" + "=" * 80)
    print("NETTOYAGE DES INTERACTIONS")
    print("=" * 80)

    # 1. Chargement
    print("\n[1/4] Chargement des données brutes...")
    interactions = pd.read_csv(interactions_raw_path)
    print(f"  Interactions brutes: {len(interactions):,}")

    # 2. Doublons
    print("\n[2/4] Suppression des doublons...")
    interactions_before = len(interactions)
    interactions = interactions.drop_duplicates(subset=["user_id", "recipe_id", "date"])
    print(f"  Doublons supprimés: {interactions_before - len(interactions):,}")

    # 3. Valider ratings
    print("\n[3/4] Validation des ratings...")
    if "rating" in interactions.columns:
        # Garder seulement ratings entre 0 et 5
        interactions = interactions[
            (interactions["rating"] >= 0) & (interactions["rating"] <= 5)
        ]
        print(f"  Ratings valides: {len(interactions):,}")

    # 4. Valeurs manquantes
    print("\n[4/4] Suppression valeurs manquantes...")
    interactions = interactions.dropna(subset=["user_id", "recipe_id", "rating"])
    print(f"  Interactions finales: {len(interactions):,}")

    # Export
    interactions.to_csv(output_path, index=False)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n  Fichier: {output_path}")
    print(f"  Taille: {file_size:.2f} MB")
    print(f"  Interactions: {len(interactions):,}")

    return interactions


def main():
    print("=" * 80)
    print("NETTOYAGE DES DONNÉES - PIPELINE COMPLET")
    print("=" * 80)

    # Chemins
    recipes_raw_path = DATA_RAW / "RAW_recipes.csv"
    interactions_raw_path = DATA_RAW / "RAW_interactions.csv"

    recipes_clean_path = DATA_PROCESSED / "recipes_clean.csv"
    interactions_clean_path = DATA_PROCESSED / "interactions_clean.csv"

    # Vérifier que les fichiers bruts existent
    if not recipes_raw_path.exists():
        print(f"\n❌ ERREUR: {recipes_raw_path} n'existe pas")
        print("   Placez les fichiers bruts dans data/raw/")
        sys.exit(1)

    if not interactions_raw_path.exists():
        print(f"\n❌ ERREUR: {interactions_raw_path} n'existe pas")
        print("   Placez les fichiers bruts dans data/raw/")
        sys.exit(1)

    try:
        # Nettoyer recettes
        recipes_clean = clean_recipes(recipes_raw_path, recipes_clean_path)

        # Nettoyer interactions
        interactions_clean = clean_interactions(
            interactions_raw_path, interactions_clean_path
        )

        # Résumé final
        print("\n" + "=" * 80)
        print("NETTOYAGE TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        print(f"\n✅ Recettes nettoyées: {len(recipes_clean):,}")
        print(f"   Fichier: {recipes_clean_path}")
        print(f"\n✅ Interactions nettoyées: {len(interactions_clean):,}")
        print(f"   Fichier: {interactions_clean_path}")

        print("\n" + "=" * 80)
        print("PROCHAINES ÉTAPES")
        print("=" * 80)
        print("\n1. Feature Engineering:")
        print("   python scripts/build_recipe_features.py")
        print("   python scripts/build_user_features.py")
        print("\n2. Clustering:")
        print("   python scripts/cluster_recipes.py")
        print("   python scripts/cluster_users.py")

    except Exception as e:
        print("\n❌ ERREUR lors du nettoyage:")
        print(f"   {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
