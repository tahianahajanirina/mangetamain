#!/usr/bin/env python3
"""
Script de déploiement - Entraîne et sauvegarde les modèles
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.recipe_features import build_recipe_features
from src.modeling.recipe_clustering import RecipeClusterer


def main():
    """Pipeline complet: Features → Clustering → Sauvegarde modèle"""
    
    print("=" * 80)
    print("PIPELINE DE DEPLOIEMENT - RECIPE CLUSTERING V3")
    print("=" * 80)
    
    # Chemins
    RECIPES_CLEAN = 'data/processed/recipes_clean.csv'
    INTERACTIONS_CLEAN = 'data/processed/interactions_clean.csv'
    FEATURES_OUTPUT = 'data/processed/recipes_features_v3.csv'
    CLUSTERED_OUTPUT = 'data/processed/recipes_clustered_v3.csv'
    MODEL_DIR = 'outputs/models'
    
    # Étape 1: Feature Engineering
    print("\n" + "=" * 80)
    print("ÉTAPE 1: FEATURE ENGINEERING")
    print("=" * 80)
    
    recipes_features = build_recipe_features(
        recipes_path=RECIPES_CLEAN,
        interactions_path=INTERACTIONS_CLEAN,
        output_path=FEATURES_OUTPUT,
        min_rating=3.5
    )
    
    print(f"\n✓ Features créées: {len(recipes_features):,} recettes")
    
    # Étape 2: Clustering
    print("\n" + "=" * 80)
    print("ÉTAPE 2: CLUSTERING K-MEANS")
    print("=" * 80)
    
    clusterer = RecipeClusterer(n_clusters=5, pca_variance=0.90, random_state=42)
    clusterer.fit(FEATURES_OUTPUT)
    
    # Nommer les clusters
    clusterer.name_clusters()
    
    # Sauvegarder résultats
    clusterer.save_results(CLUSTERED_OUTPUT)
    print(f"\n✓ Clusters sauvegardés: {CLUSTERED_OUTPUT}")
    
    # Étape 3: Sauvegarder le modèle
    print("\n" + "=" * 80)
    print("ÉTAPE 3: SAUVEGARDE DU MODÈLE")
    print("=" * 80)
    
    model_name = clusterer.save_model(MODEL_DIR)
    
    # Statistiques finales
    stats = clusterer.get_stats()
    
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"\nModèle: {model_name}")
    print(f"Silhouette: {stats['silhouette']:.4f}")
    print(f"Clusters: {stats['n_clusters']}")
    print(f"Recettes: {stats['n_recipes']:,}")
    print(f"\nFichiers générés:")
    print(f"  • {FEATURES_OUTPUT}")
    print(f"  • {CLUSTERED_OUTPUT}")
    print(f"  • {MODEL_DIR}/{model_name}_*.pkl")
    
    print("\n" + "=" * 80)
    print("✓ DÉPLOIEMENT RÉUSSI")
    print("=" * 80)
    
    print(f"\nUtilisation:")
    print(f"  from src.modeling.recipe_clustering import RecipeClusterer")
    print(f"  model = RecipeClusterer.load_model('{model_name}')")
    print(f"  predictions = model.predict(new_recipes_df)")
    
    return model_name


if __name__ == '__main__':
    model_name = main()
