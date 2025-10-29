#!/usr/bin/env python3
"""
Pipeline complet: Feature Engineering + Clustering des Recettes V3

Usage:
    python scripts/run_recipe_pipeline.py [--k N]
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import DATA_PROCESSED, OUTPUTS_FIGURES
from src.feature_engineering.recipe_features import RecipeFeatureBuilder
from src.modeling.recipe_clustering import RecipeClusterer


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet recettes V3")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Nombre de clusters (défaut: détection auto)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering (utiliser features existantes)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PIPELINE COMPLET: RECETTES V3")
    print("=" * 80)

    # Chemins
    recipes_path = DATA_PROCESSED / "recipes_clean.csv"
    interactions_path = DATA_PROCESSED / "interactions_clean.csv"
    features_path = DATA_PROCESSED / "recipes_features_v3.csv"
    output_path = DATA_PROCESSED / "recipes_clustered_v3.csv"

    try:
        # ÉTAPE 1: Feature Engineering
        if not args.skip_features:
            print("\n" + "=" * 80)
            print("ÉTAPE 1/2: FEATURE ENGINEERING")
            print("=" * 80)

            builder = RecipeFeatureBuilder(min_rating=3.5)
            _ = builder.build_features(
                recipes_path=str(recipes_path), interactions_path=str(interactions_path)
            )
            builder.save_features(str(features_path))

            stats = builder.get_feature_stats()
            print(
                f"\n✓ Features créées: {stats['n_recipes']:,} recettes, {stats['n_features']} features"
            )
        else:
            print(
                "\n⏭️  Feature engineering skippé (utilisation des features existantes)"
            )

        # ÉTAPE 2: Clustering
        print("\n" + "=" * 80)
        print("ÉTAPE 2/2: CLUSTERING")
        print("=" * 80)

        OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

        clusterer = RecipeClusterer(n_clusters=args.k, pca_variance=0.90)
        clusterer.fit(features_path=str(features_path))
        clusterer.name_clusters()
        clusterer.plot_clusters(output_dir=str(OUTPUTS_FIGURES))
        clusterer.save_results(str(output_path))

        # IMPORTANT: Sauvegarder le modèle .pkl
        model_name = clusterer.save_model()

        stats = clusterer.get_stats()

        # SYNTHÈSE FINALE
        print("\n" + "=" * 80)
        print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("=" * 80)
        print("\n📊 Résultats:")
        print(f"  • Recettes: {stats['n_recipes']:,}")
        print(f"  • Clusters: {stats['n_clusters']}")
        print(f"  • Silhouette: {stats['silhouette']:.4f}")

        print("\n📁 Fichiers générés:")
        print(f"  • Features: {features_path}")
        print(f"  • Clustering: {output_path}")
        print(f"  • Modèle: outputs/models/{model_name}_*.pkl")
        print(
            f"  • Visualisation: {OUTPUTS_FIGURES / '07c_recipes_clusters_pca_v3.png'}"
        )

        print("\n🏷️  Clusters créés:")
        for cluster_id, name in stats["cluster_names"].items():
            count = stats["cluster_distribution"][cluster_id]
            pct = 100 * count / stats["n_recipes"]
            print(f'  {cluster_id}. "{name}": {count:,} ({pct:.1f}%)')

    except Exception as e:
        print("\n❌ ERREUR dans le pipeline:")
        print(f"   {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
