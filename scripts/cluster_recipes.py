#!/usr/bin/env python3
"""
Script pour effectuer le clustering des recettes V3

Usage:
    python scripts/cluster_recipes.py [--k N]

Options:
    --k N    Nombre de clusters (défaut: détection automatique)
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import DATA_PROCESSED, OUTPUTS_FIGURES
from src.modeling.recipe_clustering import RecipeClusterer


def main():
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Clustering des recettes V3")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Nombre de clusters (défaut: détection auto)",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.90,
        help="Variance PCA à conserver (défaut: 0.90)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CLUSTERING RECETTES V3")
    print("=" * 80)

    # Chemins
    features_path = DATA_PROCESSED / "recipes_features_v3.csv"
    output_path = DATA_PROCESSED / "recipes_clustered_v3.csv"

    # Vérifier que les features existent
    if not features_path.exists():
        print(f"❌ ERREUR: {features_path} n'existe pas")
        print("   Exécutez d'abord: python scripts/build_recipe_features.py")
        sys.exit(1)

    # Créer le dossier de sortie si nécessaire
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # Clustering
    try:
        clusterer = RecipeClusterer(n_clusters=args.k, pca_variance=args.pca_variance)

        # Fit
        clusterer.fit(features_path=str(features_path))

        # Nommer les clusters
        clusterer.name_clusters()

        # Visualiser
        clusterer.plot_clusters(output_dir=str(OUTPUTS_FIGURES))

        # Sauvegarder résultats (CSV)
        clusterer.save_results(str(output_path))

        # Sauvegarder modèle (.pkl)
        model_name = clusterer.save_model()

        # Afficher les stats
        stats = clusterer.get_stats()
        print("\n" + "=" * 80)
        print("STATISTIQUES FINALES")
        print("=" * 80)
        print(f"Clusters: {stats['n_clusters']}")
        print(f"Recettes: {stats['n_recipes']:,}")
        print(f"Silhouette: {stats['silhouette']:.4f}")
        print(f"Inertia: {stats['inertia']:,.2f}")

        print("\nNoms des clusters:")
        for cluster_id, name in stats["cluster_names"].items():
            count = stats["cluster_distribution"][cluster_id]
            pct = 100 * count / stats["n_recipes"]
            print(f'  {cluster_id}. "{name}": {count:,} ({pct:.1f}%)')

        print("\n✓ Clustering terminé avec succès!")
        print(f"  Résultats CSV: {output_path}")
        print(f"  Modèle .pkl: outputs/models/{model_name}_*.pkl")
        print(f"  Visualisation: {OUTPUTS_FIGURES / '07c_recipes_clusters_pca_v3.png'}")

    except Exception as e:
        print("\n❌ ERREUR lors du clustering:")
        print(f"   {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
