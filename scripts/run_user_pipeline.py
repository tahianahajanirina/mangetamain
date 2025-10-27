#!/usr/bin/env python3
"""
Pipeline complet pour le clustering des utilisateurs

Ce script exécute:
1. Construction des features utilisateurs
2. Clustering K-Means
3. Sauvegarde du modèle

Usage:
    python scripts/run_user_pipeline.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.user_features import UserFeatureBuilder
from src.modeling.user_clustering import UserClusterer
from config.config import DATA_PROCESSED, OUTPUTS_FIGURES


def main():
    print("=" * 80)
    print("PIPELINE COMPLET UTILISATEURS")
    print("=" * 80)
    
    # Chemins
    recipes_path = DATA_PROCESSED / 'recipes_clean.csv'
    interactions_path = DATA_PROCESSED / 'interactions_clean.csv'
    features_path = DATA_PROCESSED / 'users_profiles.csv'
    clustered_path = DATA_PROCESSED / 'users_clustered.csv'
    model_dir = 'outputs/models'
    
    # Vérifier les prérequis
    if not recipes_path.exists() or not interactions_path.exists():
        print("\n❌ ERREUR: Fichiers clean manquants")
        print("   Exécutez d'abord: python scripts/clean_data.py")
        sys.exit(1)
    
    try:
        # ÉTAPE 1: Feature Engineering
        print("\n" + "=" * 80)
        print("ÉTAPE 1/3: FEATURE ENGINEERING")
        print("=" * 80)
        
        builder = UserFeatureBuilder(min_interactions=5)
        users_profiles = builder.build_features(
            recipes_path=str(recipes_path),
            interactions_path=str(interactions_path)
        )
        builder.save_features(str(features_path))
        
        print(f"\n✓ Features créées: {len(users_profiles):,} utilisateurs")
        
        # ÉTAPE 2: Clustering
        print("\n" + "=" * 80)
        print("ÉTAPE 2/3: CLUSTERING K-MEANS")
        print("=" * 80)
        
        clusterer = UserClusterer(n_clusters=None, pca_variance=0.95, random_state=42)
        clusterer.fit(features_path=str(features_path))
        clusterer.name_clusters()
        
        # Visualisation
        clusterer.plot_clusters(output_dir=str(OUTPUTS_FIGURES))
        
        # Sauvegarder résultats
        clusterer.save_results(str(clustered_path))
        
        # ÉTAPE 3: Sauvegarde du modèle
        print("\n" + "=" * 80)
        print("ÉTAPE 3/3: SAUVEGARDE DU MODÈLE")
        print("=" * 80)
        
        model_name = clusterer.save_model(model_dir)
        
        # Statistiques finales
        stats = clusterer.get_stats()
        
        print("\n" + "=" * 80)
        print("PIPELINE TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        
        print(f"\n✓ Modèle: {model_name}")
        print(f"✓ Silhouette: {stats['silhouette']:.4f}")
        print(f"✓ Clusters: {stats['n_clusters']}")
        print(f"✓ Utilisateurs: {stats['n_users']:,}")
        
        print(f"\n✓ Fichiers générés:")
        print(f"  • {features_path}")
        print(f"  • {clustered_path}")
        print(f"  • outputs/models/{model_name}_*.pkl")
        print(f"  • outputs/figures/user_clusters_pca.png")
        
        print("\n" + "=" * 80)
        print("UTILISATION EN PRODUCTION")
        print("=" * 80)
        print(f"\n  from src.modeling.user_clustering import UserClusterer")
        print(f"  model = UserClusterer.load_model('{model_name}')")
        print(f"  predictions = model.predict(new_users_df)")
        
        return model_name
        
    except Exception as e:
        print(f"\n❌ ERREUR dans le pipeline:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    model_name = main()
