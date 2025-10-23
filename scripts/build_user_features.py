#!/usr/bin/env python3
"""
Script pour construire les features utilisateurs

Usage:
    python scripts/build_user_features.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.user_features import UserFeatureBuilder
from config.config import DATA_PROCESSED


def main():
    print("=" * 80)
    print("BUILD USER FEATURES")
    print("=" * 80)
    
    # Chemins
    recipes_path = DATA_PROCESSED / 'recipes_clean.csv'
    interactions_path = DATA_PROCESSED / 'interactions_clean.csv'
    output_path = DATA_PROCESSED / 'users_profiles.csv'
    
    # Vérifier que les fichiers existent
    if not recipes_path.exists():
        print(f"❌ ERREUR: {recipes_path} n'existe pas")
        print("   Exécutez d'abord le preprocessing")
        sys.exit(1)
    
    if not interactions_path.exists():
        print(f"❌ ERREUR: {interactions_path} n'existe pas")
        print("   Exécutez d'abord le preprocessing")
        sys.exit(1)
    
    # Construire les features
    builder = UserFeatureBuilder(min_interactions=5)
    
    try:
        users_profiles = builder.build_features(
            recipes_path=str(recipes_path),
            interactions_path=str(interactions_path)
        )
        
        # Sauvegarder
        builder.save_features(str(output_path))
        
        # Afficher les stats
        stats = builder.get_feature_stats()
        print("\n" + "=" * 80)
        print("STATISTIQUES")
        print("=" * 80)
        print(f"Utilisateurs: {stats['n_users']:,}")
        print(f"Features: {stats['n_features']}")
        print(f"\nMétriques moyennes:")
        print(f"  Interactions: {stats['interactions_mean']:.1f} (médiane: {stats['interactions_median']:.0f})")
        print(f"  Rating: {stats['rating_mean']:.2f}")
        
        print("\n✓ Features utilisateurs créées avec succès!")
        print(f"  Fichier: {output_path}")
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de la construction des features:")
        print(f"   {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
