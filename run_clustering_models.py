#!/usr/bin/env python3
"""
Simple script to run clustering models for both users and recipes
This script generates the missing clustering models for deployment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.modeling.user_clustering import UserClusterer
from src.modeling.recipe_clustering import RecipeClusterer

def build_user_features():
    """Build user features from raw data"""
    print("="*80)
    print("BUILDING USER FEATURES")
    print("="*80)

    data_dir = Path("data")

    # Load data
    print("\nLoading data...")
    recipes = pd.read_csv(data_dir / "processed_recipes.csv")
    interactions = pd.read_csv(data_dir / "RAW_interactions.csv")

    print(f"Recipes: {len(recipes):,}")
    print(f"Interactions: {len(interactions):,}")

    # Merge
    print("\nMerging data...")
    merged = interactions.merge(
        recipes[['id', 'minutes', 'n_steps', 'n_ingredients', 'calories']],
        left_on='recipe_id',
        right_on='id',
        how='left'
    )

    # Build features
    print("\nBuilding features...")
    users = merged.groupby('user_id').agg({
        'recipe_id': 'count',
        'rating': 'mean',
        'minutes': ['mean', 'std'],
        'n_steps': ['mean', 'std'],
        'n_ingredients': ['mean', 'std'],
        'calories': ['mean', 'std']
    })

    # Flatten columns
    users.columns = ['_'.join(col).strip() if col[1] else col[0] for col in users.columns.values]
    users = users.rename(columns={
        'recipe_id_count': 'n_interactions',
        'rating_mean': 'avg_rating',
        'minutes_mean': 'avg_minutes',
        'minutes_std': 'std_minutes',
        'n_steps_mean': 'avg_n_steps',
        'n_steps_std': 'std_n_steps',
        'n_ingredients_mean': 'avg_n_ingredients',
        'n_ingredients_std': 'std_n_ingredients',
        'calories_mean': 'avg_calories',
        'calories_std': 'std_calories'
    })

    # Fill NaN
    users = users.fillna(0)
    users = users.reset_index()

    # Filter users with minimum interactions
    users = users[users['n_interactions'] >= 5]

    print(f"\nUsers with >= 5 interactions: {len(users):,}")
    print(f"Features: {len(users.columns) - 1}")

    # Save
    output_path = data_dir / "users_profiles.csv"
    users.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return output_path


def build_recipe_features():
    """Build recipe features from processed data"""
    print("\n" + "="*80)
    print("BUILDING RECIPE FEATURES")
    print("="*80)

    data_dir = Path("data")

    # Load data
    print("\nLoading data...")
    recipes = pd.read_csv(data_dir / "processed_recipes.csv")

    print(f"Recipes: {len(recipes):,}")

    # Build features for clustering
    print("\nBuilding features...")
    features = recipes[['id', 'name']].copy()

    # Log transform minutes
    features['log_minutes'] = np.log1p(recipes['minutes'].clip(lower=0))

    # Time complexity (minutes * n_steps)
    features['time_complexity'] = recipes['minutes'] * recipes['n_steps']

    # Efficiency (calories per minute)
    features['efficiency'] = recipes['calories'] / (recipes['minutes'] + 1)

    # Health category (based on multiple nutrition factors)
    # Simple health score: higher protein, lower fat/sugar
    health_score = (
        (recipes['protein_pdv'] / 10) -
        (recipes['total_fat_pdv'] / 20) -
        (recipes['sugar_pdv'] / 20)
    )
    features['health_category'] = health_score.clip(lower=0, upper=5)

    # Popularity score (simple version based on contributor_id frequency)
    contributor_counts = recipes['contributor_id'].value_counts()
    features['popularity_score'] = recipes['contributor_id'].map(contributor_counts).fillna(1)
    features['popularity_score'] = np.log1p(features['popularity_score'])

    print(f"\nFeatures created: {len(features.columns) - 2}")  # -2 for id and name

    # Save
    output_path = data_dir / "recipes_features_v3.csv"
    features.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return output_path


def run_user_clustering():
    """Run user clustering"""
    print("\n" + "="*80)
    print("USER CLUSTERING")
    print("="*80)

    # Build features first
    features_path = build_user_features()

    # Run clustering
    clusterer = UserClusterer(n_clusters=None, pca_variance=0.95)
    clusterer.fit(features_path=str(features_path))

    # Name clusters
    clusterer.name_clusters()

    # Visualize
    clusterer.plot_clusters(output_dir="outputs/figures")

    # Save results
    clusterer.save_results("data/users_clustered.csv")

    # Save model
    model_name = clusterer.save_model(output_dir="outputs/models")

    # Stats
    stats = clusterer.get_stats()
    print("\n" + "="*80)
    print("USER CLUSTERING COMPLETE")
    print("="*80)
    print(f"Clusters: {stats['n_clusters']}")
    print(f"Users: {stats['n_users']:,}")
    print(f"Silhouette: {stats['silhouette']:.4f}")
    print(f"Model: {model_name}")


def run_recipe_clustering():
    """Run recipe clustering"""
    print("\n" + "="*80)
    print("RECIPE CLUSTERING")
    print("="*80)

    # Build features first
    features_path = build_recipe_features()

    # Run clustering
    clusterer = RecipeClusterer(n_clusters=None, pca_variance=0.90)
    clusterer.fit(features_path=str(features_path))

    # Name clusters
    clusterer.name_clusters()

    # Visualize
    clusterer.plot_clusters(output_dir="outputs/figures")

    # Save results
    clusterer.save_results("data/recipes_clustered_v3.csv")

    # Save model
    model_name = clusterer.save_model(output_dir="outputs/models")

    # Stats
    stats = clusterer.get_stats()
    print("\n" + "="*80)
    print("RECIPE CLUSTERING COMPLETE")
    print("="*80)
    print(f"Clusters: {stats['n_clusters']}")
    print(f"Recipes: {stats['n_recipes']:,}")
    print(f"Silhouette: {stats['silhouette']:.4f}")
    print(f"Model: {model_name}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CLUSTERING MODEL GENERATION")
    print("="*80)
    print("\nGenerating clustering models for deployment...")
    print("This will create:")
    print("  1. User clustering model")
    print("  2. Recipe clustering model")
    print()

    try:
        # Run user clustering
        run_user_clustering()

        # Run recipe clustering
        run_recipe_clustering()

        print("\n" + "="*80)
        print("ALL CLUSTERING MODELS GENERATED SUCCESSFULLY!")
        print("="*80)
        print("\nModels saved in: outputs/models/")
        print("Visualizations saved in: outputs/figures/")
        print("\nReady for deployment!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
