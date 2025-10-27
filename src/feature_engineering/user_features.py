"""
Feature Engineering pour les Profils Utilisateurs

Ce module construit les features utilisateurs à partir des données nettoyées
pour le clustering de segmentation utilisateurs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserFeatureBuilder:
    """Construit les features pour le clustering utilisateurs"""
    
    def __init__(self, min_interactions: int = 5):
        """
        Args:
            min_interactions: Nombre minimum d'interactions pour garder un utilisateur
        """
        self.min_interactions = min_interactions
        self.users_profiles = None
        
    def load_data(self, recipes_path: str, interactions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Charge les données nettoyées"""
        logger.info("Chargement des données...")
        recipes = pd.read_csv(recipes_path)
        interactions = pd.read_csv(interactions_path)
        
        logger.info(f"Recettes: {len(recipes):,}")
        logger.info(f"Interactions: {len(interactions):,}")
        
        return recipes, interactions
    
    def merge_data(self, recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
        """Fusionne interactions et recettes"""
        logger.info("Fusion interactions ↔ recettes...")
        
        merged = interactions.merge(
            recipes,
            left_on='recipe_id',
            right_on='id',
            how='left'
        )
        
        logger.info(f"Dataset fusionné: {len(merged):,} lignes")
        return merged
    
    def build_behavioral_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Construit les features comportementales par utilisateur"""
        logger.info("Création des features comportementales...")
        
        behavioral = merged.groupby('user_id').agg({
            'recipe_id': 'count',  # n_interactions
            'rating': 'mean'        # avg_rating
        }).rename(columns={
            'recipe_id': 'n_interactions',
            'rating': 'avg_rating'
        })
        
        return behavioral
    
    def build_temporal_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Construit les features temporelles (préférences de temps de préparation)"""
        logger.info("Création des features temporelles...")
        
        temporal = merged.groupby('user_id').agg({
            'minutes': ['mean', 'std', 'min', 'max'],
            'n_steps': ['mean', 'std'],
            'n_ingredients': ['mean', 'std']
        })
        
        # Flatten column names
        temporal.columns = ['_'.join(col).strip() for col in temporal.columns.values]
        temporal = temporal.rename(columns={
            'minutes_mean': 'avg_minutes',
            'minutes_std': 'std_minutes',
            'minutes_min': 'min_minutes',
            'minutes_max': 'max_minutes',
            'n_steps_mean': 'avg_n_steps',
            'n_steps_std': 'std_n_steps',
            'n_ingredients_mean': 'avg_n_ingredients',
            'n_ingredients_std': 'std_n_ingredients'
        })
        
        # Fill NaN std with 0
        temporal = temporal.fillna(0)
        
        return temporal
    
    def build_nutritional_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Construit les features nutritionnelles moyennes"""
        logger.info("Création des features nutritionnelles...")
        
        nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 
                         'saturated_fat', 'carbohydrates']
        
        nutritional = merged.groupby('user_id')[nutrition_cols].mean()
        nutritional.columns = [f'avg_{col}' for col in nutritional.columns]
        
        return nutritional
    
    def build_tag_features(self, merged: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Construit les features basées sur les tags préférés"""
        logger.info(f"Création des features tags (top {top_n})...")
        
        # Définir les catégories de tags principales
        tag_categories = {
            'quick': ['60-minutes-or-less', '30-minutes-or-less', '15-minutes-or-less', 'easy'],
            'healthy': ['healthy', 'low-fat', 'low-calorie', 'low-cholesterol'],
            'comfort': ['comfort-food', 'main-dish', 'one-dish-meal'],
            'dessert': ['desserts', 'cookies-and-brownies', 'cakes'],
            'vegetarian': ['vegetarian', 'vegan'],
            'meat': ['meat', 'beef', 'chicken', 'pork'],
            'seasonal': ['summer', 'winter', 'holiday'],
            'cuisine': ['north-american', 'european', 'asian', 'mexican'],
            'baking': ['oven', 'baking', 'breads'],
            'equipment': ['stove-top', 'crock-pot-slow-cooker', 'microwave']
        }
        
        # Initialiser les compteurs par utilisateur
        tag_features = pd.DataFrame(index=merged['user_id'].unique())
        
        for category, tags in tag_categories.items():
            # Compter les interactions avec des recettes contenant ces tags
            category_col = f'tag_{category}'
            tag_features[category_col] = 0
            
            for user_id in tag_features.index:
                user_interactions = merged[merged['user_id'] == user_id]
                
                # Vérifier si les tags sont présents (si tags_parsed existe)
                if 'tags_parsed' in user_interactions.columns:
                    count = user_interactions['tags_parsed'].apply(
                        lambda x: any(tag in str(x).lower() for tag in tags) if pd.notna(x) else False
                    ).sum()
                else:
                    count = 0
                
                # Proportion par rapport au total d'interactions
                tag_features.loc[user_id, category_col] = count / len(user_interactions)
        
        return tag_features
    
    def build_features(self, recipes_path: str, interactions_path: str) -> pd.DataFrame:
        """Pipeline complet de construction des features utilisateurs"""
        logger.info("=" * 80)
        logger.info("CONSTRUCTION DES FEATURES UTILISATEURS")
        logger.info("=" * 80)
        
        # 1. Charger et fusionner
        recipes, interactions = self.load_data(recipes_path, interactions_path)
        merged = self.merge_data(recipes, interactions)
        
        # 2. Construire les features
        behavioral = self.build_behavioral_features(merged)
        temporal = self.build_temporal_features(merged)
        nutritional = self.build_nutritional_features(merged)
        tags = self.build_tag_features(merged)
        
        # 3. Combiner toutes les features
        logger.info("Combinaison de toutes les features...")
        users_profiles = behavioral.join([temporal, nutritional, tags], how='inner')
        
        # 4. Filtrer utilisateurs avec peu d'interactions
        logger.info(f"Filtrage utilisateurs (min {self.min_interactions} interactions)...")
        initial_count = len(users_profiles)
        users_profiles = users_profiles[users_profiles['n_interactions'] >= self.min_interactions]
        logger.info(f"Utilisateurs conservés: {len(users_profiles):,} / {initial_count:,}")
        
        # 5. Reset index pour avoir user_id comme colonne
        users_profiles = users_profiles.reset_index()
        
        self.users_profiles = users_profiles
        
        logger.info("=" * 80)
        logger.info(f"FEATURES UTILISATEURS CRÉÉES")
        logger.info(f"  Utilisateurs: {len(users_profiles):,}")
        logger.info(f"  Features: {len(users_profiles.columns) - 1}")
        logger.info("=" * 80)
        
        return users_profiles
    
    def save_features(self, output_path: str) -> None:
        """Sauvegarde les features dans un fichier CSV"""
        if self.users_profiles is None:
            raise ValueError("Features non construites. Appelez build_features() d'abord.")
        
        logger.info(f"Sauvegarde: {output_path}")
        self.users_profiles.to_csv(output_path, index=False)
        
        # Afficher les stats
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"  Shape: {self.users_profiles.shape}")
        logger.info(f"  Taille: {file_size:.2f} MB")
        logger.info("✓ Sauvegarde terminée")
    
    def get_feature_stats(self) -> Dict:
        """Retourne les statistiques des features"""
        if self.users_profiles is None:
            raise ValueError("Features non construites.")
        
        return {
            'n_users': len(self.users_profiles),
            'n_features': len(self.users_profiles.columns) - 1,
            'columns': list(self.users_profiles.columns),
            'interactions_mean': self.users_profiles['n_interactions'].mean(),
            'interactions_median': self.users_profiles['n_interactions'].median(),
            'rating_mean': self.users_profiles['avg_rating'].mean()
        }


def main():
    """Fonction principale pour exécution en ligne de commande"""
    from config.config import DATA_PROCESSED
    
    # Chemins
    recipes_path = DATA_PROCESSED / 'recipes_clean.csv'
    interactions_path = DATA_PROCESSED / 'interactions_clean.csv'
    output_path = DATA_PROCESSED / 'users_profiles.csv'
    
    # Construire les features
    builder = UserFeatureBuilder(min_interactions=5)
    users_profiles = builder.build_features(
        recipes_path=str(recipes_path),
        interactions_path=str(interactions_path)
    )
    
    # Sauvegarder
    builder.save_features(str(output_path))
    
    # Afficher les stats
    stats = builder.get_feature_stats()
    print("\nSTATISTIQUES:")
    for key, value in stats.items():
        if key != 'columns':
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
