"""Recipe recommendation system using time prediction model.

This module provides functionality to recommend recipes based on:
- Available cooking time
- Dietary preferences
- Difficulty level
- Course type
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from src.modeling.time_predictor import TimePredictionModel

logger = logging.getLogger(__name__)


class RecipeRecommender:
    """Recommend recipes based on time and preferences."""

    def __init__(
        self,
        recipes_df: pd.DataFrame,
        time_model: TimePredictionModel,
        feature_cols: List[str]
    ):
        """Initialize the recommender.

        Args:
            recipes_df: DataFrame with recipe data and features.
            time_model: Trained time prediction model.
            feature_cols: List of feature column names.
        """
        self.recipes_df = recipes_df.copy()
        self.time_model = time_model
        self.feature_cols = feature_cols

        # Predict time for all recipes if not already predicted
        if 'predicted_time' not in self.recipes_df.columns:
            self._predict_all_times()

        logger.info(f"RecipeRecommender initialized with {len(self.recipes_df)} recipes")

    def _predict_all_times(self):
        """Predict cooking time for all recipes."""
        logger.info("Predicting cooking times for all recipes")

        X = self.recipes_df[self.feature_cols].fillna(0)
        predictions = self.time_model.predict(X, scale_features=True)

        self.recipes_df['predicted_time'] = predictions
        logger.info("Predictions completed")

    def recommend_by_time(
        self,
        available_minutes: int,
        max_recipes: int = 10,
        time_buffer: int = 10
    ) -> pd.DataFrame:
        """Recommend recipes that fit within available time.

        Args:
            available_minutes: Available cooking time in minutes.
            max_recipes: Maximum number of recommendations.
            time_buffer: Buffer time in minutes (flexibility).

        Returns:
            DataFrame with recommended recipes.
        """
        # Filter recipes within time limit
        time_limit = available_minutes + time_buffer
        filtered = self.recipes_df[
            self.recipes_df['predicted_time'] <= time_limit
        ].copy()

        # Sort by how close they are to available time (use it well)
        filtered['time_score'] = (
            available_minutes - abs(filtered['predicted_time'] - available_minutes * 0.8)
        )

        # Return top recipes
        recommendations = filtered.nlargest(max_recipes, 'time_score')

        return recommendations[[
            'name', 'predicted_time', 'minutes', 'n_steps',
            'n_ingredients', 'tags'
        ]].reset_index(drop=True)

    def recommend_advanced(
        self,
        available_minutes: int,
        dietary_tags: Optional[List[str]] = None,
        course_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        exclude_ingredients: Optional[List[str]] = None,
        max_recipes: int = 10
    ) -> pd.DataFrame:
        """Advanced recipe recommendations with multiple filters.

        Args:
            available_minutes: Available cooking time in minutes.
            dietary_tags: List of dietary requirements (e.g., ['vegan', 'gluten-free']).
            course_type: Course type (e.g., 'main-dish', 'dessert', 'appetizer').
            difficulty: Difficulty level ('easy', 'moderate', 'hard').
            exclude_ingredients: List of ingredients to exclude.
            max_recipes: Maximum number of recommendations.

        Returns:
            DataFrame with recommended recipes.
        """
        filtered = self.recipes_df.copy()

        # Filter by time (with 15 min buffer)
        filtered = filtered[filtered['predicted_time'] <= available_minutes + 15]

        # Filter by dietary requirements
        if dietary_tags:
            for tag in dietary_tags:
                tag_lower = tag.lower()
                filtered = filtered[
                    filtered['tags'].apply(
                        lambda x: any(tag_lower in str(t).lower() for t in x) if isinstance(x, list) else False
                    )
                ]

        # Filter by course type
        if course_type:
            filtered = filtered[
                filtered['tags'].apply(
                    lambda x: any(course_type.lower() in str(t).lower() for t in x) if isinstance(x, list) else False
                )
            ]

        # Filter by difficulty
        if difficulty == 'easy':
            filtered = filtered[
                (filtered['n_steps'] <= 10) &
                (filtered['n_ingredients'] <= 12)
            ]
        elif difficulty == 'moderate':
            filtered = filtered[
                (filtered['n_steps'] > 10) & (filtered['n_steps'] <= 15)
            ]
        elif difficulty == 'hard':
            filtered = filtered[filtered['n_steps'] > 15]

        # Exclude recipes with certain ingredients
        if exclude_ingredients:
            for ingredient in exclude_ingredients:
                ingredient_lower = ingredient.lower()
                filtered = filtered[
                    ~filtered['ingredients'].apply(
                        lambda x: any(ingredient_lower in str(ing).lower() for ing in x) if isinstance(x, list) else False
                    )
                ]

        if len(filtered) == 0:
            logger.warning("No recipes found matching criteria")
            return pd.DataFrame()

        # Score recipes
        filtered['time_score'] = (
            available_minutes - abs(filtered['predicted_time'] - available_minutes * 0.8)
        )

        # Return top recipes
        recommendations = filtered.nlargest(max_recipes, 'time_score')

        return recommendations[[
            'name', 'predicted_time', 'minutes', 'n_steps',
            'n_ingredients', 'tags', 'ingredients', 'steps', 'description'
        ]].reset_index(drop=True)

    def get_recipe_details(self, recipe_name: str) -> Optional[Dict]:
        """Get full details for a specific recipe.

        Args:
            recipe_name: Name of the recipe.

        Returns:
            Dictionary with recipe details or None if not found.
        """
        recipe = self.recipes_df[
            self.recipes_df['name'].str.lower() == recipe_name.lower()
        ]

        if len(recipe) == 0:
            return None

        recipe = recipe.iloc[0]

        return {
            'name': recipe['name'],
            'predicted_time': recipe['predicted_time'],
            'actual_time': recipe['minutes'],
            'description': recipe.get('description', 'No description available'),
            'n_steps': recipe['n_steps'],
            'n_ingredients': recipe['n_ingredients'],
            'ingredients': recipe['ingredients'],
            'steps': recipe['steps'],
            'tags': recipe['tags'],
            'nutrition': {
                'calories': recipe.get('calories', 0),
                'protein': recipe.get('protein_pdv', 0),
                'carbs': recipe.get('carbs_pdv', 0),
                'fat': recipe.get('total_fat_pdv', 0)
            }
        }

    def get_statistics(self) -> Dict:
        """Get statistics about available recipes.

        Returns:
            Dictionary with recipe statistics.
        """
        return {
            'total_recipes': len(self.recipes_df),
            'avg_time': self.recipes_df['predicted_time'].mean(),
            'median_time': self.recipes_df['predicted_time'].median(),
            'time_ranges': {
                'quick (< 30 min)': len(self.recipes_df[self.recipes_df['predicted_time'] < 30]),
                'moderate (30-60 min)': len(self.recipes_df[
                    (self.recipes_df['predicted_time'] >= 30) &
                    (self.recipes_df['predicted_time'] < 60)
                ]),
                'long (60+ min)': len(self.recipes_df[self.recipes_df['predicted_time'] >= 60])
            },
            'avg_steps': self.recipes_df['n_steps'].mean(),
            'avg_ingredients': self.recipes_df['n_ingredients'].mean()
        }
