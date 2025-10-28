"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_recipes_df():
    """Create a sample recipes dataframe for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': [
            'Chocolate Chip Cookies',
            'Caesar Salad',
            'Spaghetti Carbonara',
            'Green Smoothie',
            'Beef Stew'
        ],
        'minutes': [30, 15, 25, 5, 120],
        'tags': [
            "['desserts', 'cookies', 'baking']",
            "['salads', 'vegetables', 'quick']",
            "['pasta', 'italian', 'main-dish']",
            "['beverages', 'healthy', 'vegan']",
            "['meat', 'stews', 'comfort-food']"
        ],
        'nutrition': [
            '[320.5, 45.0, 15.0, 8.0, 25.0, 12.0, 5.0]',  # high calories, sugar
            '[150.0, 10.0, 12.0, 5.0, 8.0, 15.0, 3.0]',   # balanced
            '[450.0, 55.0, 18.0, 12.0, 30.0, 10.0, 8.0]', # high calories, fat
            '[120.0, 25.0, 2.0, 1.0, 3.0, 45.0, 2.0]',    # low calories, high fiber
            '[380.0, 35.0, 22.0, 15.0, 35.0, 8.0, 12.0]'  # high protein
        ],
        'n_steps': [8, 5, 7, 3, 12],
        'steps': [
            "['Mix dry ingredients', 'Add wet ingredients', 'Form cookies', 'Bake at 350F']",
            "['Chop lettuce', 'Make dressing', 'Toss together', 'Add croutons', 'Serve']",
            "['Boil pasta', 'Cook bacon', 'Mix eggs and cheese', 'Combine', 'Season']",
            "['Add spinach to blender', 'Add banana', 'Add liquid', 'Blend']",
            "['Brown meat', 'Add vegetables', 'Add broth', 'Simmer 2 hours']"
        ],
        'description': [
            'Classic homemade chocolate chip cookies',
            'Fresh caesar salad with homemade dressing',
            'Traditional Italian pasta carbonara',
            'Healthy green smoothie packed with nutrients',
            'Hearty beef stew perfect for cold days'
        ],
        'ingredients': [
            "['flour', 'sugar', 'chocolate chips', 'butter', 'eggs']",
            "['romaine lettuce', 'parmesan', 'croutons', 'caesar dressing']",
            "['spaghetti', 'eggs', 'bacon', 'parmesan', 'black pepper']",
            "['spinach', 'banana', 'almond milk', 'honey']",
            "['beef', 'carrots', 'potatoes', 'onions', 'beef broth']"
        ],
        'n_ingredients': [5, 4, 5, 4, 5]
    })


@pytest.fixture
def sample_interactions_df():
    """Create a sample interactions dataframe for testing."""
    return pd.DataFrame({
        'user_id': [101, 102, 103, 101, 104, 102, 105, 103, 104, 105],
        'recipe_id': [1, 1, 2, 3, 2, 4, 1, 5, 3, 4],
        'date': pd.date_range('2024-01-01', periods=10),
        'rating': [5, 4, 5, 3, 4, 5, 5, 2, 4, 5],
        'review': [
            'Delicious cookies!',
            'Great recipe',
            'Perfect salad',
            'Good but too salty',
            'Nice and fresh',
            'Love this smoothie',
            'Best cookies ever',
            'Too much work',
            'Easy and tasty',
            'Refreshing drink'
        ]
    })


@pytest.fixture
def sample_recipes_with_features():
    """Create recipes with engineered features."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Recipe 1', 'Recipe 2', 'Recipe 3'],
        'minutes': [30, 45, 20],
        'n_steps': [5, 8, 3],
        'n_ingredients': [6, 10, 4],
        'calories': [300.0, 450.0, 200.0],
        'total_fat': [15.0, 25.0, 8.0],
        'sugar': [10.0, 5.0, 20.0],
        'sodium': [500.0, 800.0, 300.0],
        'protein': [12.0, 20.0, 5.0],
        'saturated_fat': [5.0, 10.0, 3.0],
        'carbohydrates': [35.0, 40.0, 30.0],
        'avg_rating': [4.5, 4.0, 3.5],
        'num_ratings': [100, 50, 25],
        'is_healthy': [1, 0, 0],
        'is_dessert': [0, 0, 1],
        'is_quick': [1, 0, 1],
    })


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    
    return {
        'data': data_dir,
        'raw': raw_dir,
        'processed': processed_dir
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory structure."""
    output_dir = tmp_path / "outputs"
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    
    models_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    
    return {
        'output': output_dir,
        'models': models_dir,
        'figures': figures_dir
    }
