"""Visualization module for exploratory data analysis.

This module provides visualization functions optimized for:
1. Recipe Preparation Time Prediction
2. Nutritional Value Estimation/Tagging
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class RecipeEDA:
    """Exploratory Data Analysis for recipe data.

    Provides visualization and analysis methods focused on
    time prediction and nutrition estimation tasks.

    Attributes:
        df: Input DataFrame.
        figure_dir: Directory to save figures.
        style: Matplotlib style.
        palette: Seaborn color palette.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        figure_dir: Optional[Path] = None,
        style: str = "seaborn-v0_8-darkgrid",
        palette: str = "husl"
    ):
        """Initialize the EDA class.

        Args:
            df: Input DataFrame.
            figure_dir: Directory to save figures.
            style: Matplotlib style.
            palette: Seaborn color palette.
        """
        self.df = df
        self.figure_dir = figure_dir
        plt.style.use(style)
        sns.set_palette(palette)

        if figure_dir:
            figure_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, filename: str, dpi: int = 300):
        """Save figure to file.

        Args:
            filename: Output filename.
            dpi: Figure DPI.
        """
        if self.figure_dir:
            filepath = self.figure_dir / filename
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")

    def plot_time_distribution(self, figsize: Tuple[int, int] = (14, 5)):
        """Plot distribution of recipe preparation times.

        Args:
            figsize: Figure size.
        """
        logger.info("Plotting time distribution")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Raw distribution
        axes[0].hist(self.df['minutes'], bins=100, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Minutes')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Recipe Time Distribution (All)')
        axes[0].axvline(
            self.df['minutes'].median(),
            color='red',
            linestyle='--',
            label=f'Median: {self.df["minutes"].median():.0f} min'
        )
        axes[0].legend()

        # Filtered distribution (< 500 minutes for detail)
        filtered_data = self.df[self.df['minutes'] <= 500]['minutes']
        axes[1].hist(filtered_data, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Minutes')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Recipe Time Distribution (≤ 500 min)')
        axes[1].axvline(
            filtered_data.median(),
            color='red',
            linestyle='--',
            label=f'Median: {filtered_data.median():.0f} min'
        )
        axes[1].legend()

        plt.tight_layout()
        self._save_figure('time_distribution.png')
        plt.show()

    def plot_time_vs_features(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot time vs key features for time prediction.

        Args:
            figsize: Figure size.
        """
        logger.info("Plotting time vs features")

        # Filter for visualization
        df_viz = self.df[self.df['minutes'] <= 500].copy()

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Minutes vs n_steps
        axes[0, 0].scatter(
            df_viz['n_steps'], df_viz['minutes'], alpha=0.3, s=10
        )
        axes[0, 0].set_xlabel('Number of Steps')
        axes[0, 0].set_ylabel('Minutes')
        axes[0, 0].set_title('Time vs Number of Steps')

        # Minutes vs n_ingredients
        axes[0, 1].scatter(
            df_viz['n_ingredients'], df_viz['minutes'], alpha=0.3, s=10
        )
        axes[0, 1].set_xlabel('Number of Ingredients')
        axes[0, 1].set_ylabel('Minutes')
        axes[0, 1].set_title('Time vs Number of Ingredients')

        # Minutes vs cooking_verb_count
        if 'cooking_verb_count' in df_viz.columns:
            axes[0, 2].scatter(
                df_viz['cooking_verb_count'], df_viz['minutes'], alpha=0.3, s=10
            )
            axes[0, 2].set_xlabel('Cooking Verb Count')
            axes[0, 2].set_ylabel('Minutes')
            axes[0, 2].set_title('Time vs Cooking Verbs')

        # Minutes vs equipment_count
        if 'equipment_count' in df_viz.columns:
            axes[1, 0].scatter(
                df_viz['equipment_count'], df_viz['minutes'], alpha=0.3, s=10
            )
            axes[1, 0].set_xlabel('Equipment Count')
            axes[1, 0].set_ylabel('Minutes')
            axes[1, 0].set_title('Time vs Equipment Count')

        # Box plot by n_steps bins
        df_viz['steps_bin'] = pd.cut(
            df_viz['n_steps'],
            bins=[0, 5, 10, 15, 100],
            labels=['1-5', '6-10', '11-15', '15+']
        )
        df_viz.boxplot(column='minutes', by='steps_bin', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Number of Steps')
        axes[1, 1].set_ylabel('Minutes')
        axes[1, 1].set_title('Time Distribution by Step Count')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=0)

        # Box plot by n_ingredients bins
        df_viz['ing_bin'] = pd.cut(
            df_viz['n_ingredients'],
            bins=[0, 5, 10, 15, 50],
            labels=['1-5', '6-10', '11-15', '15+']
        )
        df_viz.boxplot(column='minutes', by='ing_bin', ax=axes[1, 2])
        axes[1, 2].set_xlabel('Number of Ingredients')
        axes[1, 2].set_ylabel('Minutes')
        axes[1, 2].set_title('Time Distribution by Ingredient Count')
        plt.sca(axes[1, 2])
        plt.xticks(rotation=0)

        plt.tight_layout()
        self._save_figure('time_vs_features.png')
        plt.show()

    def plot_nutrition_correlation(
        self,
        figsize: Tuple[int, int] = (10, 8),
        nutrition_cols: Optional[List[str]] = None
    ):
        """Plot correlation matrix of nutritional features.

        Args:
            figsize: Figure size.
            nutrition_cols: List of nutrition columns to include.
        """
        logger.info("Plotting nutrition correlation matrix")

        if nutrition_cols is None:
            nutrition_cols = [
                'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv',
                'protein_pdv', 'saturated_fat_pdv', 'carbs_pdv'
            ]

        # Filter available columns
        nutrition_cols = [col for col in nutrition_cols if col in self.df.columns]

        corr_matrix = self.df[nutrition_cols].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1
        )
        plt.title('Nutritional Features Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        self._save_figure('nutrition_correlation.png')
        plt.show()

    def plot_nutrition_distributions(
        self,
        figsize: Tuple[int, int] = (15, 10),
        nutrition_cols: Optional[List[str]] = None
    ):
        """Plot distributions of nutritional values.

        Args:
            figsize: Figure size.
            nutrition_cols: List of nutrition columns to plot.
        """
        logger.info("Plotting nutrition distributions")

        if nutrition_cols is None:
            nutrition_cols = [
                'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv',
                'protein_pdv', 'saturated_fat_pdv', 'carbs_pdv'
            ]

        nutrition_cols = [col for col in nutrition_cols if col in self.df.columns]

        n_cols = 3
        n_rows = (len(nutrition_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(nutrition_cols):
            axes[idx].hist(
                self.df[col].dropna(),
                bins=50,
                edgecolor='black',
                alpha=0.7
            )
            axes[idx].set_xlabel(col.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{col.replace("_", " ").title()} Distribution')
            axes[idx].axvline(
                self.df[col].median(),
                color='red',
                linestyle='--',
                label=f'Median: {self.df[col].median():.1f}'
            )
            axes[idx].legend()

        # Hide extra subplots
        for idx in range(len(nutrition_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self._save_figure('nutrition_distributions.png')
        plt.show()

    def plot_nutrition_pca(self, figsize: Tuple[int, int] = (14, 5)):
        """Plot PCA analysis of nutrition features.

        Args:
            figsize: Figure size.
        """
        logger.info("Plotting nutrition PCA")

        if 'nutrition_pc1' not in self.df.columns:
            logger.warning("PCA features not found. Skipping PCA plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Scatter plot of PC1 vs PC2
        scatter = axes[0].scatter(
            self.df['nutrition_pc1'],
            self.df['nutrition_pc2'],
            c=self.df['calories'],
            alpha=0.3,
            s=10,
            cmap='viridis'
        )
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('Nutrition PCA: PC1 vs PC2 (colored by calories)')
        plt.colorbar(scatter, ax=axes[0], label='Calories')

        # PC1 vs PC3
        scatter2 = axes[1].scatter(
            self.df['nutrition_pc1'],
            self.df['nutrition_pc3'],
            c=self.df['protein_pdv'],
            alpha=0.3,
            s=10,
            cmap='plasma'
        )
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC3')
        axes[1].set_title('Nutrition PCA: PC1 vs PC3 (colored by protein)')
        plt.colorbar(scatter2, ax=axes[1], label='Protein PDV')

        plt.tight_layout()
        self._save_figure('nutrition_pca.png')
        plt.show()

    def plot_feature_importance_preview(
        self,
        target: str = 'minutes',
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot feature correlation with target.

        Args:
            target: Target column name.
            top_n: Number of top features to show.
            figsize: Figure size.
        """
        logger.info(f"Plotting feature importance preview for {target}")

        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target from features
        if target in numeric_cols:
            numeric_cols.remove(target)

        # Calculate correlations
        correlations = self.df[numeric_cols].corrwith(self.df[target]).abs()
        top_features = correlations.nlargest(top_n)

        # Plot
        plt.figure(figsize=figsize)
        top_features.sort_values().plot(kind='barh')
        plt.xlabel('Absolute Correlation')
        plt.title(f'Top {top_n} Features Correlated with {target}')
        plt.tight_layout()
        self._save_figure(f'feature_correlation_{target}.png')
        plt.show()

    def generate_time_prediction_report(self):
        """Generate complete EDA report for time prediction task."""
        logger.info("Generating time prediction EDA report")

        self.plot_time_distribution()
        self.plot_time_vs_features()
        self.plot_feature_importance_preview(target='minutes')

        logger.info("Time prediction EDA report completed")

    def generate_nutrition_report(self):
        """Generate complete EDA report for nutrition estimation task."""
        logger.info("Generating nutrition estimation EDA report")

        self.plot_nutrition_distributions()
        self.plot_nutrition_correlation()
        self.plot_nutrition_pca()
        self.plot_feature_importance_preview(target='calories')

        logger.info("Nutrition estimation EDA report completed")
