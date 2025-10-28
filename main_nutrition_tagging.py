"""Main pipeline for nutritional value tagging task.

This script runs the complete pipeline for estimating and tagging
nutritional values:
1. Load and preprocess data
2. Engineer features
3. Perform EDA
4. Train and evaluate models for multiple nutrition tags
"""

import logging
import logging.config
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    FEATURE_CONFIG,
    EDA_CONFIG,
    MODEL_CONFIG,
    FIGURE_DIR,
    MODEL_DIR,
    LOGGING_CONFIG
)
from src.preprocessing.data_loader import RecipeDataLoader
from src.feature_engineering.nutrition_features import NutritionFeatureEngineer
from src.eda.visualization import RecipeEDA
from src.modeling.nutrition_tagger import train_and_evaluate_nutrition_model

# Setup logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    """Run the complete nutrition tagging pipeline."""
    logger.info("="*80)
    logger.info("RECIPE NUTRITION TAGGING PIPELINE")
    logger.info("="*80)

    # Step 1: Load and preprocess data
    logger.info("\n[STEP 1] Loading and preprocessing data")
    logger.info("-"*80)

    data_loader = RecipeDataLoader(RAW_DATA_FILE)
    df = data_loader.preprocess()

    logger.info(f"Loaded {len(df)} recipes")

    # Step 2: Engineer features for nutrition tagging
    logger.info("\n[STEP 2] Engineering features for nutrition tagging")
    logger.info("-"*80)

    nutrition_engineer = NutritionFeatureEngineer(
        ingredient_categories=FEATURE_CONFIG['ingredient_categories'],
        dietary_patterns=FEATURE_CONFIG['dietary_patterns'],
        nutrition_cols=FEATURE_CONFIG['nutrition_cols']
    )

    # Add n_ingredients if not present
    if 'n_ingredients' not in df.columns:
        df['n_ingredients'] = df['ingredients'].apply(len)

    df = nutrition_engineer.engineer_features(df, include_pca=True)
    feature_names = nutrition_engineer.get_feature_names(include_pca=True)

    logger.info(f"Engineered {len(feature_names)} features")

    # Step 3: Exploratory Data Analysis
    logger.info("\n[STEP 3] Performing EDA")
    logger.info("-"*80)

    eda = RecipeEDA(df, figure_dir=FIGURE_DIR)
    eda.generate_nutrition_report()

    # Step 4: Train and evaluate models for multiple nutrition tags
    logger.info("\n[STEP 4] Training and evaluating models")
    logger.info("-"*80)

    task_config = MODEL_CONFIG['nutrition_tagging']

    # Define target tags to predict
    target_tags = [
        'high_calorie',
        'low_calorie',
        'high_protein',
        'low_fat',
        'low_sugar',
        'healthy_recipe'
    ]

    # Filter features for nutrition prediction
    # Exclude the target tags themselves from features
    nutrition_feature_cols = [
        col for col in feature_names
        if col not in target_tags and col in df.columns
    ]

    logger.info(f"Using {len(nutrition_feature_cols)} features for prediction")

    all_results = {}

    for target_tag in target_tags:
        if target_tag not in df.columns:
            logger.warning(f"Target tag '{target_tag}' not found. Skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for: {target_tag}")
        logger.info(f"{'='*60}")

        # Check class distribution
        class_dist = df[target_tag].value_counts()
        logger.info(f"Class distribution: {class_dist.to_dict()}")

        # Skip if class imbalance is too extreme
        if len(class_dist) < 2 or min(class_dist) < 100:
            logger.warning(
                f"Skipping {target_tag} due to insufficient samples "
                f"in minority class"
            )
            continue

        tag_results = {}

        for model_name, model_params in task_config['models'].items():
            logger.info(f"\nTraining {model_name} model for {target_tag}...")

            # Extract model type
            model_type = model_name
            model_kwargs = {k: v for k, v in model_params.items()
                          if k not in ['name', 'random_state']}

            try:
                # Train model
                model, metrics = train_and_evaluate_nutrition_model(
                    df=df,
                    feature_cols=nutrition_feature_cols,
                    target_col=target_tag,
                    model_type=model_type,
                    test_size=task_config['test_size'],
                    random_state=task_config['random_state'],
                    **model_kwargs
                )

                tag_results[model_name] = metrics

                # Save model
                model_path = MODEL_DIR / f"nutrition_{target_tag}_{model_name}.pkl"
                model.save_model(model_path)

                # Display feature importance if available
                try:
                    importance = model.get_feature_importance(top_n=10)
                    logger.info(f"\nTop 10 features for {model_name} ({target_tag}):")
                    logger.info(f"\n{importance.to_string()}")
                except ValueError:
                    pass

            except Exception as e:
                logger.error(f"Error training {model_name} for {target_tag}: {e}")
                continue

        all_results[target_tag] = tag_results

    # Step 5: Summary
    logger.info("\n[STEP 5] Summary")
    logger.info("="*80)

    for target_tag, tag_results in all_results.items():
        if tag_results:
            logger.info(f"\n{target_tag.upper()} - Model Comparison:")
            results_df = pd.DataFrame(tag_results).T
            logger.info(f"\n{results_df.to_string()}")

            best_model = results_df['f1'].idxmax()
            logger.info(f"\nBest model: {best_model}")
            logger.info(f"F1 Score: {results_df.loc[best_model, 'f1']:.4f}")
            logger.info(f"Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")

    # Save processed data
    logger.info(f"\n\nSaving processed data to {PROCESSED_DATA_FILE}")
    df.to_csv(PROCESSED_DATA_FILE, index=False)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
