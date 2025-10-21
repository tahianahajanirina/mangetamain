"""Main pipeline for recipe time prediction task.

This script runs the complete pipeline for predicting recipe
preparation time:
1. Load and preprocess data
2. Engineer features
3. Perform EDA
4. Train and evaluate models
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
from src.feature_engineering.time_features import TimeFeatureEngineer
from src.eda.visualization import RecipeEDA
from src.modeling.time_predictor import train_and_evaluate_time_model

# Setup logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    """Run the complete time prediction pipeline."""
    logger.info("="*80)
    logger.info("RECIPE TIME PREDICTION PIPELINE")
    logger.info("="*80)

    # Step 1: Load and preprocess data
    logger.info("\n[STEP 1] Loading and preprocessing data")
    logger.info("-"*80)

    data_loader = RecipeDataLoader(RAW_DATA_FILE)
    df = data_loader.preprocess()

    logger.info(f"Loaded {len(df)} recipes")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Step 2: Engineer features for time prediction
    logger.info("\n[STEP 2] Engineering features for time prediction")
    logger.info("-"*80)

    time_engineer = TimeFeatureEngineer(
        cooking_verbs=FEATURE_CONFIG['cooking_verbs'],
        equipment_terms=FEATURE_CONFIG['equipment_terms']
    )

    df = time_engineer.engineer_features(df)
    feature_names = time_engineer.get_feature_names()

    logger.info(f"Engineered {len(feature_names)} features")
    logger.info(f"Features: {feature_names}")

    # Step 3: Exploratory Data Analysis
    logger.info("\n[STEP 3] Performing EDA")
    logger.info("-"*80)

    eda = RecipeEDA(df, figure_dir=FIGURE_DIR)
    eda.generate_time_prediction_report()

    # Step 4: Train and evaluate models
    logger.info("\n[STEP 4] Training and evaluating models")
    logger.info("-"*80)

    target_col = 'minutes'
    task_config = MODEL_CONFIG['time_prediction']

    results = {}

    for model_name, model_params in task_config['models'].items():
        logger.info(f"\nTraining {model_name} model...")

        # Extract model type
        model_type = model_name
        # Exclude 'name' and 'random_state' as they are handled separately
        model_kwargs = {k: v for k, v in model_params.items() if k not in ('name', 'random_state')}

        # Train model
        model, metrics = train_and_evaluate_time_model(
            df=df,
            feature_cols=feature_names,
            target_col=target_col,
            model_type=model_type,
            test_size=task_config['test_size'],
            random_state=task_config['random_state'],
            **model_kwargs
        )

        results[model_name] = metrics

        # Save model
        model_path = MODEL_DIR / f"time_{model_name}.pkl"
        model.save_model(model_path)

        # Display feature importance if available
        try:
            importance = model.get_feature_importance(top_n=10)
            logger.info(f"\nTop 10 features for {model_name}:")
            logger.info(f"\n{importance.to_string()}")
        except ValueError:
            pass

    # Step 5: Summary
    logger.info("\n[STEP 5] Summary")
    logger.info("="*80)

    results_df = pd.DataFrame(results).T
    logger.info("\nModel Comparison:")
    logger.info(f"\n{results_df.to_string()}")

    # Find best model
    best_model = results_df['r2'].idxmax()
    logger.info(f"\nBest model: {best_model}")
    logger.info(f"R²: {results_df.loc[best_model, 'r2']:.4f}")
    logger.info(f"MAE: {results_df.loc[best_model, 'mae']:.2f} minutes")
    logger.info(f"RMSE: {results_df.loc[best_model, 'rmse']:.2f} minutes")

    # Save processed data
    logger.info(f"\nSaving processed data to {PROCESSED_DATA_FILE}")
    df.to_csv(PROCESSED_DATA_FILE, index=False)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
