#!/usr/bin/env python3
"""
Upload trained models to Hugging Face Hub.

This script uploads all trained models from the outputs/ directory
to a Hugging Face repository for easy distribution.

Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login: huggingface-cli login
    3. Create repo on https://huggingface.co/

Usage:
    python scripts/upload_models_to_hf.py
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def check_huggingface_setup():
    """Check if Hugging Face CLI is set up."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()
        logger.info(f"✓ Logged in as: {user['name']}")
        return True

    except ImportError:
        logger.error("✗ huggingface_hub not installed")
        logger.error("  Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"✗ Not logged in to Hugging Face: {e}")
        logger.error("  Run: huggingface-cli login")
        return False


def upload_to_huggingface(repo_id: str):
    """Upload models to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi

        logger.info(f"Uploading models to: {repo_id}")

        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            logger.info(f"✓ Repository ready: {repo_id}")
        except Exception as e:
            logger.warning(f"Repo might already exist: {e}")

        # Upload entire outputs directory
        api.upload_folder(
            folder_path=str(OUTPUTS_DIR),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="outputs",
        )

        logger.info("✓ Models uploaded successfully!")
        logger.info(f"  View at: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        logger.error(f"✗ Error uploading to Hugging Face: {e}")
        return False


def create_model_card(repo_id: str):
    """Create a README.md for the model repository."""
    readme_content = f"""---
license: mit
tags:
- recipe-recommendation
- food
- machine-learning
- classification
- clustering
---

# MangeTaMain - ML Models

This repository contains trained machine learning models for the MangeTaMain recipe recommendation system.

## Models Included

1. **Sentiment Analysis Model** (`outputs/sentiment_model/`)
   - Fine-tuned BERT for recipe review sentiment analysis
   - Predicts positive/negative sentiment from reviews

2. **Nutrition Classifier** (`outputs/models/nutrition_classifier_*.pkl`)
   - Multi-label classification for nutrition tags
   - Tags: healthy, low-calorie, high-protein, low-fat, low-sugar, high-calorie

3. **Recipe Clustering** (`outputs/models/recipe_clustering_*.pkl`)
   - K-Means clustering of recipes by features
   - Groups similar recipes together

4. **User Clustering** (`outputs/models/user_clustering_*.pkl`)
   - User preference clustering
   - Identifies user groups with similar tastes

## Usage

```python
from huggingface_hub import snapshot_download

# Download all models
snapshot_download(
    repo_id="{repo_id}",
    local_dir="./outputs",
    local_dir_use_symlinks=False,
)
```

## Training Data

Models trained on the [Food.com Recipes and Interactions Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

## Repository

Source code: https://github.com/tahianahajanirina/mangetamain
"""

    readme_path = OUTPUTS_DIR / "README.md"
    readme_path.write_text(readme_content)
    logger.info(f"✓ Model card created: {readme_path}")
    return readme_path


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("HUGGING FACE MODEL UPLOADER")
    logger.info("=" * 80)

    # Check if outputs directory exists and has models
    if not OUTPUTS_DIR.exists():
        logger.error(f"✗ Outputs directory not found: {OUTPUTS_DIR}")
        return 1

    model_files = list(OUTPUTS_DIR.glob("**/*.pkl")) + list(
        OUTPUTS_DIR.glob("**/*.safetensors")
    )
    if not model_files:
        logger.error("✗ No model files found in outputs/")
        logger.error("  Train models first before uploading")
        return 1

    logger.info(f"✓ Found {len(model_files)} model files to upload")

    # Check Hugging Face setup
    if not check_huggingface_setup():
        return 1

    # Get repo ID
    repo_id = os.getenv("HF_MODEL_REPO")
    if not repo_id:
        repo_id = input(
            "Enter Hugging Face repo ID (e.g., username/mangetamain-models): "
        ).strip()

    if not repo_id or "/" not in repo_id:
        logger.error("✗ Invalid repo ID. Format: username/repo-name")
        return 1

    # Create model card
    create_model_card(repo_id)

    # Confirm upload
    print(f"\nAbout to upload to: {repo_id}")
    response = input("Continue? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        logger.info("Upload cancelled")
        return 0

    # Upload
    success = upload_to_huggingface(repo_id)

    if success:
        logger.info("=" * 80)
        logger.info("✓ UPLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info(f"  1. View models: https://huggingface.co/{repo_id}")
        logger.info(f"  2. Set environment variable: HF_MODEL_REPO={repo_id}")
        logger.info("  3. Download with: python scripts/download_models.py")
        return 0
    else:
        logger.error("=" * 80)
        logger.error("✗ UPLOAD FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
