#!/usr/bin/env python3
"""
Download ML models from Hugging Face Hub or cloud storage.

This script downloads all trained models at startup if they're not present locally.
Used for deployment without storing large model files in the Git repository.

Usage:
    python scripts/download_models.py
    
Environment variables:
    MODEL_SOURCE: 'huggingface' (default) or 'gcs' or 'aws'
    HF_MODEL_REPO: Hugging Face repo (e.g., 'tahianahajanirina/mangetamain-models')
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SENTIMENT_DIR = OUTPUTS_DIR / "sentiment_model"


def ensure_directories():
    """Create output directories if they don't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Directories ready: {OUTPUTS_DIR}")


def check_models_exist():
    """Check if models are already downloaded."""
    # Check for key model files
    key_files = [
        MODELS_DIR / "nutrition_classifier_20251027_232241_model.pkl",
        SENTIMENT_DIR / "config.json",
    ]
    
    existing = [f.exists() for f in key_files]
    if all(existing):
        logger.info("✓ All models already exist locally")
        return True
    
    logger.info(f"✗ Models missing: {sum(not e for e in existing)}/{len(key_files)}")
    return False


def download_from_huggingface():
    """Download models from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        
        repo_id = os.getenv("HF_MODEL_REPO", "tahianahajanirina/mangetamain-models")
        logger.info(f"Downloading models from Hugging Face: {repo_id}")
        
        # Download all files to outputs directory
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(OUTPUTS_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        logger.info("✓ Models downloaded successfully from Hugging Face")
        return True
        
    except ImportError:
        logger.error("✗ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"✗ Error downloading from Hugging Face: {e}")
        return False


def download_from_gcs():
    """Download models from Google Cloud Storage."""
    try:
        from google.cloud import storage
        
        bucket_name = os.getenv("GCS_BUCKET", "mangetamain-models")
        logger.info(f"Downloading models from GCS: {bucket_name}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Download all blobs with prefix 'models/' or 'sentiment_model/'
        blobs = list(bucket.list_blobs(prefix="outputs/"))
        
        for blob in blobs:
            local_path = PROJECT_ROOT / blob.name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            logger.info(f"  Downloaded: {blob.name}")
        
        logger.info("✓ Models downloaded successfully from GCS")
        return True
        
    except ImportError:
        logger.error("✗ google-cloud-storage not installed. Run: pip install google-cloud-storage")
        return False
    except Exception as e:
        logger.error(f"✗ Error downloading from GCS: {e}")
        return False


def download_from_aws():
    """Download models from AWS S3."""
    try:
        import boto3
        
        bucket_name = os.getenv("AWS_BUCKET", "mangetamain-models")
        logger.info(f"Downloading models from S3: {bucket_name}")
        
        s3 = boto3.client('s3')
        
        # List and download all objects with prefix 'outputs/'
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='outputs/')
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                local_path = PROJECT_ROOT / key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket_name, key, str(local_path))
                logger.info(f"  Downloaded: {key}")
        
        logger.info("✓ Models downloaded successfully from S3")
        return True
        
    except ImportError:
        logger.error("✗ boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        logger.error(f"✗ Error downloading from S3: {e}")
        return False


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("MODEL DOWNLOADER")
    logger.info("=" * 80)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if models already exist
    if check_models_exist():
        logger.info("Skipping download - models already present")
        return 0
    
    # Get model source from environment
    model_source = os.getenv("MODEL_SOURCE", "huggingface").lower()
    
    # Download from specified source
    success = False
    if model_source == "huggingface":
        success = download_from_huggingface()
    elif model_source == "gcs":
        success = download_from_gcs()
    elif model_source == "aws":
        success = download_from_aws()
    else:
        logger.error(f"✗ Unknown MODEL_SOURCE: {model_source}")
        logger.error("  Valid options: huggingface, gcs, aws")
        return 1
    
    if success:
        logger.info("=" * 80)
        logger.info("✓ ALL MODELS READY")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("✗ DOWNLOAD FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
