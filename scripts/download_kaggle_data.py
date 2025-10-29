#!/usr/bin/env python3
"""
Download Food.com dataset from Kaggle.

This script downloads the Food.com Recipes and User Interactions dataset
using the Kaggle API.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials:
       - Go to https://www.kaggle.com/account
       - Create API token (downloads kaggle.json)
       - Place kaggle.json in ~/.kaggle/
       - On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_kaggle_data.py
"""

import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    try:
        import kaggle  # noqa: F401

        print("✓ Kaggle package found")
        return True
    except ImportError:
        print("✗ Kaggle package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def download_dataset():
    """Download the dataset from Kaggle."""
    print("\n" + "=" * 80)
    print("DOWNLOADING FOOD.COM DATASET FROM KAGGLE")
    print("=" * 80)

    # Ensure data directory exists
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Dataset details
    dataset = "shuyangli94/food-com-recipes-and-user-interactions"

    print(f"\nDataset: {dataset}")
    print(f"Destination: {DATA_RAW}")

    try:
        import kaggle

        # Download dataset
        print("\nDownloading dataset...")
        kaggle.api.dataset_download_files(
            dataset, path=str(DATA_RAW), unzip=True, quiet=False
        )

        print("\n✓ Download complete!")

        # List downloaded files
        print("\nDownloaded files:")
        for file in sorted(DATA_RAW.glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")

        # Check for expected files
        expected_files = [
            "RAW_recipes.csv",
            "RAW_interactions.csv",
            "PP_recipes.csv",
            "PP_users.csv",
            "interactions_train.csv",
            "interactions_validation.csv",
            "interactions_test.csv",
        ]

        print("\nExpected files status:")
        all_found = True
        for filename in expected_files:
            filepath = DATA_RAW / filename
            if filepath.exists():
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} - NOT FOUND")
                all_found = False

        if all_found:
            print("\n✓ All expected files downloaded successfully!")
            return True
        else:
            print("\n⚠ Some files are missing. Please check the download.")
            return False

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have accepted the dataset terms on Kaggle")
        print("  2. Check your Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("  3. Verify your internet connection")
        return False


def main():
    """Main function."""
    print("Food.com Dataset Downloader")
    print("-" * 80)

    # Check Kaggle setup
    if not check_kaggle_setup():
        print(
            "\n✗ Kaggle setup failed. Please install kaggle and configure API credentials."
        )
        sys.exit(1)

    # Check for existing data
    existing_files = list(DATA_RAW.glob("*.csv"))
    if existing_files:
        print(f"\n⚠ Found {len(existing_files)} existing CSV files in {DATA_RAW}")
        response = input(
            "Do you want to download again? This will overwrite existing files. [y/N]: "
        )
        if response.lower() not in ["y", "yes"]:
            print("Download cancelled.")
            return

    # Download dataset
    success = download_dataset()

    if success:
        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run data cleaning: python scripts/clean_data.py")
        print("  2. Run pipelines: python scripts/run_recipe_pipeline.py")
    else:
        print("\n" + "=" * 80)
        print("DOWNLOAD FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
