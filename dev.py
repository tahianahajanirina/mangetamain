#!/usr/bin/env python3
"""
Alternative to Makefile for environments without make.
Provides common development commands.

Usage:
    python dev.py <command>

Commands:
    install         Install project dependencies
    install-dev     Install dev dependencies
    test            Run tests
    test-cov        Run tests with coverage
    lint            Run linting
    format          Format code
    clean           Clean generated files
    download-data   Download Kaggle dataset
    pipeline        Run full ML pipeline
    docker-build    Build Docker image
    docker-up       Start Docker containers
    docker-down     Stop Docker containers
    help            Show this help message
"""

import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description=None):
    """Run a shell command."""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)
    
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    return result


def install():
    """Install project dependencies."""
    run_command("pip install -e .", "Installing project dependencies")


def install_dev():
    """Install dev dependencies."""
    run_command('pip install -e ".[dev,test,sentiment]"', "Installing dev dependencies")


def test():
    """Run tests."""
    run_command("pytest tests/ -v", "Running tests")


def test_cov():
    """Run tests with coverage."""
    run_command(
        "pytest tests/ -v --cov=src --cov-report=html --cov-report=term",
        "Running tests with coverage"
    )


def lint():
    """Run linting."""
    print("\n" + "="*60)
    print("Running linting checks")
    print("="*60)
    
    print("\n[1/2] Checking with ruff...")
    subprocess.run("ruff check src/ scripts/ tests/", shell=True)
    
    print("\n[2/2] Checking with black...")
    subprocess.run("black --check src/ scripts/ tests/", shell=True)


def format_code():
    """Format code."""
    print("\n" + "="*60)
    print("Formatting code")
    print("="*60)
    
    print("\n[1/2] Running black...")
    run_command("black src/ scripts/ tests/")
    
    print("\n[2/2] Running ruff --fix...")
    subprocess.run("ruff check --fix src/ scripts/ tests/", shell=True)


def clean():
    """Clean generated files."""
    print("\n" + "="*60)
    print("Cleaning generated files")
    print("="*60)
    
    patterns = [
        "build/",
        "dist/",
        "*.egg-info",
        ".pytest_cache",
        ".coverage",
        "htmlcov/",
        "__pycache__",
        "*.pyc",
    ]
    
    for pattern in patterns:
        if pattern.endswith('/'):
            # Directory
            for path in Path('.').rglob(pattern.rstrip('/')):
                if path.is_dir():
                    print(f"  Removing {path}")
                    shutil.rmtree(path, ignore_errors=True)
        else:
            # File or pattern
            for path in Path('.').rglob(pattern):
                if path.is_file():
                    print(f"  Removing {path}")
                    path.unlink()
    
    print("\n✓ Cleanup complete")


def download_data():
    """Download Kaggle dataset."""
    run_command("python scripts/download_kaggle_data.py", "Downloading Kaggle dataset")


def pipeline():
    """Run full ML pipeline."""
    print("\n" + "="*60)
    print("Running full ML pipeline")
    print("="*60)
    
    steps = [
        ("Downloading data", "python scripts/download_kaggle_data.py"),
        ("Cleaning data", "python scripts/clean_data.py"),
        ("Recipe pipeline", "python scripts/run_recipe_pipeline.py"),
        ("Nutrition pipeline", "python scripts/run_nutrition_pipeline.py"),
        ("Nutrition tagging", "python main_nutrition_tagging.py"),
        ("Time prediction", "python main_time_prediction.py"),
        ("Sentiment analysis", "python train_sentiment_model.py"),
    ]
    
    for i, (desc, cmd) in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {desc}")
        print("-" * 60)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"\n❌ Step failed: {desc}")
            sys.exit(result.returncode)
    
    print("\n" + "="*60)
    print("✓ Pipeline complete!")
    print("="*60)


def docker_build():
    """Build Docker image."""
    run_command("docker build -t recipe-ml:latest .", "Building Docker image")


def docker_up():
    """Start Docker containers."""
    run_command("docker-compose up -d", "Starting Docker containers")


def docker_down():
    """Stop Docker containers."""
    run_command("docker-compose down", "Stopping Docker containers")


def show_help():
    """Show help message."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)
    
    command = sys.argv[1]
    
    commands = {
        'install': install,
        'install-dev': install_dev,
        'test': test,
        'test-cov': test_cov,
        'lint': lint,
        'format': format_code,
        'clean': clean,
        'download-data': download_data,
        'pipeline': pipeline,
        'docker-build': docker_build,
        'docker-up': docker_up,
        'docker-down': docker_down,
        'help': show_help,
    }
    
    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print(f"\nAvailable commands: {', '.join(commands.keys())}")
        sys.exit(1)
    
    commands[command]()


if __name__ == '__main__':
    main()
