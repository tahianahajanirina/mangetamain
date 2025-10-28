#!/usr/bin/env python3
"""
Verify CI/CD setup and configuration.

This script checks if all requirements for CI/CD are properly configured.
"""

import sys
import os
from pathlib import Path
import subprocess


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✓" if condition else "✗"


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 9)
    is_ok = version >= required
    
    print(f"{check_mark(is_ok)} Python version: {version.major}.{version.minor}.{version.micro}")
    if not is_ok:
        print(f"   Required: >= {required[0]}.{required[1]}")
    return is_ok


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pytest',
        'pytest-cov',
        'black',
        'ruff',
        'pandas',
        'scikit-learn',
        'numpy'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Check if project structure is correct."""
    required_paths = [
        'src/',
        'tests/',
        'scripts/',
        'data/',
        'outputs/',
        '.github/workflows/',
        'pyproject.toml',
        'Makefile'
    ]
    
    all_ok = True
    project_root = Path(__file__).parent.parent
    
    for path in required_paths:
        full_path = project_root / path
        exists = full_path.exists()
        print(f"{check_mark(exists)} {path}")
        if not exists:
            all_ok = False
    
    return all_ok


def check_workflows():
    """Check if GitHub Actions workflows are present."""
    project_root = Path(__file__).parent.parent
    workflows_dir = project_root / '.github' / 'workflows'
    
    required_workflows = [
        'ci.yml',
        'ml-pipeline.yml',
        'docker.yml'
    ]
    
    all_ok = True
    for workflow in required_workflows:
        workflow_path = workflows_dir / workflow
        exists = workflow_path.exists()
        print(f"{check_mark(exists)} {workflow}")
        if not exists:
            all_ok = False
    
    return all_ok


def check_test_structure():
    """Check test structure."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / 'tests'
    
    required_files = [
        'conftest.py',
        'test_data_loader.py',
        'test_feature_engineering/test_recipe_features.py',
        'test_modeling/test_recipe_clustering.py',
    ]
    
    all_ok = True
    for file in required_files:
        file_path = tests_dir / file
        exists = file_path.exists()
        print(f"{check_mark(exists)} tests/{file}")
        if not exists:
            all_ok = False
    
    return all_ok


def check_kaggle_setup():
    """Check if Kaggle API is configured."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    exists = kaggle_json.exists()
    
    print(f"{check_mark(exists)} Kaggle credentials (~/.kaggle/kaggle.json)")
    
    if exists:
        # Check permissions
        mode = oct(os.stat(kaggle_json).st_mode)[-3:]
        correct_perms = mode == '600'
        print(f"{check_mark(correct_perms)} Correct permissions (600)")
        return correct_perms
    else:
        print("   To setup: https://www.kaggle.com/account")
        return False


def check_git_repo():
    """Check if in a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Git repository initialized")
        
        # Check remote
        result = subprocess.run(
            ['git', 'remote', '-v'],
            capture_output=True,
            text=True
        )
        has_remote = bool(result.stdout.strip())
        print(f"{check_mark(has_remote)} Git remote configured")
        
        return True
    except subprocess.CalledProcessError:
        print("✗ Not a git repository")
        return False


def run_quick_test():
    """Run a quick test to verify setup."""
    print("\nRunning quick test...")
    try:
        result = subprocess.run(
            ['pytest', 'tests/', '--collect-only', '-q'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 or 'collected' in result.stdout:
            # Count tests
            import re
            match = re.search(r'(\d+) test', result.stdout)
            if match:
                count = match.group(1)
                print(f"✓ Found {count} tests")
                return True
            print("✓ Tests found")
            return True
        else:
            print("✗ Test collection failed")
            return False
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False


def main():
    """Main verification function."""
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "CI/CD SETUP VERIFICATION" + " "*19 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {}
    
    # Check Python
    print_section("1. Python Environment")
    results['python'] = check_python_version()
    
    # Check dependencies
    print_section("2. Python Dependencies")
    results['dependencies'] = check_dependencies()
    
    # Check project structure
    print_section("3. Project Structure")
    results['structure'] = check_project_structure()
    
    # Check workflows
    print_section("4. GitHub Actions Workflows")
    results['workflows'] = check_workflows()
    
    # Check test structure
    print_section("5. Test Structure")
    results['tests'] = check_test_structure()
    
    # Check Kaggle
    print_section("6. Kaggle API (Optional)")
    results['kaggle'] = check_kaggle_setup()
    
    # Check git
    print_section("7. Git Repository")
    results['git'] = check_git_repo()
    
    # Quick test
    print_section("8. Quick Test")
    results['quick_test'] = run_quick_test()
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print()
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check}")
    
    print("\n" + "="*60)
    
    if all(results.values()):
        print("\n🎉 All checks passed! CI/CD is ready to use.")
        print("\nNext steps:")
        print("  1. Commit and push to trigger CI")
        print("  2. Configure GitHub secrets (KAGGLE_USERNAME, KAGGLE_KEY)")
        print("  3. Run ML pipeline manually or wait for schedule")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        if not results['dependencies']:
            print("  - Run: pip install -e '.[dev,test]'")
        if not results['kaggle']:
            print("  - Setup Kaggle: bash scripts/setup_cicd.sh")
        if not results['git']:
            print("  - Initialize git: git init && git remote add origin <URL>")
        return 1


if __name__ == '__main__':
    sys.exit(main())
