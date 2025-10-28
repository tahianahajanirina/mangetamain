# GitHub Configuration

This directory contains GitHub-specific configuration files for CI/CD automation.

## 📁 Structure

```
.github/
├── workflows/           # GitHub Actions workflows
│   ├── ci.yml          # Continuous Integration
│   ├── ml-pipeline.yml # ML Training Pipeline
│   └── docker.yml      # Docker Build & Push
└── CI_CD_GUIDE.md      # Complete CI/CD documentation
```

## 🔄 Workflows

### 1. CI - Tests and Linting (`ci.yml`)
**Trigger**: Push/PR on `main` and `develop`

Runs:
- Unit tests on Python 3.9, 3.10, 3.11
- Code linting (ruff + black)
- Code coverage reporting

### 2. ML Pipeline (`ml-pipeline.yml`)
**Trigger**: 
- Manual (workflow_dispatch)
- Weekly schedule (Sunday 2 AM)
- Push to `main` (when src/ or scripts/ modified)

Executes complete ML training pipeline:
1. Download Kaggle dataset
2. Clean data
3. Recipe pipeline
4. Nutrition classification
5. Nutrition tagging
6. Time prediction
7. Sentiment analysis

Artifacts: Trained models saved for 30 days

### 3. Docker Build (`docker.yml`)
**Trigger**: 
- Push to `main`
- Tags matching `v*`
- Pull requests (test only)

Builds and pushes Docker image to GitHub Container Registry.

## 🔐 Required Secrets

Configure these in: Repository Settings → Secrets → Actions

| Secret | Description | Where to get |
|--------|-------------|--------------|
| `KAGGLE_USERNAME` | Kaggle username | https://www.kaggle.com/account |
| `KAGGLE_KEY` | Kaggle API key | https://www.kaggle.com/account → API Token |

## 📚 Documentation

See [CI_CD_GUIDE.md](CI_CD_GUIDE.md) for complete documentation.

## 🚀 Quick Start

1. Configure secrets (see above)
2. Push to trigger CI
3. Manually run ML Pipeline from Actions tab

## 🔗 Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)
