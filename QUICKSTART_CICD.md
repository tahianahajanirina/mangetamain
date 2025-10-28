# 🚀 Quick Start - CI/CD Pipeline

## ⚡ Installation Rapide

```bash
# 1. Cloner le repo
git clone https://github.com/tahianahajanirina/mangetamain.git
cd mangetamain

# 2. Installer les dépendances
make install-dev

# 3. Setup CI/CD
bash scripts/setup_cicd.sh
```

## 🧪 Tests

```bash
# Lancer tous les tests
make test

# Tests avec couverture
make test-cov

# Tests spécifiques
pytest tests/test_data_loader.py -v
```

## 🤖 Pipeline ML Complet

```bash
# Option 1: Avec Makefile (recommandé)
make pipeline

# Option 2: Commandes séparées
python scripts/download_kaggle_data.py
python scripts/clean_data.py
python scripts/run_recipe_pipeline.py
python scripts/run_nutrition_pipeline.py
python main_nutrition_tagging.py
python main_time_prediction.py
python train_sentiment_model.py
```

## 🐳 Docker

```bash
# Build
make docker-build

# Run
make docker-up

# Stop
make docker-down
```

## 🔧 GitHub Actions

### Configuration des Secrets

1. **Aller sur GitHub**: Settings → Secrets → Actions
2. **Ajouter**:
   - `KAGGLE_USERNAME`: votre username Kaggle
   - `KAGGLE_KEY`: votre clé API Kaggle

### Workflows Disponibles

| Workflow | Déclencheur | Durée |
|----------|------------|-------|
| **CI Tests** | Push/PR | ~5-10 min |
| **ML Pipeline** | Manuel/Schedule | ~60-120 min |
| **Docker Build** | Push main | ~10-15 min |

### Lancer Manuellement

1. GitHub → Actions
2. Sélectionner workflow
3. "Run workflow" → Choisir branche

## 📊 Artefacts

Les modèles entraînés sont sauvegardés comme artefacts GitHub:

1. GitHub → Actions → Workflow run
2. Scroll → "Artifacts"
3. Télécharger les modèles

## 📚 Documentation Complète

- **Guide CI/CD**: [.github/CI_CD_GUIDE.md](.github/CI_CD_GUIDE.md)
- **Guide Tests**: [tests/README.md](tests/README.md)
- **Setup Projet**: [SETUP_GUIDE.md](SETUP_GUIDE.md)

## 🆘 Troubleshooting

### Tests échouent
```bash
# Réinstaller proprement
pip install -e ".[dev,test]"
pytest tests/ -v
```

### Import errors
```bash
# Vérifier l'installation
pip list | grep recipe-ml
python -c "import src; print(src.__file__)"
```

### Kaggle API errors
```bash
# Vérifier credentials
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 🎯 Workflow de Développement

```bash
# 1. Créer branche feature
git checkout -b feature/ma-feature

# 2. Développer
# ... coder ...

# 3. Tester localement
make test

# 4. Commit et push
git add .
git commit -m "feat: description"
git push origin feature/ma-feature

# 5. Créer PR sur GitHub
# → CI s'exécute automatiquement

# 6. Après merge
# → ML Pipeline s'exécute (si sur main)
```

## 📈 Commandes Utiles

```bash
# Linting
make lint

# Formatage
make format

# Nettoyer
make clean

# Tout voir
make help
```

## 🔗 Liens Rapides

- 📦 [Dataset Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- 🐙 [Repository GitHub](https://github.com/tahianahajanirina/mangetamain)
- 📖 [Documentation pytest](https://docs.pytest.org/)
- 🚀 [GitHub Actions](https://docs.github.com/en/actions)

---

**Dernière mise à jour**: 2025-10-28
