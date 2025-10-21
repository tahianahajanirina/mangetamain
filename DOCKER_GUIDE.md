# 🐳 Guide Docker pour MangeTaMain (VS Code Dev Containers)

## 🚀 Démarrage Rapide avec VS Code

### 1. **Prérequis**
- Docker Desktop installé et démarré
- VS Code avec l'extension "Dev Containers" installée

### 2. **Ouvrir dans Dev Container**
1. Ouvrez le projet dans VS Code
2. Appuyez sur `Ctrl+Shift+P` (ou `Cmd+Shift+P` sur Mac)
3. Tapez "Dev Containers: Reopen in Container"
4. Sélectionnez cette option

### 3. **Placer vos données**
Copiez vos fichiers CSV dans le dossier `data/raw/` :
```
data/raw/
├── interactions_test.csv
├── interactions_train.csv  
├── interactions_validation.csv
├── PP_recipes.csv
├── PP_users.csv
├── RAW_interactions.csv
└── RAW_recipes.csv
```

## 📋 Commandes Utiles

### Gestion du container
```bash
# Lancer en arrière-plan
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down

# Reconstruire l'image
docker-compose build --no-cache

# Entrer dans le container
docker-compose exec mangetamain bash
```

### Développement dans VS Code
```bash
# Installer de nouveaux packages (dans le terminal intégré VS Code)
pip install nom-du-package

# Ajouter à pyproject.toml si permanent
# Éditez la section [project] dependencies dans pyproject.toml

# Lancer des tests (dans le terminal VS Code)
pytest
```

## 🔧 Configuration

### Variables d'environnement
Créez un fichier `.env` pour personnaliser :
```bash
JUPYTER_TOKEN=votre-token-secret
JUPYTER_PORT=8888
```

### Volumes montés
- `./data` → `/home/app/data` (données persistantes)
- `./notebooks` → `/home/app/notebooks` (notebooks)
- `./outputs` → `/home/app/outputs` (résultats)
- `.` → `/home/app` (code source pour développement)

## 🐛 Dépannage

### Port déjà utilisé
```bash
# Changer le port dans docker-compose.yml
ports:
  - "8889:8888"  # Utiliser 8889 au lieu de 8888
```

### Problèmes de permissions
```bash
# Donner les permissions au dossier
sudo chown -R $USER:$USER data/ outputs/ notebooks/
```

### Container ne démarre pas
```bash
# Voir les logs détaillés
docker-compose logs mangetamain

# Reconstruire proprement
docker-compose down
docker system prune -f
docker-compose up --build
```

## 🎯 Workflow Recommandé (VS Code)

1. **Ouvrir le projet** dans VS Code
2. **Reopen in Container** (`Ctrl+Shift+P` → "Dev Containers: Reopen in Container")
3. **Placer les données**: Copiez vos CSV dans `data/raw/`
4. **Développer**: Créez et modifiez vos notebooks directement dans VS Code
5. **Analyser**: Utilisez le notebook `01_data_exploration.ipynb` intégré
6. **Exécuter**: Les cellules s'exécutent directement dans VS Code avec Jupyter

## 📦 Packages Inclus (via pyproject.toml)

- **Data Science**: pandas, numpy, scipy, scikit-learn
- **Visualisation**: matplotlib, seaborn
- **ML**: joblib
- **NLP**: nltk, textblob, textstat, wordcloud
- **Utilitaires**: python-dateutil
- **Dev**: jupyter, ipython (pour les notebooks VS Code)

## ✨ Avantages Dev Containers

- ✅ **Environnement isolé** - Pas d'impact sur votre système
- ✅ **Notebooks intégrés** - Directement dans VS Code
- ✅ **IntelliSense complet** - Auto-complétion Python
- ✅ **Debugging intégré** - Points d'arrêt dans les notebooks
- ✅ **Git intégré** - Contrôle de version facile
- ✅ **Extensions Python** - Pylance, Black, etc.

---

**💡 Conseil**: Utilisez `Ctrl+Shift+P` → "Dev Containers: Reopen in Container" pour démarrer rapidement !