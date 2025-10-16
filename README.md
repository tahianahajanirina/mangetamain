# MangeTaMain - Système de Recommandation de Recettes

## 📋 Vue d'ensemble du projet

**MangeTaMain** est un projet de Data Science visant à développer un système intelligent de recommandation de recettes personnalisées en exploitant les données historiques d'interactions utilisateurs et les caractéristiques des recettes de Food.com.

### Problématique

*Comment exploiter les données historiques d'interactions et les évaluations des plats et ingrédients afin d'identifier et anticiper les préférences culinaires des utilisateurs, dans le but de recommander des recettes personnalisées et d'optimiser l'expérience ainsi que l'engagement sur une plateforme de recettes en ligne ?*

---

## 📊 Dataset

**Source** : [Kaggle - Food.com Recipes and User Interactions](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)

### Fichiers de données

#### RAW_recipes.csv (231 638 lignes, 12 colonnes)
- `id` : Identifiant unique de la recette
- `name` : Nom de la recette
- `contributor_id` : ID du contributeur
- `submitted` : Date de soumission
- `tags` : Tags associés à la recette
- `minutes` : Temps de préparation en minutes
- `nutrition` : 7 valeurs nutritionnelles (calories, lipides, sucre, sodium, protéines, graisses saturées, glucides)
- `n_steps` : Nombre d'étapes
- `steps` : Étapes de préparation
- `ingredients` : Liste des ingrédients
- `n_ingredients` : Nombre d'ingrédients
- `description` : Description de la recette

#### RAW_interactions.csv (1 000 000+ lignes, 5 colonnes)
- `user_id` : Identifiant utilisateur
- `recipe_id` : Identifiant de la recette
- `date` : Date de l'interaction
- `rating` : Note de 1 à 5 étoiles
- `review` : Avis textuel

**Période couverte** : 18 ans (2000–2018)

---

## 🎯 Objectifs du projet

### 1. Personnalisation et Rétention Utilisateur

**Objectif Business**
- Augmenter le temps passé sur la plateforme
- Réduire le taux de rebond
- Améliorer la satisfaction et fidélisation client

**Objectif Data Science**
- Développer un système de recommandation personnalisé
- Techniques : Filtrage collaboratif, factorisation matricielle (SVD, NMF), modèles hybrides
- Métriques : RMSE, MAE, Precision@K, Recall@K, Hit Rate

### 2. Optimisation du Contenu et Qualité des Recettes

**Objectif Business**
- Identifier les caractéristiques des recettes à succès
- Guider les créateurs de contenu
- Réduire les recettes mal notées

**Objectif Data Science**
- Prédire la note moyenne d'une recette avant publication
- Techniques : Régression (Random Forest, XGBoost, Gradient Boosting)
- Métriques : MSE, RMSE, R², MAE

### 3. Compréhension Client et Amélioration Continue

**Objectif Business**
- Comprendre les facteurs de satisfaction/insatisfaction
- Détecter rapidement les problèmes de qualité
- Améliorer le service client et la modération

**Objectif Data Science**
- Classification automatique des sentiments et identification des thèmes
- Techniques : NLP (TF-IDF, Word2Vec, BERT), Classification (Naive Bayes, SVM, LSTM, Transformers), Topic Modeling (LDA)
- Métriques : Accuracy, F1-Score, AUC-ROC, Confusion Matrix

---

## ⚠️ Défis techniques identifiés

- **Déséquilibre des classes** : Majorité de notes 5 étoiles
- **Valeurs aberrantes** : Durées irréalistes, nutrition extrême
- **Parsing nécessaire** : Colonne nutrition au format string → conversion en liste
- **Jointure requise** : Entre RAW_recipes.csv et RAW_interactions.csv pour analyses complètes

---

## 🌿 Stratégie de branches Git

Ce projet utilise le modèle **Git Flow** pour une collaboration structurée et efficace.

### Structure des branches

```
master (main)
  └── develop
       ├── feature/data-exploration
       ├── feature/data-preprocessing
       ├── feature/recommendation-system
       ├── feature/prediction-model
       ├── feature/sentiment-analysis
       └── feature/...
```

### Description des branches

#### `master`
- **Rôle** : Branche de production contenant le code stable et testé
- **Protection** : Protégée, accepte uniquement les merges depuis `develop` via Pull Requests
- **Règle** : Ne jamais commiter directement sur `master`

#### `develop`
- **Rôle** : Branche d'intégration principale pour le développement
- **Protection** : Protégée, accepte les merges depuis les branches `feature/`
- **Workflow** : Toutes les fonctionnalités convergent ici avant d'être déployées en production

#### `feature/*`
- **Rôle** : Branches de développement pour des fonctionnalités spécifiques
- **Convention de nommage** : `feature/nom-descriptif-de-la-feature`
- **Cycle de vie** :
  1. Créée depuis `develop`
  2. Développement de la fonctionnalité
  3. Pull Request vers `develop`
  4. Supprimée après merge

### Branches suggérées pour le projet

- `feature/data-exploration` : Analyse exploratoire des données
- `feature/data-preprocessing` : Nettoyage et préparation des données
- `feature/recommendation-system` : Système de recommandation
- `feature/prediction-model` : Modèle de prédiction de notes
- `feature/sentiment-analysis` : Analyse de sentiments des avis
- `feature/api-development` : Développement API
- `feature/deployment` : Configuration déploiement

---

## 🚀 Guide de démarrage

### Prérequis

- Git installé
- Python 3.8+
- Compte GitHub avec accès au repository

### Configuration initiale

#### 1. Cloner le repository

```bash
git clone https://github.com/[votre-organisation]/mongetamain.git
cd mongetamain
```

#### 2. Vérifier les branches disponibles

```bash
git branch -a
```

#### 3. Créer et configurer les branches principales

**Créer la branche `develop`** (à faire une seule fois par le chef de projet)

```bash
git checkout -b develop
git push -u origin develop
```

---

## 📖 Workflow de développement

### Pour démarrer une nouvelle fonctionnalité

#### 1. Se positionner sur `develop` et mettre à jour

```bash
git checkout develop
git pull origin develop
```

#### 2. Créer une branche feature

```bash
git checkout -b feature/nom-de-votre-feature
```

#### 3. Développer et commiter régulièrement

```bash
git add .
git commit -m "Description claire du changement"
```

#### 4. Pousser la branche sur GitHub

```bash
git push -u origin feature/nom-de-votre-feature
```

#### 5. Créer une Pull Request

1. Aller sur GitHub
2. Cliquer sur "Compare & pull request"
3. Base : `develop` ← Compare : `feature/nom-de-votre-feature`
4. Ajouter une description détaillée
5. Assigner des reviewers
6. Soumettre la PR

#### 6. Après approbation et merge

```bash
git checkout develop
git pull origin develop
git branch -d feature/nom-de-votre-feature
```

### Pour mettre en production

#### 1. Depuis `develop` vers `master`

```bash
git checkout master
git pull origin master
git merge develop
git push origin master
```

#### 2. Créer un tag de version (optionnel mais recommandé)

```bash
git tag -a v1.0.0 -m "Version 1.0.0 - Système de recommandation initial"
git push origin v1.0.0
```

---

## 🔄 Bonnes pratiques Git

### Commits

- **Messages clairs et descriptifs** en français ou anglais (selon convention équipe)
- **Format suggéré** :
  ```
  [TYPE] Sujet court (50 chars max)

  Description détaillée si nécessaire

  Type: feat, fix, docs, refactor, test, chore
  ```

### Pull Requests

- **Titre explicite** résumant la fonctionnalité
- **Description complète** incluant :
  - Objectif de la PR
  - Changements effectués
  - Tests réalisés
  - Captures d'écran si applicable
- **Reviewers** : Au moins 1-2 membres de l'équipe
- **Résoudre les conflits** avant de merger

### Synchronisation régulière

```bash
# Récupérer les dernières modifications de develop
git checkout develop
git pull origin develop

# Mettre à jour votre branche feature
git checkout feature/votre-feature
git merge develop
# ou
git rebase develop
```

---

## 📁 Structure du projet (à définir)

```
mongetamain/
├── data/
│   ├── raw/                  # Données brutes
│   ├── processed/            # Données nettoyées
│   └── external/             # Données externes
├── notebooks/                # Jupyter notebooks pour exploration
├── src/
│   ├── data/                 # Scripts de préparation des données
│   ├── features/             # Feature engineering
│   ├── models/               # Modèles ML
│   └── visualization/        # Visualisations
├── tests/                    # Tests unitaires
├── docs/                     # Documentation
├── requirements.txt          # Dépendances Python
├── .gitignore
└── README.md
```

```
data/
├── raw/                    # Données brutes (originales)
│   ├── interactions_test.csv
│   ├── interactions_train.csv
│   ├── interactions_validation.csv
│   ├── PP_recipes.csv
│   ├── PP_users.csv
│   ├── RAW_interactions.csv
│   └── RAW_recipes.csv
├── processed/              # Données traitées
│   ├── cleaned_interactions.csv
│   ├── cleaned_recipes.csv
│   └── features/
└── README.md              # Ce fichier
```

## Instructions

1. **Placez vos fichiers CSV dans le dossier `raw/`**
2. Les données traitées seront automatiquement sauvegardées dans `processed/`
3. N'éditez jamais les fichiers dans `raw/` - ils servent de référence

---

## 👥 Organisation de l'équipe

### Rôles suggérés

- **Chef de projet** : Coordination, gestion des branches principales
- **Data Engineers** : Préparation et nettoyage des données
- **Data Scientists** : Développement des modèles
- **ML Engineers** : Déploiement et API
- **Reviewers** : Validation du code et des analyses

### Communication

- **Daily standup** : Synchronisation quotidienne
- **Code review** : Obligatoire sur toutes les PR
- **Documentation** : Mise à jour continue dans `/docs`

---

## 📚 Ressources

- [Documentation Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Dataset Kaggle](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)
- [Guide des Pull Requests GitHub](https://docs.github.com/en/pull-requests)

---

## 📝 Prochaines étapes

1. ✅ Créer les branches `master` et `develop`
2. ⬜ Télécharger et placer les datasets dans `/data/raw/`
3. ⬜ Créer la première branche `feature/data-exploration`
4. ⬜ Effectuer l'analyse exploratoire initiale
5. ⬜ Documenter les findings
6. ⬜ Première PR vers `develop`

---

## 🤝 Contribution

Chaque membre de l'équipe est encouragé à :
- Suivre le workflow Git Flow
- Documenter son code
- Écrire des tests
- Participer aux code reviews
- Partager ses connaissances

---

## 📞 Contact

Pour toute question sur le projet ou le workflow Git, contactez le chef de projet ou créez une issue GitHub.

---

**Dernière mise à jour** : 2025-10-07
