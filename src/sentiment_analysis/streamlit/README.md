# 🍽️ Recipe Review Sentiment Analyzer - Streamlit App

Web application pour analyser les sentiments des avis de recettes culinaires.

## 🚀 Démarrage Rapide

### 1. Installer les dépendances
```powershell
pip install -e .
# OU
uv pip install -e .
```

### 2. Lancer l'application
```powershell
# Depuis la racine du projet
streamlit run src/sentiment_analysis/streamlit/app.py

# OU utiliser le script de lancement
python -m streamlit run src/sentiment_analysis/streamlit/app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

---

## ✨ Fonctionnalités

### 📝 Analyse Simple
- Entrez un avis de recette
- Obtenez instantanément le sentiment (Positif/Neutre/Négatif)
- Visualisez le score de confiance
- Exemples pré-chargés pour tester

### 📋 Analyse par Lot
- **Option 1 :** Uploadez un fichier CSV avec une colonne `review`
- **Option 2 :** Entrez plusieurs avis manuellement (un par ligne)
- Téléchargez les résultats en CSV

### 📈 Statistiques
- Informations sur le modèle
- Conseils d'utilisation
- Performance et métriques

---

## 📁 Structure

```
src/sentiment_analysis/streamlit/
├── app.py              # Application Streamlit principale
└── README.md           # Cette documentation
```

---

## 🎨 Interface

```
┌─────────────────────────────────────────────────────────┐
│  🍽️ Recipe Review Sentiment Analyzer                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📝 Single Review  |  📋 Batch Analysis  |  📈 Stats   │
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │ Enter a recipe review:                        │    │
│  │ ┌──────────────────────────────────────────┐ │    │
│  │ │ This recipe was amazing! Will make       │ │    │
│  │ │ again...                                 │ │    │
│  │ └──────────────────────────────────────────┘ │    │
│  │                                               │    │
│  │        [🔍 Analyze Sentiment]                │    │
│  └───────────────────────────────────────────────┘    │
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │ 🟢 Positive                                   │    │
│  │ ━━━━━━━━━━━━━━━━━━━━━━━ 95%                │    │
│  └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Exemples d'utilisation

### Analyse d'un avis simple
1. Allez dans l'onglet **"📝 Single Review"**
2. Cliquez sur un exemple ou tapez votre propre avis
3. Cliquez sur **"🔍 Analyze Sentiment"**
4. Visualisez le résultat instantanément

### Analyse par lot (CSV)
1. Allez dans l'onglet **"📋 Batch Analysis"**
2. Préparez un CSV avec une colonne `review` :
   ```csv
   review
   "Amazing recipe! Delicious."
   "Too salty for my taste."
   "It was okay, nothing special."
   ```
3. Uploadez le fichier
4. Cliquez sur **"🔍 Analyze All Reviews"**
5. Téléchargez les résultats

---

## 🎯 Modèle ML

- **Modèle :** RoBERTa fine-tuné sur Food.com
- **Source :** `TahianaAndriambahoaka/sentiment-analysis-food-reviews`
- **Framework :** Hugging Face Transformers
- **Classes :** Positive, Neutral, Negative

---

## 🔧 Configuration

### Variables d'environnement (optionnel)
```powershell
# Pour forcer CPU (si pas de GPU)
$env:CUDA_VISIBLE_DEVICES = "-1"

# Pour désactiver les warnings
$env:TRANSFORMERS_VERBOSITY = "error"
```

### Personnalisation
Vous pouvez modifier le fichier `app.py` pour :
- Changer les couleurs (CSS personnalisé)
- Ajouter d'autres visualisations
- Modifier les exemples
- Ajouter de nouvelles fonctionnalités

---

## 🚀 Déploiement

### En local
```powershell
streamlit run src/sentiment_analysis/streamlit/app.py
```

### Sur Streamlit Cloud (gratuit)
1. Push votre code sur GitHub
2. Allez sur [share.streamlit.io](https://share.streamlit.io)
3. Connectez votre repo GitHub
4. Sélectionnez `src/sentiment_analysis/streamlit/app.py` comme fichier principal
5. Déployez !

### Sur Heroku / Railway / Render
Créez un `Procfile` à la racine :
```
web: streamlit run src/sentiment_analysis/streamlit/app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🐛 Résolution de problèmes

### Erreur : "No module named 'streamlit'"
```powershell
pip install streamlit
```

### Erreur : "No module named 'src'"
```powershell
# Assurez-vous d'être à la racine du projet
pip install -e .
```

### Le modèle se charge lentement
- Au premier lancement, le modèle (~500MB) est téléchargé depuis Hugging Face
- Les lancements suivants sont instantanés (modèle mis en cache)

### Port déjà utilisé
```powershell
streamlit run src/sentiment_analysis/streamlit/app.py --server.port 8502
```

---

## 🤝 Contribution

Pour ajouter des fonctionnalités :
1. Modifiez `app.py`
2. Testez localement avec `streamlit run src/sentiment_analysis/streamlit/app.py`
3. Committez et pushez vos changements

---

## 📝 License

MIT License - MangeTaMain Project

---

## 🎉 Crédits

- **Modèle ML :** Fine-tuné sur Food.com dataset
- **Framework :** Streamlit + Hugging Face Transformers
- **Équipe :** MangeTaMain - Télécom Paris
