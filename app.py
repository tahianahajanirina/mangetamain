#!/usr/bin/env python3
"""
Application Streamlit - Version Ultra Légère
Utilise SEULEMENT les métadonnées du modèle .pkl

Commande:
    streamlit run app_light.py
"""

import streamlit as st
import joblib
import json
import numpy as np
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Recipe Clustering",
    page_icon="🍳",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Charge le modèle .pkl"""
    model_dir = Path('outputs/models')
    metadata_files = list(model_dir.glob('recipe_clustering_*_metadata.json'))
    
    if not metadata_files:
        st.error("❌ Aucun modèle trouvé!")
        st.stop()
    
    latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_name']
    
    # Charger modèle
    scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")
    pca = joblib.load(model_dir / f"{model_name}_pca.pkl")
    kmeans = joblib.load(model_dir / f"{model_name}_kmeans.pkl")
    
    return scaler, pca, kmeans, metadata

# Chargement
st.title("🍳 Recipe Clustering - ML Demo")
st.markdown("### Prédiction en Temps Réel avec Modèle .pkl")
st.markdown("---")

with st.spinner('Chargement du modèle...'):
    scaler, pca, kmeans, metadata = load_model()

st.success("✅ Modèle chargé avec succès!")

# Info modèle
st.sidebar.header("📊 Modèle")
st.sidebar.metric("Silhouette", f"{metadata['metrics']['silhouette']:.4f}")
st.sidebar.metric("Clusters", metadata['n_clusters'])
st.sidebar.metric("Recettes", f"{metadata['metrics']['n_recipes']:,}")

# Métriques
col1, col2, col3 = st.columns(3)
col1.metric("Silhouette Score", f"{metadata['metrics']['silhouette']:.4f}")
col2.metric("Nb Clusters", metadata['n_clusters'])
col3.metric("PCA Components", metadata['pca_n_components'])

st.markdown("---")

# Noms des clusters
st.subheader("📋 Clusters Identifiés")

for cluster_id, name in metadata['cluster_names'].items():
    st.markdown(f"**Cluster {cluster_id}:** {name}")

st.markdown("---")

# Section Prédiction
st.subheader("🔬 Prédire le Cluster d'une Nouvelle Recette")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Features de la Recette:**")
    log_minutes = st.slider("Temps (log minutes)", 1.0, 7.0, 4.0, 0.1)
    complexity = st.slider("Complexité (nb étapes)", 1, 100, 10)

with col2:
    st.markdown("**Suite:**")
    efficiency = st.slider("Efficacité", 0.0, 5.0, 1.0, 0.1)
    health = st.slider("Santé (0-5)", 0, 5, 3)

st.info("ℹ️ Le modèle utilise 4 features: temps, complexité, efficacité, santé")

if st.button("🎯 Prédire le Cluster", type="primary"):
    # Créer vecteur avec les 4 features dans le bon ordre
    X_new = np.array([[log_minutes, complexity, efficiency, health]])
    
    # Pipeline
    X_scaled = scaler.transform(X_new)
    X_pca = pca.transform(X_scaled)
    prediction = kmeans.predict(X_pca)[0]
    
    cluster_name = metadata['cluster_names'].get(str(prediction), f"Cluster {prediction}")
    
    st.success(f"### 🎉 Résultat: **{cluster_name}**")
    
    st.markdown(f"""
    **Détails:**
    - Cluster ID: {prediction}
    - Nom: {cluster_name}
    - Features PCA utilisées: {metadata['pca_n_components']}
    """)

# Footer
st.markdown("---")
st.caption(f"🤖 Modèle: {metadata['model_name']} | 📅 {metadata['timestamp']}")
