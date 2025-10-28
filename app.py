#!/usr/bin/env python3
"""
Application Streamlit - Clustering & Classification Nutritionnelle
Deux modèles: Recipe Clustering + Nutrition Classification

Commande:
    streamlit run app.py
"""

import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Recipe ML - Clustering & Nutrition",
    page_icon="🍳",
    layout="wide"
)

@st.cache_resource
def load_clustering_model():
    """Charge le modèle de clustering"""
    model_dir = Path('outputs/models')
    metadata_files = list(model_dir.glob('recipe_clustering_*_metadata.json'))
    
    if not metadata_files:
        return None, None, None, None
    
    latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_name']
    
    # Charger modèle
    scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")
    pca = joblib.load(model_dir / f"{model_name}_pca.pkl")
    kmeans = joblib.load(model_dir / f"{model_name}_kmeans.pkl")
    
    return scaler, pca, kmeans, metadata

@st.cache_resource
def load_nutrition_model():
    """Charge le modèle de classification nutritionnelle"""
    model_dir = Path('outputs/models')
    metadata_files = list(model_dir.glob('nutrition_classifier_*_metadata.json'))
    
    if not metadata_files:
        return None, None, None
    
    latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_name']
    
    # Charger modèle
    scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")
    model = joblib.load(model_dir / f"{model_name}_model.pkl")
    
    return scaler, model, metadata

# Header
st.title("🍳 Recipe Machine Learning Platform")
st.markdown("### Clustering de Recettes & Classification Nutritionnelle")
st.markdown("---")

# Sélection du modèle
tab1, tab2 = st.tabs(["🔵 Clustering de Recettes", "🥗 Classification Nutritionnelle"])

# ============================================================================
# TAB 1: CLUSTERING
# ============================================================================
with tab1:
    st.header("🔵 Clustering de Recettes par Temps/Complexité")
    
    with st.spinner('Chargement du modèle de clustering...'):
        scaler_clust, pca, kmeans, metadata_clust = load_clustering_model()
    
    if scaler_clust is None:
        st.warning("⚠️ Aucun modèle de clustering trouvé. Entraînez d'abord le modèle:")
        st.code("python scripts/run_recipe_pipeline.py --k 5")
    else:
        st.success("✅ Modèle de clustering chargé!")
        
        # Info modèle
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{metadata_clust['metrics']['silhouette']:.4f}")
        col2.metric("Nb Clusters", metadata_clust['n_clusters'])
        col3.metric("PCA Components", metadata_clust['pca_n_components'])
        
        st.markdown("---")
        
        # Noms des clusters
        st.subheader("📋 Clusters Identifiés")
        for cluster_id, name in metadata_clust['cluster_names'].items():
            st.markdown(f"**Cluster {cluster_id}:** {name}")
        
        st.markdown("---")
        
        # Section Prédiction
        st.subheader("🔬 Prédire le Cluster d'une Nouvelle Recette")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Features de la Recette:**")
            log_minutes = st.slider("Temps (log minutes)", 1.0, 7.0, 4.0, 0.1, key="clust_time")
            complexity = st.slider("Complexité (nb étapes)", 1, 100, 10, key="clust_complex")
        
        with col2:
            st.markdown("**Suite:**")
            efficiency = st.slider("Efficacité", 0.0, 5.0, 1.0, 0.1, key="clust_eff")
            health = st.slider("Santé (0-5)", 0, 5, 3, key="clust_health")
        
        st.info("ℹ️ Le modèle utilise 4 features: temps, complexité, efficacité, santé")
        
        if st.button("🎯 Prédire le Cluster", type="primary", key="predict_cluster"):
            # Créer vecteur avec les 4 features dans le bon ordre
            X_new = np.array([[log_minutes, complexity, efficiency, health]])
            
            # Pipeline
            X_scaled = scaler_clust.transform(X_new)
            X_pca = pca.transform(X_scaled)
            prediction = kmeans.predict(X_pca)[0]
            
            cluster_name = metadata_clust['cluster_names'].get(str(prediction), f"Cluster {prediction}")
            
            st.success(f"### 🎉 Résultat: **{cluster_name}**")
            
            st.markdown(f"""
            **Détails:**
            - Cluster ID: {prediction}
            - Nom: {cluster_name}
            - Features PCA utilisées: {metadata_clust['pca_n_components']}
            """)

# ============================================================================
# TAB 2: CLASSIFICATION NUTRITIONNELLE
# ============================================================================
with tab2:
    st.header("🥗 Classification Nutritionnelle des Recettes")
    
    with st.spinner('Chargement du modèle de classification...'):
        scaler_nutr, model_nutr, metadata_nutr = load_nutrition_model()
    
    if scaler_nutr is None:
        st.warning("⚠️ Aucun modèle de classification trouvé. Entraînez d'abord le modèle:")
        st.code("python scripts/run_nutrition_pipeline.py")
    else:
        st.success("✅ Modèle de classification nutritionnelle chargé!")
        
        # Info modèle
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metadata_nutr.get('test_accuracy', 0.999):.2%}")
        col2.metric("F1-Score", f"{metadata_nutr.get('cv_f1_mean', 0.999):.4f}")
        col3.metric("Features", metadata_nutr['n_features'])
        
        st.markdown("---")
        
        # Catégories
        st.subheader("📋 Catégories Nutritionnelles")
        class_names = metadata_nutr['class_names']
        
        cols = st.columns(4)
        emojis = ['🥗', '🍎', '⚖️', '🍰']
        colors = ['green', 'blue', 'orange', 'red']
        
        for i, (col, emoji, color) in enumerate(zip(cols, emojis, colors)):
            col.markdown(f":{color}[**{emoji} {class_names[i]}**]")
        
        st.markdown("---")
        
        # Section Prédiction
        st.subheader("🔬 Classifier une Nouvelle Recette")
        
        st.markdown("#### 📊 Valeurs Nutritionnelles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            calories = st.number_input("Calories (kcal)", 0, 2000, 300, 10)
            total_fat = st.number_input("Graisses totales (g)", 0.0, 200.0, 15.0, 0.5)
            saturated_fat = st.number_input("Graisses saturées (g)", 0.0, 100.0, 5.0, 0.5)
        
        with col2:
            sodium = st.number_input("Sodium (mg)", 0, 5000, 400, 10)
            carbohydrates = st.number_input("Glucides (g)", 0.0, 500.0, 30.0, 1.0)
            sugar = st.number_input("Sucre (g)", 0.0, 200.0, 10.0, 0.5)
        
        with col3:
            protein = st.number_input("Protéines (g)", 0.0, 200.0, 20.0, 0.5)
            n_steps = st.number_input("Nombre d'étapes", 1, 100, 10, 1)
            n_ingredients = st.number_input("Nombre d'ingrédients", 1, 50, 8, 1)
            minutes = st.number_input("Temps (minutes)", 1, 500, 30, 1)
        
        st.info("ℹ️ Les autres features (PDV%, ratios, etc.) sont calculées automatiquement")
        
        if st.button("🎯 Classifier la Recette", type="primary", key="predict_nutrition"):
            # Calculer les features dérivées
            daily_values = {
                'calories': 2000, 'total_fat': 78, 'saturated_fat': 20,
                'sodium': 2300, 'carbohydrates': 275, 'sugar': 50, 'protein': 50
            }
            
            # PDV%
            calories_pdv = (calories / daily_values['calories']) * 100
            fat_pdv = (total_fat / daily_values['total_fat']) * 100
            saturated_fat_pdv = (saturated_fat / daily_values['saturated_fat']) * 100
            sodium_pdv = (sodium / daily_values['sodium']) * 100
            carbs_pdv = (carbohydrates / daily_values['carbohydrates']) * 100
            sugar_pdv = (sugar / daily_values['sugar']) * 100
            protein_pdv = (protein / daily_values['protein']) * 100
            
            # Ratios
            protein_density = (protein * 100) / calories if calories > 0 else 0
            sugar_ratio = (sugar * 4 * 100) / calories if calories > 0 else 0
            sodium_density = (sodium * 100) / calories if calories > 0 else 0
            saturated_fat_ratio = (saturated_fat / total_fat * 100) if total_fat > 0 else 0
            macro_balance = np.std([fat_pdv, carbs_pdv, protein_pdv])
            
            # Health score
            score = 100.0
            score -= np.clip(calories_pdv - 20, 0, 30)
            score -= np.clip(fat_pdv - 20, 0, 25)
            score -= np.clip(saturated_fat_pdv - 15, 0, 20)
            score -= np.clip(sodium_pdv - 15, 0, 20)
            score -= np.clip(sugar_pdv - 15, 0, 25)
            score += np.clip(protein_pdv, 0, 15)
            score += np.clip(protein_density / 2, 0, 10)
            health_score = np.clip(score, 0, 100)
            
            # Features binaires
            is_low_calorie = 1 if calories < 200 else 0
            is_high_protein = 1 if protein_pdv > 30 else 0
            is_low_fat = 1 if fat_pdv < 10 else 0
            is_low_sodium = 1 if sodium_pdv < 10 else 0
            is_low_sugar = 1 if sugar_pdv < 10 else 0
            
            # Complexité et interactions
            complexity_score = np.log1p(n_steps) + np.log1p(n_ingredients)
            calorie_protein_interaction = calories * protein_pdv
            fat_sodium_interaction = fat_pdv * sodium_pdv
            
            # Créer le vecteur de features (31 features dans le bon ordre)
            X_new = np.array([[
                calories, total_fat, saturated_fat, sodium, carbohydrates, sugar, protein,
                calories_pdv, fat_pdv, saturated_fat_pdv, sodium_pdv, carbs_pdv, sugar_pdv, protein_pdv,
                protein_density, sugar_ratio, sodium_density, saturated_fat_ratio, macro_balance,
                health_score,
                is_low_calorie, is_high_protein, is_low_fat, is_low_sodium, is_low_sugar,
                complexity_score, calorie_protein_interaction, fat_sodium_interaction,
                n_steps, n_ingredients, minutes
            ]])
            
            # Prédire
            X_scaled = scaler_nutr.transform(X_new)
            prediction = model_nutr.predict(X_scaled)[0]
            probabilities = model_nutr.predict_proba(X_scaled)[0]
            
            category_name = class_names[prediction]
            confidence = probabilities[prediction]
            
            # Afficher résultat
            emoji = emojis[prediction]
            color = colors[prediction]
            
            st.markdown(f"## {emoji} **Catégorie: :{color}[{category_name}]**")
            st.metric("Confiance", f"{confidence*100:.1f}%")
            
            # Health score
            st.markdown(f"### 💚 Health Score: **{health_score:.1f}/100**")
            st.progress(health_score / 100)
            
            st.markdown("---")
            
            # Probabilités
            st.subheader("📊 Probabilités par Catégorie")
            
            prob_df = pd.DataFrame({
                'Catégorie': [f"{emojis[i]} {class_names[i]}" for i in range(4)],
                'Probabilité': probabilities
            })
            
            st.bar_chart(prob_df.set_index('Catégorie'))
            
            # Détails
            with st.expander("📋 Détails des Valeurs Nutritionnelles"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Valeurs brutes:**")
                    st.write(f"- Calories: {calories} kcal ({calories_pdv:.1f}% DV)")
                    st.write(f"- Protéines: {protein}g ({protein_pdv:.1f}% DV)")
                    st.write(f"- Graisses: {total_fat}g ({fat_pdv:.1f}% DV)")
                    st.write(f"- Sucre: {sugar}g ({sugar_pdv:.1f}% DV)")
                
                with col2:
                    st.markdown("**Ratios calculés:**")
                    st.write(f"- Densité protéique: {protein_density:.2f}")
                    st.write(f"- Ratio sucre: {sugar_ratio:.2f}")
                    st.write(f"- Densité sodium: {sodium_density:.2f}")
                    st.write(f"- Équilibre macros: {macro_balance:.2f}")
            
            # Interprétation
            st.markdown("---")
            st.subheader("💡 Interprétation")
            
            if prediction == 0:
                st.success("✅ Cette recette est **très saine** avec peu de calories et riche en nutriments essentiels. Excellente pour une alimentation quotidienne.")
            elif prediction == 1:
                st.info("👍 Cette recette est **équilibrée et saine**, adaptée à une alimentation régulière et variée.")
            elif prediction == 2:
                st.warning("⚠️ Cette recette est **modérée**, à consommer avec attention dans le cadre d'une alimentation équilibrée.")
            else:
                st.error("🍰 Cette recette est **indulgente**, riche en calories/sucre/graisses. À réserver pour les occasions spéciales!")


# Footer
st.markdown("---")
st.caption("🤖 ML Platform | Clustering + Nutrition Classification | 📅 2025")
