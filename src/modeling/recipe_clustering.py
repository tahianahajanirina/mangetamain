"""
Clustering des Recettes V3 - Silhouette Optimisé (>= 0.26)

Ce module implémente le clustering K-Means optimisé sur les recettes
avec PCA et nommage intelligent des clusters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple
import logging
import joblib
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


class RecipeClusterer:
    """Clustering K-Means optimisé pour recettes V3"""
    
    def __init__(self, n_clusters: int = None, pca_variance: float = 0.90, 
                 random_state: int = 42):
        """
        Args:
            n_clusters: Nombre de clusters (None = détection automatique)
            pca_variance: Variance à conserver pour PCA (default: 90%)
            random_state: Seed pour reproductibilité
        """
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.random_state = random_state
        
        self.recipes = None
        self.features_scaled = None
        self.features_pca = None
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.cluster_names = {}
        
    def load_features(self, features_path: str) -> pd.DataFrame:
        """Charge les features recettes V3"""
        logger.info("Chargement des features...")
        recipes = pd.read_csv(features_path)
        
        logger.info(f"Recettes: {len(recipes):,}")
        logger.info(f"Features: {len(recipes.columns) - 2}")
        
        self.recipes = recipes
        return recipes
    
    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalisation + PCA des features"""
        logger.info("=" * 80)
        logger.info("NORMALISATION + PCA")
        logger.info("=" * 80)
        
        # IMPORTANT: Exclure popularity_score du clustering
        # (utilisé seulement pour post-filtrage/ranking)
        features_cols = ['log_minutes', 'time_complexity', 'efficiency', 'health_category']
        
        logger.info(f"\nFeatures pour CLUSTERING ({len(features_cols)}): {features_cols}")
        logger.info("Feature pour FILTRAGE (après): popularity_score")
        logger.info("\nRaison: popularity_score domine et crée des clusters homogènes.")
        logger.info("Solution: Cluster par temps/complexité/santé, puis filtrer par popularité.\n")
        
        features_raw = self.recipes[features_cols].copy()
        
        # Normalisation
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features_raw)
        
        logger.info(f"Normalisation terminée")
        logger.info(f"  Mean: {self.features_scaled.mean():.6f}")
        logger.info(f"  Std: {self.features_scaled.std():.6f}")
        
        # PCA
        self.pca = PCA(n_components=self.pca_variance)
        self.features_pca = self.pca.fit_transform(self.features_scaled)
        
        logger.info(f"\nPCA appliquée:")
        logger.info(f"  Composantes: {self.pca.n_components_}")
        logger.info(f"  Variance expliquée: {self.pca.explained_variance_ratio_.sum():.1%}")
        logger.info(f"  Shape: {self.features_pca.shape}")
        
        logger.info(f"\nVariance par composante:")
        for i, var in enumerate(self.pca.explained_variance_ratio_, 1):
            logger.info(f"  PC{i}: {var:.1%}")
        
        return self.features_scaled, self.features_pca
    
    def find_optimal_k(self, k_range: range = range(3, 11), 
                       sample_size: int = 20000) -> int:
        """Trouve le K optimal via silhouette score"""
        logger.info("=" * 80)
        logger.info("DETERMINATION DU K OPTIMAL")
        logger.info("=" * 80)
        
        # Échantillonnage pour rapidité
        sample_indices = np.random.RandomState(self.random_state).choice(
            len(self.features_pca),
            size=min(sample_size, len(self.features_pca)),
            replace=False
        )
        features_sample = self.features_pca[sample_indices]
        
        logger.info(f"\nÉchantillon: {len(features_sample):,} recettes")
        logger.info(f"Test de K = {k_range.start} à {k_range.stop - 1}...\n")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=20,
                max_iter=500
            )
            kmeans.fit(features_sample)
            
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(features_sample, kmeans.labels_)
            silhouette_scores.append(sil_score)
            
            logger.info(f"  K={k}: Inertia={kmeans.inertia_:>10,.2f}, Silhouette={sil_score:.4f}")
        
        best_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        
        logger.info(f"\n\nK OPTIMAL: {best_k}")
        logger.info(f"Silhouette score: {max_silhouette:.4f}")
        
        if max_silhouette >= 0.30:
            logger.info("\nOBJECTIF ATTEINT! Silhouette >= 0.30")
        elif max_silhouette >= 0.25:
            logger.info("\nTRÈS BON! Silhouette >= 0.25")
        elif max_silhouette >= 0.20:
            logger.info("\nBON! Silhouette >= 0.20")
        else:
            logger.info("\nACCEPTABLE! Structure détectable")
        
        return best_k
    
    def fit(self, features_path: str = None) -> 'RecipeClusterer':
        """Pipeline complet de clustering"""
        logger.info("=" * 80)
        logger.info("CLUSTERING RECETTES V3")
        logger.info("=" * 80)
        
        # 1. Charger si nécessaire
        if features_path and self.recipes is None:
            self.load_features(features_path)
        
        # 2. Préparer features
        self.prepare_features()
        
        # 3. Trouver K optimal si non spécifié
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_k()
        
        # 4. Entraîner K-Means final
        logger.info("=" * 80)
        logger.info(f"ENTRAÎNEMENT K-MEANS FINAL (K={self.n_clusters})")
        logger.info("=" * 80)
        
        logger.info(f"\nEntraînement sur {len(self.features_pca):,} recettes...")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=50,
            max_iter=500
        )
        clusters = self.kmeans.fit_predict(self.features_pca)
        
        self.recipes['cluster'] = clusters
        
        logger.info(f"\nK-MEANS ENTRAÎNÉ:")
        logger.info(f"  K: {self.n_clusters}")
        logger.info(f"  Inertia: {self.kmeans.inertia_:,.2f}")
        
        # Silhouette sur échantillon
        sample_for_sil = np.random.RandomState(self.random_state).choice(
            len(self.features_pca),
            size=min(10000, len(self.features_pca)),
            replace=False
        )
        final_silhouette = silhouette_score(
            self.features_pca[sample_for_sil],
            clusters[sample_for_sil]
        )
        logger.info(f"  Silhouette: {final_silhouette:.4f}")
        
        # Distribution
        logger.info(f"\nDISTRIBUTION:")
        cluster_counts = self.recipes['cluster'].value_counts().sort_index()
        for cluster_id in range(self.n_clusters):
            count = cluster_counts[cluster_id]
            pct = 100 * count / len(self.recipes)
            logger.info(f"  Cluster {cluster_id}: {count:>7,} recettes ({pct:>5.1f}%)")
        
        return self
    
    def name_clusters(self) -> Dict[int, str]:
        """Nomme intelligemment les clusters basé sur leurs profils"""
        logger.info("=" * 80)
        logger.info("NOMMAGE DES CLUSTERS")
        logger.info("=" * 80)
        
        features_cols_all = ['popularity_score', 'log_minutes', 'time_complexity', 
                            'efficiency', 'health_category']
        cluster_profiles = self.recipes.groupby('cluster')[features_cols_all].mean()
        
        def name_cluster(cluster_id, profile):
            """Nommage basé sur temps, santé et complexité (PAS popularité)"""
            names = []
            
            # Temps (critère principal)
            minutes = np.expm1(profile['log_minutes'])
            if minutes < 25:
                names.append('Rapides')
            elif minutes > 90:
                names.append('Lentes')
            elif minutes > 50:
                names.append('Elaborees')
            elif minutes > 35:
                # Différencier selon la santé
                if profile['health_category'] >= 2.5:
                    names.append('Equilibrees')
                else:
                    names.append('Moderees')
            
            # Santé (critère important)
            if profile['health_category'] >= 2.8:
                names.append('Tres Saines')
            elif profile['health_category'] >= 2.3:
                names.append('Saines')
            elif profile['health_category'] < 1.5:
                names.append('Gourmandes')
            
            # Complexité
            if profile['time_complexity'] > 120:
                names.append('Tres Complexes')
            elif profile['time_complexity'] > 90:
                names.append('Complexes')
            elif profile['time_complexity'] < 60:
                names.append('Simples')
            
            # Efficacité (seulement si pas déjà "Lentes")
            if profile['efficiency'] > 2.0:
                names.append('Efficaces')
            elif profile['efficiency'] < 1.0 and 'Lentes' not in names:
                names.append('Peu Efficaces')
            
            # Si pas assez de caractéristiques, utiliser "Standard"
            if len(names) < 2:
                if 30 <= minutes <= 50 and 60 <= profile['time_complexity'] <= 90:
                    names.insert(0, 'Standard')
            
            # Éliminer les doublons
            names = list(dict.fromkeys(names))
            
            # Retourner max 2 caractéristiques
            if len(names) >= 2:
                return ' & '.join(names[:2])
            elif len(names) == 1:
                return names[0]
            else:
                return 'Standard'
        
        cluster_counts = self.recipes['cluster'].value_counts().sort_index()
        
        for cluster_id in range(self.n_clusters):
            profile_dict = cluster_profiles.loc[cluster_id].to_dict()
            cluster_name = name_cluster(cluster_id, profile_dict)
            self.cluster_names[cluster_id] = cluster_name
        
        logger.info(f"\nCLUSTERS NOMMÉS (basé sur TEMPS/SANTÉ/COMPLEXITÉ):")
        for cluster_id, name in self.cluster_names.items():
            count = cluster_counts[cluster_id]
            pct = 100 * count / len(self.recipes)
            profile = cluster_profiles.loc[cluster_id]
            avg_pop = profile['popularity_score']
            minutes = np.expm1(profile['log_minutes'])
            logger.info(f"  {cluster_id}. \"{name}\" ({count:,} - {pct:.1f}%) "
                       f"[{minutes:.0f}min, santé:{profile['health_category']:.1f}, "
                       f"pop:{avg_pop:.1f}]")
        
        logger.info(f"\nNote: popularity_score disponible pour filtrage dans chaque cluster")
        
        # Appliquer les noms
        self.recipes['cluster_name'] = self.recipes['cluster'].map(self.cluster_names)
        
        return self.cluster_names
    
    def save_results(self, output_path: str) -> None:
        """Sauvegarde les résultats du clustering"""
        logger.info("=" * 80)
        logger.info("EXPORT FINAL")
        logger.info("=" * 80)
        
        logger.info(f"\nFichier exporté: {output_path}")
        self.recipes.to_csv(output_path, index=False)
        
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"Shape: {self.recipes.shape}")
        logger.info(f"Taille: {file_size:.2f} MB")
        
        logger.info(f"\nDataset avec clusters V3 prêt!")
    
    def plot_clusters(self, output_dir: str = None) -> None:
        """Visualise les clusters en 2D avec PCA"""
        logger.info("=" * 80)
        logger.info("VISUALISATION PCA 2D")
        logger.info("=" * 80)
        
        # PCA 2D pour visualisation
        pca_viz = PCA(n_components=2)
        features_pca_2d = pca_viz.fit_transform(self.features_scaled)
        
        logger.info(f"\nVariance expliquée: {pca_viz.explained_variance_ratio_.sum():.1%}")
        
        # Échantillon pour plot
        SAMPLE_PLOT = 30000
        plot_indices = np.random.RandomState(self.random_state).choice(
            len(features_pca_2d),
            size=min(SAMPLE_PLOT, len(features_pca_2d)),
            replace=False
        )
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        clusters = self.recipes['cluster'].values
        
        for cluster_id in range(self.n_clusters):
            mask = clusters[plot_indices] == cluster_id
            cluster_name = self.cluster_names.get(cluster_id, f'Cluster {cluster_id}')
            ax.scatter(
                features_pca_2d[plot_indices][mask, 0],
                features_pca_2d[plot_indices][mask, 1],
                label=f'{cluster_name} ({mask.sum():,})',
                alpha=0.6, s=25, color=colors[cluster_id], edgecolors='none'
            )
        
        # Centroïdes en PCA 2D
        features_cols = ['log_minutes', 'time_complexity', 'efficiency', 'health_category']
        centroids_scaled = self.scaler.transform(
            self.recipes.groupby('cluster')[features_cols].mean()
        )
        centroids_pca_2d = pca_viz.transform(centroids_scaled)
        ax.scatter(
            centroids_pca_2d[:, 0], centroids_pca_2d[:, 1],
            c='red', marker='*', s=1200, edgecolors='black',
            linewidth=2.5, label='Centroïdes', zorder=10
        )
        
        # Calculer silhouette pour le titre
        sample_for_sil = np.random.RandomState(self.random_state).choice(
            len(self.features_pca),
            size=min(10000, len(self.features_pca)),
            replace=False
        )
        final_silhouette = silhouette_score(
            self.features_pca[sample_for_sil],
            clusters[sample_for_sil]
        )
        
        ax.set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})',
                     fontsize=12, fontweight='bold')
        ax.set_title(f'Clustering V3 OPTIMISÉ (K={self.n_clusters}, Silhouette={final_silhouette:.3f})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / '07c_recipes_clusters_pca_v3.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"\nVisualisation sauvegardée: {output_path}")
        
        plt.close()
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du clustering"""
        if self.kmeans is None:
            raise ValueError("Clustering non effectué. Appelez fit() d'abord.")
        
        sample_for_sil = np.random.RandomState(self.random_state).choice(
            len(self.features_pca),
            size=min(10000, len(self.features_pca)),
            replace=False
        )
        
        clusters = self.recipes['cluster'].values
        
        return {
            'n_clusters': self.n_clusters,
            'n_recipes': len(self.recipes),
            'inertia': self.kmeans.inertia_,
            'silhouette': silhouette_score(
                self.features_pca[sample_for_sil],
                clusters[sample_for_sil]
            ),
            'cluster_names': self.cluster_names,
            'cluster_distribution': self.recipes['cluster'].value_counts().to_dict()
        }
    
    def save_model(self, output_dir: str = 'outputs/models') -> str:
        """
        Sauvegarde le modèle complet pour déploiement.
        
        Args:
            output_dir: Dossier de sortie
            
        Returns:
            str: Nom du modèle sauvegardé
        """
        if self.kmeans is None:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"recipe_clustering_v3_{timestamp}"
        
        # Sauvegarder les composants du pipeline
        joblib.dump(self.scaler, output_dir / f"{model_name}_scaler.pkl")
        joblib.dump(self.pca, output_dir / f"{model_name}_pca.pkl")
        joblib.dump(self.kmeans, output_dir / f"{model_name}_kmeans.pkl")
        
        # Sauvegarder les métadonnées (convertir numpy types en Python types)
        stats = self.get_stats()
        cluster_names_serializable = {str(k): str(v) for k, v in self.cluster_names.items()}
        
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'n_clusters': int(self.n_clusters),
            'pca_variance': float(self.pca_variance),
            'random_state': int(self.random_state),
            'pca_n_components': int(self.pca.n_components_),
            'pca_explained_variance': float(self.pca.explained_variance_ratio_.sum()),
            'cluster_names': cluster_names_serializable,
            'metrics': {
                'silhouette': float(stats['silhouette']),
                'inertia': float(self.kmeans.inertia_),
                'n_recipes': int(len(self.recipes))
            }
        }
        
        with open(output_dir / f"{model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✓ Modèle sauvegardé: {output_dir / model_name}")
        logger.info(f"  - Scaler: {model_name}_scaler.pkl")
        logger.info(f"  - PCA: {model_name}_pca.pkl")
        logger.info(f"  - KMeans: {model_name}_kmeans.pkl")
        logger.info(f"  - Metadata: {model_name}_metadata.json")
        
        return model_name
    
    @classmethod
    def load_model(cls, model_name: str, model_dir: str = 'outputs/models'):
        """
        Charge un modèle sauvegardé pour inference.
        
        Args:
            model_name: Nom du modèle (ex: 'recipe_clustering_v3_20250123_143022')
            model_dir: Dossier contenant le modèle
            
        Returns:
            RecipeClusterer: Instance avec modèle chargé
        """
        model_dir = Path(model_dir)
        
        # Charger métadonnées
        with open(model_dir / f"{model_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Créer instance
        instance = cls(
            n_clusters=metadata['n_clusters'],
            pca_variance=metadata['pca_variance'],
            random_state=metadata['random_state']
        )
        
        # Charger les composants
        instance.scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")
        instance.pca = joblib.load(model_dir / f"{model_name}_pca.pkl")
        instance.kmeans = joblib.load(model_dir / f"{model_name}_kmeans.pkl")
        
        # Convertir cluster_names back to int keys
        instance.cluster_names = {int(k): v for k, v in metadata['cluster_names'].items()}
        instance.n_clusters = metadata['n_clusters']
        
        logger.info(f"\n✓ Modèle chargé: {model_name}")
        logger.info(f"  Silhouette: {metadata['metrics']['silhouette']:.4f}")
        logger.info(f"  K={instance.n_clusters}, PCA={metadata['pca_n_components']} components")
        
        return instance
    
    def predict(self, recipes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les clusters pour de nouvelles recettes.
        
        Args:
            recipes_df: DataFrame avec les features V3
            
        Returns:
            DataFrame avec colonnes 'cluster' et 'cluster_name'
        """
        if self.kmeans is None:
            raise ValueError("Le modèle doit être chargé ou entraîné avant de prédire.")
        
        # Features pour clustering (SANS popularity_score)
        features_for_clustering = [
            'log_minutes', 'time_complexity', 'efficiency', 'health_category'
        ]
        
        X = recipes_df[features_for_clustering]
        
        # Pipeline: Scale → PCA → Predict
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        clusters = self.kmeans.predict(X_pca)
        
        # Ajouter cluster names
        result = recipes_df.copy()
        result['cluster'] = clusters
        result['cluster_name'] = result['cluster'].map(self.cluster_names)
        
        return result


def main():
    """Fonction principale pour exécution en ligne de commande"""
    from config.config import DATA_PROCESSED, OUTPUTS_FIGURES
    
    # Chemins
    features_path = DATA_PROCESSED / 'recipes_features_v3.csv'
    output_path = DATA_PROCESSED / 'recipes_clustered_v3.csv'
    
    # Clustering
    clusterer = RecipeClusterer(n_clusters=None, pca_variance=0.90)
    clusterer.fit(features_path=str(features_path))
    
    # Nommer les clusters
    clusterer.name_clusters()
    
    # Visualiser
    clusterer.plot_clusters(output_dir=str(OUTPUTS_FIGURES))
    
    # Sauvegarder
    clusterer.save_results(str(output_path))
    
    # Afficher les stats
    stats = clusterer.get_stats()
    print("\n" + "=" * 80)
    print("STATISTIQUES FINALES")
    print("=" * 80)
    print(f"Clusters: {stats['n_clusters']}")
    print(f"Recettes: {stats['n_recipes']:,}")
    print(f"Silhouette: {stats['silhouette']:.4f}")
    print(f"Inertia: {stats['inertia']:,.2f}")


if __name__ == '__main__':
    main()
