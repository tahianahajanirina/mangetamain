"""
Clustering des Utilisateurs - K-Means

Ce module implémente le clustering K-Means sur les profils utilisateurs
pour segmenter les utilisateurs selon leurs préférences culinaires.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class UserClusterer:
    """Clustering K-Means pour profils utilisateurs"""

    def __init__(
        self, n_clusters: int = None, pca_variance: float = 0.95, random_state: int = 42
    ):
        """
        Args:
            n_clusters: Nombre de clusters (None = détection automatique)
            pca_variance: Variance à conserver pour PCA (default: 95%)
            random_state: Seed pour reproductibilité
        """
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.random_state = random_state

        self.users = None
        self.features_scaled = None
        self.features_pca = None
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.cluster_names = {}

    def load_features(self, features_path: str) -> pd.DataFrame:
        """Charge les profils utilisateurs"""
        logger.info("Chargement des profils utilisateurs...")
        users = pd.read_csv(features_path)

        logger.info(f"Utilisateurs: {len(users):,}")
        logger.info(f"Features: {len(users.columns) - 1}")  # -1 pour user_id

        self.users = users
        return users

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalisation + PCA des features"""
        logger.info("=" * 80)
        logger.info("NORMALISATION + PCA")
        logger.info("=" * 80)

        # Exclure user_id et colonnes non-numériques
        feature_cols = [
            col
            for col in self.users.columns
            if col != "user_id" and self.users[col].dtype in ["float64", "int64"]
        ]

        logger.info(f"\nFeatures pour clustering ({len(feature_cols)}):")
        logger.info(f"  {', '.join(feature_cols[:5])}...")

        features_raw = self.users[feature_cols].copy()

        # Normalisation
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features_raw)

        logger.info("\nNormalisation terminée")
        logger.info(f"  Mean: {self.features_scaled.mean():.6f}")
        logger.info(f"  Std: {self.features_scaled.std():.6f}")

        # PCA
        self.pca = PCA(n_components=self.pca_variance)
        self.features_pca = self.pca.fit_transform(self.features_scaled)

        logger.info("\nPCA appliquée:")
        logger.info(f"  Composantes: {self.pca.n_components_}")
        logger.info(
            f"  Variance expliquée: {self.pca.explained_variance_ratio_.sum():.1%}"
        )
        logger.info(f"  Shape: {self.features_pca.shape}")

        logger.info("\nVariance par composante:")
        for i, var in enumerate(self.pca.explained_variance_ratio_[:5], 1):
            logger.info(f"  PC{i}: {var:.1%}")

        return self.features_scaled, self.features_pca

    def find_optimal_k(self, k_range: range = range(2, 11)) -> int:
        """Trouve le K optimal via silhouette score"""
        logger.info("=" * 80)
        logger.info("DETERMINATION DU K OPTIMAL")
        logger.info("=" * 80)

        logger.info(f"\nTest de K = {k_range.start} à {k_range.stop - 1}...\n")

        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k, random_state=self.random_state, n_init=20, max_iter=500
            )
            kmeans.fit(self.features_pca)

            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(self.features_pca, kmeans.labels_)
            silhouette_scores.append(sil_score)

            logger.info(
                f"  K={k}: Inertia={kmeans.inertia_:>10,.2f}, Silhouette={sil_score:.4f}"
            )

        best_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)

        logger.info(f"\n\nK OPTIMAL: {best_k}")
        logger.info(f"Silhouette score: {max_silhouette:.4f}")

        return best_k

    def fit(self, features_path: str = None) -> "UserClusterer":
        """Pipeline complet de clustering"""
        logger.info("=" * 80)
        logger.info("CLUSTERING UTILISATEURS")
        logger.info("=" * 80)

        # 1. Charger si nécessaire
        if features_path and self.users is None:
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

        logger.info(f"\nEntraînement sur {len(self.features_pca):,} utilisateurs...")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=50,
            max_iter=500,
        )
        clusters = self.kmeans.fit_predict(self.features_pca)

        self.users["cluster"] = clusters

        logger.info("\nK-MEANS ENTRAÎNÉ:")
        logger.info(f"  K: {self.n_clusters}")
        logger.info(f"  Inertia: {self.kmeans.inertia_:,.2f}")

        # Silhouette
        final_silhouette = silhouette_score(self.features_pca, clusters)
        logger.info(f"  Silhouette: {final_silhouette:.4f}")

        # Distribution
        logger.info("\nDISTRIBUTION:")
        cluster_counts = self.users["cluster"].value_counts().sort_index()
        for cluster_id in range(self.n_clusters):
            count = cluster_counts[cluster_id]
            pct = 100 * count / len(self.users)
            logger.info(
                f"  Cluster {cluster_id}: {count:>7,} utilisateurs ({pct:>5.1f}%)"
            )

        return self

    def name_clusters(self) -> Dict[int, str]:
        """Nomme intelligemment les clusters basé sur leurs profils"""
        logger.info("=" * 80)
        logger.info("NOMMAGE DES CLUSTERS")
        logger.info("=" * 80)

        # Colonnes à analyser pour nommage
        key_features = [
            "avg_rating",
            "avg_minutes",
            "avg_calories",
            "avg_n_steps",
            "n_interactions",
        ]

        cluster_profiles = self.users.groupby("cluster")[key_features].mean()

        def name_cluster(cluster_id, profile):
            """Nommage basé sur les préférences utilisateur"""
            names = []

            # Niveau d'activité
            if profile["n_interactions"] > 50:
                names.append("Très Actifs")
            elif profile["n_interactions"] > 20:
                names.append("Actifs")
            elif profile["n_interactions"] < 10:
                names.append("Occasionnels")

            # Préférence temps
            if profile["avg_minutes"] < 30:
                names.append("Rapides")
            elif profile["avg_minutes"] > 60:
                names.append("Patients")

            # Préférence calories
            if profile["avg_calories"] < 300:
                names.append("Légers")
            elif profile["avg_calories"] > 500:
                names.append("Gourmands")

            # Exigence (rating)
            if profile["avg_rating"] > 4.5:
                names.append("Exigeants")
            elif profile["avg_rating"] < 3.5:
                names.append("Tolérants")

            # Éliminer doublons
            names = list(dict.fromkeys(names))

            # Retourner max 2 caractéristiques
            if len(names) >= 2:
                return " & ".join(names[:2])
            elif len(names) == 1:
                return names[0]
            else:
                return f"Groupe {cluster_id + 1}"

        cluster_counts = self.users["cluster"].value_counts().sort_index()

        for cluster_id in range(self.n_clusters):
            profile_dict = cluster_profiles.loc[cluster_id].to_dict()
            cluster_name = name_cluster(cluster_id, profile_dict)
            self.cluster_names[cluster_id] = cluster_name

        logger.info("\nCLUSTERS NOMMÉS:")
        for cluster_id, name in self.cluster_names.items():
            count = cluster_counts[cluster_id]
            pct = 100 * count / len(self.users)
            profile = cluster_profiles.loc[cluster_id]
            logger.info(
                f'  {cluster_id}. "{name}" ({count:,} - {pct:.1f}%) '
                f"[{profile['n_interactions']:.0f} interactions, "
                f"rating:{profile['avg_rating']:.1f}, "
                f"{profile['avg_minutes']:.0f}min, "
                f"{profile['avg_calories']:.0f}cal]"
            )

        # Appliquer les noms
        self.users["cluster_name"] = self.users["cluster"].map(self.cluster_names)

        return self.cluster_names

    def save_results(self, output_path: str) -> None:
        """Sauvegarde les résultats du clustering"""
        logger.info("=" * 80)
        logger.info("EXPORT FINAL")
        logger.info("=" * 80)

        logger.info(f"\nFichier exporté: {output_path}")
        self.users.to_csv(output_path, index=False)

        file_size = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"Shape: {self.users.shape}")
        logger.info(f"Taille: {file_size:.2f} MB")

        logger.info("\nDataset avec clusters utilisateurs prêt!")

    def plot_clusters(self, output_dir: str = None) -> None:
        """Visualise les clusters en 2D avec PCA"""
        logger.info("=" * 80)
        logger.info("VISUALISATION PCA 2D")
        logger.info("=" * 80)

        # PCA 2D pour visualisation
        pca_viz = PCA(n_components=2)
        features_pca_2d = pca_viz.fit_transform(self.features_scaled)

        logger.info(
            f"\nVariance expliquée: {pca_viz.explained_variance_ratio_.sum():.1%}"
        )

        fig, ax = plt.subplots(figsize=(14, 10))

        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        clusters = self.users["cluster"].values

        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            cluster_name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            ax.scatter(
                features_pca_2d[mask, 0],
                features_pca_2d[mask, 1],
                label=f"{cluster_name} ({mask.sum():,})",
                alpha=0.6,
                s=30,
                color=colors[cluster_id],
                edgecolors="none",
            )

        # Centroïdes en PCA 2D
        feature_cols = [
            col
            for col in self.users.columns
            if col not in ["user_id", "cluster", "cluster_name"]
            and self.users[col].dtype in ["float64", "int64"]
        ]
        centroids_scaled = self.scaler.transform(
            self.users.groupby("cluster")[feature_cols].mean()
        )
        centroids_pca_2d = pca_viz.transform(centroids_scaled)
        ax.scatter(
            centroids_pca_2d[:, 0],
            centroids_pca_2d[:, 1],
            c="red",
            marker="*",
            s=1000,
            edgecolors="black",
            linewidth=2,
            label="Centroïdes",
            zorder=10,
        )

        # Calculer silhouette pour le titre
        final_silhouette = silhouette_score(self.features_pca, clusters)

        ax.set_xlabel(
            f"PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel(
            f"PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_title(
            f"Clustering Utilisateurs (K={self.n_clusters}, Silhouette={final_silhouette:.3f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / "user_clusters_pca.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"\nVisualisation sauvegardée: {output_path}")

        plt.close()

    def get_stats(self) -> Dict:
        """Retourne les statistiques du clustering"""
        if self.kmeans is None:
            raise ValueError("Clustering non effectué. Appelez fit() d'abord.")

        clusters = self.users["cluster"].values

        return {
            "n_clusters": self.n_clusters,
            "n_users": len(self.users),
            "inertia": self.kmeans.inertia_,
            "silhouette": silhouette_score(self.features_pca, clusters),
            "cluster_names": self.cluster_names,
            "cluster_distribution": self.users["cluster"].value_counts().to_dict(),
        }

    def save_model(self, output_dir: str = "outputs/models") -> str:
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
        model_name = f"user_clustering_{timestamp}"

        # Sauvegarder les composants du pipeline
        joblib.dump(self.scaler, output_dir / f"{model_name}_scaler.pkl")
        joblib.dump(self.pca, output_dir / f"{model_name}_pca.pkl")
        joblib.dump(self.kmeans, output_dir / f"{model_name}_kmeans.pkl")

        # Sauvegarder les métadonnées
        stats = self.get_stats()
        cluster_names_serializable = {
            str(k): str(v) for k, v in self.cluster_names.items()
        }

        metadata = {
            "model_name": model_name,
            "timestamp": timestamp,
            "n_clusters": int(self.n_clusters),
            "pca_variance": float(self.pca_variance),
            "random_state": int(self.random_state),
            "pca_n_components": int(self.pca.n_components_),
            "pca_explained_variance": float(self.pca.explained_variance_ratio_.sum()),
            "cluster_names": cluster_names_serializable,
            "metrics": {
                "silhouette": float(stats["silhouette"]),
                "inertia": float(self.kmeans.inertia_),
                "n_users": int(len(self.users)),
            },
        }

        with open(output_dir / f"{model_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Modèle sauvegardé: {output_dir / model_name}")
        logger.info(f"  - Scaler: {model_name}_scaler.pkl")
        logger.info(f"  - PCA: {model_name}_pca.pkl")
        logger.info(f"  - KMeans: {model_name}_kmeans.pkl")
        logger.info(f"  - Metadata: {model_name}_metadata.json")

        return model_name

    @classmethod
    def load_model(cls, model_name: str, model_dir: str = "outputs/models"):
        """
        Charge un modèle sauvegardé pour inference.

        Args:
            model_name: Nom du modèle (ex: 'user_clustering_20250123_143022')
            model_dir: Dossier contenant le modèle

        Returns:
            UserClusterer: Instance avec modèle chargé
        """
        model_dir = Path(model_dir)

        # Charger métadonnées
        with open(model_dir / f"{model_name}_metadata.json", "r") as f:
            metadata = json.load(f)

        # Créer instance
        instance = cls(
            n_clusters=metadata["n_clusters"],
            pca_variance=metadata["pca_variance"],
            random_state=metadata["random_state"],
        )

        # Charger les composants
        instance.scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")
        instance.pca = joblib.load(model_dir / f"{model_name}_pca.pkl")
        instance.kmeans = joblib.load(model_dir / f"{model_name}_kmeans.pkl")

        # Convertir cluster_names back to int keys
        instance.cluster_names = {
            int(k): v for k, v in metadata["cluster_names"].items()
        }
        instance.n_clusters = metadata["n_clusters"]

        logger.info(f"\n✓ Modèle chargé: {model_name}")
        logger.info(f"  Silhouette: {metadata['metrics']['silhouette']:.4f}")
        logger.info(
            f"  K={instance.n_clusters}, PCA={metadata['pca_n_components']} components"
        )

        return instance

    def predict(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les clusters pour de nouveaux utilisateurs.

        Args:
            users_df: DataFrame avec les features utilisateurs

        Returns:
            DataFrame avec colonnes 'cluster' et 'cluster_name'
        """
        if self.kmeans is None:
            raise ValueError("Le modèle doit être chargé ou entraîné avant de prédire.")

        # Features (exclure user_id)
        feature_cols = [
            col
            for col in users_df.columns
            if col != "user_id" and users_df[col].dtype in ["float64", "int64"]
        ]

        X = users_df[feature_cols]

        # Pipeline: Scale → PCA → Predict
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        clusters = self.kmeans.predict(X_pca)

        # Ajouter cluster names
        result = users_df.copy()
        result["cluster"] = clusters
        result["cluster_name"] = result["cluster"].map(self.cluster_names)

        return result


def main():
    """Fonction principale pour exécution en ligne de commande"""
    from config.config import DATA_PROCESSED, OUTPUTS_FIGURES

    # Chemins
    features_path = DATA_PROCESSED / "users_profiles.csv"
    output_path = DATA_PROCESSED / "users_clustered.csv"

    # Clustering
    clusterer = UserClusterer(n_clusters=None, pca_variance=0.95)
    clusterer.fit(features_path=str(features_path))

    # Nommer les clusters
    clusterer.name_clusters()

    # Visualiser
    clusterer.plot_clusters(output_dir=str(OUTPUTS_FIGURES))

    # Sauvegarder
    clusterer.save_results(str(output_path))

    # Sauvegarder le modèle
    model_name = clusterer.save_model()

    # Afficher les stats
    stats = clusterer.get_stats()
    print("\n" + "=" * 80)
    print("STATISTIQUES FINALES")
    print("=" * 80)
    print(f"Clusters: {stats['n_clusters']}")
    print(f"Utilisateurs: {stats['n_users']:,}")
    print(f"Silhouette: {stats['silhouette']:.4f}")
    print(f"Inertia: {stats['inertia']:,.2f}")
    print(f"\nModèle sauvegardé: {model_name}")


if __name__ == "__main__":
    main()
