"""
Modèle de Classification Nutritionnelle Supervisée

Ce module implémente un classificateur multi-classes pour catégoriser
les recettes en 4 catégories nutritionnelles:
- 0: Very Healthy
- 1: Healthy
- 2: Moderate
- 3: Indulgent

Auteur: Équipe Data Science
Date: 2025-10-27
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NutritionClassifier:
    """
    Classificateur supervisé pour catégoriser les recettes par nutrition.

    Architecture:
    - Feature scaling avec StandardScaler
    - Random Forest ou Gradient Boosting
    - Validation croisée stratifiée
    - Sauvegarde modèle + métadonnées
    """

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        """
        Initialise le classificateur.

        Args:
            model_type: 'random_forest' ou 'gradient_boosting'
            random_state: Seed pour reproductibilité
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.class_names = ["Very Healthy", "Healthy", "Moderate", "Indulgent"]
        self.metadata = {}

        # Initialiser le modèle
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",  # Important pour classes déséquilibrées
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge les features et la cible.

        Args:
            filepath: Chemin vers recipes_nutrition_features.csv

        Returns:
            Tuple (X features, y target)
        """
        logger.info(f"Chargement des données depuis {filepath}...")
        df = pd.read_csv(filepath)

        # Features pour le modèle (exclure id, name, target)
        feature_cols = [
            # Nutrition brute
            "calories",
            "total_fat",
            "saturated_fat",
            "sodium",
            "carbohydrates",
            "sugar",
            "protein",
            # PDV%
            "calories_pdv",
            "fat_pdv",
            "saturated_fat_pdv",
            "sodium_pdv",
            "carbs_pdv",
            "sugar_pdv",
            "protein_pdv",
            # Ratios
            "protein_density",
            "sugar_ratio",
            "sodium_density",
            "saturated_fat_ratio",
            "macro_balance",
            # Score
            "health_score",
            # Features binaires
            "is_low_calorie",
            "is_high_protein",
            "is_low_fat",
            "is_low_sodium",
            "is_low_sugar",
            # Complexité et interactions
            "complexity_score",
            "calorie_protein_interaction",
            "fat_sodium_interaction",
            # Autres
            "n_steps",
            "n_ingredients",
            "minutes",
        ]

        X = df[feature_cols].copy()
        y = df["nutrition_category"].copy()

        self.feature_names = feature_cols

        logger.info(f"  ✓ {len(X):,} échantillons")
        logger.info(f"  ✓ {len(feature_cols)} features")
        logger.info("  Distribution des classes:")
        for cat_id, count in y.value_counts().sort_index().items():
            pct = (count / len(y)) * 100
            logger.info(
                f"    {cat_id} - {self.class_names[cat_id]}: {count:,} ({pct:.1f}%)"
            )

        return X, y

    def prepare_train_test(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split stratifié train/test.

        Args:
            X: Features
            y: Target
            test_size: Proportion du test set

        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        logger.info(
            f"\nSéparation train/test ({int((1-test_size)*100)}/{int(test_size*100)})..."
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,  # Préserve la distribution des classes
        )

        logger.info(f"  Train: {len(X_train):,} échantillons")
        logger.info(f"  Test:  {len(X_test):,} échantillons")

        return X_train, X_test, y_train, y_test

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalise les features avec StandardScaler.

        Args:
            X_train: Features d'entraînement
            X_test: Features de test

        Returns:
            Tuple (X_train_scaled, X_test_scaled)
        """
        logger.info("\nNormalisation des features...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("  ✓ Features normalisées (μ=0, σ=1)")

        return X_train_scaled, X_test_scaled

    def train(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Entraîne le modèle avec validation croisée.

        Args:
            X_train: Features d'entraînement normalisées
            y_train: Target d'entraînement
        """
        logger.info(f"\nEntraînement du modèle ({self.model_type})...")

        # Validation croisée stratifiée (5-fold)
        logger.info("  Validation croisée (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        logger.info(f"    F1-Score CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Entraînement final sur tout le train set
        logger.info("  Entraînement final...")
        self.model.fit(X_train, y_train)

        logger.info("  ✓ Modèle entraîné")

        # Stocker métadonnées
        self.metadata["cv_f1_mean"] = float(cv_scores.mean())
        self.metadata["cv_f1_std"] = float(cv_scores.std())

    def evaluate(
        self, X_test: np.ndarray, y_test: pd.Series, dataset_name: str = "Test"
    ) -> Dict:
        """
        Évalue le modèle sur un dataset.

        Args:
            X_test: Features de test
            y_test: Target de test
            dataset_name: Nom du dataset (pour logging)

        Returns:
            Dictionnaire de métriques
        """
        logger.info(f"\nÉvaluation sur {dataset_name} Set...")

        # Prédictions
        y_pred = self.model.predict(X_test)
        _ = self.model.predict_proba(X_test)

        # Métriques globales
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")

        # Rapport par classe
        logger.info("\n  Classification Report:")
        report = classification_report(
            y_test, y_pred, target_names=self.class_names, zero_division=0
        )
        print(report)

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\n  Matrice de Confusion:")
        logger.info(
            f"  {'':15} " + " ".join(f"{name:>12}" for name in self.class_names)
        )
        for i, row in enumerate(cm):
            logger.info(
                f"  {self.class_names[i]:15} " + " ".join(f"{val:>12}" for val in row)
            )

        # Métriques par classe
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "per_class": {
                self.class_names[i]: {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1": float(f1_per_class[i]),
                    "support": int(support_per_class[i]),
                }
                for i in range(len(self.class_names))
            },
            "confusion_matrix": cm.tolist(),
        }

        return metrics

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Récupère l'importance des features.

        Args:
            top_n: Nombre de features à retourner

        Returns:
            DataFrame avec features et importances
        """
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Le modèle ne supporte pas feature_importances_")
            return pd.DataFrame()

        importance = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        logger.info(f"\nTop {top_n} Features Importantes:")
        for idx, row in importance.iterrows():
            logger.info(f"  {row['feature']:30} {row['importance']:.4f}")

        return importance

    def save_model(self, output_dir: str, version: str = None) -> str:
        """
        Sauvegarde le modèle complet.

        Args:
            output_dir: Répertoire de sortie
            version: Version du modèle (auto si None)

        Returns:
            Nom du fichier sauvegardé
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_name = f"nutrition_classifier_{version}"

        # Sauvegarder scaler
        scaler_path = output_path / f"{model_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"  ✓ Scaler sauvegardé: {scaler_path}")

        # Sauvegarder modèle
        model_path = output_path / f"{model_name}_model.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"  ✓ Modèle sauvegardé: {model_path}")

        # Sauvegarder métadonnées
        metadata = {
            "model_name": model_name,
            "model_type": self.model_type,
            "version": version,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "created_at": datetime.now().isoformat(),
            **self.metadata,
        }

        metadata_path = output_path / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Métadonnées sauvegardées: {metadata_path}")

        return model_name

    def load_model(self, model_dir: str, model_name: str) -> None:
        """
        Charge un modèle sauvegardé.

        Args:
            model_dir: Répertoire contenant le modèle
            model_name: Nom du modèle (sans extension)
        """
        model_path = Path(model_dir)

        # Charger scaler
        scaler_path = model_path / f"{model_name}_scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        logger.info("  ✓ Scaler chargé")

        # Charger modèle
        model_file = model_path / f"{model_name}_model.pkl"
        self.model = joblib.load(model_file)
        logger.info("  ✓ Modèle chargé")

        # Charger métadonnées
        metadata_path = model_path / f"{model_name}_metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]
        logger.info("  ✓ Métadonnées chargées")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fait des prédictions.

        Args:
            X: Features à prédire

        Returns:
            Tuple (classes prédites, probabilités)
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities

    def predict_single(self, recipe_features: Dict) -> Dict:
        """
        Prédit la catégorie nutritionnelle d'une seule recette.

        Args:
            recipe_features: Dictionnaire avec les features

        Returns:
            Dictionnaire avec prédiction et probabilités
        """
        # Créer DataFrame avec les features
        X = pd.DataFrame([recipe_features])[self.feature_names]

        # Prédire
        pred, proba = self.predict(X)

        result = {
            "category_id": int(pred[0]),
            "category_name": self.class_names[pred[0]],
            "confidence": float(proba[0][pred[0]]),
            "probabilities": {
                self.class_names[i]: float(proba[0][i])
                for i in range(len(self.class_names))
            },
        }

        return result


def main():
    """Point d'entrée pour test rapide."""
    classifier = NutritionClassifier(model_type="random_forest")

    # Charger données
    X, y = classifier.load_data("data/processed/recipes_nutrition_features.csv")

    # Train/test split
    X_train, X_test, y_train, y_test = classifier.prepare_train_test(X, y)

    # Normaliser
    X_train_scaled, X_test_scaled = classifier.scale_features(X_train, X_test)

    # Entraîner
    classifier.train(X_train_scaled, y_train)

    # Évaluer
    _ = classifier.evaluate(X_test_scaled, y_test)

    # Feature importance
    classifier.get_feature_importance(top_n=15)

    # Sauvegarder
    model_name = classifier.save_model("outputs/models")

    print(f"\n✓ Modèle sauvegardé: {model_name}")


if __name__ == "__main__":
    main()
