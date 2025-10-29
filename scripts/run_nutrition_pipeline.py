#!/usr/bin/env python3
"""
Pipeline Complet - Classification Nutritionnelle des Recettes

Ce script orchestre l'ensemble du pipeline:
1. Feature Engineering (features nutritionnelles)
2. Entraînement du modèle supervisé
3. Évaluation et métriques
4. Sauvegarde du modèle

Usage:
    python scripts/run_nutrition_pipeline.py [--model-type random_forest|gradient_boosting]
                                              [--skip-features]
                                              [--test-size 0.2]

Auteur: Équipe Data Science
Date: 2025-10-27
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.nutrition_classification_features import (
    NutritionClassificationFeatures,
)
from src.modeling.nutrition_classifier import NutritionClassifier

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Classification Nutritionnelle"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Type de modèle à entraîner (défaut: random_forest)",
    )

    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip le feature engineering (utilise fichier existant)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du test set (défaut: 0.2)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Répertoire de sortie pour le modèle (défaut: outputs/models)",
    )

    return parser.parse_args()


def print_header(title: str):
    """Affiche un en-tête formaté."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Affiche un titre de section."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80 + "\n")


def run_feature_engineering(skip: bool = False) -> str:
    """
    Exécute le feature engineering.

    Args:
        skip: Si True, skip cette étape

    Returns:
        Chemin vers le fichier de features
    """
    features_path = "data/processed/recipes_nutrition_features.csv"

    if skip:
        if Path(features_path).exists():
            logger.info(f"⏭️  Feature engineering skipped (utilise {features_path})")
            return features_path
        else:
            logger.warning(
                f"Fichier {features_path} introuvable, exécution du feature engineering..."
            )

    print_section("ÉTAPE 1/4 - FEATURE ENGINEERING")

    engineer = NutritionClassificationFeatures()

    input_path = "data/processed/recipes_clean.csv"

    # Vérifier que le fichier d'entrée existe
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Fichier {input_path} introuvable. "
            f"Exécutez d'abord: python scripts/clean_data.py"
        )

    # Créer les features
    df, stats = engineer.build_features(input_path, features_path)

    logger.info(f"✅ Features créées: {len(df):,} recettes, {df.shape[1]} colonnes")

    return features_path


def train_model(
    features_path: str, model_type: str, test_size: float, random_state: int
) -> NutritionClassifier:
    """
    Entraîne le modèle de classification.

    Args:
        features_path: Chemin vers les features
        model_type: Type de modèle
        test_size: Taille du test set
        random_state: Seed

    Returns:
        Modèle entraîné
    """
    print_section("ÉTAPE 2/4 - ENTRAÎNEMENT DU MODÈLE")

    # Initialiser le classificateur
    classifier = NutritionClassifier(model_type=model_type, random_state=random_state)

    # Charger les données
    X, y = classifier.load_data(features_path)

    # Split train/test
    X_train, X_test, y_train, y_test = classifier.prepare_train_test(
        X, y, test_size=test_size
    )

    # Normaliser
    X_train_scaled, X_test_scaled = classifier.scale_features(X_train, X_test)

    # Entraîner avec validation croisée
    classifier.train(X_train_scaled, y_train)

    # Stocker les données de test pour évaluation
    classifier.X_test = X_test_scaled
    classifier.y_test = y_test

    logger.info("✅ Modèle entraîné avec succès")

    return classifier


def evaluate_model(classifier: NutritionClassifier) -> dict:
    """
    Évalue le modèle sur le test set.

    Args:
        classifier: Modèle entraîné

    Returns:
        Dictionnaire de métriques
    """
    print_section("ÉTAPE 3/4 - ÉVALUATION DU MODÈLE")

    # Évaluation sur test set
    metrics = classifier.evaluate(
        classifier.X_test, classifier.y_test, dataset_name="Test"
    )

    # Feature importance
    importance_df = classifier.get_feature_importance(top_n=15)

    logger.info("✅ Évaluation terminée")

    return metrics


def save_model_and_results(
    classifier: NutritionClassifier, metrics: dict, output_dir: str
) -> str:
    """
    Sauvegarde le modèle et les résultats.

    Args:
        classifier: Modèle entraîné
        metrics: Métriques d'évaluation
        output_dir: Répertoire de sortie

    Returns:
        Nom du modèle sauvegardé
    """
    print_section("ÉTAPE 4/4 - SAUVEGARDE DU MODÈLE")

    # Sauvegarder le modèle
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = classifier.save_model(output_dir, version=version)

    # Sauvegarder les métriques dans un fichier séparé
    metrics_path = Path(output_dir) / f"{model_name}_evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  ✓ Métriques sauvegardées: {metrics_path}")

    logger.info(f"✅ Modèle sauvegardé: {model_name}")

    return model_name


def print_summary(
    model_name: str, metrics: dict, model_type: str, features_path: str, output_dir: str
):
    """Affiche un résumé final."""
    print_header("RÉSUMÉ DU PIPELINE")

    print("📊 PERFORMANCE DU MODÈLE")
    print(f"  Type de modèle:     {model_type}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")

    print("\n📁 FICHIERS GÉNÉRÉS")
    print(f"  Features:           {features_path}")
    print(f"  Modèle:             {output_dir}/{model_name}_model.pkl")
    print(f"  Scaler:             {output_dir}/{model_name}_scaler.pkl")
    print(f"  Métadonnées:        {output_dir}/{model_name}_metadata.json")
    print(f"  Évaluation:         {output_dir}/{model_name}_evaluation.json")

    print("\n🎯 PERFORMANCE PAR CLASSE")
    for class_name, class_metrics in metrics["per_class"].items():
        print(
            f"  {class_name:15} - F1: {class_metrics['f1']:.4f}, "
            f"Support: {class_metrics['support']:>6,}"
        )

    print("\n✨ PROCHAINES ÉTAPES")
    print("  1. Analyser les métriques dans le fichier d'évaluation")
    print("  2. Utiliser le modèle pour prédire de nouvelles recettes")
    print("  3. Intégrer dans l'application Streamlit")
    print("  4. Affiner les seuils de catégorisation si nécessaire")

    print("\n" + "=" * 80 + "\n")


def main():
    """Point d'entrée principal du pipeline."""
    args = parse_args()

    print_header("PIPELINE DE CLASSIFICATION NUTRITIONNELLE")

    start_time = datetime.now()

    try:
        # 1. Feature Engineering
        features_path = run_feature_engineering(skip=args.skip_features)

        # 2. Entraînement du modèle
        classifier = train_model(
            features_path=features_path,
            model_type=args.model_type,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        # 3. Évaluation
        metrics = evaluate_model(classifier)

        # 4. Sauvegarde
        model_name = save_model_and_results(
            classifier=classifier, metrics=metrics, output_dir=args.output_dir
        )

        # Résumé final
        elapsed = datetime.now() - start_time

        print_summary(
            model_name=model_name,
            metrics=metrics,
            model_type=args.model_type,
            features_path=features_path,
            output_dir=args.output_dir,
        )

        print(f"⏱️  Durée totale: {elapsed.total_seconds():.1f}s")
        print("✅ Pipeline terminé avec succès!\n")

        return 0

    except Exception as e:
        logger.error(
            f"❌ Erreur lors de l'exécution du pipeline: {str(e)}", exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
