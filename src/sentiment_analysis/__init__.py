from functools import wraps

from transformers import pipeline


def load_model(func):
    """Décorateur qui s'assure que le modèle est chargé avant d'exécuter la fonction"""

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if cls._sentiment_analyzer is None:
            cls._sentiment_analyzer = pipeline(
                "sentiment-analysis", model=cls._model_name
            )
        return func(cls, *args, **kwargs)

    return wrapper


class SentimentAnalyzer:
    """Analyseur de sentiment pour les avis de recettes"""

    _sentiment_analyzer = None
    _model_name = "TahianaAndriambahoaka/sentiment-analysis-food-reviews"

    @classmethod
    @load_model
    def predict_sentiment(cls, text):
        """
        Prédit le sentiment d'un texte

        Args:
            text (str): Le texte à analyser

        Returns:
            tuple: (label, confidence_score)
                - label (str): "negative", "neutral" ou "positive"
                - confidence_score (float): Score de confiance entre 0 et 1
        """
        # Prédiction
        result = cls._sentiment_analyzer(text)

        return result[0]["label"], result[0]["score"]

    @classmethod
    def unload_model(cls):
        """
        Décharge le modèle de sentiment pour libérer la RAM (~500MB).
        Utilisé pour Streamlit Cloud avec limite de RAM.
        """
        if cls._sentiment_analyzer is not None:
            del cls._sentiment_analyzer
            cls._sentiment_analyzer = None
            import gc

            gc.collect()
