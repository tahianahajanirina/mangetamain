from functools import wraps

from transformers import pipeline


def load_model(func):
    """Décorateur qui s'assure que le modèle est chargé avant d'exécuter la fonction"""

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if cls._sentiment_analyzer is None:
            # Optimisations pour réduire l'utilisation de la RAM:
            # 1. device=-1 pour forcer CPU (évite allocation GPU inutile)
            # 2. model_kwargs pour optimiser le chargement
            cls._sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=cls._model_name,
                device=-1,  # Force CPU pour économiser RAM
                model_kwargs={
                    "low_cpu_mem_usage": True,  # Optimisation mémoire
                    "use_cache": False,  # Désactive le cache KV pour économiser RAM
                }
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
        # Tronquer le texte si trop long pour économiser RAM
        max_length = 512  # Limite de tokens
        if len(text) > max_length:
            text = text[:max_length]
        
        # Prédiction avec truncation pour éviter erreurs de longueur
        result = cls._sentiment_analyzer(text, truncation=True, max_length=512)

        return result[0]["label"], result[0]["score"]
    
    @classmethod
    def unload_model(cls):
        """
        Décharge le modèle de la mémoire pour libérer de la RAM.
        Utile si le modèle n'est plus nécessaire.
        """
        if cls._sentiment_analyzer is not None:
            del cls._sentiment_analyzer
            cls._sentiment_analyzer = None
            # Force garbage collection
            import gc
            gc.collect()
