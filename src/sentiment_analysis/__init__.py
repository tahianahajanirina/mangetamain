from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SentimentAnalyzer:
    """Analyseur de sentiment pour les avis de recettes"""
    
    _model = None
    _tokenizer = None
    _device = None
    _label_map = {"LABEL_0": "Négatif", "LABEL_1": "Neutre", "LABEL_2": "Positif"}
    _model_name = "TahianaAndriambahoaka/sentiment-analysis-food-reviews"
    
    @classmethod
    def _load_model(cls):
        """Charge le modèle et le tokenizer une seule fois depuis Hugging Face"""
        if cls._model is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained(cls._model_name)
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            cls._device = torch.device('cpu')
            cls._model.eval()
            cls._model.to(cls._device)
    
    @classmethod
    def predict_sentiment(cls, text):
        """
        Prédit le sentiment d'un texte
        
        Args:
            text (str): Le texte à analyser
            
        Returns:
            tuple: (sentiment, confidence_score)
                - sentiment (str): "Négatif", "Neutre" ou "Positif"
                - confidence_score (float): Score de confiance entre 0 et 1
        """
        # Charger le modèle si ce n'est pas déjà fait
        cls._load_model()
        
        # Tokenization
        inputs = cls._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Déplacer sur le bon device
        inputs = {k: v.to(cls._device) for k, v in inputs.items()}
        
        # Prédiction
        with torch.no_grad():
            outputs = cls._model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Mapper le label prédit
        label_key = f"LABEL_{predicted_class}"
        return cls._label_map[label_key], confidence
