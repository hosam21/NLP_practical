import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_dir: str = "models", dataset_dir: str = "dataset"):
        """
        Initialize the model loader with paths to model and dataset directories.
        
        Args:
            model_dir: Directory containing the model files
            dataset_dir: Directory containing the dataset files
        """
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        
        # Verify directories exist
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Load models
        logger.info("Loading sentiment model...")
        sentiment_model_path = os.path.join(model_dir, "sentiment_svm_model.joblib")
        if not os.path.exists(sentiment_model_path):
            raise FileNotFoundError(f"Sentiment model not found: {sentiment_model_path}")
        self.sentiment_model = joblib.load(sentiment_model_path)
        
        logger.info("Loading sarcasm model...")
        sarcasm_model_path = os.path.join(model_dir, "sarcasm_svm_model.joblib")
        if not os.path.exists(sarcasm_model_path):
            raise FileNotFoundError(f"Sarcasm model not found: {sarcasm_model_path}")
        self.sarcasm_model = joblib.load(sarcasm_model_path)
        
        # Load label encoders
        logger.info("Loading label encoders...")
        sentiment_encoder_path = os.path.join(model_dir, "sentiment_label_encoder.joblib")
        sarcasm_encoder_path = os.path.join(model_dir, "sarcasm_label_encoder.joblib")
        
        if not os.path.exists(sentiment_encoder_path):
            raise FileNotFoundError(f"Sentiment label encoder not found: {sentiment_encoder_path}")
        if not os.path.exists(sarcasm_encoder_path):
            raise FileNotFoundError(f"Sarcasm label encoder not found: {sarcasm_encoder_path}")
            
        self.sentiment_encoder = joblib.load(sentiment_encoder_path)
        self.sarcasm_encoder = joblib.load(sarcasm_encoder_path)
        
        # Load feature names and class mappings
        logger.info("Loading feature names and class mappings...")
        feature_names_path = os.path.join(dataset_dir, "feature_names.npy")
        sentiment_classes_path = os.path.join(dataset_dir, "sentiment_classes.npy")
        sarcasm_classes_path = os.path.join(dataset_dir, "sarcasm_classes.npy")
        
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names not found: {feature_names_path}")
        if not os.path.exists(sentiment_classes_path):
            raise FileNotFoundError(f"Sentiment classes not found: {sentiment_classes_path}")
        if not os.path.exists(sarcasm_classes_path):
            raise FileNotFoundError(f"Sarcasm classes not found: {sarcasm_classes_path}")
            
        self.feature_names = np.load(feature_names_path, allow_pickle=True)
        self.sentiment_classes = np.load(sentiment_classes_path, allow_pickle=True)
        self.sarcasm_classes = np.load(sarcasm_classes_path, allow_pickle=True)
        
        # Initialize TF-IDF vectorizer with the same vocabulary as training
        logger.info("Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.feature_names,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        logger.info("Model loader initialized successfully")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make predictions for sentiment and sarcasm.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment and sarcasm predictions with their probabilities
        """
        # Transform text to TF-IDF features
        features = self.vectorizer.transform([text])
        features_dense = features.toarray()
        
        # Get predictions
        sentiment_pred = self.sentiment_model.predict_proba(features_dense)
        sarcasm_pred = self.sarcasm_model.predict_proba(features_dense)
        
        # Get class labels and probabilities
        sentiment_idx = np.argmax(sentiment_pred)
        sarcasm_idx = np.argmax(sarcasm_pred)
        
        sentiment_label = self.sentiment_classes[sentiment_idx]
        sarcasm_label = self.sarcasm_classes[sarcasm_idx]
        
        sentiment_prob = np.max(sentiment_pred)
        sarcasm_prob = np.max(sarcasm_pred)
        
        return {
            'sentiment': {
                'label': sentiment_label,
                'probability': float(sentiment_prob)
            },
            'sarcasm': {
                'label': sarcasm_label,
                'probability': float(sarcasm_prob)
            }
        } 
