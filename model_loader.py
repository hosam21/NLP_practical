import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_dir: str, dataset_dir: str):
        """
        Initialize the model loader with paths to model and dataset directories.
        
        Args:
            model_dir: Directory containing the model files
            dataset_dir: Directory containing the dataset files
        """
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        
        # Load models
        logger.info("Loading sentiment model...")
        self.sentiment_model = joblib.load(f'{model_dir}/sentiment_svm_model.joblib')
        
        logger.info("Loading sarcasm model...")
        self.sarcasm_model = joblib.load(f'{model_dir}/sarcasm_svm_model.joblib')
        
        # Load label encoders
        logger.info("Loading label encoders...")
        self.sentiment_encoder = joblib.load(f'{model_dir}/sentiment_label_encoder.joblib')
        self.sarcasm_encoder = joblib.load(f'{model_dir}/sarcasm_label_encoder.joblib')
        
        # Load feature names and class mappings
        logger.info("Loading feature names and class mappings...")
        self.feature_names = np.load(f'{dataset_dir}/feature_names.npy', allow_pickle=True)
        self.sentiment_classes = np.load(f'{dataset_dir}/sentiment_classes.npy', allow_pickle=True)
        self.sarcasm_classes = np.load(f'{dataset_dir}/sarcasm_classes.npy', allow_pickle=True)
        
        # Initialize TF-IDF vectorizer
        logger.info("Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.vectorizer.fit(self.feature_names)
        
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
