import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
from preprocessor import ArabicTextPreprocessor
from model_loader import ModelLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Arabic Text Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        direction: rtl;
        text-align: right;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .result-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .sentiment-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .sarcasm-true {
        background-color: #cce5ff;
        border: 1px solid #b8daff;
    }
    .sarcasm-false {
        background-color: #e2e3e5;
        border: 1px solid #d6d8db;
    }
    .confidence-bar {
        height: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        background-color: #007bff;
        transition: width 0.5s ease;
    }
    .title {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        color: #34495e;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .processed-text {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
        color: #2c3e50;  /* Dark blue-gray color for better visibility */
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_components():
    """Initialize the preprocessor and model loader"""
    try:
        preprocessor = ArabicTextPreprocessor(
            stopwords_path="list.txt",
            sentiment_words_path="arabic_sentiment_words.txt"
        )
        model_loader = ModelLoader(
            model_dir="models",
            dataset_dir="dataset"
        )
        return preprocessor, model_loader
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error("Error initializing the application. Please check the logs.")
        return None, None

def display_results(results, processed_text):
    """Display the analysis results in a beautiful format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Analysis")
        sentiment_label = results['sentiment']['label']
        sentiment_prob = results['sentiment']['probability']
        
        sentiment_class = {
            'positive': 'sentiment-positive',
            'negative': 'sentiment-negative',
            'neutral': 'sentiment-neutral'
        }.get(sentiment_label, '')
        
        st.markdown(f"""
            <div class="result-box {sentiment_class}">
                <h3>Sentiment: {sentiment_label}</h3>
                <p>Confidence: {sentiment_prob:.2%}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {sentiment_prob:.0%}"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Sarcasm Detection")
        sarcasm_label = results['sarcasm']['label']
        sarcasm_prob = results['sarcasm']['probability']
        
        sarcasm_class = 'sarcasm-true' if sarcasm_label else 'sarcasm-false'
        
        st.markdown(f"""
            <div class="result-box {sarcasm_class}">
                <h3>Sarcasm: {sarcasm_label}</h3>
                <p>Confidence: {sarcasm_prob:.2%}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {sarcasm_prob:.0%}"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Display processed text
    st.subheader("Processed Text")
    st.markdown(f"""
        <div class="processed-text">
            {processed_text}
        </div>
    """, unsafe_allow_html=True)

def main():
    # Display title and description
    st.markdown('<div class="title">Arabic Text Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze the sentiment and sarcasm in Arabic text</div>', unsafe_allow_html=True)
    
    # Initialize components
    preprocessor, model_loader = initialize_components()
    if preprocessor is None or model_loader is None:
        return
    
    # Text input
    text = st.text_area("Enter Arabic text:", height=150)
    
    if st.button("Analyze"):
        if text:
            try:
                # Preprocess text
                processed_text = preprocessor.preprocess(text)
                
                # Get predictions
                results = model_loader.predict(processed_text)
                
                # Display results
                display_results(results, processed_text)
                
            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                st.error("An error occurred during analysis. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 
