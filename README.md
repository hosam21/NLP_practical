# Arabic Text Analysis App

This Streamlit application analyzes Arabic text for sentiment and sarcasm using deep learning models.

## Features

- Sentiment Analysis (Positive, Negative, Neutral)
- Sarcasm Detection (True/False)
- Real-time text analysis
- Beautiful and responsive UI
- Support for Arabic text input

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the trained models in the correct directories:
   - `models/sentiment_dl_model.keras`
   - `models/sarcasm_dl_model.keras`
   - `models/sentiment_label_encoder.joblib`
   - `models/sarcasm_label_encoder.joblib`
   - `dataset/feature_names.npy`
   - `dataset/sentiment_classes.npy`
   - `dataset/sarcasm_classes.npy`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

4. Enter Arabic text in the text area and click "Analyze" to get the results

## Model Information

The app uses LSTM-based deep learning models for both sentiment analysis and sarcasm detection. The models were trained on a dataset of Arabic text and achieve high accuracy in both tasks.

## Requirements

- Python 3.8 or higher
- All dependencies listed in requirements.txt

## Note

Make sure you have enough RAM and a decent CPU/GPU as the models require significant computational resources to run efficiently. 