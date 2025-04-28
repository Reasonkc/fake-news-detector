#!/usr/bin/env python3
"""
inference_api.py

A Flask-based API server for fake news detection inference.
Provides two endpoints:
  - POST /predict_baseline: TF-IDF + Logistic Regression/Random Forest
  - POST /predict_bert: BERT fine-tuned model

Usage:
  1. Ensure `baseline.joblib` and `bert_output/` exist in the working directory.
  2. Install requirements:
     pip install flask joblib nltk torch transformers
  3. Run:
     python inference_api.py

Example request:
  curl -X POST http://localhost:5000/predict_baseline \
       -H "Content-Type: application/json" \
       -d '{"text": "Breaking news: ..."}'
"""
from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load baseline model and vectorizer
data = joblib.load('baseline.joblib')
baseline_model = data['model']
baseline_vec = data['vectorizer']

# Load XGBoost model and vectorizer
try:
    xgb_data = joblib.load('xgb_model.joblib')
    xgb_model = xgb_data['model']
    xgb_vec = xgb_data['vectorizer']
    app.logger.info("Loaded XGBoost model from xgb_model.joblib")
except Exception as e:
    xgb_model, xgb_vec = None, None
    app.logger.warning(f"Could not load xgb_model.joblib: {e}")

# Load BERT tokenizer and model
# bert_dir = 'bert_output'
# bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
# bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
# bert_model.eval()

# Load DistilBERT (sentiment model)
try:
    distilbert_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    app.logger.info("Loaded distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    distilbert_classifier = None
    app.logger.warning(f"Could not load DistilBERT pipeline: {e}")

# Load Fake News Detection BERT model
try:
    fake_news_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-fake-news-detection")
    fake_news_model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-fake-news-detection")
    fake_news_pipeline = pipeline("text-classification", model=fake_news_model, tokenizer=fake_news_tokenizer)
    app.logger.info("Loaded mrm8488/bert-tiny-fake-news-detection model")
except Exception as e:
    fake_news_pipeline = None
    app.logger.warning(f"Could not load Fake News Detection model: {e}")

# Prepare NLP tools for baseline preprocessing
nltk.download('stopwords')
nltk_downloads = ['wordnet', 'omw-1.4']
for pkg in nltk_downloads:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Lowercase, remove digits/punctuation, lemmatize, drop stopwords."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stops]
    return ' '.join(tokens)

@app.route('/predict_baseline', methods=['POST'])
def predict_baseline():
    """Run baseline model inference."""
    payload = request.get_json(force=True)
    text = payload.get('text', '')
    clean = preprocess_text(text)
    vec_text = baseline_vec.transform([clean])
    pred = baseline_model.predict(vec_text)[0]
    prob = baseline_model.predict_proba(vec_text)[0].tolist()
    label = 'REAL' if pred == 1 else 'FAKE'
    return jsonify({'label': label, 'probabilities': prob})

@app.route('/predict_xgb', methods=['POST'])
def predict_xgb():
    """Run XGBoost model inference."""
    if xgb_model is None or xgb_vec is None:
        return jsonify({'error': 'XGBoost model not loaded.'}), 500

    payload = request.get_json(force=True)
    text = payload.get('text', '')
    clean = preprocess_text(text)
    vec_text = xgb_vec.transform([clean])
    pred = xgb_model.predict(vec_text)[0]
    prob = xgb_model.predict_proba(vec_text)[0].tolist()
    label = 'REAL' if pred == 1 else 'FAKE'
    return jsonify({'label': label, 'probabilities': prob})

@app.route('/predict_distilbert', methods=['POST'])
def predict_distilbert():
    """Run DistilBERT (sentiment) model inference."""
    if distilbert_classifier is None:
        return jsonify({'error': 'DistilBERT model not loaded.'}), 500

    payload = request.get_json(force=True)
    text = payload.get('text', '')
    outputs = distilbert_classifier(text)[0]
    label_raw = outputs['label']  # POSITIVE or NEGATIVE
    score = outputs['score']

    # Map: POSITIVE => REAL, NEGATIVE => FAKE (rough assumption)
    label = 'REAL' if label_raw == 'POSITIVE' else 'FAKE'
    probabilities = [1 - score, score] if label == 'REAL' else [score, 1 - score]

    return jsonify({'label': label, 'probabilities': probabilities})

@app.route('/predict_fakenews', methods=['POST'])
def predict_fakenews():
    """Run Fake News Detection fine-tuned BERT model inference."""
    if fake_news_pipeline is None:
        return jsonify({'error': 'Fake news BERT model not loaded.'}), 500

    payload = request.get_json(force=True)
    text = payload.get('text', '')
    outputs = fake_news_pipeline(text)[0]
    label_raw = outputs['label'].upper()
    score = outputs['score']

    # Map labels depending on model output
    if 'FAKE' in label_raw:
        label = 'FAKE'
        probabilities = [score, 1 - score]
    else:
        label = 'REAL'
        probabilities = [1 - score, score]

    return jsonify({'label': label, 'probabilities': probabilities})

# @app.route('/predict_bert', methods=['POST'])
# def predict_bert():
#     """Run BERT model inference."""
#     payload = request.get_json(force=True)
#     text = payload.get('text', '')
#     inputs = bert_tokenizer(
#         text,
#         return_tensors='pt',
#         truncation=True,
#         padding=True,
#         max_length=512
#     )
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     logits = outputs.logits
#     probs = F.softmax(logits, dim=-1).squeeze().tolist()
#     pred_id = int(logits.argmax(dim=-1))
#     label = 'REAL' if pred_id == 1 else 'FAKE'
#     return jsonify({'label': label, 'probabilities': probs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
