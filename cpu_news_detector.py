#!/usr/bin/env python3
"""
cpu_news_detector.py

A fake news detection tool optimized for CPU training using classical ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Linear SVM
  - Multinomial Naive Bayes

Supports training and inference modes, with local CSV inputs for Fake.csv and True.csv.

Usage examples:
  # Train a model (e.g., XGBoost) combining Fake.csv and True.csv
  python cpu_news_detector.py --mode train --fake-file Fake.csv --true-file True.csv --model xgb --max-features 5000 --test-size 0.2 --random-state 42 --save-model xgb_model.joblib

  # Inference on new text
  python cpu_news_detector.py --mode infer --load-model xgb_model.joblib --input-text "Breaking news: ..."
"""
import argparse
import logging
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import joblib

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_resources():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)


def load_data(fake_file, true_file):
    """Load Fake.csv and True.csv, assign labels, concatenate."""
    if not fake_file or not true_file:
        logging.error("Both --fake-file and --true-file must be provided in train mode.")
        sys.exit(1)
    fake_df = pd.read_csv(fake_file, encoding='latin-1', engine='python', on_bad_lines='skip')
    true_df = pd.read_csv(true_file, encoding='latin-1', engine='python', on_bad_lines='skip')
    fake_df['label'] = 0  # fake
    true_df['label'] = 1  # real
    df = pd.concat([fake_df[['text','label']], true_df[['text','label']]], ignore_index=True)
    return df.dropna(subset=['text','label']).reset_index(drop=True)


def preprocess_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return " ".join(lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words)


def get_model(name, args):
    name = name.lower()
    if name == 'logistic':
        return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=args.random_state)
    if name == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=args.random_state)
    if name == 'xgb':
        if XGBClassifier is None:
            logging.error("XGBoost is not installed. Install xgboost or choose another model.")
            sys.exit(1)
        return XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False,
                             eval_metric='logloss', n_jobs=-1, random_state=args.random_state)
    if name == 'svm':
        return LinearSVC(class_weight='balanced', max_iter=5000)
    if name == 'nb':
        return MultinomialNB(alpha=0.1)
    logging.error(f"Unknown model: {name}")
    sys.exit(1)


def train(args):
    # Download NLP resources
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))

    # Load data
    df = load_data(args.fake_file, args.true_file)
    tqdm.pandas(desc="Preprocessing texts")
    df['clean'] = df['text'].progress_apply(lambda x: preprocess_text(str(x), lemmatizer, stops))
    X = df['clean']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    # Vectorize
    vec = TfidfVectorizer(max_features=args.max_features, stop_words='english')
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    # Model
    model = get_model(args.model, args)
    logging.info(f"Training {args.model} model...")
    model.fit(X_tr, y_train)

    # Evaluate
    preds = model.predict(X_te)
    print(classification_report(y_test, preds, digits=4))

    # Save
    if args.save_model:
        joblib.dump({'model': model, 'vectorizer': vec}, args.save_model)
        logging.info(f"Saved model to {args.save_model}")


def infer(args):
    # Load model
    data = joblib.load(args.load_model)
    model = data['model']
    vec = data['vectorizer']

    # Preprocess input
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    clean = preprocess_text(args.input_text, lemmatizer, stops)
    X_vec = vec.transform([clean])

    # Predict
    pred = model.predict(X_vec)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_vec)[0].tolist()
    label = 'REAL' if pred == 1 else 'FAKE'
    out = {'label': label}
    if prob is not None:
        out['probabilities'] = {'fake': prob[0], 'real': prob[1]}
    print(out)


def main():
    parser = argparse.ArgumentParser(description="CPU-based Fake News Detector")
    parser.add_argument('--mode', choices=['train','infer'], required=True)
    parser.add_argument('--fake-file', type=str, help='Path to Fake.csv for training')
    parser.add_argument('--true-file', type=str, help='Path to True.csv for training')
    parser.add_argument('--model', type=str, choices=['logistic','rf','xgb','svm','nb'],
                        default='logistic', help='Model type for training')
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--save-model', type=str, default='')
    parser.add_argument('--load-model', type=str, help='Path to saved model for inference')
    parser.add_argument('--input-text', type=str, default='', help='Text for inference')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == 'train':
        train(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
