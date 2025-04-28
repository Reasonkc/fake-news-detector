#!/usr/bin/env python3
"""
fake_news_detector.py

A two-stage fake news detection toolkit:
 1. Training models (baseline or BERT)
 2. Inference with a saved model

This version includes tqdm progress bars for text preprocessing.

Usage:
  # Train baseline with progress bars
  python fake_news_detector.py --mode train_baseline --use-kaggle --max-features 5000 --ngram-min 1 --ngram-max 2 --model-type logistic --test-size 0.2 --save-model baseline.joblib

  # Infer with baseline model
  python fake_news_detector.py --mode infer_baseline --load-model baseline.joblib --input-text "Your news text here"

  # Train BERT (unchanged)
  python fake_news_detector.py --mode train_bert --use-kaggle --bert-model bert-base-uncased --epochs 3 --train-batch 8 --eval-batch 16 --output-dir bert_output

  # Infer with BERT model
  python fake_news_detector.py --mode infer_bert --output-dir bert_output --input-text "Your news text here"
"""
import argparse
import logging
import re
import sys

import pandas as pd
import joblib
from tqdm.auto import tqdm

# Baseline ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Transformer imports
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Optional Kagglehub loader
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    kagglehub = None


def download_nltk_resources():
    """Download necessary NLTK corpora if missing."""
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)


def load_dataframe(use_kaggle, file_path):
    """
    Load and concatenate fake vs real CSVs, then ensure 'text' and 'label' columns exist.
    """
    if use_kaggle:
        if kagglehub is None:
            logging.error("kagglehub not installed; install it or disable --use-kaggle.")
            sys.exit(1)
        # Load both Fake.csv and True.csv
        logging.info("Loading Fake.csv and True.csv from Kaggle dataset")
        fake_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "clmentbisaillon/fake-and-real-news-dataset",
            "Fake.csv",
            pandas_kwargs={"encoding": "latin-1", "compression": "zip"}
        )
        true_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "clmentbisaillon/fake-and-real-news-dataset",
            "True.csv",
            pandas_kwargs={"encoding": "latin-1", "compression": "zip"}
        )

        fake_df['label'] = 'fake'
        true_df['label'] = 'real'
        df = pd.concat([fake_df, true_df], ignore_index=True)
    else:
        if not file_path:
            logging.error("Provide --file-path when not using Kaggle.")
            sys.exit(1)
        logging.info(f"Loading dataset from local file: {file_path}")
        df = pd.read_csv(
            file_path,
            encoding='latin-1',
            engine='python',
            on_bad_lines='skip'
        )
        # If file has no 'label', expect CSV contains both label and text
        if 'label' not in df.columns:
            logging.error(f"No 'label' column found in {file_path}. Please include a label column.")
            sys.exit(1)
    # Normalize 'text' column name
    if 'text' not in df.columns:
        candidates = [c for c in df.columns if 'text' in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: 'text'})
            logging.info(f"Renamed column '{candidates[0]}' to 'text'.")
        else:
            logging.error(f"No text column found; available columns = {df.columns.tolist()}")
            sys.exit(1)
    # Drop rows missing essential data
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
    return df


def preprocess_text(text, lemmatizer, stop_words):
    """Clean, tokenize, lemmatize, and remove stopwords from text."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(tokens)


def train_baseline(args):
    """Train and evaluate a baseline model (TF-IDF + LR or RF)."""
    download_nltk_resources()
    df = load_dataframe(args.use_kaggle, args.file_path)

    # Preprocessing with progress bar
    tqdm.pandas(desc="Preprocessing texts")
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    df['clean_text'] = df['text'].astype(str).progress_apply(
        lambda x: preprocess_text(x, lemmatizer, stops)
    )
    # Binary labels
    df['label_bin'] = df['label'].str.lower().map({'real':1, 'fake':0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label_bin'],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df['label_bin']
    )
    # Vectorize
    vec = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        stop_words='english'
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    # Model selection
    if args.model_type == 'logistic':
        model = LogisticRegression(C=args.logistic_c, max_iter=args.max_iter, random_state=args.random_state)
    else:
        model = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            random_state=args.random_state
        )
    # Train
    logging.info("Training baseline model...")
    model.fit(X_tr, y_train)
    # Evaluate
    preds = model.predict(X_te)
    print(classification_report(y_test, preds, digits=4))
    # Save
    if args.save_model:
        joblib.dump({'model': model, 'vectorizer': vec}, args.save_model)
        logging.info(f"Saved baseline model to {args.save_model}")


def train_bert(args):
    """Fine-tune a BERT model for fake news classification."""
    # Load and prepare data
    df = load_dataframe(args.use_kaggle, args.file_path)
    df['label_bin'] = df['label'].str.lower().map({'real':1, 'fake':0})
    dataset = Dataset.from_pandas(df[['text','label_bin']])
    dataset = dataset.train_test_split(test_size=args.test_size, seed=args.random_state)

    # Tokenization & label rename
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    dataset = dataset.map(tokenize_fn, batched=True)
    # rename label_bin → labels so Trainer can compute loss
    dataset = dataset.rename_column('label_bin', 'labels')
    dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)

    # Training arguments without evaluation_strategy for compatibility
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        logging_dir=f"{args.output_dir}/logs"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer
    )

    # Train and manually evaluate
    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Evaluation metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value}")

    # Save model if output_dir exists
    try:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logging.info(f"Saved fine-tuned BERT model to {args.output_dir}")
    except Exception as e:
        logging.error(f"Failed to save BERT model: {e}")


def infer_baseline(args):
    """Load a saved baseline model and make a prediction."""
    data = joblib.load(args.load_model)
    model = data['model']
    vec = data['vectorizer']
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    clean = preprocess_text(args.input_text, lemmatizer, stops)
    vec_text = vec.transform([clean])
    pred = model.predict(vec_text)[0]
    prob = model.predict_proba(vec_text)[0]
    label = 'REAL' if pred==1 else 'FAKE'
    print(f"Prediction: {label}, Probabilities: {prob}")


def infer_bert(args):
    """Load a fine-tuned BERT model and make a prediction."""
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    inputs = tokenizer(args.input_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(logits.argmax(dim=-1))
    label = 'REAL' if pred_id==1 else 'FAKE'
    print(f"Prediction: {label}, Probabilities: {probs}")


def main():
    parser = argparse.ArgumentParser(description="Fake News Detector: train/infer baseline or BERT")
    parser.add_argument('--mode', required=True, choices=['train_baseline','train_bert','infer_baseline','infer_bert'])
    parser.add_argument('--use-kaggle', action='store_true', help='Load dataset from Kagglehub')
    parser.add_argument('--file-path', type=str, default='', help='Local CSV path or Kaggle file name')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    # Baseline args
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--ngram-min', type=int, default=1)
    parser.add_argument('--ngram-max', type=int, default=2)
    parser.add_argument('--model-type', choices=['logistic','rf'], default='logistic')
    parser.add_argument('--logistic-c', type=float, default=1.0)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--rf-estimators', type=int, default=100)
    parser.add_argument('--rf-max-depth', type=int, default=None)
    parser.add_argument('--save-model', type=str, default='')
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--input-text', type=str, default='', help='Text for inference')
    # BERT args
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch', type=int, default=8)
    parser.add_argument('--eval-batch', type=int, default=16)
    parser.add_argument('--output-dir', type=str, default='bert_output')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == 'train_baseline':
        train_baseline(args)
    elif args.mode == 'train_bert':
        train_bert(args)
    elif args.mode == 'infer_baseline':
        infer_baseline(args)
    elif args.mode == 'infer_bert':
        infer_bert(args)


if __name__ == '__main__':
    main()
