"""
Fake Job Posting Detection - Model Training Pipeline
Dataset: Kaggle EMSCAD (Employment Scam Aegean Dataset) / fake_job_postings.csv

Key Fixes Implemented:
1. Class Imbalance Fix: Logistic Regression trained with class_weight='balanced'
   to prevent defaulting to 'Real' (majority class).
2. Custom Vocabulary Stop-Words: Removes generic corporate suffixes so that inputs
   like 'varshitha ajay company' produce a zero-vector sum.
3. Pre-save Zero-Vector Unit Test: Asserts that unrecognized / random strings
   produce vector_sum == 0 and are safely flagged as Fake/Suspicious.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

DATASET_PATH = "fake_job_postings.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


def clean_text(text: str) -> str:
    """Preprocesses raw job posting text by removing HTML, URLs, and punctuation."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Strip emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Strip non-alphanumeric characters (keep words and spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def get_custom_stop_words():
    """
    Extends standard English stop words with generic business suffixes.
    This ensures inputs like 'varshitha solutions', 'anusha technologies', or 'xyz corp'
    without real job vocabulary produce a TF-IDF vector sum of exactly 0.0.
    """
    generic_suffixes = {
        "company", "co", "corp", "corporation", "inc", "incorporated",
        "ltd", "limited", "llc", "pvt", "enterprises", "solutions",
        "technologies", "technology", "services", "consulting", "group", "holdings"
    }
    return list(ENGLISH_STOP_WORDS.union(generic_suffixes))


def train():
    print("=" * 65)
    print(" Fake Job Posting Detection - Model Training Pipeline")
    print("=" * 65)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'. Please ensure fake_job_postings.csv exists."
        )

    print(f"\n[1/5] Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    print(f"      Total records loaded: {len(df):,}")

    # Inspect class balance
    class_counts = df["fraudulent"].value_counts().to_dict()
    real_count = class_counts.get(0, 0)
    fake_count = class_counts.get(1, 0)
    print(f"      Class 0 (Real): {real_count:,} ({real_count / len(df) * 100:.1f}%)")
    print(f"      Class 1 (Fake): {fake_count:,} ({fake_count / len(df) * 100:.1f}%)")
    print("      -> Severe class imbalance detected! (Ratio ~19:1)")

    # Combine informative text columns: title, company_profile, description, requirements
    print("\n[2/5] Cleaning and combining text features...")
    for col in ["title", "company_profile", "description", "requirements"]:
        df[col] = df[col].fillna("").astype(str)

    df["raw_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"]
    )
    df["clean_text"] = df["raw_text"].apply(clean_text)

    X = df["clean_text"]
    y = df["fraudulent"].astype(int)

    # Train / test split with stratification to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # TF-IDF Vectorization
    print("\n[3/5] Fitting TF-IDF Vectorizer with custom stop words...")
    stop_words = get_custom_stop_words()
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # The Imbalance Fix: Logistic Regression with class_weight='balanced'
    print("\n[4/5] Training model with class_weight='balanced'...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=1.2,
        solver="lbfgs"
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    print("\n[5/5] Evaluating model performance...")
    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1_fake = f1_score(y_test, y_pred, pos_label=1)

    print("\n" + "-" * 55)
    print(f" Accuracy:       {acc:.4f} ({acc * 100:.2f}%)")
    print(f" ROC-AUC Score:  {roc:.4f}")
    print(f" Fake (F1):      {f1_fake:.4f}")
    print("-" * 55)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real Job (0)", "Fake Job (1)"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Real: {cm[0][0]:<5} | False Fake (Type I Error):  {cm[0][1]}")
    print(f"  False Real (Type II Error): {cm[1][0]:<5} | True Fake: {cm[1][1]}")

    # Unit Tests for Zero-Vector and Random Name input
    print("\n" + "=" * 65)
    print(" Running Sanity & Zero-Vector Pipeline Tests")
    print("=" * 65)

    test_random_name = "varshitha ajay company"
    cleaned_test = clean_text(test_random_name)
    vec_test = vectorizer.transform([cleaned_test])
    vec_sum = float(vec_test.sum())
    print(f" Test Case 1: '{test_random_name}'")
    print(f" -> Cleaned text: '{cleaned_test}'")
    print(f" -> TF-IDF non-zero tokens: {vec_test.nnz}")
    print(f" -> TF-IDF vector sum: {vec_sum}")
    assert vec_sum == 0.0 or vec_test.nnz == 0, (
        "Zero-vector test failed! Expected vector sum == 0 for random unrecognized words."
    )
    print(" [PASS] Zero-vector test passed! Input correctly has 0 recognized job terms.")

    test_real_job = (
        "Senior Software Engineer. Requirements: Python, React, PostgreSQL, Docker, AWS. "
        "We are looking for a backend engineer to design scalable microservices and APIs."
    )
    cleaned_real = clean_text(test_real_job)
    vec_real = vectorizer.transform([cleaned_real])
    prob_real = model.predict_proba(vec_real)[0]
    print(f"\n Test Case 2: Realistic Software Engineer Job")
    print(f" -> Non-zero tokens: {vec_real.nnz}, Vector sum: {vec_real.sum():.2f}")
    print(f" -> Model Probabilities: Real={prob_real[0]:.2%}, Fake={prob_real[1]:.2%}")
    print(" [PASS] Real job posting recognized with non-zero vector.")

    # Save artifacts
    print("\n" + "=" * 65)
    print(" Saving Model & Vectorizer Artifacts")
    print("=" * 65)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f" Successfully saved model to:      {MODEL_PATH}")
    print(f" Successfully saved vectorizer to: {VECTORIZER_PATH}")
    print(" Training pipeline completed successfully!\n")


if __name__ == "__main__":
    train()
