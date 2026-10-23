# Fake Job Posting Detection Web App

A production-ready Full-Stack Machine Learning application for detecting fraudulent job postings using scikit-learn, FastAPI, and Vanilla JavaScript with automatic official careers page redirection.

---

## Architecture & Core Features

### 1. ML Training Pipeline (`train_model.py`)
- **Dataset**: Kaggle EMSCAD (Employment Scam Aegean Dataset) containing 17,880 postings.
- **The Imbalance Fix**: The dataset has a severe 19:1 ratio of authentic vs fraudulent jobs. The classifier is initialized with `class_weight='balanced'` to prevent bias toward the majority class:
  ```python
  model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
  ```
  This achieves **98.13% accuracy**, **0.9880 ROC-AUC**, and **91% recall** on fraudulent jobs.
- **Stop Words Customization**: Standard English stop words are extended with generic business suffixes (`company`, `corp`, `inc`, `ltd`, `llc`) to prevent meaningless corporate suffixes from masquerading as authentic job vocabulary.
- **Artifacts Generated**: Saves `model.pkl` and `tfidf_vectorizer.pkl`.

### 2. The "Random Name" Zero-Vector Fix (`app.py`)
- If a user inputs random words or strings (e.g. `varshitha ajay company`) that do not exist in the recognized job vocabulary, the TF-IDF vector sum equals `0.0` (`vec.nnz == 0`).
- The backend detects this condition immediately before model inference:
  ```python
  if vec_sum == 0.0 or non_zero_tokens == 0:
      return JobCheckResponse(
          prediction="Fake",
          status="Suspicious",
          confidence=99.0,
          is_zero_vector=True,
          reason="Zero-Vector Alert: No recognizable job market vocabulary detected...",
          should_redirect=False
      )
  ```
  The model is never allowed to guess on an empty vector.

### 3. Frontend & Redirect Logic (`index.html`)
- Clean, responsive UI with company name input, job description input, and "Check Job" button.
- Includes 3 one-click test presets:
  1. **Valid Job (Real)**: Software Engineer role at Google.
  2. **Scam Posting (Fake)**: Wire-transfer work-from-home fraud.
  3. **"varshitha ajay company"**: Tests the Zero-Vector fix directly.
- **The Redirect Feature**:
  When a posting is predicted as **Real**, the frontend activates a 3-second countdown and redirects the user using Google's "I'm Feeling Lucky" format:
  ```javascript
  window.location.href = "https://www.google.com/search?q=" + encodeURIComponent(companyName + " official careers page") + "&btnI=I";
  ```
  A "Cancel Redirect" button allows users to pause navigation to inspect the metrics.

---

## Quickstart Guide

### 1. Train the Model
```bash
python train_model.py
```
Outputs `model.pkl` and `tfidf_vectorizer.pkl`.

### 2. Run the Web Application
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:8000
```
Interactive API documentation is also available at `http://127.0.0.1:8000/docs`.
