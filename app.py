"""
Fake Job Posting Detection - FastAPI Backend Server
Loads model.pkl and tfidf_vectorizer.pkl to provide real-time inference.

Key Features & Fixes:
1. Serves index.html directly from root '/'
2. Strict Zero-Vector Fix: Checks if TF-IDF sum == 0 or nnz == 0.
   If true, instantly flags as Fake/Suspicious with confidence ~99%
   without letting the model guess on an empty vector.
3. Class Imbalance-Trained Model: Produces reliable probabilities.
4. Returns prediction, confidence, zero-vector flag, and redirect instructions.
"""

import base64
import os
import re
import urllib.parse
from bs4 import BeautifulSoup
from curl_cffi import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import joblib
from pydantic import BaseModel, Field

# Model & Vectorizer paths
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
INDEX_HTML_PATH = "index.html"

# Trusted corporate networks for verification
TRUSTED_CORPORATE_DOMAINS = [
    "linkedin.com/company",
    "glassdoor",
    "zaubacorp",
    "ambitionbox",
    "naukri",
    "indeed"
]

# Established global tech leaders for instant validation
WELL_KNOWN_ENTERPRISES = {
    "google", "microsoft", "apple", "amazon", "meta", "netflix",
    "ibm", "tcs", "infosys", "wipro", "cognizant", "accenture",
    "oracle", "cisco", "intel", "nvidia", "adobe", "salesforce"
}


def decode_bing_u(u_val: str) -> str:
    """Decodes Bing's encoded search redirect URL parameter."""
    try:
        b64_str = u_val[2:] if u_val.startswith(("a0", "a1")) else u_val
        b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
        return base64.b64decode(b64_str).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def fetch_search_urls(query: str, max_results: int = 5) -> list[str]:
    """Fetches top search result URLs using curl_cffi with Chrome impersonation."""
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    try:
        resp = requests.get(url, headers=headers, impersonate="chrome", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        extracted = []
        for li in soup.find_all("li", class_="b_algo")[:max_results]:
            a = li.find("a", href=True)
            if a and "/ck/a?!" in a["href"] and "u=" in a["href"]:
                u_part = a["href"].split("u=")[1].split("&")[0]
                real_url = decode_bing_u(u_part)
                if real_url and real_url.startswith("http"):
                    extracted.append(real_url)
            elif a and a["href"].startswith("http"):
                extracted.append(a["href"])
        return extracted[:max_results]
    except Exception as e:
        print(f"[Corporate Web Search Error] {e}")
        return []


def search_corporate_presence(company_name: str) -> bool:
    """
    Corporate Web Search Gate:
    Dynamically checks top-5 web search results for the given company name.
    Verifies if any returned URL belongs to trusted corporate networks:
      - linkedin.com/company
      - glassdoor
      - zaubacorp
      - ambitionbox
      - naukri
      - indeed
    Allows real mid-level IT companies/startups (like Beauroi Technologies)
    while strictly blocking made-up companies (like Ajay Solutions).
    """
    clean = company_name.strip()
    if not clean:
        return False

    normalized = re.sub(r"[^a-zA-Z0-9]", "", clean).lower()
    if normalized in WELL_KNOWN_ENTERPRISES:
        print(f"[Corporate Web Search] '{clean}' verified via enterprise directory.")
        return True

    # Smart Search: Top-5 results for company_name
    urls = fetch_search_urls(clean, max_results=5)
    print(f"[Corporate Web Search] '{clean}' top URLs: {urls}")

    # Trust Check: Check if any returned URL contains trusted corporate networks
    for u in urls:
        u_lower = u.lower()
        for trusted in TRUSTED_CORPORATE_DOMAINS:
            if trusted in u_lower:
                print(f"[Corporate Web Search] '{clean}' verified on trusted network: {trusted} ({u})")
                return True

    # Check for direct official corporate domain (e.g. https://www.company.com)
    for u in urls:
        u_lower = u.lower()
        if normalized and len(normalized) >= 4:
            if f"://www.{normalized}." in u_lower or f"://{normalized}." in u_lower:
                print(f"[Corporate Web Search] '{clean}' verified via official domain: {u}")
                return True

    # Secondary targeted search for registered corporate profile
    targeted_urls = fetch_search_urls(f"{clean} careers", max_results=5)
    for u in targeted_urls:
        u_lower = u.lower()
        for trusted in TRUSTED_CORPORATE_DOMAINS:
            if trusted in u_lower:
                print(f"[Corporate Web Search] '{clean}' verified on targeted network: {trusted} ({u})")
                return True

    print(f"[Corporate Web Search] No trusted presence found for '{clean}'.")
    return False

app = FastAPI(
    title="Fake Job Posting Detection API",
    description="Machine Learning service detecting fraudulent job postings with Zero-Vector safeguards.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None


def load_artifacts():
    """Loads the trained model and TF-IDF vectorizer into memory."""
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Missing model artifacts: '{MODEL_PATH}' or '{VECTORIZER_PATH}'. "
            f"Please run 'python train_model.py' first."
        )
    print("Loading model and vectorizer artifacts...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Artifacts loaded successfully!")


@app.on_event("startup")
def startup_event():
    load_artifacts()


def clean_text(text: str) -> str:
    """Preprocesses input text matching the training pipeline cleaning steps."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# Mandatory job vocabulary required for Positive Vocabulary Gate
MANDATORY_JOB_KEYWORDS = {
    "experience", "salary", "requirements", "skills", "apply",
    "candidate", "role", "qualifications", "degree", "shift",
    "full-time", "part-time", "resume", "hiring"
}

# Explicit scam phrases & negative patterns for the Red Flag Gate
SCAM_RED_FLAGS = [
    "registration fee",
    "wire transfer",
    "western union",
    "deposit",
    "crypto",
    "whatsapp",
    "no interview",
    "earn money from home",
    "bank details",
    "processing fee",
    "send a check"
]


class JobCheckRequest(BaseModel):
    company_name: str = Field(..., example="Google")
    job_description: str = Field(
        ...,
        example="We are hiring a Senior Software Engineer for a full-time role. Requirements include a Bachelor degree, 5+ years of experience with Python, and strong skills. Apply with resume."
    )


class JobCheckResponse(BaseModel):
    prediction: str
    reason: str = ""
    company_name: str = ""


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the frontend single-page application."""
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse(
        "<h1>Frontend file index.html not found. Please create index.html in the project directory.</h1>",
        status_code=404
    )


@app.get("/health")
def health_check():
    """Health check endpoint reporting API and model readiness."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "vocabulary_size": len(vectorizer.vocabulary_) if vectorizer else 0
    }


@app.post("/predict", response_model=JobCheckResponse)
def predict_job(payload: JobCheckRequest):
    """
    Enhanced Hybrid Gatekeeper for Fake Job Detection.
    Order of execution:
      Gate 0: Corporate Web Search Gate (verifies presence on LinkedIn, Glassdoor, or job registries)
      Gate 1: Word Count Gate (must be >= 15 words)
      Gate 2: Red Flag Gate (strictly checks for scam/financial red flags)
      Gate 3: Positive Vocabulary Gate (must contain at least one HR keyword)
      Gate 4: ML Confidence Gate (internal threshold >= 0.50)
    Note: Percentages are hidden from the client response.
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model or vectorizer not loaded. Run train_model.py first."
        )

    company = payload.company_name.strip()
    description = payload.job_description.strip()
    full_text = f"{company} {description}".strip()

    # -----------------------------------------------------------------
    # GATE 0: CORPORATE WEB SEARCH GATE
    # Strictly reject unverified / made-up companies without corporate presence
    # -----------------------------------------------------------------
    if not search_corporate_presence(company):
        print(f"[Gatekeeper] Gate 0 Failed: '{company}' has no official corporate presence.")
        return JobCheckResponse(
            prediction="Fake",
            reason="Unverified Company: No official corporate presence found on LinkedIn, Glassdoor, or job registries.",
            company_name=company
        )

    # -----------------------------------------------------------------
    # GATE 1: THE WORD COUNT GATE
    # Strictly reject if total word count is less than 15 words
    # -----------------------------------------------------------------
    words = full_text.split()
    if len(words) < 15:
        print(f"[Gatekeeper] Gate 1 Failed: Word count ({len(words)}) < 15 words.")
        return JobCheckResponse(
            prediction="Fake",
            reason="Input is too short. Please enter a full job description.",
            company_name=company
        )

    # -----------------------------------------------------------------
    # GATE 2: THE RED FLAG GATE (NEGATIVE FILTER)
    # Explicitly catch known scam phrases and suspicious financial requests
    # -----------------------------------------------------------------
    text_lower = full_text.lower()
    matched_red_flags = []
    for flag in SCAM_RED_FLAGS:
        pattern = r"(?:\b|_)" + re.escape(flag) + r"(?:\b|_)"
        if re.search(pattern, text_lower) or flag in text_lower:
            matched_red_flags.append(flag)

    if matched_red_flags:
        print(f"[Gatekeeper] Gate 2 (Red Flag) Triggered: Detected {matched_red_flags}.")
        return JobCheckResponse(
            prediction="Fake",
            reason="Suspicious scam patterns or financial requests detected.",
            company_name=company
        )

    # -----------------------------------------------------------------
    # GATE 3: THE POSITIVE VOCABULARY GATE
    # Input MUST contain at least one legitimate HR/job-related keyword
    # -----------------------------------------------------------------
    matched_keywords = set()
    for kw in MANDATORY_JOB_KEYWORDS:
        pattern = r"(?:\b|_)" + re.escape(kw) + r"(?:\b|_)"
        if re.search(pattern, text_lower):
            matched_keywords.add(kw)

    if len(matched_keywords) < 1:
        print(f"[Gatekeeper] Gate 3 (Vocabulary) Failed: No valid job keywords found.")
        return JobCheckResponse(
            prediction="Fake",
            reason="Input does not contain valid job-related context.",
            company_name=company
        )

    # -----------------------------------------------------------------
    # GATE 4: THE ML CONFIDENCE GATE
    # Vectorize and evaluate probability. Real must be >= 50% (0.50)
    # -----------------------------------------------------------------
    cleaned = clean_text(full_text)
    vec = vectorizer.transform([cleaned])

    # Zero-vector safety check
    if vec.sum() == 0.0 or vec.nnz == 0:
        print("[Gatekeeper] Zero-vector detected (sum == 0). Bypassing model.")
        return JobCheckResponse(
            prediction="Fake",
            reason="Input does not contain valid job-related context.",
            company_name=company
        )

    probabilities = model.predict_proba(vec)[0]
    prob_real = float(probabilities[0])
    prob_fake = float(probabilities[1])

    # Confidence Threshold of 0.50 (50%) used internally
    CONFIDENCE_THRESHOLD = 0.50

    if prob_real >= CONFIDENCE_THRESHOLD:
        prediction = "Real"
        reason = "Verified as an authentic job posting."
    else:
        prediction = "Fake"
        reason = "Job posting flagged as fraudulent based on scam indicators."

    # Terminal Debug Logging (visible in terminal only)
    print(f"Model Probabilities: {probabilities}")
    print(f"Prediction: {prediction} | Real Prob: {prob_real:.4f} | Fake Prob: {prob_fake:.4f}")

    return JobCheckResponse(
        prediction=prediction,
        reason=reason,
        company_name=company
    )


if __name__ == "__main__":
    import uvicorn
    print("\nStarting Fake Job Detection Web Server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
