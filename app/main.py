# app/main.py
import os
import json
import sqlite3
import datetime
import shutil
import numpy as np
import joblib
import shap
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "training"
DB_PATH = BASE_DIR / "decisions.db"


from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    FileResponse,
    RedirectResponse,
)

from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .bootstrap import ensure_baseline_bootstrap
from .schemas import CreditScoreRequest, CreditScoreResponse, Factor
import train_from_export  # training pipeline


import re
from typing import Optional
import httpx

from dotenv import load_dotenv

load_dotenv()  # loads .env from current project folder

def log_llm_startup_status():
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        print("[LLM] WARNING: Groq not active (LLM_API_KEY missing)")
        return

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

     
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0
        )

        print(f"[LLM] Groq connected | model={model}")

    except Exception as e:
        print(f"[LLM] ERROR: Groq unreachable → fallback enabled | {e}")


log_llm_startup_status()

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="AI Credit & Inclusion Platform - MVP",
    version="0.5.0",
)


@app.get("/")
def read_root():
    # Redirect root to dashboard UI
    return RedirectResponse(url="/dashboard")


# -------------------------------------------------------------------
# Training metrics globals
# -------------------------------------------------------------------

TRAINING_METRICS: dict = {}
TRAINING_METRICS_PATH = str(MODELS_DIR / "training_metrics.json")
MODEL_HISTORY_LOG = str(MODELS_DIR / "model_train_log.jsonl")



def load_training_metrics() -> None:
    """Load last training metrics into TRAINING_METRICS."""
    global TRAINING_METRICS
    if os.path.exists(TRAINING_METRICS_PATH):
        try:
            with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
                TRAINING_METRICS = json.load(f)
        except Exception:
            TRAINING_METRICS = {}
    else:
        TRAINING_METRICS = {}

# -------------------------------------------------------------------
# CORS + templates
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "app" / "templates")
)

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Model loading (ACTIVE model is used for scoring)
# -------------------------------------------------------------------



MODELS_DIR.mkdir(exist_ok=True)

BASE_MODEL_PATH = MODELS_DIR / "base_model.pkl"
ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.pkl"

# Your very first model file (existing in your project)
FIRST_MODEL = MODELS_DIR / "credit_model.pkl"

# One-time bootstrap:
# - create base_model.pkl from credit_model.pkl (only once)
# - create active_model.pkl (only once)
if FIRST_MODEL.exists():
    if not BASE_MODEL_PATH.exists():
        shutil.copy2(FIRST_MODEL, BASE_MODEL_PATH)
        print("✅ Base model created:", BASE_MODEL_PATH)

    if not ACTIVE_MODEL_PATH.exists():
        shutil.copy2(FIRST_MODEL, ACTIVE_MODEL_PATH)
        print("✅ Active model initialized:", ACTIVE_MODEL_PATH)

# Always score using ACTIVE model (not credit_model.pkl)
MODEL_PATH = os.getenv("MODEL_PATH", str(ACTIVE_MODEL_PATH))

def build_shap_explainer():
    # Works for tree models (XGBoost/LightGBM/RandomForest)
    # background should be a small sample array saved in your bundle
    try:
        if background is not None:
            return shap.TreeExplainer(model, data=background)
        return shap.TreeExplainer(model)
    except Exception:
        return shap.TreeExplainer(model)


def load_model_bundle(path: str | None = None) -> None:
    """Load model bundle + rebuild SHAP explainer so scoring uses latest ACTIVE model."""
    global model_bundle, model, feature_cols, background, fraud_model, explainer, MODEL_PATH

    if path is not None:
        MODEL_PATH = path

    p = Path(MODEL_PATH)
    if not p.exists():
        print(f"❌ MODEL_PATH missing: {p}")
        return

    model_bundle = joblib.load(str(p)) 
    model = model_bundle["model"]
    feature_cols = model_bundle["features"]
    background = model_bundle.get("background")
    fraud_model = model_bundle.get("fraud_model")
    explainer = build_shap_explainer()


# ---------------------------------------------------------
# Startup initialization (OUTSIDE all functions)
# ---------------------------------------------------------
@app.on_event("startup")
def _startup():
    # 1️⃣ DB tables (must be first)
    init_db()
    init_chat_db()

    # 2️⃣ Ensure model + metrics exist (Railway cold start safe)
    ensure_baseline_bootstrap()

    # 3️⃣ Load metrics + active model into memory
    load_training_metrics()
    load_model_bundle()


# -------------------------------------------------------------------
# Chat (LLM) - grounded on ExplainChain ledger
# -------------------------------------------------------------------




def init_chat_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_id TEXT,
            decision_id INTEGER,
            user_message TEXT,
            assistant_message TEXT,
            created_at TEXT
        );
        """
    )

    decision_id = cur.lastrowid

        conn.commit()
    conn.close()

    return decision_id


class ChatRequest(BaseModel):
    message: str
    borrower_id: Optional[str] = None
    decision_id: Optional[int] = None
    mode: str = "borrower"   # borrower | officer | auditor
    language: str = "en"     # en | bn


def _row_to_decision_dict(row: tuple) -> dict:
    return {
        "id": row[0],
        "borrower_id": row[1],
        "risk_score": row[2],
        "decision": row[3],
        "explanation": row[4],
        "shap_factors": row[5],
        "fraud_score": row[6],
        "fraud_label": row[7],
        "created_at": row[8],
    }



def fetch_decision_by_id(decision_id: int) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
SELECT id, borrower_id, risk_score, decision, explanation, shap_factors, fraud_score, fraud_label, created_at
FROM decisions
WHERE id = ?
""", (decision_id,))

    row = cur.fetchone()
    conn.close()
    return _row_to_decision_dict(row) if row else None


def fetch_latest_decision_for_borrower(borrower_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id, borrower_id, risk_score, decision, explanation,
            shap_factors, fraud_score, fraud_label, created_at
        FROM decisions
        WHERE borrower_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (borrower_id,))
    row = cur.fetchone()
    conn.close()
    return _row_to_decision_dict(row) if row else None


def _safe_json_loads(s: str) -> list:
    try:
        return json.loads(s) if s else []
    except Exception:
        return []

def build_grounded_context(decision: dict) -> str:
    shap_list = _safe_json_loads(decision.get("shap_factors") or "[]")

    # ✅ Backward compatibility: if pct_influence missing, compute from impacts
    total_abs = 0.0
    for f in shap_list:
        try:
            total_abs += abs(float(f.get("impact") or 0.0))
        except Exception:
            total_abs += 0.0
    if total_abs == 0:
        total_abs = 1.0

    # Keep it compact (LLM context should be small & factual)
    top = []
    for f in shap_list[:6]:
        name = f.get("feature") or f.get("name") or "unknown_feature"
        val = f.get("value")
        impact = f.get("impact")
        direction = f.get("direction")

        pct = f.get("pct_influence")
        if pct is None:
            # compute pct from impact if not stored
            try:
                pct = round((abs(float(impact or 0.0)) / total_abs) * 100, 2)
            except Exception:
                pct = 0.0

        top.append(
            f"- {name}: value={val}, pct_influence={pct}%, direction={direction}"
        )

    top_txt = "\n".join(top) if top else "- (no shap factors logged)"

    return f"""
DECISION FACTS (ground truth):
- decision_id: {decision.get("id")}
- borrower_id: {decision.get("borrower_id")}
- risk_score_PD: {decision.get("risk_score")}
- decision_label: {decision.get("decision")}
- explanation_from_model: {decision.get("explanation")}
- fraud_score: {decision.get("fraud_score")}
- fraud_label: {decision.get("fraud_label")}
- created_at: {decision.get("created_at")}

TOP_SHAP_FACTORS (model explainability):
{top_txt}

RULES:
- Only use the DECISION FACTS + TOP_SHAP_FACTORS.
- Do not invent new reasons.
- If user asks something not covered, say you don't have enough info.
""".strip()


def deterministic_fallback_answer(user_msg: str, decision: dict, mode: str, language: str) -> str:
    # Simple non-LLM fallback to keep demo working without API keys
    rs = decision.get("risk_score")
    dec = decision.get("decision")
    expl = decision.get("explanation") or ""
    fraud = decision.get("fraud_label")
    if language == "bn":
        return (
            f"সিদ্ধান্ত: {dec}\n"
            f"ঝুঁকি (PD): {rs}\n"
            f"ফ্রড ক্যাটাগরি: {fraud}\n\n"
            f"কারণ (মডেল): {expl}\n\n"
            f"আপনি চাইলে জিজ্ঞেস করতে পারেন: “কোন ৩টা ফ্যাক্টর সবচেয়ে বেশি প্রভাব ফেলেছে?”"
        )
    return (
        f"Decision: {dec}\n"
        f"Risk (PD): {rs}\n"
        f"Fraud category: {fraud}\n\n"
        f"Model reason: {expl}\n\n"
        f"You can ask: “Which top 3 factors drove this decision?”"
    )


async def call_llm_openai_compatible(system_prompt: str, user_prompt: str) -> str:
    """
    OpenAI-compatible Chat Completions client via HTTPX.

    ✅ Configured to work with Groq (https://console.groq.com/keys)
    ✅ Still compatible with any OpenAI-style provider

    Required environment variables (Groq example):
      LLM_BASE_URL = https://api.groq.com/openai/v1
      LLM_API_KEY  = <your_groq_api_key>
      LLM_MODEL    = llama-3.1-70b-versatile   (or another Groq model)

    Notes:
    - Endpoint used: POST /chat/completions
    - Temperature kept low for explainability use-cases
    """

    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set (Groq key required)")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,   # low = stable, explainable answers
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload
        )

        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"].strip()



def log_chat(borrower_id: str | None, decision_id: int | None, user_message: str, assistant_message: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    cur.execute(
        """
        INSERT INTO chat_logs (borrower_id, decision_id, user_message, assistant_message, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (borrower_id, decision_id, user_message, assistant_message, created_at),
    )
    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# SQLite ExplainChain ledger
# -------------------------------------------------------------------




def init_db():
    """Create SQLite table if it doesn't exist and upgrade schema if needed."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Base schema (used when table is first created)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_id TEXT,
            monthly_income REAL,
            monthly_expense REAL,
            age INTEGER,
            has_smartphone INTEGER,
            has_wallet INTEGER,
            avg_wallet_balance REAL,
            on_time_payment_ratio REAL,
            num_loans_taken INTEGER,
            risk_score REAL,
            decision TEXT,
            explanation TEXT,
            shap_factors TEXT,
            fraud_score REAL,
            fraud_label TEXT,
            created_at TEXT
        );
        """
    )

    # ---- schema upgrade for existing DBs ----
    cur.execute("PRAGMA table_info(decisions)")
    cols = [row[1] for row in cur.fetchall()]

    # Columns that might be missing in older versions
    needed_cols = {
        "shap_factors": "TEXT",
        "fraud_score": "REAL",
        "fraud_label": "TEXT",
        "repayment_status": "INTEGER",
        "days_late": "INTEGER",
        "outcome_recorded_at": "TEXT",
    }

    for col_name, col_type in needed_cols.items():
        if col_name not in cols:
            cur.execute(f"ALTER TABLE decisions ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()



def log_decision(
    req: CreditScoreRequest,
    risk_score: float,
    decision: str,
    explanation: str,
    top_factors: list[Factor],
    fraud_score: float,
    fraud_label: str,
) -> int:
    """Insert one decision row into SQLite ledger."""
    shap_factors_json = json.dumps([f.dict() for f in top_factors])
    created_at = datetime.datetime.utcnow().isoformat() + "Z"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO decisions (
            borrower_id,
            monthly_income,
            monthly_expense,
            age,
            has_smartphone,
            has_wallet,
            avg_wallet_balance,
            on_time_payment_ratio,
            num_loans_taken,
            risk_score,
            decision,
            explanation,
            shap_factors,
            fraud_score,
            fraud_label,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            req.borrower_id,
            req.monthly_income,
            req.monthly_expense,
            req.age,
            1 if req.has_smartphone else 0,
            1 if req.has_wallet else 0,
            req.avg_wallet_balance,
            req.on_time_payment_ratio,
            req.num_loans_taken,
            float(risk_score),
            decision,
            explanation,
            shap_factors_json,
            float(fraud_score),
            fraud_label,
            created_at,
        ),
    )
    conn.commit()
    conn.close()


def fetch_recent_decisions(limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            borrower_id,
            risk_score,
            decision,
            fraud_score,
            fraud_label,
            created_at
        FROM decisions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "borrower_id": r[1],
                "risk_score": r[2],
                "decision": r[3],
                "fraud_score": r[4],
                "fraud_label": r[5],
                "created_at": r[6],
            }
        )
    return out


class OutcomeUpdate(BaseModel):
    repaid: bool              # True = good, False = default/bad
    days_late: int | None = None


# initialize DB at startup
#init_db()

# -------------------------------------------------------------------
# Core credit & fraud helpers
# -------------------------------------------------------------------

def normalize_shap_percentages(shap_row, feature_cols):
    """
    Convert SHAP values to relative percentage influence.
    """
    abs_vals = np.abs(shap_row)
    total = abs_vals.sum() or 1.0

    pct_map = {}
    for i, feat in enumerate(feature_cols):
        pct_map[feat] = round((abs_vals[i] / total) * 100, 2)

    return pct_map






def compute_credit(req: CreditScoreRequest):
    """Return (pd_default, decision, explanation, top_factors[list[Factor]])"""
    features_values = [
        req.monthly_income,
        req.monthly_expense,
        req.age,
        1 if req.has_smartphone else 0,
        1 if req.has_wallet else 0,
        req.avg_wallet_balance,
        req.on_time_payment_ratio,
        req.num_loans_taken,
    ]

    X = np.array(features_values).reshape(1, -1)
    pd_default = float(model.predict_proba(X)[0, 1])  # probability of default

    # decision policy
    if pd_default < 0.2:
        decision = "APPROVE"
        explanation = "Low predicted default risk based on income and repayment behavior."
    elif pd_default < 0.4:
        decision = "REVIEW"
        explanation = "Moderate risk. MFI officer should review and potentially adjust limit."
    else:
        decision = "REJECT_OR_SMALL_LIMIT"
        explanation = "High predicted default risk. Consider smaller loan or financial coaching first."

    # SHAP explanations (robust with fallback)
    try:
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_row = np.array(shap_values[1][0])
        else:
            shap_row = np.array(shap_values[0])
    except Exception as e:
        # If SHAP fails for any reason, fall back to zeros so scoring still works
        shap_row = np.zeros(len(feature_cols))

    pct_map = normalize_shap_percentages(shap_row, feature_cols)


    abs_vals = np.abs(shap_row)
    top_idx = np.argsort(abs_vals)[::-1][:3]

    top_factors: list[Factor] = []
    for idx in top_idx:
        feat_name = feature_cols[idx]
        impact = float(shap_row[idx])
        direction = "increased_risk" if impact > 0 else "decreased_risk"
        detail = (
            f"{feat_name} with value {features_values[idx]} "
            f"{'raises' if impact > 0 else 'lowers'} the default risk."
        )
        top_factors.append(
            Factor(
                feature=feat_name,
                impact=impact,
                direction=direction,
                detail=detail,
                pct_influence=pct_map.get(feat_name, 0.0)
            )
        )

    return pd_default, decision, explanation, top_factors


def compute_fraud(req: CreditScoreRequest):
    """Return (fraud_score[0-1], fraud_label, flags[list[str]])"""
    features_values = [
        req.monthly_income,
        req.monthly_expense,
        req.age,
        1 if req.has_smartphone else 0,
        1 if req.has_wallet else 0,
        req.avg_wallet_balance,
        req.on_time_payment_ratio,
        req.num_loans_taken,
    ]
    X = np.array(features_values).reshape(1, -1)

    try:
        pred = fraud_model.predict(X)[0]  # -1 = anomaly, 1 = normal
        score = fraud_model.decision_function(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fraud model failed: {e}")

    # decision_function: higher = less anomalous
    # robust normalization using a squashing function
    normalized_score = 1 / (1 + np.exp(score))   # higher anomaly -> closer to 1 (usually)
    normalized_score = float(max(0.0, min(1.0, normalized_score)))


    flags: list[str] = []

    if req.on_time_payment_ratio < 0.5:
        flags.append("Low repayment behavior history.")
    if req.avg_wallet_balance < 500:
        flags.append("Wallet balance is very low relative to profile.")
    if req.monthly_income < req.monthly_expense:
        flags.append("Expense exceeds income — unsustainable pattern.")
    if req.num_loans_taken == 0:
        flags.append("No prior loan history — risk unvalidated.")
    if pred == -1:
        flags.append("Anomalous profile detected by IsolationForest.")

    if normalized_score >= 0.75:
        risk_label = "HIGHLY_ANOMALOUS"
    elif normalized_score >= 0.55:
        risk_label = "UNUSUAL"
    elif normalized_score >= 0.49:
        risk_label = "BORDERLINE"
    else:
        risk_label = "NORMAL"

    return normalized_score, risk_label, flags


# -------------------------------------------------------------------
# Routes – outcomes & admin metrics
# -------------------------------------------------------------------

@app.post("/decisions/{decision_id}/outcome")
def update_outcome(decision_id: int, outcome: OutcomeUpdate):
    """
    Attach real-world repayment outcome to a past decision.
    - repaid = True  -> target = 0 (good)
    - repaid = False -> target = 1 (bad/default)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    target_val = 0 if outcome.repaid else 1
    outcome_time = datetime.datetime.utcnow().isoformat() + "Z"

    cur.execute(
        """
        UPDATE decisions
        SET repayment_status = ?,
            days_late = ?,
            outcome_recorded_at = ?
        WHERE id = ?
        """,
        (target_val, outcome.days_late, outcome_time, decision_id),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()

    if updated == 0:
        raise HTTPException(status_code=404, detail="Decision not found")

    return {"status": "ok", "updated": updated}


@app.get("/admin/metrics")
def get_admin_metrics():
    """Return last training metrics for dashboard admin card."""
    return TRAINING_METRICS


@app.get("/admin/model_history")
def get_model_history():
    """
    Return full model training history from model_train_log.jsonl
    for the dashboard heartbeat chart & table.
    """
    history: list[dict] = []
    if os.path.exists(MODEL_HISTORY_LOG):
        try:
            with open(MODEL_HISTORY_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        history.append(rec)
                    except Exception:
                        continue
        except Exception:
            history = []

    # oldest → newest
    history.sort(key=lambda r: r.get("trained_at", ""))

    return {"history": history}


@app.get("/admin/download_model")
def download_model(filename: str):
    """
    Download a specific versioned model (by *filename only*).
    Example: /admin/download_model?filename=credit_model_20251209_161248.pkl
    """
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    model_path = os.path.join("models", "versions", filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@app.post("/admin/deploy_model")
def deploy_model(version: str):
    """
    Make this model version ACTIVE.

    1) Copy its .pkl → models/credit_model.pkl
    2) Update training_metrics.json + in-memory TRAINING_METRICS
    """
    # find record in history
    record = None
    if os.path.exists(MODEL_HISTORY_LOG):
        with open(MODEL_HISTORY_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("model_version") == version:
                    record = rec
                    break

    if not record:
        raise HTTPException(status_code=404, detail="Version not found in history log")

    model_path = record.get("model_file_path") or os.path.join(
        "models", f"credit_model_{version}.pkl"
    )

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    # deploy -> ACTIVE model
    shutil.copy2(model_path, str(ACTIVE_MODEL_PATH))

    # reload in-memory model + SHAP explainer
    load_model_bundle(str(ACTIVE_MODEL_PATH))

    global explainer
    
    metrics = {
        "last_auc": record.get("auc"),
        "rows": record.get("rows_used"),
        "last_trained_at": record.get("trained_at"),
        "version_path": record.get("model_file_path", model_path),
        "version_tag": record.get("model_version", version),
        "drift_score": record.get("drift_score"),
        "drift_label": record.get("drift_label"),
    }

    # persist metrics
    try:
        with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    # refresh in-memory
    global TRAINING_METRICS
    TRAINING_METRICS = metrics

    return {"status": "ok", "metrics": metrics}


@app.post("/admin/delete_model")
def delete_model(version: str):
    """
    Delete a specific model version:

    - Remove its .pkl version file
    - Remove its record from model_train_log.jsonl

    Active model (TRAINING_METRICS.version_tag) cannot be deleted.
    """
    active_version = (TRAINING_METRICS or {}).get("version_tag")
    if active_version == version:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the active model. Deploy another version first.",
        )

    if not os.path.exists(MODEL_HISTORY_LOG):
        raise HTTPException(status_code=404, detail="History log does not exist")

    kept_lines: list[str] = []
    deleted_record = None

    with open(MODEL_HISTORY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                kept_lines.append(line)
                continue

            if rec.get("model_version") == version:
                deleted_record = rec
            else:
                kept_lines.append(line)

    if not deleted_record:
        raise HTTPException(status_code=404, detail="Version not found in history log")

    # rewrite log without deleted record
    with open(MODEL_HISTORY_LOG, "w", encoding="utf-8") as f:
        for l in kept_lines:
            f.write(l if l.endswith("\n") else l.rstrip() + "\n")

    # delete its .pkl
    model_path = deleted_record.get("model_file_path") or os.path.join(
        "models", f"credit_model_{version}.pkl"
    )
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
        except Exception:
            pass

    return {"status": "ok", "deleted_version": version}


# -------------------------------------------------------------------
# Healthcheck + templates
# -------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_features": feature_cols,
        "has_fraud_model": fraud_model is not None,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/history_view", response_class=HTMLResponse)
def history_view(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


# -------------------------------------------------------------------
# History API for ExplainChain
# -------------------------------------------------------------------

@app.get("/history")
def history():
    """Recent decisions ledger (ExplainChain)."""
    return {"decisions": fetch_recent_decisions()}


@app.delete("/history/{decision_id}")
def delete_decision(decision_id: int):
    """Delete a single decision from the ledger."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM decisions WHERE id = ?", (decision_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Decision not found")
    return {"status": "ok", "deleted": deleted}


@app.delete("/history")
def delete_all_history():
    """Delete all decisions from the ledger."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM decisions")
    conn.commit()
    conn.close()
    return {"status": "ok"}


# -------------------------------------------------------------------
# Training export + scoring-only endpoints
# -------------------------------------------------------------------

@app.get("/training_export", response_class=PlainTextResponse)
def training_export():
    """
    Export labeled rows (with repayment_status) as CSV for model retraining.
    target = repayment_status (0 = repaid, 1 = default/bad)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            monthly_income,
            monthly_expense,
            age,
            has_smartphone,
            has_wallet,
            avg_wallet_balance,
            on_time_payment_ratio,
            num_loans_taken,
            repayment_status
        FROM decisions
        WHERE repayment_status IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()

    header = (
        "monthly_income,monthly_expense,age,has_smartphone,has_wallet,"
        "avg_wallet_balance,on_time_payment_ratio,num_loans_taken,target\n"
    )
    lines = [header]
    for r in rows:
        line = ",".join(
            [
                str(r[0]),
                str(r[1]),
                str(r[2]),
                str(r[3]),
                str(r[4]),
                str(r[5]),
                str(r[6]),
                str(r[7]),
                str(r[8]),
            ]
        )
        lines.append(line + "\n")

    return "".join(lines)


@app.post("/score", response_model=CreditScoreResponse)
def score_credit(req: CreditScoreRequest):
    """Credit scoring only (kept for API compatibility)."""
    pd_default, decision_label, explanation, top_factors = compute_credit(req)
    return CreditScoreResponse(
        borrower_id=req.borrower_id,
        risk_score=pd_default,
        decision=decision_label,
        explanation=explanation,
        top_factors=top_factors,
    )


@app.post("/fraud_check")
def fraud_check(req: CreditScoreRequest):
    """Fraud scoring only (kept for API compatibility)."""
    fraud_score, fraud_label, flags = compute_fraud(req)
    return {
        "fraud_score": round(fraud_score, 3),
        "risk_label": fraud_label,
        "flags": flags,
    }

# -------------------------------------------------------------------
# Admin: retrain and CSV upload
# -------------------------------------------------------------------

@app.post("/admin/retrain")
def admin_retrain():
    """
    Trigger retraining from latest CSV in data/training.
    After training completes, automatically activate the newest trained model.
    """
    try:
        _ = train_from_export.run()

        # refresh metrics from disk (training writes training_metrics.json)
        load_training_metrics()

        # training_metrics.json should contain the path of the newly saved version
        # training_metrics.json from train_from_export.py uses candidate_path
        newest_path = (TRAINING_METRICS or {}).get("candidate_path") or (TRAINING_METRICS or {}).get("version_path")

        if newest_path and os.path.exists(newest_path):
            # activate it
            shutil.copy2(newest_path, str(ACTIVE_MODEL_PATH))
            # reload in-memory model
            load_model_bundle(str(ACTIVE_MODEL_PATH))
        else:
            # fallback: just reload active as-is
            load_model_bundle(str(ACTIVE_MODEL_PATH))

        return {
            "status": "ok",
            "message": "Retraining complete and activated.",
            "metrics": TRAINING_METRICS,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")



UPLOAD_DIR = DATA_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@app.post("/admin/upload_training_csv")
async def upload_training_csv(file: UploadFile = File(...)):
    """
    Upload CSV for training and save under data/training directory.
    Frontend sends multipart/form-data with key 'file'.
    """
    try:
        filename = file.filename or "training.csv"

        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files allowed.")

        save_path = os.path.join(UPLOAD_DIR, filename)

        content = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(content)

        return {
            "status": "uploaded",
            "file": filename,
            "path": save_path,
            "message": "Training CSV uploaded successfully.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Combined decision endpoint
# -------------------------------------------------------------------

@app.post("/decision")
def decision(req: CreditScoreRequest):
    """
    Combined endpoint:
    - Computes credit score + SHAP explanations
    - Computes fraud score + flags
    - Logs everything into SQLite ExplainChain ledger
    """
    try:
        pd_default, decision_label, explanation, top_factors = compute_credit(req)
        fraud_score, fraud_label, flags = compute_fraud(req)

        # log to ExplainChain
        decision_id = log_decision(
            req=req,
            risk_score=pd_default,
            decision=decision_label,
            explanation=explanation,
            top_factors=top_factors,
            fraud_score=fraud_score,
            fraud_label=fraud_label,
        )

        return {
            "decision_id": decision_id,
            "borrower_id": req.borrower_id,
            "credit": {
                "decision_id": decision_id,
                "risk_score": pd_default,
                "decision": decision_label,
                "explanation": explanation,
                "top_factors": [f.dict() for f in top_factors],
            },
            "fraud": {
                "fraud_score": fraud_score,
                "risk_label": fraud_label,
                "flags": flags,
            },
        }

    except HTTPException:
        # rethrow FastAPI errors (e.g., fraud model failure)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/decision failed: {e}")




# ============================================================
# FAQ / Policy answers (Deterministic) — for judge/demo questions
# ============================================================

# --- Configurable thresholds (env override friendly) ---
PD_APPROVE_MAX = float(os.getenv("PD_APPROVE_MAX", "0.35"))      # PD <= this => APPROVE
PD_REJECT_MIN  = float(os.getenv("PD_REJECT_MIN",  "0.60"))      # PD >= this => REJECT (else REVIEW zone)

FRAUD_OK_MAX      = float(os.getenv("FRAUD_OK_MAX", "0.30"))     # <= ok
FRAUD_REVIEW_MIN  = float(os.getenv("FRAUD_REVIEW_MIN", "0.30")) # >= review
FRAUD_REJECT_MIN  = float(os.getenv("FRAUD_REJECT_MIN", "0.70")) # >= reject

# On-time payment ratio (0..1)
OTP_GOOD_MIN   = float(os.getenv("OTP_GOOD_MIN", "0.80"))
OTP_WARN_MIN   = float(os.getenv("OTP_WARN_MIN", "0.60"))

# Model performance thresholds
AUC_ACCEPT_MIN = float(os.getenv("AUC_ACCEPT_MIN", "0.70"))      # acceptable
AUC_GOOD_MIN   = float(os.getenv("AUC_GOOD_MIN",   "0.80"))      # good
AUC_REJECT_MAX = float(os.getenv("AUC_REJECT_MAX", "0.60"))      # below this => reject

# Drift thresholds (generic; if you use PSI, these map well)
DRIFT_OK_MAX     = float(os.getenv("DRIFT_OK_MAX", "0.10"))
DRIFT_WARN_MAX   = float(os.getenv("DRIFT_WARN_MAX", "0.25"))

# Gini acceptance threshold (common)
GINI_ACCEPT_MIN = float(os.getenv("GINI_ACCEPT_MIN", "0.40"))

# Model training data size (optional: set via env)
MODEL_TRAIN_ROWS = os.getenv("MODEL_TRAIN_ROWS", "").strip()
MODEL_TOTAL_ROWS = os.getenv("MODEL_TOTAL_ROWS", "").strip()


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)


def _faq_answer(user_msg: str, language: str) -> str | None:
    """
    Deterministic answers for common judge/auditor questions:
    - definitions
    - acceptance/rejection ranges
    - model metrics interpretation
    """
    q = (user_msg or "").strip().lower()
    if not q:
        return None

    # helpers
    def bn(s_en: str, s_bn: str) -> str:
        return s_bn if language == "bn" else s_en

    # --- On Time Payment Ratio ---
    if ("on time payment ratio" in q) or ("ontime payment ratio" in q) or ("on-time payment" in q) or ("otp ratio" in q):
        # meaning
        if ("what is" in q) or ("meaning" in q) or ("meant" in q) or ("মানে" in q) or ("কি" in q):
            return bn(
                "On Time Payment Ratio = (# of payments made on/before due date) / (total scheduled payments). "
                "Range: 0 to 1 (or 0% to 100%). Higher is better.",
                "On Time Payment Ratio = (সময়মতো/ডিউ ডেটের আগে পরিশোধ করা কিস্তির সংখ্যা) / (মোট নির্ধারিত কিস্তি)। "
                "রেঞ্জ: 0 থেকে 1 (বা 0% থেকে 100%)। বেশি হলে ভালো।"
            )

        # range / accept reject
        if ("range" in q) or ("accept" in q) or ("reject" in q) or ("threshold" in q) or ("কত হলে" in q) or ("রেঞ্জ" in q) or ("গ্রহণ" in q) or ("বাতিল" in q):
            return bn(
                f"On Time Payment Ratio policy ranges:\n"
                f"- Good: >= {OTP_GOOD_MIN:.2f}\n"
                f"- Warning/Medium: {OTP_WARN_MIN:.2f} to {OTP_GOOD_MIN:.2f}\n"
                f"- Risky: < {OTP_WARN_MIN:.2f}\n"
                f"Note: This is a guideline threshold; final approval depends on the overall PD (risk score) and other factors.",
                f"On Time Payment Ratio নীতিগত রেঞ্জ:\n"
                f"- ভালো: >= {OTP_GOOD_MIN:.2f}\n"
                f"- মাঝারি/সতর্ক: {OTP_WARN_MIN:.2f} থেকে {OTP_GOOD_MIN:.2f}\n"
                f"- ঝুঁকিপূর্ণ: < {OTP_WARN_MIN:.2f}\n"
                f"নোট: এটি গাইডলাইন থ্রেশহোল্ড; চূড়ান্ত সিদ্ধান্ত PD (risk score) ও অন্যান্য ফ্যাক্টরের উপর নির্ভর করে।"
            )

    # --- Risk score (PD) ---
    if ("risk score" in q) or ("pd" in q) or ("probability of default" in q) or ("ঝুঁকি" in q):
        if ("what is" in q) or ("meaning" in q) or ("meant" in q) or ("কি" in q) or ("মানে" in q):
            return bn(
                "Risk score (PD) = Probability of Default. Range: 0 to 1. Higher PD means higher chance of default (more risk).",
                "Risk score (PD) = Probability of Default (ডিফল্ট হওয়ার সম্ভাবনা)। রেঞ্জ: 0 থেকে 1। PD যত বেশি, ঝুঁকি তত বেশি।"
            )
        if ("range" in q) or ("accept" in q) or ("reject" in q) or ("threshold" in q) or ("কত হলে" in q) or ("রেঞ্জ" in q):
            return bn(
                f"Decision policy by PD (risk score):\n"
                f"- APPROVE: PD <= {PD_APPROVE_MAX:.2f}\n"
                f"- REJECT:  PD >= {PD_REJECT_MIN:.2f}\n"
                f"- REVIEW:  between {PD_APPROVE_MAX:.2f} and {PD_REJECT_MIN:.2f}",
                f"PD (risk score) অনুযায়ী সিদ্ধান্ত নীতি:\n"
                f"- APPROVE: PD <= {PD_APPROVE_MAX:.2f}\n"
                f"- REJECT:  PD >= {PD_REJECT_MIN:.2f}\n"
                f"- REVIEW:  {PD_APPROVE_MAX:.2f} থেকে {PD_REJECT_MIN:.2f} এর মধ্যে"
            )

    # --- Fraud score ---
    if ("fraud score" in q) or ("fraud" in q) or ("ফ্রড" in q) or ("জালিয়াত" in q):
        if ("what is" in q) or ("meaning" in q) or ("meant" in q) or ("কি" in q) or ("মানে" in q):
            return bn(
                "Fraud score = a model score estimating likelihood of fraud. Typically range 0 to 1. Higher means more fraud risk.",
                "Fraud score = জালিয়াতির সম্ভাবনা নির্দেশ করা একটি মডেল স্কোর। সাধারণত রেঞ্জ 0 থেকে 1। বেশি মানে বেশি ফ্রড ঝুঁকি।"
            )
        if ("range" in q) or ("accept" in q) or ("reject" in q) or ("threshold" in q) or ("কত হলে" in q) or ("রেঞ্জ" in q):
            return bn(
                f"Fraud policy ranges:\n"
                f"- OK:     <= {FRAUD_OK_MAX:.2f}\n"
                f"- REVIEW: >= {FRAUD_REVIEW_MIN:.2f} and < {FRAUD_REJECT_MIN:.2f}\n"
                f"- REJECT: >= {FRAUD_REJECT_MIN:.2f}",
                f"Fraud নীতিগত রেঞ্জ:\n"
                f"- OK:     <= {FRAUD_OK_MAX:.2f}\n"
                f"- REVIEW: >= {FRAUD_REVIEW_MIN:.2f} এবং < {FRAUD_REJECT_MIN:.2f}\n"
                f"- REJECT: >= {FRAUD_REJECT_MIN:.2f}"
            )

    # --- AUC ---
    if ("auc" in q) or ("area under the curve" in q):
        if ("what is" in q) or ("meaning" in q) or ("meant" in q) or ("কি" in q) or ("মানে" in q):
            return bn(
                "AUC (Area Under ROC Curve) measures how well the model separates good vs bad borrowers. Range: 0.5 to 1.0 (higher is better).",
                "AUC (Area Under ROC Curve) মডেল ভালো/খারাপ borrower আলাদা করতে কতটা সক্ষম তা মাপে। রেঞ্জ: 0.5 থেকে 1.0 (বেশি হলে ভালো)।"
            )
        if ("range" in q) or ("accept" in q) or ("reject" in q) or ("threshold" in q) or ("কত হলে" in q) or ("রেঞ্জ" in q):
            return bn(
                f"AUC guideline ranges:\n"
                f"- Reject/poor: < {AUC_REJECT_MAX:.2f}\n"
                f"- Acceptable: >= {AUC_ACCEPT_MIN:.2f}\n"
                f"- Good:       >= {AUC_GOOD_MIN:.2f}\n"
                f"(0.50 ≈ random; 1.00 = perfect)",
                f"AUC গাইডলাইন রেঞ্জ:\n"
                f"- Reject/খারাপ: < {AUC_REJECT_MAX:.2f}\n"
                f"- Acceptable/গ্রহণযোগ্য: >= {AUC_ACCEPT_MIN:.2f}\n"
                f"- Good/ভালো:     >= {AUC_GOOD_MIN:.2f}\n"
                f"(0.50 ≈ র‍্যান্ডম; 1.00 = পারফেক্ট)"
            )

    # --- Drift score ---
    if ("drift score" in q) or ("data drift" in q) or ("drift" in q) or ("ড্রিফ্ট" in q):
        if ("what is" in q) or ("meaning" in q) or ("meant" in q) or ("কি" in q) or ("মানে" in q):
            return bn(
                "Drift score measures how much current (live) data distribution differs from training data. Higher means the model may be less reliable.",
                "Drift score বর্তমান (লাইভ) ডেটা ট্রেনিং ডেটা থেকে কতটা বদলেছে তা মাপে। বেশি হলে মডেল কম নির্ভরযোগ্য হতে পারে।"
            )
        if ("range" in q) or ("accept" in q) or ("reject" in q) or ("threshold" in q) or ("কত হলে" in q) or ("রেঞ্জ" in q):
            return bn(
                f"Drift guideline ranges:\n"
                f"- OK/stable:     <= {DRIFT_OK_MAX:.2f}\n"
                f"- Warning:       > {DRIFT_OK_MAX:.2f} and <= {DRIFT_WARN_MAX:.2f}\n"
                f"- High drift:    > {DRIFT_WARN_MAX:.2f} (retrain/check recommended)",
                f"Drift গাইডলাইন রেঞ্জ:\n"
                f"- OK/স্টেবল:     <= {DRIFT_OK_MAX:.2f}\n"
                f"- Warning:       > {DRIFT_OK_MAX:.2f} এবং <= {DRIFT_WARN_MAX:.2f}\n"
                f"- High drift:    > {DRIFT_WARN_MAX:.2f} (রিট্রেন/চেক রেকমেন্ডেড)"
            )

    # --- Gini ---
    if ("gini" in q) or ("gini score" in q) or ("জিনি" in q):
        if ("what is" in q) or ("meaning" in q) or ("meant" in q) or ("কি" in q) or ("মানে" in q):
            return bn(
                "Gini score is another ranking/separation metric. For binary models, Gini = 2*AUC - 1. Range: 0 to 1 (higher is better).",
                "Gini score আরেকটি র‍্যাঙ্কিং/সেপারেশন মেট্রিক। বাইনারি মডেলে Gini = 2*AUC - 1। রেঞ্জ: 0 থেকে 1 (বেশি হলে ভালো)।"
            )
        if ("range" in q) or ("accept" in q) or ("reject" in q) or ("threshold" in q) or ("কত হলে" in q) or ("রেঞ্জ" in q):
            return bn(
                f"Gini guideline ranges:\n"
                f"- Acceptable: >= {GINI_ACCEPT_MIN:.2f}\n"
                f"- Weak/Review: 0.20 to {GINI_ACCEPT_MIN:.2f}\n"
                f"- Poor/Reject: < 0.20",
                f"Gini গাইডলাইন রেঞ্জ:\n"
                f"- Acceptable/গ্রহণযোগ্য: >= {GINI_ACCEPT_MIN:.2f}\n"
                f"- Weak/Review: 0.20 থেকে {GINI_ACCEPT_MIN:.2f}\n"
                f"- Poor/Reject: < 0.20"
            )

    # --- Training data size ---
    if ("how many" in q and ("data" in q or "persons" in q or "people" in q)) or ("dataset size" in q) or ("কত জন" in q) or ("ডেটা" in q and "কত" in q):
        if MODEL_TRAIN_ROWS or MODEL_TOTAL_ROWS:
            return bn(
                f"Active model data usage:\n"
                f"- Training rows: {MODEL_TRAIN_ROWS or 'unknown'}\n"
                f"- Total rows (if tracked): {MODEL_TOTAL_ROWS or 'unknown'}",
                f"Active model ডেটা ব্যবহার:\n"
                f"- Training rows: {MODEL_TRAIN_ROWS or 'unknown'}\n"
                f"- Total rows (যদি ট্র্যাক করা হয়): {MODEL_TOTAL_ROWS or 'unknown'}"
            )
        return bn(
            "This deployment does not store the training dataset size in the app. "
            "Set env vars MODEL_TRAIN_ROWS / MODEL_TOTAL_ROWS to show it in chat.",
            "এই ডিপ্লয়মেন্টে ট্রেনিং ডেটাসেট সাইজ অ্যাপে সংরক্ষণ করা নেই। "
            "চ্যাটে দেখাতে MODEL_TRAIN_ROWS / MODEL_TOTAL_ROWS env সেট করুন।"
        )

    return None



def fetch_last_chat_turn(borrower_id: str | None, decision_id: int | None) -> tuple[str | None, str | None]:
    """
    Returns (last_user_message, last_assistant_message)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT user_message, assistant_message
        FROM chat_logs
        WHERE (? IS NULL OR borrower_id = ?)
          AND (? IS NULL OR decision_id = ?)
        ORDER BY id DESC
        LIMIT 1
        """,
        (borrower_id, borrower_id, decision_id, decision_id)
    )

    row = cur.fetchone()
    conn.close()
    return (row[0], row[1]) if row else (None, None)


def is_followup_question(msg: str) -> bool:
    msg = msg.lower().strip()
    return msg in {
        "more detail",
        "give me more detail",
        "explain more",
        "tell me more",
        "why",
        "why so",
        "why is that",
        "elaborate",
        "details",
    }


# ============================================================
# AI Credit Officer – Embedded Chat (Explainable & Grounded)
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Grounded chatbot:
    - Retrieves decision facts from ExplainChain (SQLite)
    - Generates borrower/officer/auditor-friendly explanation
    """
    try:
        msg = (req.message or "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="message is required")
        raw_user_msg = msg



        decision = None
        if req.decision_id is not None:
            decision = fetch_decision_by_id(req.decision_id)
        elif req.borrower_id:
            decision = fetch_latest_decision_for_borrower(req.borrower_id)

        if not decision:
            return {
                "answer": "No decision found yet. Please score a borrower first.",
                "grounded": False,
            }

        # ✅ Conversational memory (1-turn)
        last_user_msg, last_bot_msg = fetch_last_chat_turn(req.borrower_id, req.decision_id)

        if is_followup_question(msg) and last_user_msg:
            msg = f"Previous question:\n{last_user_msg}\n\nFollow-up:\n{raw_user_msg}"






        # ✅ FAQ-first: answer common metric/policy questions without LLM
        faq = _faq_answer(msg, req.language)
        if faq:
            log_chat(req.borrower_id, req.decision_id, raw_user_msg, faq)

            return {"answer": faq, "grounded": True}


        context = build_grounded_context(decision)

        if req.mode == "auditor":
            style = "Write formal audit-ready explanation. Use PD and SHAP. Include impact and pct_influence."
        elif req.mode == "officer":
            style = "Write concise loan-officer explanation with improvement tips. Prefer pct_influence; impact optional."
        else:
            style = "Write simple borrower-friendly explanation. Use pct_influence (%) not impact. Do not show raw SHAP impact unless asked."


        lang = "Respond in Bangla." if req.language == "bn" else "Respond in English."

        system_prompt = (
            "You are AI Credit Officer Assistant.\n"
            "You must ONLY use the provided decision facts.\n"
            "Never invent reasons.\n"
        )


        user_prompt = f"""
{context}

USER QUESTION:
{msg}

OUTPUT RULES:
- {style}
- {lang}
- If information is missing, say so clearly.
""".strip()

        try:
            answer = await call_llm_openai_compatible(system_prompt, user_prompt)
        except Exception:
            answer = deterministic_fallback_answer(msg, decision, req.mode, req.language)

        log_chat(decision.get("borrower_id"), decision.get("id"), raw_user_msg, answer)


        return {
            "answer": answer,
            "decision_id": decision.get("id"),
            "borrower_id": decision.get("borrower_id"),
            "grounded": True
        }

    except HTTPException:
        raise
    except Exception as e:
        # This makes the error visible in the front-end instead of a blank 500
        raise HTTPException(status_code=500, detail=f"/chat failed: {e}")


