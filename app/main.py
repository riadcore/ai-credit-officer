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

from .schemas import CreditScoreRequest, CreditScoreResponse, Factor
import train_from_export  # training pipeline


# -------------------------------------------------------------------
# Base paths (Railway/Linux safe)
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "training"
VERSIONS_DIR = MODELS_DIR / "versions"

# Ensure folders exist BEFORE any file writes (Railway cold boot safe)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = DATA_DIR / "training_master.csv"
if not MASTER_PATH.exists():
    MASTER_PATH.write_text("", encoding="utf-8")  # create empty file


# -------------------------------------------------------------------
# Model files
# -------------------------------------------------------------------
BASE_MODEL_PATH = MODELS_DIR / "base_model.pkl"
ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.pkl"
FIRST_MODEL = MODELS_DIR / "credit_model.pkl"  # initial seed model (if present)

# One-time bootstrap: create base + active from the first model
if FIRST_MODEL.exists():
    if not BASE_MODEL_PATH.exists():
        shutil.copy2(FIRST_MODEL, BASE_MODEL_PATH)
        print("✅ Base model created:", BASE_MODEL_PATH)

    if not ACTIVE_MODEL_PATH.exists():
        shutil.copy2(FIRST_MODEL, ACTIVE_MODEL_PATH)
        print("✅ Active model initialized:", ACTIVE_MODEL_PATH)

# Always score using ACTIVE model unless overridden by env var
MODEL_PATH = os.getenv("MODEL_PATH", str(ACTIVE_MODEL_PATH))


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="AI Credit & Inclusion Platform - MVP",
    version="0.5.0",
)


@app.get("/")
def read_root():
    return RedirectResponse(url="/dashboard")


# -------------------------------------------------------------------
# Training metrics globals (use absolute paths)
# -------------------------------------------------------------------
TRAINING_METRICS: dict = {}
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"
MODEL_HISTORY_LOG = MODELS_DIR / "model_train_log.jsonl"


def load_training_metrics() -> None:
    """Load last training metrics into TRAINING_METRICS."""
    global TRAINING_METRICS
    if TRAINING_METRICS_PATH.exists():
        try:
            TRAINING_METRICS = json.loads(TRAINING_METRICS_PATH.read_text(encoding="utf-8"))
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

templates = Jinja2Templates(directory="app/templates")


# -------------------------------------------------------------------
# In-memory model globals
# -------------------------------------------------------------------
model_bundle = None
model = None
feature_cols: list[str] = []
background = None
fraud_model = None
explainer = None


def build_shap_explainer():
    """Build a SHAP explainer robustly for tree models."""
    try:
        if background is not None:
            return shap.TreeExplainer(model, data=background)
        return shap.TreeExplainer(model)
    except Exception:
        # last-resort
        return shap.TreeExplainer(model)


def load_model_bundle(path: str | None = None) -> None:
    """
    Load model bundle + rebuild SHAP explainer so scoring uses latest ACTIVE model.
    Railway-safe: guard if file doesn't exist yet.
    """
    global model_bundle, model, feature_cols, background, fraud_model, explainer, MODEL_PATH

    if path is not None:
        MODEL_PATH = path

    p = Path(MODEL_PATH)
    if not p.exists():
        # Don't crash the whole app on cold boot; endpoints will error nicely if used
        print(f"❌ MODEL_PATH not found on disk: {p}")
        model_bundle = None
        model = None
        feature_cols = []
        background = None
        fraud_model = None
        explainer = None
        return

    model_bundle = joblib.load(str(p))
    model = model_bundle["model"]
    feature_cols = model_bundle["features"]
    background = model_bundle.get("background")
    fraud_model = model_bundle.get("fraud_model")

    explainer = build_shap_explainer()
    print(f"✅ Loaded ACTIVE model bundle from: {p}")


# -------------------------------------------------------------------
# SQLite ExplainChain ledger
# -------------------------------------------------------------------
DB_PATH = BASE_DIR / "decisions.db"


def init_db():
    """Create SQLite table if it doesn't exist and upgrade schema if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

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

    # schema upgrade for existing DBs
    cur.execute("PRAGMA table_info(decisions)")
    cols = [row[1] for row in cur.fetchall()]

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


# -------------------------------------------------------------------
# Startup initialization (Railway-safe)
# -------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    init_db()
    load_training_metrics()
    load_model_bundle()


# -------------------------------------------------------------------
# Decision logging + fetch
# -------------------------------------------------------------------
def log_decision(
    req: CreditScoreRequest,
    risk_score: float,
    decision: str,
    explanation: str,
    top_factors: list[Factor],
    fraud_score: float,
    fraud_label: str,
) -> None:
    shap_factors_json = json.dumps([f.dict() for f in top_factors])
    created_at = datetime.datetime.utcnow().isoformat() + "Z"

    conn = sqlite3.connect(str(DB_PATH))
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
    conn = sqlite3.connect(str(DB_PATH))
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
    repaid: bool
    days_late: int | None = None


# -------------------------------------------------------------------
# Core credit & fraud helpers
# -------------------------------------------------------------------
def _ensure_model_loaded():
    if model is None or not feature_cols:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Ensure models/active_model.pkl exists and restart.",
        )


def compute_credit(req: CreditScoreRequest):
    """Return (pd_default, decision, explanation, top_factors[list[Factor]])"""
    _ensure_model_loaded()

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
    pd_default = float(model.predict_proba(X)[0, 1])

    if pd_default < 0.2:
        decision = "APPROVE"
        explanation = "Low predicted default risk based on income and repayment behavior."
    elif pd_default < 0.4:
        decision = "REVIEW"
        explanation = "Moderate risk. MFI officer should review and potentially adjust limit."
    else:
        decision = "REJECT_OR_SMALL_LIMIT"
        explanation = "High predicted default risk. Consider smaller loan or financial coaching first."

    # SHAP explanations (robust fallback)
    try:
        if explainer is None:
            shap_row = np.zeros(len(feature_cols))
        else:
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_row = np.array(shap_values[1][0])
            else:
                shap_row = np.array(shap_values[0])
    except Exception:
        shap_row = np.zeros(len(feature_cols))

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
            )
        )

    return pd_default, decision, explanation, top_factors


def compute_fraud(req: CreditScoreRequest):
    """Return (fraud_score[0-1], fraud_label, flags[list[str]])"""
    _ensure_model_loaded()

    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Fraud model is not available in the model bundle.")

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
    normalized_score = 1 / (1 + np.exp(score))
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
    conn = sqlite3.connect(str(DB_PATH))
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
    return TRAINING_METRICS


@app.get("/admin/model_history")
def get_model_history():
    history: list[dict] = []
    if MODEL_HISTORY_LOG.exists():
        try:
            with MODEL_HISTORY_LOG.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            history = []

    history.sort(key=lambda r: r.get("trained_at", ""))
    return {"history": history}


@app.get("/admin/download_model")
def download_model(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    model_path = VERSIONS_DIR / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=filename,
    )


@app.post("/admin/deploy_model")
def deploy_model(version: str):
    record = None
    if MODEL_HISTORY_LOG.exists():
        with MODEL_HISTORY_LOG.open("r", encoding="utf-8") as f:
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

    model_path = record.get("model_file_path")
    if model_path:
        model_path = Path(model_path)
    else:
        model_path = MODELS_DIR / f"credit_model_{version}.pkl"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    # deploy -> ACTIVE model
    shutil.copy2(str(model_path), str(ACTIVE_MODEL_PATH))
    load_model_bundle(str(ACTIVE_MODEL_PATH))

    metrics = {
        "last_auc": record.get("auc"),
        "rows": record.get("rows_used"),
        "last_trained_at": record.get("trained_at"),
        "version_path": record.get("model_file_path", str(model_path)),
        "version_tag": record.get("model_version", version),
        "drift_score": record.get("drift_score"),
        "drift_label": record.get("drift_label"),
    }

    try:
        TRAINING_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception:
        pass

    global TRAINING_METRICS
    TRAINING_METRICS = metrics

    return {"status": "ok", "metrics": metrics}


@app.post("/admin/delete_model")
def delete_model(version: str):
    active_version = (TRAINING_METRICS or {}).get("version_tag")
    if active_version == version:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the active model. Deploy another version first.",
        )

    if not MODEL_HISTORY_LOG.exists():
        raise HTTPException(status_code=404, detail="History log does not exist")

    kept_lines: list[str] = []
    deleted_record = None

    with MODEL_HISTORY_LOG.open("r", encoding="utf-8") as f:
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
    with MODEL_HISTORY_LOG.open("w", encoding="utf-8") as f:
        for l in kept_lines:
            f.write(l if l.endswith("\n") else l.rstrip() + "\n")

    # delete its .pkl
    model_path = deleted_record.get("model_file_path")
    if model_path:
        model_path = Path(model_path)
    else:
        model_path = MODELS_DIR / f"credit_model_{version}.pkl"

    if model_path.exists():
        try:
            model_path.unlink()
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


@app.get("/admin/active_model_info")
def active_model_info():
    p = Path(MODEL_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Model path not found: {p}")

    # hash first ~1MB to identify file quickly (fast & enough for checking changes)
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1024 * 1024))

    return {
        "MODEL_PATH": str(p),
        "mtime_utc": datetime.datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z",
        "size_bytes": p.stat().st_size,
        "sha256_1mb": h.hexdigest(),
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
    return {"decisions": fetch_recent_decisions()}


@app.delete("/history/{decision_id}")
def delete_decision(decision_id: int):
    conn = sqlite3.connect(str(DB_PATH))
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
    conn = sqlite3.connect(str(DB_PATH))
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
    conn = sqlite3.connect(str(DB_PATH))
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
        line = ",".join([str(x) for x in r])
        lines.append(line + "\n")

    return "".join(lines)


@app.post("/score", response_model=CreditScoreResponse)
def score_credit(req: CreditScoreRequest):
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

        load_training_metrics()

        newest_path = (TRAINING_METRICS or {}).get("candidate_path") or (TRAINING_METRICS or {}).get("version_path")
        if newest_path:
            newest_path = Path(newest_path)

        if newest_path and newest_path.exists():
            shutil.copy2(str(newest_path), str(ACTIVE_MODEL_PATH))
            load_model_bundle(str(ACTIVE_MODEL_PATH))
        else:
            load_model_bundle(str(ACTIVE_MODEL_PATH))

        return {
            "status": "ok",
            "message": "Retraining complete and activated.",
            "metrics": TRAINING_METRICS,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")


UPLOAD_DIR = DATA_DIR  # Railway-safe
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/admin/upload_training_csv")
async def upload_training_csv(file: UploadFile = File(...)):
    try:
        import datetime

        # ✅ NEW: timestamped filename to avoid retraining old data
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        original = file.filename or "training.csv"
        filename = f"{stamp}_{original}"

        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files allowed.")

        save_path = UPLOAD_DIR / filename

        content = await file.read()
        save_path.write_bytes(content)

        return {
            "status": "uploaded",
            "file": filename,
            "path": str(save_path),
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

        log_decision(
            req=req,
            risk_score=pd_default,
            decision=decision_label,
            explanation=explanation,
            top_factors=top_factors,
            fraud_score=fraud_score,
            fraud_label=fraud_label,
        )

        return {
            "borrower_id": req.borrower_id,
            "credit": {
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/decision failed: {e}")
