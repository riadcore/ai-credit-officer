import os
import glob
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import shutil


# ------------------------------------------------------------
# Railway-safe absolute paths (based on repo root)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "training"
VERSIONS_DIR = MODELS_DIR / "versions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_PATH = MODELS_DIR / "base_model.pkl"
ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.pkl"
CANDIDATE_META_PATH = MODELS_DIR / "candidate.json"

# Latest snapshot path (bundle)
CREDIT_MODEL_PATH = MODELS_DIR / "credit_model.pkl"

# Where the admin card reads from
METRICS_PATH = MODELS_DIR / "training_metrics.json"

# Master blended dataset (continuous learning)
MASTER_PATH = DATA_DIR / "training_master.csv"

# Model version history log (one JSON per line)
LOG_FILE = MODELS_DIR / "model_train_log.jsonl"

# Ensure master file exists (Railway cold boot safe)
if not MASTER_PATH.exists():
    MASTER_PATH.write_text("", encoding="utf-8")


# Global list of features
FEATURE_COLS = [
    "monthly_income",
    "monthly_expense",
    "age",
    "has_smartphone",
    "has_wallet",
    "avg_wallet_balance",
    "on_time_payment_ratio",
    "num_loans_taken",
]

# ------------------------------------------------------------
# BOOTSTRAP: train base model from a specific CSV file
# ------------------------------------------------------------
BOOTSTRAP_CSV = os.getenv("BOOTSTRAP_CSV", str(DATA_DIR / "data_for_training.csv"))
FORCE_BOOTSTRAP = os.getenv("FORCE_BOOTSTRAP", "0") == "1"


def compute_drift_score(old_df: pd.DataFrame | None,
                        new_df: pd.DataFrame,
                        feature_cols: list[str]) -> float:
    """Simple numeric drift score in [0, 1]."""
    if old_df is None or len(old_df) == 0:
        return 0.0

    numeric_cols = [c for c in feature_cols if c in old_df.columns]
    if not numeric_cols:
        return 0.0

    scores = []
    for col in numeric_cols:
        try:
            old = old_df[col].astype(float)
            new = new_df[col].astype(float)
        except Exception:
            continue

        if len(old) == 0 or len(new) == 0:
            continue

        m_old = float(old.mean())
        s_old = float(old.std()) or 1.0
        m_new = float(new.mean())

        z = abs(m_new - m_old) / (s_old + 1e-6)
        z = min(z, 3.0)
        scores.append(z / 3.0)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def label_drift(score: float) -> str:
    if score <= 0.0:
        return "BASELINE"
    if score < 0.2:
        return "STABLE"
    if score < 0.5:
        return "MONITOR"
    return "POSSIBLE_DRIFT"


def _load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and do minimal safety cleaning:
    - keep only expected columns + target
    - coerce to numeric
    - drop malformed rows
    """
    df = pd.read_csv(csv_path)

    expected = FEATURE_COLS + ["target"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[expected].copy()

    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna()
    removed = before - len(df)
    if removed > 0:
        print(f"[CLEAN] Dropped {removed} malformed/NaN rows")

    df["target"] = df["target"].astype(int)
    return df


def _train_bundle(df: pd.DataFrame) -> tuple[dict, float]:
    """Train credit + fraud model bundle from a dataframe. Returns (bundle, auc)."""
    if "target" not in df.columns:
        raise ValueError("Training data must contain 'target' column with 0/1 labels.")

    X = df[FEATURE_COLS].values
    y = df["target"].values

    unique, counts = np.unique(y, return_counts=True)
    print("[TRAIN] Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  target={cls}: {cnt} rows")

    if len(unique) < 2:
        raise ValueError("Need BOTH classes in target (0 and 1). Your CSV has only one class.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    credit_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    credit_model.fit(X_train, y_train)

    y_val_pred = credit_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"[TRAIN] Validation ROC-AUC: {auc:.3f}")

    good_mask = (y_train == 0)
    X_fraud_train = X_train[good_mask]
    if X_fraud_train.shape[0] < 20:
        print(f"[TRAIN] WARNING: Only {X_fraud_train.shape[0]} good rows; fallback to full X_train.")
        X_fraud_train = X_train

    fraud_model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )
    fraud_model.fit(X_fraud_train)
    print(f"[TRAIN] Fraud model trained on {X_fraud_train.shape[0]} rows.")

    bg_size = min(200, X_train.shape[0])
    bg_idx = np.random.choice(X_train.shape[0], size=bg_size, replace=False)
    background = X_train[bg_idx]

    bundle = {
        "model": credit_model,
        "features": FEATURE_COLS,
        "background": background,
        "fraud_model": fraud_model,
    }
    return bundle, float(auc)


def bootstrap_from_specific_csv(csv_path: str) -> dict:
    """
    Creates:
      - models/credit_model.pkl
      - models/base_model.pkl
      - models/active_model.pkl
      - models/versions/model_bundle_YYYYMMDD_HHMMSS.pkl
    from one chosen CSV file.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"BOOTSTRAP_CSV not found: {p}")

    print(f"[BOOTSTRAP] Using CSV: {p}")
    df = _load_and_clean_csv(str(p))

    # Initialize master from bootstrap dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_PATH, index=False)
    print(f"[BOOTSTRAP] Master dataset initialized at {MASTER_PATH} ({len(df)} rows)")

    bundle, auc = _train_bundle(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = VERSIONS_DIR / f"model_bundle_{ts}.pkl"
    joblib.dump(bundle, version_path)

    CANDIDATE_META_PATH.write_text(json.dumps({
        "candidate_bundle": str(version_path.name),
        "timestamp": ts
    }, indent=2), encoding="utf-8")

    joblib.dump(bundle, CREDIT_MODEL_PATH)

    shutil.copy2(CREDIT_MODEL_PATH, BASE_MODEL_PATH)
    shutil.copy2(CREDIT_MODEL_PATH, ACTIVE_MODEL_PATH)

    now = datetime.utcnow()

    metrics = {
        "last_auc": float(auc),
        "rows": int(len(df)),
        "last_file": str(p),
        "last_trained_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidate_bundle": str(version_path.name),
        "candidate_path": str(version_path),
        "version_tag": ts,
        "master_path": str(MASTER_PATH),
        "drift_score": 0.0,
        "drift_label": "BASELINE",
        "bootstrap": True,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dataset_str = df.sort_index(axis=1).to_csv(index=False)
    dataset_hash = hashlib.md5(dataset_str.encode("utf-8")).hexdigest()

    record = {
        "model_version": ts,
        "rows_used": int(len(df)),
        "auc": float(auc),
        "dataset_hash": dataset_hash,
        "trained_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_file_path": str(version_path),
        "master_path": str(MASTER_PATH),
        "drift_score": 0.0,
        "drift_label": "BASELINE",
        "bootstrap": True,
    }

    with LOG_FILE.open("a", encoding="utf-8") as logf:
        logf.write(json.dumps(record) + "\n")

    print(f"[BOOTSTRAP] Saved: {CREDIT_MODEL_PATH}")
    print(f"[BOOTSTRAP] Saved: {BASE_MODEL_PATH}")
    print(f"[BOOTSTRAP] Saved: {ACTIVE_MODEL_PATH}")
    print(f"[BOOTSTRAP] Saved version: {version_path}")

    return metrics


def run():
    """
    Normal retrain flow:
      - use latest uploaded CSV in data/training (excluding training_master.csv)
      - blend into training_master.csv
      - train bundle
      - save versioned candidate + metrics
    """
    all_files = glob.glob(str(DATA_DIR / "*.csv"))
    all_files = [f for f in all_files if Path(f).name != MASTER_PATH.name]

    if not all_files:
        raise FileNotFoundError("No training CSV found in data/training/ (upload one in the Admin UI).")

    latest_file = max(all_files, key=os.path.getmtime)
    print(f"[TRAIN] Using NEW training file: {latest_file}")

    new_df = _load_and_clean_csv(latest_file)
    print(f"[TRAIN] New CSV rows (after cleaning): {len(new_df)}")

    old_df = None
    if MASTER_PATH.exists() and MASTER_PATH.stat().st_size > 0:
        try:
            old_df = pd.read_csv(MASTER_PATH)
        except Exception:
            old_df = None

    if old_df is not None and len(old_df) > 0:
        blended = pd.concat([old_df, new_df], ignore_index=True)
        before_dedup = len(blended)
        blended = blended.drop_duplicates()
        removed = before_dedup - len(blended)
        print(f"[TRAIN] After merge+dedup: {len(blended)} rows (removed {removed})")
        drift_score = compute_drift_score(old_df, new_df, FEATURE_COLS)
    else:
        blended = new_df.copy()
        print(f"[TRAIN] No master yet. Starting master with {len(blended)} rows.")
        drift_score = 0.0

    drift_label = label_drift(drift_score)
    print(f"[TRAIN] Drift score={drift_score:.3f}, label={drift_label}")

    blended.to_csv(MASTER_PATH, index=False)
    print(f"[TRAIN] Master dataset updated at {MASTER_PATH}")

    bundle, auc = _train_bundle(blended)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = VERSIONS_DIR / f"model_bundle_{ts}.pkl"
    joblib.dump(bundle, version_path)

    CANDIDATE_META_PATH.write_text(json.dumps({
        "candidate_bundle": str(version_path.name),
        "timestamp": ts
    }, indent=2), encoding="utf-8")

    joblib.dump(bundle, CREDIT_MODEL_PATH)

    now = datetime.utcnow()
    metrics = {
        "last_auc": float(auc),
        "rows": int(len(blended)),
        "last_file": str(latest_file),
        "last_trained_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidate_bundle": str(version_path.name),
        "candidate_path": str(version_path),
        "version_tag": ts,
        "master_path": str(MASTER_PATH),
        "drift_score": float(drift_score),
        "drift_label": drift_label,
        "bootstrap": False,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dataset_str = blended.sort_index(axis=1).to_csv(index=False)
    dataset_hash = hashlib.md5(dataset_str.encode("utf-8")).hexdigest()

    record = {
        "model_version": ts,
        "rows_used": int(len(blended)),
        "auc": float(auc),
        "dataset_hash": dataset_hash,
        "trained_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_file_path": str(version_path),
        "master_path": str(MASTER_PATH),
        "drift_score": float(drift_score),
        "drift_label": drift_label,
        "bootstrap": False,
    }

    with LOG_FILE.open("a", encoding="utf-8") as logf:
        logf.write(json.dumps(record) + "\n")

    print(f"[TRAIN] Saved candidate bundle: {version_path}")
    return metrics


if __name__ == "__main__":
    if FORCE_BOOTSTRAP:
        info = bootstrap_from_specific_csv(BOOTSTRAP_CSV)
        print(f"[BOOTSTRAP] Finished. Validation ROC-AUC={info['last_auc']:.3f}, rows used={info['rows']}")
    else:
        info = run()
        print(f"[TRAIN] Finished. Validation ROC-AUC={info['last_auc']:.3f}, rows used={info['rows']}")
