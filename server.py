
"""
Bank RFI Prediction Bot -- single-file backend + frontend, no subfolders.
 
Everything lives here on purpose: Sumsub API client, SQLite persistence,
feature engineering, statistical pattern analysis, the predictive model, the
ingestion logic, the FastAPI routes, the background scheduler, and the
dashboard HTML itself (embedded as a string and served directly). One file,
one `requirements.txt`, one start command, zero subfolders, no Dockerfile.
 
Run locally:
    cp .env.example .env        # fill in real SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000
 
Deploy on Render:
    Build command: pip install -r requirements.txt
    Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
    Environment variables: SUMSUB_APP_TOKEN, SUMSUB_SECRET_KEY (required --
    Render already has these set with exactly these names). Everything else
    has a sensible default -- see README.md.
 
HOW DATA GETS IN (no separate/local data files -- everything comes from the
Sumsub API):
  1. Webhooks (the real mechanism for "keeps growing as tags change on
     Sumsub"): register this app's /api/webhooks/sumsub URL in Sumsub
     Dashboard -> Settings -> Webhooks for Transaction Monitoring events.
     Every lifecycle event Sumsub fires includes the transaction's Sumsub ID
     (kytTxnId, confirmed against Sumsub's real webhook payload docs) -- the
     bot fetches that transaction's tags immediately and stores it if
     they include "bank rfi".
  2. An hourly background job (co-located in this same process) re-runs a
     best-effort historical backfill, then retrains on everything in the
     database -- see WHAT'S VERIFIED VS. ASSUMED below for exactly which
     backfill strategies are confirmed-working vs. experimental.
 
WHAT'S VERIFIED VS. ASSUMED (read this before trusting a prediction) -- this
section reflects live research against Sumsub's published API docs, not
guesswork:
  - CONFIRMED: fetching one transaction + its tags/notes by ID
    (/resources/kyt/txns/{id}/one, .../tags, .../notes), the HMAC request
    signing scheme, and the webhook payload field names (kytTxnId is the
    correct ID to re-fetch; there is no "tags" field on the webhook payload
    itself, so a follow-up GET is required and is what this bot does).
  - CONFIRMED BUT NARROW: /resources/kyt/txns/query/-  only returns results
    when filtered to data.type=travelRule -- i.e. it is documented for
    crypto Travel Rule transactions specifically, not general fiat KYT
    transactions, and it has no tag filter or offset-based pagination (max
    100 items per call, sorted by date). This bot still calls it every
    cycle as a best-effort historical-backfill attempt with a date-cursor
    pagination workaround, but if your transactions aren't Travel Rule type
    it will legitimately return zero rows every time -- that's expected,
    not a bug.
  - UNCONFIRMED / EXPERIMENTAL: enumerating applicants and walking each
    one's transactions -- Sumsub's public docs do not document a "list this
    applicant's KYT transactions" endpoint, so this strategy tries a couple
    of plausible paths and silently no-ops if none exist on your account.
  - UNCONFIRMED: which field (if any) names the bank that sent an RFI. The
    bot surfaces every plausible place this could live (a second tag, a
    compliance note, a best-effort counterparty-bank field) instead of
    picking one.
  - BOTTOM LINE: for an account whose transactions are not Travel Rule/
    crypto type, historical bulk backfill has no confirmed API path in
    Sumsub's public docs today -- the reliable, fully-automatic mechanism
    is the webhook, which is exactly what makes "grows as new bank-rfi tags
    are added on Sumsub" work going forward. Open the running app's
    dashboard -- it live-checks your credentials and tells you plainly
    what's actually happening, rather than requiring you to trust this
    comment.
"""
from __future__ import annotations
 
import hashlib
import hmac
import json
import logging
import math
import os
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
 
import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from scipy import stats as scipy_stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
 
load_dotenv()
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank_rfi_bot")
 
HERE = os.path.dirname(os.path.abspath(__file__))
 
# ============================================================================
# Config (env vars) -- only SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY are required;
# everything else below has a working default.
# ============================================================================
 
SUMSUB_BASE_URL = os.environ.get("SUMSUB_BASE_URL", "https://api.sumsub.com")
BANK_RFI_TAG = os.environ.get("BANK_RFI_TAG", "bank rfi")
DB_PATH = os.environ.get("BOT_DB_PATH", os.path.join(HERE, "bank_rfi_bot.db"))
MIN_ML_SAMPLES = int(os.environ.get("MIN_ML_SAMPLES", "15"))
INGEST_INTERVAL_MINUTES = int(os.environ.get("INGEST_INTERVAL_MINUTES", "60"))  # hourly by default, per spec
CONNECTION_CHECK_CACHE_SECONDS = 60
 
 
# ============================================================================
# Sumsub API client -- HMAC-SHA256 request signing per
# https://docs.sumsub.com/reference/authentication
# ============================================================================
 
class SumsubAuthError(RuntimeError):
    pass
 
 
class SumsubAPIError(RuntimeError):
    def __init__(self, status: int, body: str, method: str, path: str):
        self.status = status
        self.body = body
        super().__init__(f"{method} {path} -> HTTP {status}: {body[:500]}")
 
 
@dataclass
class SumsubResponse:
    status: int
    json_body: Optional[Any]
    raw_body: str
 
    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300
 
 
class SumsubClient:
    def __init__(self, app_token: str | None = None, secret_key: str | None = None, base_url: str = SUMSUB_BASE_URL):
        self.app_token = app_token or os.environ.get("SUMSUB_APP_TOKEN")
        self.secret_key = secret_key or os.environ.get("SUMSUB_SECRET_KEY")
        if not self.app_token or not self.secret_key:
            raise SumsubAuthError("Missing SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY environment variables.")
        self.base_url = base_url
        self._session = requests.Session()
 
    def _sign(self, ts: int, method: str, path_with_query: str, body_bytes: bytes) -> str:
        to_sign = str(ts).encode() + method.upper().encode() + path_with_query.encode() + (body_bytes or b"")
        return hmac.new(self.secret_key.encode(), to_sign, hashlib.sha256).hexdigest()
 
    def _request(self, method: str, path_with_query: str, json_body=None, raise_on_error: bool = True) -> SumsubResponse:
        ts = int(time.time())
        body_bytes = b""
        headers = {"Accept": "application/json"}
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        headers.update({
            "X-App-Token": self.app_token,
            "X-App-Access-Sig": self._sign(ts, method, path_with_query, body_bytes),
            "X-App-Access-Ts": str(ts),
        })
        resp = self._session.request(
            method.upper(), self.base_url + path_with_query,
            data=body_bytes if body_bytes else None, headers=headers, timeout=30,
        )
        parsed = None
        try:
            parsed = resp.json()
        except ValueError:
            pass
        if raise_on_error and not (200 <= resp.status_code < 300):
            raise SumsubAPIError(resp.status_code, resp.text, method, path_with_query)
        return SumsubResponse(status=resp.status_code, json_body=parsed, raw_body=resp.text)
 
    def get(self, path_with_query: str, raise_on_error: bool = True) -> SumsubResponse:
        return self._request("GET", path_with_query, raise_on_error=raise_on_error)
 
    def get_transaction(self, txn_id: str) -> dict:
        return self.get(f"/resources/kyt/txns/{txn_id}/one").json_body
 
    def get_txn_tags(self, txn_id: str) -> list[str]:
        resp = self.get(f"/resources/kyt/txns/{txn_id}/tags")
        items = resp.json_body or []
        return [item.get("label") for item in items if isinstance(item, dict) and item.get("label")]
 
    def get_txn_notes(self, txn_id: str) -> list[dict]:
        resp = self.get(f"/resources/kyt/txns/{txn_id}/notes", raise_on_error=False)
        if not resp.ok or not resp.json_body:
            return []
        return resp.json_body.get("list", {}).get("items", [])
 
    def query_txns(self, filters: dict | None = None, limit: int = 100, order: str = "-createdAt") -> SumsubResponse:
        """Best-effort call to Sumsub's flexible txn search endpoint. Publicly
        documented only for Travel Rule (crypto) transactions -- unconfirmed
        whether it generalizes. Treated as diagnostic, never trusted blindly."""
        path_filters = ""
        if filters:
            path_filters = ";" + ";".join(f"{k}={v}" for k, v in filters.items())
        return self.get(f"/resources/kyt/txns/query/-{path_filters}?limit={limit}&order={order}", raise_on_error=False)
 
    def list_applicants(self, offset: int = 0, limit: int = 50) -> SumsubResponse:
        return self.get(f"/resources/applicants/-;offset={offset};limit={limit}", raise_on_error=False)
 
 
# ============================================================================
# SQLite persistence -- the durable, ever-growing store the bot learns from.
# ============================================================================
 
SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    applicant_id TEXT,
    external_user_id TEXT,
    direction TEXT,
    amount REAL,
    currency TEXT,
    txn_type TEXT,
    counterparty_country TEXT,
    counterparty_id TEXT,
    counterparty_bank_name TEXT,
    payment_method TEXT,
    review_status TEXT,
    txn_created_at TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    tags TEXT,
    notes TEXT,
    notes_text TEXT,
    is_bank_rfi INTEGER DEFAULT 0,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_txn_applicant ON transactions(applicant_id);
CREATE INDEX IF NOT EXISTS idx_txn_is_bank_rfi ON transactions(is_bank_rfi);
 
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT, finished_at TEXT, strategy_used TEXT,
    scanned_count INTEGER, new_bank_rfi_count INTEGER, total_bank_rfi_count INTEGER, notes TEXT
);
 
CREATE TABLE IF NOT EXISTS model_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT, n_bank_rfi INTEGER, n_other INTEGER, mode TEXT,
    metrics_json TEXT, feature_importances_json TEXT
);
 
CREATE TABLE IF NOT EXISTS webhook_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT DEFAULT CURRENT_TIMESTAMP, event_type TEXT, kyt_txn_id TEXT,
    raw_json TEXT, processed INTEGER DEFAULT 0
);
"""
 
OPEN_STATUSES = ("init", "onHold", "awaitingUser", "pending", "queued", None, "")
 
 
def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        for new_col in ("notes", "notes_text", "counterparty_bank_name"):
            if new_col not in cols:
                conn.execute(f"ALTER TABLE transactions ADD COLUMN {new_col} TEXT")
 
 
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
 
 
def upsert_transaction(row: dict):
    row = dict(row)
    row["tags"] = json.dumps(row.get("tags") or [])
    row["notes"] = json.dumps(row.get("notes") or [])
    row["raw_json"] = json.dumps(row.get("raw_json") or {})
    cols = [
        "txn_id", "applicant_id", "external_user_id", "direction", "amount", "currency",
        "txn_type", "counterparty_country", "counterparty_id", "counterparty_bank_name", "payment_method",
        "review_status", "txn_created_at", "tags", "notes", "notes_text", "is_bank_rfi", "raw_json",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "txn_id")
    sql = f"""INSERT INTO transactions ({", ".join(cols)}) VALUES ({placeholders})
              ON CONFLICT(txn_id) DO UPDATE SET {updates}, ingested_at=CURRENT_TIMESTAMP"""
    with get_conn() as conn:
        conn.execute(sql, [row.get(c) for c in cols])
 
 
def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    d["notes"] = json.loads(d.get("notes") or "[]")
    d["raw_json"] = json.loads(d.get("raw_json") or "{}")
    d["is_bank_rfi"] = bool(d.get("is_bank_rfi"))
    return d
 
 
def all_transactions() -> list[dict]:
    with get_conn() as conn:
        return [_row_to_dict(r) for r in conn.execute("SELECT * FROM transactions").fetchall()]
 
 
def open_transactions() -> list[dict]:
    placeholders = ",".join("?" for _ in OPEN_STATUSES)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM transactions WHERE is_bank_rfi=0 AND review_status IN ({placeholders})", OPEN_STATUSES
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
 
 
def get_transaction(txn_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM transactions WHERE txn_id=?", (txn_id,)).fetchone()
        return _row_to_dict(row) if row else None
 
 
def record_ingestion_run(started_at, finished_at, strategy_used, scanned_count, new_bank_rfi_count, notes=""):
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM transactions WHERE is_bank_rfi=1").fetchone()["c"]
        conn.execute(
            """INSERT INTO ingestion_runs
               (started_at, finished_at, strategy_used, scanned_count, new_bank_rfi_count, total_bank_rfi_count, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (started_at, finished_at, strategy_used, scanned_count, new_bank_rfi_count, total, notes),
        )
 
 
def recent_ingestion_runs(limit=20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM ingestion_runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
 
 
def record_model_version(mode, n_bank_rfi, n_other, metrics: dict, feature_importances: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO model_versions
               (trained_at, n_bank_rfi, n_other, mode, metrics_json, feature_importances_json)
               VALUES (datetime('now'), ?, ?, ?, ?, ?)""",
            (n_bank_rfi, n_other, mode, json.dumps(metrics), json.dumps(feature_importances)),
        )
 
 
def latest_model_version() -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM model_versions ORDER BY version_id DESC LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics_json"] = json.loads(d["metrics_json"] or "{}")
        d["feature_importances_json"] = json.loads(d["feature_importances_json"] or "{}")
        return d
 
 
def model_version_history(limit=50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT version_id, trained_at, n_bank_rfi, n_other, mode, metrics_json FROM model_versions "
            "ORDER BY version_id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics_json"] = json.loads(d["metrics_json"] or "{}")
            out.append(d)
        return out
 
 
def record_webhook_event(event_type: str, kyt_txn_id: str, raw_json: dict):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO webhook_events (event_type, kyt_txn_id, raw_json) VALUES (?,?,?)",
            (event_type, kyt_txn_id, json.dumps(raw_json)),
        )
 
 
def webhook_event_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM webhook_events").fetchone()["c"]
 
 
# ============================================================================
# Feature engineering -- every feature here should be explainable in one
# sentence to a compliance reviewer.
# ============================================================================
 
HIGH_RISK_COUNTRIES = {"IR", "KP", "SY", "AF", "MM", "VE", "RU", "BY"}
STRUCTURING_THRESHOLDS = [3000, 10000]
CATEGORICAL_FEATURES = ["currency", "txn_type", "counterparty_country"]
NUMERIC_FEATURES = [
    "amount", "log_amount", "direction_out", "counterparty_high_risk", "is_round_amount",
    "near_reporting_threshold", "applicant_prior_txn_count", "applicant_prior_bank_rfi_count",
    "applicant_prior_bank_rfi_rate", "applicant_txn_count_24h", "applicant_txn_count_7d",
    "applicant_avg_prior_amount", "amount_vs_applicant_avg", "hour_of_day", "is_weekend",
    "is_new_applicant",
]
 
 
def _parse_amount(row: dict) -> float:
    try:
        return float(row.get("amount") or 0.0)
    except (TypeError, ValueError):
        return 0.0
 
 
def _parse_dt(row: dict):
    ts = row.get("txn_created_at")
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts[: len(fmt) + 5], fmt)
        except ValueError:
            continue
    return None
 
 
def build_applicant_history_index(transactions: list[dict]) -> dict:
    by_applicant = defaultdict(list)
    for t in transactions:
        by_applicant[t.get("applicant_id")].append(t)
    for txns in by_applicant.values():
        txns.sort(key=lambda t: _parse_dt(t) or datetime.min)
    return by_applicant
 
 
def extract_features(row: dict, history_index: dict) -> dict:
    amount = _parse_amount(row)
    dt = _parse_dt(row)
    applicant_id = row.get("applicant_id")
    history = history_index.get(applicant_id, [])
 
    prior = [t for t in history if t.get("txn_id") != row.get("txn_id") and (_parse_dt(t) or datetime.min) <= (dt or datetime.max)]
    prior_amounts = [_parse_amount(t) for t in prior]
    prior_bank_rfi = sum(1 for t in prior if t.get("is_bank_rfi"))
 
    window_24h = [t for t in prior if dt and _parse_dt(t) and (dt - _parse_dt(t)).total_seconds() <= 86400]
    window_7d = [t for t in prior if dt and _parse_dt(t) and (dt - _parse_dt(t)).total_seconds() <= 7 * 86400]
 
    nearest_threshold_gap = min((abs(amount - th) for th in STRUCTURING_THRESHOLDS), default=1e9)
    just_under_threshold = any(0 < th - amount <= th * 0.1 for th in STRUCTURING_THRESHOLDS)
    counterparty = (row.get("counterparty_country") or "").upper()
 
    return {
        "txn_id": row.get("txn_id"),
        "amount": amount,
        "log_amount": math.log1p(amount),
        "direction_out": 1 if (row.get("direction") or "").lower() == "out" else 0,
        "currency": row.get("currency") or "unknown",
        "txn_type": row.get("txn_type") or "unknown",
        "counterparty_country": counterparty or "unknown",
        "counterparty_high_risk": 1 if counterparty in HIGH_RISK_COUNTRIES else 0,
        "is_round_amount": 1 if amount > 0 and amount % 100 == 0 else 0,
        "near_reporting_threshold": 1 if just_under_threshold else 0,
        "nearest_threshold_gap": nearest_threshold_gap,
        "applicant_prior_txn_count": len(prior),
        "applicant_prior_bank_rfi_count": prior_bank_rfi,
        "applicant_prior_bank_rfi_rate": (prior_bank_rfi / len(prior)) if prior else 0.0,
        "applicant_txn_count_24h": len(window_24h),
        "applicant_txn_count_7d": len(window_7d),
        "applicant_avg_prior_amount": statistics.fmean(prior_amounts) if prior_amounts else 0.0,
        "amount_vs_applicant_avg": (amount / statistics.fmean(prior_amounts)) if prior_amounts and statistics.fmean(prior_amounts) > 0 else 1.0,
        "hour_of_day": dt.hour if dt else -1,
        "is_weekend": 1 if dt and dt.weekday() >= 5 else 0,
        "is_new_applicant": 1 if len(prior) == 0 else 0,
    }
 
 
# ============================================================================
# Pattern analysis -- plain, auditable stats (no black box).
# ============================================================================
 
def analyze_patterns(bank_rfi_features: list[dict], other_features: list[dict]) -> dict:
    result = {
        "n_bank_rfi": len(bank_rfi_features), "n_other": len(other_features),
        "numeric_findings": [], "categorical_findings": [], "narrative": [],
    }
    if len(bank_rfi_features) < 3:
        result["narrative"].append(
            f'Only {len(bank_rfi_features)} transaction(s) tagged "bank rfi" so far -- too few to '
            "find statistically reliable patterns yet. The bot will keep re-analyzing as more come in."
        )
        return result
 
    for feat in NUMERIC_FEATURES:
        rfi_vals = [f[feat] for f in bank_rfi_features if feat in f]
        other_vals = [f[feat] for f in other_features if feat in f]
        if len(rfi_vals) < 3 or len(other_vals) < 3:
            continue
        try:
            _, p_val = scipy_stats.mannwhitneyu(rfi_vals, other_vals, alternative="two-sided")
        except ValueError:
            continue
        p_val = float(p_val)
        rfi_mean = float(sum(rfi_vals) / len(rfi_vals))
        other_mean = float(sum(other_vals) / len(other_vals)) if other_vals else 0.0
        finding = {
            "feature": feat, "bank_rfi_mean": round(rfi_mean, 3), "other_mean": round(other_mean, 3),
            "p_value": round(p_val, 5), "significant": bool(p_val < 0.05),
        }
        result["numeric_findings"].append(finding)
        if finding["significant"] and other_mean != 0:
            direction = "higher" if rfi_mean > other_mean else "lower"
            result["narrative"].append(
                f'"{feat}" is significantly {direction} in bank-rfi transactions '
                f"(avg {rfi_mean:.2f} vs {other_mean:.2f} overall, p={p_val:.4f})."
            )
 
    for feat in CATEGORICAL_FEATURES:
        rfi_counts = Counter(f.get(feat, "unknown") for f in bank_rfi_features)
        other_counts = Counter(f.get(feat, "unknown") for f in other_features)
        n_rfi, n_other = len(bank_rfi_features), len(other_features)
        for value, rfi_n in rfi_counts.most_common(5):
            other_n = other_counts.get(value, 0)
            rfi_rate = rfi_n / n_rfi if n_rfi else 0
            other_rate = other_n / n_other if n_other else 0
            lift = (rfi_rate / other_rate) if other_rate > 0 else (float("inf") if rfi_rate > 0 else 0)
            finding = {
                "feature": feat, "value": value, "bank_rfi_rate": round(rfi_rate, 3),
                "other_rate": round(other_rate, 3), "lift": round(lift, 2) if lift != float("inf") else None,
            }
            result["categorical_findings"].append(finding)
            if rfi_n >= 3 and (finding["lift"] is None or finding["lift"] >= 2.0):
                lift_text = "exclusively" if other_n == 0 else f"{finding['lift']:.1f}x more often"
                result["narrative"].append(
                    f'{feat}="{value}" appears {lift_text} in bank-rfi transactions '
                    f"({rfi_rate*100:.0f}% of them vs {other_rate*100:.0f}% overall)."
                )
 
    if not result["narrative"]:
        result["narrative"].append(
            "No single feature stands out strongly yet -- the model below combines several weaker "
            "signals. This narrative will sharpen as more bank-rfi examples accumulate."
        )
    return result
 
 
# ============================================================================
# Predictive model -- transparent heuristic until enough labeled data exists,
# then a calibrated logistic regression. Never a black box: every score comes
# with matched reasons and the nearest past bank-rfi transactions.
# ============================================================================
 
def _vocab_for(features_list: list[dict], key: str) -> list[str]:
    return sorted({str(f.get(key, "unknown")) for f in features_list})
 
 
def _vectorize(features_list: list[dict], vocab: dict, scaler: StandardScaler | None = None, fit_scaler: bool = False):
    numeric_matrix = np.array([[f.get(n, 0.0) for n in NUMERIC_FEATURES] for f in features_list], dtype=float)
    if scaler is not None:
        numeric_matrix = scaler.fit_transform(numeric_matrix) if fit_scaler else scaler.transform(numeric_matrix)
    cat_blocks = []
    for cat in CATEGORICAL_FEATURES:
        values = vocab[cat]
        block = np.zeros((len(features_list), len(values)))
        for i, f in enumerate(features_list):
            v = str(f.get(cat, "unknown"))
            if v in values:
                block[i, values.index(v)] = 1.0
        cat_blocks.append(block)
    return np.hstack([numeric_matrix] + cat_blocks) if cat_blocks else numeric_matrix
 
 
class ModelBundle:
    """Lives in memory only (in `_current_bundle`) -- retrained fresh from the
    database every ingestion cycle, so there's no model file to manage, no
    models/ folder, and nothing to go stale on disk."""
 
    def __init__(self, mode, classifier, scaler, vocab, feature_names, train_features, train_matrix,
                 pattern_findings, explain_classifier=None):
        self.mode = mode
        self.classifier = classifier
        self.explain_classifier = explain_classifier
        self.scaler = scaler
        self.vocab = vocab
        self.feature_names = feature_names
        self.train_features = train_features
        self.train_matrix = train_matrix
        self.pattern_findings = pattern_findings
 
 
def train_model(transactions: list[dict]) -> ModelBundle:
    history_index = build_applicant_history_index(transactions)
    feats = [extract_features(t, history_index) for t in transactions]
    for f, t in zip(feats, transactions):
        f["_is_bank_rfi"] = bool(t.get("is_bank_rfi"))
 
    bank_rfi_feats = [f for f in feats if f["_is_bank_rfi"]]
    other_feats = [f for f in feats if not f["_is_bank_rfi"]]
    pattern_findings = analyze_patterns(bank_rfi_feats, other_feats)
 
    vocab = {cat: _vocab_for(feats, cat) for cat in CATEGORICAL_FEATURES}
    feature_names = list(NUMERIC_FEATURES) + [f"{cat}={v}" for cat in CATEGORICAL_FEATURES for v in vocab[cat]]
 
    if len(bank_rfi_feats) < MIN_ML_SAMPLES:
        scaler = StandardScaler()
        if feats:
            _vectorize(feats, vocab, scaler, fit_scaler=True)
        train_matrix = _vectorize(bank_rfi_feats, vocab, scaler) if bank_rfi_feats else np.zeros((0, len(feature_names)))
        bundle = ModelBundle(
            mode="heuristic", classifier=None, scaler=scaler, vocab=vocab, feature_names=feature_names,
            train_features=bank_rfi_feats, train_matrix=train_matrix, pattern_findings=pattern_findings,
        )
    else:
        scaler = StandardScaler()
        X = _vectorize(feats, vocab, scaler, fit_scaler=True)
        y = np.array([1 if f["_is_bank_rfi"] else 0 for f in feats])
 
        explain_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
        explain_clf.fit(X, y)
 
        n_pos = int(y.sum())
        cv_folds = max(2, min(5, n_pos // 5))
        base_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
        clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv_folds)
        clf.fit(X, y)
 
        rfi_matrix = _vectorize(bank_rfi_feats, vocab, scaler)
        bundle = ModelBundle(
            mode="ml", classifier=clf, scaler=scaler, vocab=vocab, feature_names=feature_names,
            train_features=bank_rfi_feats, train_matrix=rfi_matrix, pattern_findings=pattern_findings,
            explain_classifier=explain_clf,
        )
 
    return bundle
 
 
def _heuristic_score(feat: dict, pattern_findings: dict) -> float:
    score = weight_sum = 0.0
    for finding in pattern_findings.get("categorical_findings", []):
        if finding.get("lift") and finding["lift"] >= 1.5 and feat.get(finding["feature"]) == finding["value"]:
            w = min(finding["lift"], 5.0)
            score += w
            weight_sum += w
    for finding in pattern_findings.get("numeric_findings", []):
        if finding.get("significant"):
            val = feat.get(finding["feature"], 0)
            rfi_mean, other_mean = finding["bank_rfi_mean"], finding["other_mean"]
            if (rfi_mean > other_mean and val >= rfi_mean) or (rfi_mean < other_mean and val <= rfi_mean):
                score += 1.0
                weight_sum += 1.0
    if feat.get("counterparty_high_risk"):
        score += 1.0
        weight_sum += 1.0
    if feat.get("near_reporting_threshold"):
        score += 1.0
        weight_sum += 1.0
    return (score / weight_sum) if weight_sum else 0.0
 
 
def _matched_reasons(feat: dict, pattern_findings: dict) -> list[str]:
    reasons = []
    for finding in pattern_findings.get("categorical_findings", []):
        if finding.get("lift") and finding["lift"] >= 1.5 and feat.get(finding["feature"]) == finding["value"]:
            reasons.append(
                f"{finding['feature']} = \"{finding['value']}\" appears in bank-rfi transactions "
                f"{finding['lift']:.1f}x more often than in the general population."
            )
    for finding in pattern_findings.get("numeric_findings", []):
        if finding.get("significant"):
            val = feat.get(finding["feature"], 0)
            rfi_mean, other_mean = finding["bank_rfi_mean"], finding["other_mean"]
            if (rfi_mean > other_mean and val >= rfi_mean) or (rfi_mean < other_mean and val <= rfi_mean):
                reasons.append(
                    f"{finding['feature']} ({val:.2f}) is on the bank-rfi side of the historical split "
                    f"(bank-rfi avg {rfi_mean:.2f} vs {other_mean:.2f} overall, p={finding['p_value']:.4f})."
                )
    if feat.get("counterparty_high_risk"):
        reasons.append("Counterparty country is on the elevated-risk list.")
    if feat.get("near_reporting_threshold"):
        reasons.append("Amount sits just under a common reporting/monitoring threshold.")
    if feat.get("applicant_prior_bank_rfi_rate", 0) > 0:
        reasons.append(
            f"This applicant already has a {feat['applicant_prior_bank_rfi_rate']*100:.0f}% bank-rfi rate on prior transactions."
        )
    return reasons
 
 
def _nearest_bank_rfi_examples(vec: np.ndarray, bundle: ModelBundle, top_k: int = 3) -> list[dict]:
    if bundle.train_matrix.shape[0] == 0:
        return []
    norms_a = np.linalg.norm(bundle.train_matrix, axis=1)
    norm_b = np.linalg.norm(vec)
    denom = norms_a * norm_b
    denom[denom == 0] = 1e-9
    sims = (bundle.train_matrix @ vec) / denom
    order = np.argsort(-sims)[:top_k]
    return [{"txn_id": bundle.train_features[i]["txn_id"], "similarity": round(float(sims[i]), 3)} for i in order if sims[i] > 0]
 
 
def score_transaction(feat: dict, bundle: ModelBundle) -> dict:
    vec = _vectorize([feat], bundle.vocab, bundle.scaler)[0]
    if bundle.mode == "ml":
        proba = float(bundle.classifier.predict_proba(vec.reshape(1, -1))[0][1])
    else:
        proba = _heuristic_score(feat, bundle.pattern_findings)
    return {
        "txn_id": feat["txn_id"], "risk_score": round(proba, 4), "mode": bundle.mode,
        "reasons": _matched_reasons(feat, bundle.pattern_findings),
        "similar_bank_rfi_transactions": _nearest_bank_rfi_examples(vec, bundle, top_k=3),
    }
 
 
def score_all_open(open_txns: list[dict], transactions: list[dict], bundle: ModelBundle) -> list[dict]:
    history_index = build_applicant_history_index(transactions)
    results = [score_transaction(extract_features(t, history_index), bundle) for t in open_txns]
    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return results
 
 
# ============================================================================
# Ingestion -- gets Sumsub data in, checks tags, keeps the dataset growing.
# See the module docstring at the top of this file for the strategy order.
# ============================================================================
 
def _dig(d: dict, *paths: str):
    for path in paths:
        cur = d
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None
 
 
def normalize_txn(raw: dict, tags: list[str], notes: list[dict] | None = None) -> dict:
    txn_id = _dig(raw, "id", "txnId", "kytTxnId") or raw.get("id")
    is_rfi = any((t or "").strip().lower() == BANK_RFI_TAG.strip().lower() for t in tags)
    note_texts = [n.get("note") for n in (notes or []) if isinstance(n, dict) and n.get("note")]
    return {
        "txn_id": txn_id,
        "applicant_id": _dig(raw, "applicantId", "applicant.id"),
        "external_user_id": _dig(raw, "externalUserId", "applicant.externalUserId"),
        "direction": _dig(raw, "data.info.direction", "data.direction", "direction"),
        "amount": _dig(raw, "data.info.amount", "data.amount", "amount"),
        "currency": _dig(raw, "data.info.currencyCode", "data.currencyCode", "currency"),
        "txn_type": _dig(raw, "data.type", "type"),
        "counterparty_country": _dig(raw, "data.counterparty.paymentMethod.country", "data.counterparty.country", "counterparty.country"),
        "counterparty_id": _dig(raw, "data.counterparty.paymentMethod.accountId", "data.counterparty.id"),
        "counterparty_bank_name": _dig(
            raw, "data.counterparty.paymentMethod.bankName", "data.counterparty.paymentMethod.bank.name",
            "data.counterparty.bankName", "data.counterparty.institution", "data.counterparty.paymentMethod.institutionName",
        ),
        "payment_method": _dig(raw, "data.applicant.paymentMethod.type", "data.paymentMethod.type"),
        "review_status": _dig(raw, "reviewStatus", "review.reviewStatus", "status"),
        "txn_created_at": _dig(raw, "data.txnDate", "createdAt", "createdAtMs"),
        "tags": tags, "notes": notes or [], "notes_text": " | ".join(note_texts),
        "is_bank_rfi": 1 if is_rfi else 0, "raw_json": raw,
    }
 
 
def ingest_single_txn(client: SumsubClient, txn_id: str) -> dict | None:
    try:
        raw = client.get_transaction(txn_id)
        tags = client.get_txn_tags(txn_id)
    except SumsubAPIError as e:
        logger.warning("Failed to fetch txn %s: %s", txn_id, e)
        return None
    notes = client.get_txn_notes(txn_id)
    row = normalize_txn(raw, tags, notes)
    if not row.get("txn_id"):
        row["txn_id"] = txn_id
    upsert_transaction(row)
    return row
 
 
def backfill_from_travelrule_query(client: SumsubClient, limit: int = 100, max_pages: int = 20) -> dict:
    """Historical-backfill attempt via Sumsub's documented transaction-query
    endpoint (docs.sumsub.com/reference/find-specific-tr-transactions). This
    endpoint is confirmed to work ONLY when filtered to data.type=travelRule
    (Travel Rule / crypto transactions) -- it has no tag filter and no
    offset/cursor pagination, just `limit` (max 100) + `order`. To still walk
    past the first 100 results, this uses a date-cursor trick: after each
    page, re-query with data.txnDate__lte set just before the oldest item
    seen so far. If your account's transactions aren't Travel Rule type,
    Sumsub will legitimately return zero items every time -- that's expected
    given what's actually documented, not a bug in this code."""
    scanned = new_rfi = pages = 0
    cursor = None
    seen_ids = set()
    while pages < max_pages:
        filters = {"data.type": "travelRule"}
        if cursor:
            filters["data.txnDate__lte"] = cursor
        resp = client.query_txns(filters=filters, limit=limit, order="-data.txnDate")
        pages += 1
        if not resp.ok:
            return {"strategy": "travelrule_query", "ran": pages > 1, "status": resp.status,
                    "body": resp.raw_body[:500], "scanned": scanned, "new_bank_rfi": new_rfi, "pages": pages}
        items = (resp.json_body or {}).get("items") or (resp.json_body or {}).get("list", {}).get("items") or []
        if not items:
            break
        oldest_date = None
        new_this_page = 0
        for item in items:
            txn_id = _dig(item, "id", "txnId", "kytTxnId")
            item_date = _dig(item, "data.txnDate", "createdAt")
            if item_date and (oldest_date is None or item_date < oldest_date):
                oldest_date = item_date
            if not txn_id or txn_id in seen_ids:
                continue
            seen_ids.add(txn_id)
            new_this_page += 1
            row = ingest_single_txn(client, txn_id)
            if row:
                scanned += 1
                new_rfi += 1 if row["is_bank_rfi"] else 0
        if len(items) < limit or new_this_page == 0 or not oldest_date:
            break
        cursor = oldest_date
    return {"strategy": "travelrule_query", "ran": pages > 0 and scanned > 0, "scanned": scanned,
            "new_bank_rfi": new_rfi, "pages": pages}
 
 
def backfill_from_applicant_walk(client: SumsubClient, max_applicants: int = 200) -> dict:
    """EXPERIMENTAL: Sumsub's public docs don't document a "list this
    applicant's KYT transactions" endpoint, so this tries a couple of
    plausible path shapes per applicant and silently moves on if neither
    exists on your account (404s are expected and harmless here)."""
    scanned = new_rfi = applicants_seen = 0
    offset = 0
    while applicants_seen < max_applicants:
        resp = client.list_applicants(offset=offset, limit=50)
        if not resp.ok:
            return {"strategy": "applicant_walk", "ran": applicants_seen > 0, "status": resp.status,
                    "body": resp.raw_body[:500], "scanned": scanned, "new_bank_rfi": new_rfi}
        items = (resp.json_body or {}).get("list", {}).get("items") or (resp.json_body or {}).get("items") or []
        if not items:
            break
        for applicant in items:
            applicant_id = applicant.get("id")
            applicants_seen += 1
            for path_tpl in ("/resources/applicants/{}/kyt/txns", "/resources/applicants/{}/kyt/txns/-/data"):
                txn_resp = client.get(path_tpl.format(applicant_id), raise_on_error=False)
                if txn_resp.ok and txn_resp.json_body:
                    txn_items = txn_resp.json_body.get("items") or txn_resp.json_body.get("list", {}).get("items") or []
                    for t in txn_items:
                        txn_id = _dig(t, "id", "txnId", "kytTxnId")
                        if not txn_id:
                            continue
                        row = ingest_single_txn(client, txn_id)
                        if row:
                            scanned += 1
                            new_rfi += 1 if row["is_bank_rfi"] else 0
                    break
        offset += 50
    return {"strategy": "applicant_walk", "ran": applicants_seen > 0, "scanned": scanned, "new_bank_rfi": new_rfi, "applicants_seen": applicants_seen}
 
 
def run_full_backfill(client: SumsubClient) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    results = []
    for fn in (backfill_from_travelrule_query, backfill_from_applicant_walk):
        try:
            results.append(fn(client))
        except Exception as e:  # noqa: BLE001 - one bad strategy must not kill the run
            logger.exception("Backfill strategy %s crashed", fn.__name__)
            results.append({"strategy": fn.__name__, "ran": False, "error": str(e)})
 
    scanned = sum(r.get("scanned", 0) for r in results)
    new_rfi = sum(r.get("new_bank_rfi", 0) for r in results)
    finished = datetime.now(timezone.utc).isoformat()
    strategies_that_ran = [r["strategy"] for r in results if r.get("ran")]
    record_ingestion_run(started, finished, ",".join(strategies_that_ran) or "none_worked", scanned, new_rfi, notes=str(results))
    return {"started": started, "finished": finished, "results": results, "scanned": scanned, "new_bank_rfi": new_rfi}
 
 
def process_webhook_event(client: SumsubClient, event: dict) -> dict | None:
    """Note: the webhook route already records receipt of every event (see
    api_sumsub_webhook) before calling this -- this function is purely about
    actually fetching and storing the transaction once credentials exist."""
    kyt_txn_id = event.get("kytTxnId") or event.get("txnId")
    if not kyt_txn_id:
        return None
    return ingest_single_txn(client, kyt_txn_id)
 
 
# ============================================================================
# FastAPI app -- routes, background scheduler, live credential check.
# ============================================================================
 
app = FastAPI(title="Bank RFI Prediction Bot")
init_db()
 
_current_bundle: ModelBundle | None = None
_client: SumsubClient | None = None
_connection_cache = {"checked_at": None, "status": "unknown", "detail": ""}
 
 
def get_client() -> SumsubClient | None:
    global _client
    if _client is None:
        try:
            _client = SumsubClient()
        except SumsubAuthError:
            return None
    return _client
 
 
def check_sumsub_connection(force: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    cached_at = _connection_cache["checked_at"]
    if not force and cached_at and (now - cached_at).total_seconds() < CONNECTION_CHECK_CACHE_SECONDS:
        return _connection_cache
    client = get_client()
    if client is None:
        _connection_cache.update(checked_at=now, status="not_configured",
                                  detail="SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY are not set yet.")
        return _connection_cache
    try:
        resp = client.list_applicants(offset=0, limit=1)
        if resp.status == 401:
            status, detail = "auth_failed", "Sumsub returned 401 Unauthorized -- token/secret pair is wrong or was rotated."
        elif resp.ok:
            status, detail = "ok", "Credentials work -- Sumsub accepted a signed request."
        else:
            status, detail = "ok", f"Credentials appear valid (HTTP {resp.status} on a basic request)."
    except Exception as e:  # noqa: BLE001
        status, detail = "unreachable", f"Could not reach api.sumsub.com: {e}"
    _connection_cache.update(checked_at=now, status=status, detail=detail)
    return _connection_cache
 
 
def _feature_importances(bundle: ModelBundle) -> dict:
    if bundle.mode != "ml" or bundle.explain_classifier is None:
        return {}
    try:
        coefs = bundle.explain_classifier.coef_[0]
        return dict(sorted(zip(bundle.feature_names, [round(float(c), 4) for c in coefs]), key=lambda kv: -abs(kv[1]))[:20])
    except Exception:
        return {}
 
 
def run_ingest_and_retrain():
    """The recurring job: pull/refresh data, then retrain on everything seen
    so far. This is what makes the bot keep learning as each new bank-rfi
    tag shows up."""
    global _current_bundle
    logger.info("Starting ingestion + retrain cycle")
    client = get_client()
    if client is None:
        logger.warning("Credentials not set yet -- skipping ingestion, will retrain on whatever's already stored.")
    else:
        try:
            result = run_full_backfill(client)
            logger.info("Ingestion result: %s", {k: v for k, v in result.items() if k != "results"})
        except Exception:
            logger.exception("Ingestion failed; will still try to retrain on existing data")
    try:
        txns = all_transactions()
        if txns:
            _current_bundle = train_model(txns)
            record_model_version(
                mode=_current_bundle.mode, n_bank_rfi=len(_current_bundle.train_features),
                n_other=len(txns) - len(_current_bundle.train_features),
                metrics={"n_training_rows": len(txns)}, feature_importances=_feature_importances(_current_bundle),
            )
            logger.info("Retrained model in %s mode on %d transactions", _current_bundle.mode, len(txns))
        else:
            logger.info("No transactions in the database yet -- nothing to train on.")
    except Exception:
        logger.exception("Retrain failed")
 
 
def _ensure_bundle_loaded():
    global _current_bundle
    if _current_bundle is None:
        txns = all_transactions()
        if txns:
            _current_bundle = train_model(txns)
 
 
@app.on_event("startup")
def on_startup():
    _ensure_bundle_loaded()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_ingest_and_retrain, "interval", minutes=INGEST_INTERVAL_MINUTES,
                       next_run_time=datetime.now(timezone.utc))
    scheduler.start()
    app.state.scheduler = scheduler
    logger.info("Scheduler started: ingest+retrain every %d minute(s)", INGEST_INTERVAL_MINUTES)
 
 
@app.get("/api/setup")
def api_setup(request: Request):
    conn = check_sumsub_connection()
    txns = all_transactions()
    base = str(request.base_url).rstrip("/")
    return {
        "credentials_configured": conn["status"] != "not_configured",
        "connection_status": conn["status"],
        "connection_detail": conn["detail"],
        "webhook_url": f"{base}/api/webhooks/sumsub",
        "webhook_events_received": webhook_event_count(),
        "total_transactions": len(txns),
        "total_bank_rfi": len([t for t in txns if t["is_bank_rfi"]]),
        "bank_rfi_tag": BANK_RFI_TAG,
        "ingest_interval_minutes": INGEST_INTERVAL_MINUTES,
    }
 
 
@app.get("/api/health")
def api_health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "bank_rfi_tag": BANK_RFI_TAG}
 
 
@app.get("/api/summary")
def api_summary():
    txns = all_transactions()
    rfi = [t for t in txns if t["is_bank_rfi"]]
    latest_version = latest_model_version()
    return {
        "total_transactions": len(txns), "total_bank_rfi": len(rfi),
        "open_pending_transactions": len(open_transactions()),
        "model_mode": latest_version["mode"] if latest_version else "untrained",
        "model_trained_at": latest_version["trained_at"] if latest_version else None,
        "recent_ingestion_runs": recent_ingestion_runs(5),
    }
 
 
@app.get("/api/patterns")
def api_patterns():
    _ensure_bundle_loaded()
    if _current_bundle is None:
        return {"narrative": ["No data ingested yet -- check the Setup panel above, or POST /api/ingest/run."]}
    return _current_bundle.pattern_findings
 
 
@app.get("/api/predictions")
def api_predictions():
    _ensure_bundle_loaded()
    if _current_bundle is None:
        return {"predictions": [], "note": "No data ingested yet."}
    txns = all_transactions()
    open_txns = open_transactions()
    scored = score_all_open(open_txns, txns, _current_bundle)
    by_id = {t["txn_id"]: t for t in txns}
    for s in scored:
        t = by_id.get(s["txn_id"], {})
        s["amount"] = t.get("amount")
        s["currency"] = t.get("currency")
        s["counterparty_country"] = t.get("counterparty_country")
        s["applicant_id"] = t.get("applicant_id")
        for sim in s.get("similar_bank_rfi_transactions", []):
            past = by_id.get(sim["txn_id"], {})
            sim["tags"] = past.get("tags", [])
            sim["notes_text"] = past.get("notes_text") or ""
            sim["counterparty_bank_name"] = past.get("counterparty_bank_name")
            sim["counterparty_country"] = past.get("counterparty_country")
            sim["amount"] = past.get("amount")
            sim["currency"] = past.get("currency")
    return {"predictions": scored, "model_mode": _current_bundle.mode}
 
 
@app.get("/api/transactions/{txn_id}")
def api_transaction_detail(txn_id: str):
    t = get_transaction(txn_id)
    if not t:
        return JSONResponse({"error": "not found"}, status_code=404)
    return t
 
 
@app.get("/api/model/history")
def api_model_history():
    return {"versions": model_version_history()}
 
 
@app.post("/api/ingest/run")
def api_trigger_ingest():
    run_ingest_and_retrain()
    return {"status": "completed", "summary": api_summary()}
 
 
@app.post("/api/webhooks/sumsub")
async def api_sumsub_webhook(request: Request):
    """Register this URL in Sumsub Dashboard -> Settings -> Webhooks for
    Transaction Monitoring events."""
    event = await request.json()
    # Record receipt unconditionally (even without credentials or a
    # recognizable txn id) so the Setup panel can confirm "yes, Sumsub is
    # reaching this URL" independently of whether ingestion succeeds.
    record_webhook_event(event.get("type", "unknown"), event.get("kytTxnId") or event.get("txnId") or "unknown", event)
    client = get_client()
    if client is None:
        logger.warning("Webhook received but credentials aren't set yet: %s", event)
        return {"received": True, "processed": False, "reason": "credentials not configured"}
    try:
        row = process_webhook_event(client, event)
        if row and row.get("is_bank_rfi"):
            logger.info("New bank-rfi transaction via webhook: %s -- retraining", row["txn_id"])
            run_ingest_and_retrain()
    except Exception:
        logger.exception("Failed to process webhook event: %s", event)
    return {"received": True}
 
 
@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND_HTML
 
 
# ============================================================================
# Frontend -- a single embedded HTML page, no separate static files.
# ============================================================================
 
FRONTEND_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light" />
<title>Bank RFI Prediction Bot</title>
<style>
  /* Always light -- intentionally does not follow the OS/browser dark-mode
     preference, so this looks the same for everyone viewing it. */
  .viz-root {
    color-scheme: light only;
    --surface-1:      #ffffff;
    --page:           #f6f6f4;
    --text-primary:   #14140f;
    --text-secondary: #52514e;
    --text-muted:     #83817a;
    --gridline:       #e6e5de;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;
    --series-2:       #eb6834;
    --series-3:       #1baf7a;
    --status-good:    #0ca30c;
    --status-warning: #fab219;
    --status-serious: #ec835a;
    --status-critical:#d03b3b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--page); color: var(--text-primary); font-size: 15px; }
  header { padding: 28px 32px 10px; }
  header h1 { font-size: 24px; margin: 0 0 6px; }
  header p { margin: 0; color: var(--text-secondary); font-size: 14px; }
  .wrap { padding: 8px 32px 56px; max-width: 1100px; margin: 0 auto; }
  .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 4px 0 16px; }
  .toolbar .hint { margin: 0; font-size: 13.5px; color: var(--text-secondary); }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 0 0 24px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }
  .tile .label { font-size: 12.5px; color: var(--text-muted); margin-bottom: 6px; }
  .tile .value { font-size: 28px; font-weight: 700; }
  .tile .sub { font-size: 12.5px; color: var(--text-secondary); margin-top: 4px; }
  section, details.section-collapse { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 20px; }
  section { padding: 20px 22px; }
  section h2 { font-size: 17px; margin: 0 0 6px; }
  section .desc, .details-body .desc { font-size: 13.5px; color: var(--text-secondary); margin: 0 0 16px; }
  details.section-collapse { padding: 0; overflow: hidden; }
  details.section-collapse summary { padding: 16px 22px; font-size: 15px; font-weight: 600; cursor: pointer; list-style: none; display: flex; align-items: center; }
  details.section-collapse summary::-webkit-details-marker { display: none; }
  details.section-collapse summary::before { content: "\25B8"; margin-right: 10px; color: var(--text-muted); display: inline-block; transition: transform .15s ease; font-weight: 400; }
  details.section-collapse[open] summary { border-bottom: 1px solid var(--border); }
  details.section-collapse[open] summary::before { transform: rotate(90deg); }
  details.section-collapse .details-body { padding: 4px 22px 20px; }
  ul.narrative { margin: 0; padding-left: 20px; font-size: 14px; line-height: 1.75; }
  .barrow { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
  .barrow .barlabel { width: 260px; font-size: 13px; color: var(--text-secondary); flex-shrink: 0; text-align: right; }
  .barrow .bartrack { flex: 1; height: 16px; background: var(--gridline); border-radius: 8px; position: relative; overflow: hidden; }
  .barrow .barfill { height: 100%; border-radius: 8px 4px 4px 8px; background: var(--series-1); }
  .barrow .barval { width: 60px; font-size: 12.5px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--gridline); }
  th { color: var(--text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; cursor: pointer; }
  th.sorted { color: var(--series-1); }
  .risk-badge { display: inline-block; min-width: 50px; text-align: center; padding: 3px 9px; border-radius: 6px; font-variant-numeric: tabular-nums; font-size: 13px; font-weight: 700; }
  .risk-high { background: color-mix(in srgb, var(--status-critical) 18%, transparent); color: var(--status-critical); }
  .risk-med  { background: color-mix(in srgb, var(--status-warning) 22%, transparent); color: #8a5a00; }
  .risk-low  { background: color-mix(in srgb, var(--status-good) 16%, transparent); color: var(--status-good); }
  .reasons { font-size: 13px; color: var(--text-secondary); margin: 6px 0 0; padding-left: 18px; line-height: 1.6; }
  .pill { display: inline-block; font-size: 11.5px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); color: var(--text-secondary); margin-right: 4px; }
  .legend { font-size: 12.5px; color: var(--text-muted); margin-bottom: 12px; }
  .legend .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
  .empty { color: var(--text-muted); font-size: 14px; padding: 8px 0; }
  .expand-row td { background: var(--page); font-size: 13px; }
  tr.clickable { cursor: pointer; }
  tr.clickable td:first-child::after { content: ""; }
  tr.clickable:hover { background: var(--page); }
  button.refresh { font-size: 13.5px; padding: 9px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary); cursor: pointer; font-weight: 600; white-space: nowrap; }
  button.refresh:hover { background: var(--gridline); }
  #setupBanner { margin-bottom: 20px; }
  .setup-row { display:flex; align-items:flex-start; gap:10px; margin: 10px 0; font-size: 14px; }
  .setup-icon { flex-shrink:0; width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; color:#fff; font-weight:700; margin-top:1px; }
  .setup-icon.good { background: var(--status-good); }
  .setup-icon.warn { background: var(--status-warning); color:#4a3400; }
  .setup-icon.crit { background: var(--status-critical); }
  code.copyline { display:inline-flex; align-items:center; gap:8px; background: var(--page); border:1px solid var(--border); border-radius:6px; padding:5px 9px; font-size:13px; margin-top:6px; }
  code.copyline button { font-size: 12px; padding: 3px 9px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary); cursor: pointer; }
</style>
</head>
<body>
<div class="viz-root">
  <header>
    <h1>Bank RFI Prediction Bot</h1>
    <p>Learns from every Sumsub transaction tagged &ldquo;bank rfi&rdquo; and scores open transactions by how likely they are to get the same tag.</p>
  </header>
  <div class="wrap">
    <section id="setupBanner" style="display:none;"></section>
 
    <div class="toolbar">
      <p class="hint">Click any transaction row below to see why it's flagged and which past &ldquo;bank rfi&rdquo; cases support it.</p>
      <button class="refresh" id="refreshBtn">Run ingestion + retrain now</button>
    </div>
 
    <div class="tiles" id="tiles"></div>
 
    <section>
      <h2>Predicted next bank-rfi candidates</h2>
      <p class="desc">Open, not-yet-tagged transactions ranked by predicted likelihood of receiving a &ldquo;bank rfi&rdquo; tag.</p>
      <div class="legend"><span class="dot" style="background:var(--status-critical)"></span>Highest risk &nbsp; <span class="dot" style="background:var(--status-warning)"></span>Medium &nbsp; <span class="dot" style="background:var(--status-good)"></span>Lower &nbsp; <span style="margin-left:8px;">(relative to this batch; exact score always shown)</span></div>
      <table id="predTable">
        <thead>
          <tr>
            <th data-key="risk_score">Risk score</th>
            <th data-key="txn_id">Transaction</th>
            <th data-key="amount">Amount</th>
            <th data-key="counterparty_country">Counterparty</th>
            <th>Top reason</th>
          </tr>
        </thead>
        <tbody id="predBody"></tbody>
      </table>
      <div class="empty" id="predEmpty" style="display:none;">No open transactions to score yet.</div>
    </section>
 
    <details class="section-collapse">
      <summary>What distinguishes bank-rfi transactions</summary>
      <div class="details-body">
        <p class="desc">Statistically significant differences between transactions that were tagged &ldquo;bank rfi&rdquo; and everything else the bot has seen so far.</p>
        <ul class="narrative" id="narrative"></ul>
        <div id="categoricalBars" style="margin-top:16px;"></div>
      </div>
    </details>
 
    <details class="section-collapse">
      <summary>Model &amp; ingestion history</summary>
      <div class="details-body">
        <p class="desc">Every retrain is logged for audit purposes &mdash; this is a compliance tool, so nothing here is a black box.</p>
        <table id="historyTable">
          <thead><tr><th>Trained at</th><th>Mode</th><th>Bank-rfi examples</th><th>Other transactions</th></tr></thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>
    </details>
  </div>
</div>
<script>
const fmt = (n) => (n === null || n === undefined) ? "–" : (typeof n === "number" ? n.toLocaleString(undefined, {maximumFractionDigits: 2}) : n);
 
function riskClass(score, rank, total) {
  if (total <= 1) return "risk-med";
  const pct = rank / (total - 1);
  if (pct <= 0.2) return "risk-high";
  if (pct <= 0.5) return "risk-med";
  return "risk-low";
}
 
async function loadAll() {
  const [setup, summary, patterns, predictions, history] = await Promise.all([
    fetch("/api/setup").then(r => r.json()),
    fetch("/api/summary").then(r => r.json()),
    fetch("/api/patterns").then(r => r.json()),
    fetch("/api/predictions").then(r => r.json()),
    fetch("/api/model/history").then(r => r.json()),
  ]);
  renderSetup(setup);
  renderTiles(summary);
  renderPatterns(patterns);
  renderPredictions(predictions.predictions || []);
  renderHistory(history.versions || []);
}
 
function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = original; }, 1500);
  });
}
window.copyToClipboard = copyToClipboard;
 
function renderSetup(s) {
  const banner = document.getElementById("setupBanner");
  const rows = [];
  if (!s.credentials_configured) {
    rows.push({icon: "crit", text: `<strong>No Sumsub credentials set.</strong> Add SUMSUB_APP_TOKEN and SUMSUB_SECRET_KEY as environment variables and redeploy.`});
  } else if (s.connection_status === "ok") {
    rows.push({icon: "good", text: `<strong>Connected to Sumsub.</strong> ${s.connection_detail}`});
  } else if (s.connection_status === "auth_failed") {
    rows.push({icon: "crit", text: `<strong>Sumsub rejected the credentials.</strong> ${s.connection_detail}`});
  } else {
    rows.push({icon: "warn", text: `<strong>Can't reach Sumsub right now.</strong> ${s.connection_detail}`});
  }
  const webhookRegistered = s.webhook_events_received > 0;
  rows.push({
    icon: webhookRegistered ? "good" : "warn",
    text: (webhookRegistered
      ? `<strong>Webhook is live</strong> -- ${s.webhook_events_received.toLocaleString()} event(s) received so far. This is the mechanism that keeps ingesting new data straight from Sumsub with no separate data files.`
      : `<strong>Register this webhook URL</strong> in Sumsub Dashboard &rarr; Settings &rarr; Webhooks (Transaction Monitoring events) so every new "${s.bank_rfi_tag}" tag is picked up automatically -- no manual data entry needed:`)
      + `<br/><code class="copyline"><span>${s.webhook_url}</span><button onclick="copyToClipboard('${s.webhook_url}', this)">Copy</button></code>`
  });
  if (s.total_transactions === 0) {
    rows.push({icon: "warn", text: `<strong>No data yet.</strong> All data comes straight from the Sumsub API -- either wait for "${s.bank_rfi_tag}"-tagged transactions to arrive via the webhook above, or trigger an ingestion run now (button below) to try the automatic historical-backfill strategies.`});
  } else {
    rows.push({icon: "good", text: `<strong>${s.total_transactions.toLocaleString()} transaction(s) ingested</strong>, ${s.total_bank_rfi.toLocaleString()} tagged "${s.bank_rfi_tag}" so far -- retraining every ${s.ingest_interval_minutes} minute(s), all pulled from the Sumsub API.`});
  }
  const allGood = s.connection_status === "ok" && s.total_transactions > 0;
  banner.style.display = "block";
  banner.innerHTML = `<h2>${allGood ? "Status" : "Setup"}</h2>${rows.map(r => `<div class="setup-row"><span class="setup-icon ${r.icon}">${r.icon === "good" ? "&#10003;" : "!"}</span><span>${r.text}</span></div>`).join("")}`;
}
 
function renderTiles(s) {
  const tiles = [
    {label: "Transactions scanned", value: fmt(s.total_transactions)},
    {label: "Tagged “bank rfi”", value: fmt(s.total_bank_rfi)},
    {label: "Open pending transactions", value: fmt(s.open_pending_transactions)},
    {label: "Model mode", value: s.model_mode || "untrained", sub: s.model_trained_at ? ("trained " + s.model_trained_at) : "not trained yet"},
  ];
  document.getElementById("tiles").innerHTML = tiles.map(t => `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}</div>${t.sub ? `<div class="sub">${t.sub}</div>` : ""}</div>`).join("");
}
 
function renderPatterns(p) {
  const narrative = p.narrative || [];
  document.getElementById("narrative").innerHTML = narrative.length ? narrative.map(n => `<li>${n}</li>`).join("") : '<li class="empty">No narrative yet.</li>';
  const cats = (p.categorical_findings || []).filter(f => f.lift === null || f.lift >= 1.2).slice(0, 8);
  const maxLift = Math.max(1, ...cats.map(c => c.lift || 5));
  document.getElementById("categoricalBars").innerHTML = cats.length ? cats.map(c => `
    <div class="barrow">
      <div class="barlabel">${c.feature} = "${c.value}"</div>
      <div class="bartrack"><div class="barfill" style="width:${Math.min(100, ((c.lift||5)/maxLift)*100)}%"></div></div>
      <div class="barval">${c.lift === null ? "only in RFI" : c.lift + "x"}</div>
    </div>`).join("") : '<p class="empty">Not enough data yet for a lift breakdown.</p>';
}
 
let currentPredictions = [];
let sortKey = "risk_score", sortDir = -1;
 
function renderPredictions(preds) {
  const byRisk = [...preds].sort((a, b) => b.risk_score - a.risk_score);
  byRisk.forEach((r, i) => { r._riskRank = i; });
  currentPredictions = preds;
  drawPredTable();
}
 
function drawPredTable() {
  const body = document.getElementById("predBody");
  const empty = document.getElementById("predEmpty");
  const rows = [...currentPredictions].sort((a,b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av === bv) return 0;
    return (av > bv ? 1 : -1) * sortDir;
  });
  if (!rows.length) { body.innerHTML = ""; empty.style.display = "block"; return; }
  empty.style.display = "none";
  body.innerHTML = rows.map((r, i) => `
    <tr class="clickable" data-idx="${i}">
      <td><span class="risk-badge ${riskClass(r.risk_score, r._riskRank, rows.length)}">${(r.risk_score*100).toFixed(0)}%</span></td>
      <td>${r.txn_id}</td>
      <td>${fmt(r.amount)} ${r.currency||""}</td>
      <td>${r.counterparty_country || "–"}</td>
      <td>${(r.reasons && r.reasons[0]) || "–"}</td>
    </tr>
    <tr class="expand-row" id="expand-${i}" style="display:none;"><td colspan="5">
      <strong>Why:</strong>
      <ul class="reasons">${(r.reasons||[]).map(x => `<li>${x}</li>`).join("") || "<li>No specific reasons matched.</li>"}</ul>
      <strong>Similar past bank-rfi transactions:</strong>
      <div>${(r.similar_bank_rfi_transactions||[]).map(s => `
        <div style="margin:6px 0; padding:6px 8px; border:1px solid var(--border); border-radius:6px;">
          <span class="pill">${s.txn_id}</span> similarity ${s.similarity} &middot;
          ${fmt(s.amount)} ${s.currency||""} &middot; ${s.counterparty_country||"–"}
          ${s.counterparty_bank_name ? ` &middot; bank: <strong>${s.counterparty_bank_name}</strong>` : ""}
          ${(s.tags && s.tags.length) ? `<div style="margin-top:3px;">tags: ${s.tags.map(t => `<span class="pill">${t}</span>`).join(" ")}</div>` : ""}
          ${s.notes_text ? `<div style="margin-top:3px; color:var(--text-secondary);">note: "${s.notes_text}"</div>` : ""}
        </div>`).join("") || '<span class="empty">none yet</span>'}</div>
    </td></tr>
  `).join("");
  body.querySelectorAll("tr.clickable").forEach(tr => {
    tr.addEventListener("click", () => {
      const el = document.getElementById("expand-" + tr.dataset.idx);
      el.style.display = el.style.display === "none" ? "table-row" : "none";
    });
  });
}
 
document.querySelectorAll("#predTable th[data-key]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    sortDir = (sortKey === key) ? -sortDir : -1;
    sortKey = key;
    document.querySelectorAll("#predTable th").forEach(x => x.classList.remove("sorted"));
    th.classList.add("sorted");
    drawPredTable();
  });
});
 
function renderHistory(versions) {
  const body = document.getElementById("historyBody");
  body.innerHTML = versions.length ? versions.map(v => `<tr><td>${v.trained_at}</td><td>${v.mode}</td><td>${v.n_bank_rfi}</td><td>${v.n_other}</td></tr>`).join("") : '<tr><td colspan="4" class="empty">No trained model versions yet.</td></tr>';
}
 
document.getElementById("refreshBtn").addEventListener("click", async (e) => {
  e.target.disabled = true;
  e.target.textContent = "Running…";
  try {
    await fetch("/api/ingest/run", {method: "POST"});
    await loadAll();
  } finally {
    e.target.disabled = false;
    e.target.textContent = "Run ingestion + retrain now";
  }
});
 
loadAll();
setInterval(loadAll, 60000);
</script>
</body>
</html>
"""
