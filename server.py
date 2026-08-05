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
  1. Webhooks (the real, event-driven mechanism -- this is what keeps the
     bot "constantly running" rather than waiting on a timer): register
     this app's /api/webhooks/sumsub URL in Sumsub Dashboard -> Settings ->
     Webhooks for Transaction Monitoring events. Every lifecycle event
     Sumsub fires includes the transaction's Sumsub ID (kytTxnId, confirmed
     against Sumsub's real webhook payload docs) -- the bot fetches that
     transaction's tags immediately, stores it, and retrains on the spot
     (see retrain_only()) so the model and predictions are current within
     seconds of the event, not the next scheduled cycle.
  2. A background safety-net job (co-located in this same process, default
     hourly, see INGEST_INTERVAL_MINUTES) re-runs the best-effort historical
     backfill and retrains -- this exists only to catch anything the webhook
     missed (not registered yet, a dropped delivery, etc.), it is not the
     primary way the model stays current -- see WHAT'S VERIFIED VS. ASSUMED
     below for exactly which backfill strategies are confirmed-working vs.
     experimental.

WHAT'S VERIFIED VS. ASSUMED (read this before trusting a prediction) -- this
section reflects live research against Sumsub's published API docs, not
guesswork:
  - CONFIRMED: fetching one transaction + its tags/notes by ID
    (/resources/kyt/txns/{id}/one, .../tags, .../notes), the HMAC request
    signing scheme, and the webhook payload field names (kytTxnId is the
    correct ID to re-fetch; there is no "tags" field on the webhook payload
    itself, so a follow-up GET is required and is what this bot does).
  - CORRECTED (was wrong earlier): /resources/kyt/txns/query/- -- the public
    docs only show examples filtered to data.type=travelRule and describe
    that parameter as "Must be travelRule", which reads like the endpoint is
    Travel-Rule/crypto-only. It isn't. A separate, already-working internal
    Kira Financial tool queries this exact endpoint filtered ONLY by date
    range (data.txnDate__gte/__lte, no data.type at all) with plain
    ?offset=N pagination, against this same Sumsub account, and gets
    transactions of every type back. That's confirmed-in-production
    behavior, so this bot now queries the same way: date range + offset,
    no type filter. See backfill_from_txn_query (hourly, last couple of
    days -- the safety net) and run_deep_historical_backfill (triggered
    on demand from the dashboard, walks the full account history month by
    month). The old data.type=travelRule-only assumption baked into an
    earlier version of this file was real docs-based research, but it was
    still wrong -- treat this file's own past comments with the same
    skepticism as any other doc.
  - UNCONFIRMED / EXPERIMENTAL: enumerating applicants and walking each
    one's transactions -- Sumsub's public docs do not document a "list this
    applicant's KYT transactions" endpoint, so this strategy tries a couple
    of plausible paths and silently no-ops if none exist on your account.
    Now a secondary/supplementary strategy since the date-range query above
    covers the general case.
  - UNCONFIRMED: which field (if any) names the bank that sent an RFI. The
    bot surfaces every plausible place this could live (a second tag, a
    compliance note, a best-effort counterparty-bank field) instead of
    picking one.
  - The webhook remains the real-time mechanism for "grows as new bank-rfi
    tags are added on Sumsub" -- it fires on every transaction lifecycle
    event regardless of type. Open the running app's dashboard -- it
    live-checks your credentials and tells you plainly what's actually
    happening, rather than requiring you to trust this comment.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import sqlite3
import statistics
import time
import urllib.parse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
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
INGEST_INTERVAL_MINUTES = int(os.environ.get("INGEST_INTERVAL_MINUTES", "60"))  # safety-net backfill sweep only -- real retraining is event-driven via the webhook, see retrain_only()
CONNECTION_CHECK_CACHE_SECONDS = 60


def _normalize_tag(text: str) -> str:
    """Case-insensitive, whitespace-collapsed tag comparison key -- so a tag
    stored on Sumsub as "Bank RFI", " bank  rfi ", or "BANK RFI" all still
    match BANK_RFI_TAG. This is the ONLY thing that decides whether a fetched
    transaction counts as "bank rfi" -- every transaction the bot pulls in
    (via webhook or the manual ID import) is checked against this, and it's
    an exact match on the tag's label text, not a fuzzy/partial one."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


BANK_RFI_TAG_NORMALIZED = _normalize_tag(BANK_RFI_TAG)


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
        """Look up by SUMSUB'S OWN internal transaction ID (the one it
        generates and returns in its API responses / webhook payloads --
        confirmed field name `kytTxnId`)."""
        return self.get(f"/resources/kyt/txns/{txn_id}/one").json_body

    def get_transaction_by_external_id(self, external_txn_id: str) -> dict:
        """Look up by the CLIENT-supplied external transaction ID instead --
        i.e. whatever ID your own system assigned when the transaction was
        originally submitted to Sumsub (docs.sumsub.com/reference/get-
        transaction-information-by-txnid: "unique transaction identifier in
        YOUR system"). This is a different endpoint shape and a different ID
        space from get_transaction() above -- e.g. a transfer_uuid pulled
        from your own records is this kind of ID, not Sumsub's internal one."""
        return self.get(f"/resources/kyt/txns/-;data.txnId={external_txn_id}/one").json_body

    def get_txn_tags(self, txn_id: str) -> list[str]:
        resp = self.get(f"/resources/kyt/txns/{txn_id}/tags")
        items = resp.json_body or []
        return [item.get("label") for item in items if isinstance(item, dict) and item.get("label")]

    def get_txn_notes(self, txn_id: str) -> list[dict]:
        resp = self.get(f"/resources/kyt/txns/{txn_id}/notes", raise_on_error=False)
        if not resp.ok or not resp.json_body:
            return []
        return resp.json_body.get("list", {}).get("items", [])

    def query_txns(self, filters: dict | None = None, limit: int = 100, order: str = "-createdAt",
                    offset: int = 0) -> SumsubResponse:
        """Sumsub's transaction-query endpoint. Its public docs only show
        examples filtered to data.type=travelRule and don't document offset
        pagination -- but a separate, already-working internal tool queries
        this exact endpoint filtered ONLY by date range (no data.type at
        all) with plain ?offset=N paging and gets transactions of every
        type back. That's confirmed-in-production behavior, not a guess, so
        offset support is treated as real even though the public docs are
        silent on it.

        Filter VALUES are percent-encoded (Python's urllib.quote, safe="")
        to match that proven request byte-for-byte -- the working reference
        implementation runs each value through JS's encodeURIComponent
        before inserting it into the path, which escapes ':', '+', and
        spaces (e.g. a date filter becomes '2026-08-01%2000%3A00%3A00%2B0000').
        An earlier version of this method inserted filter values into the
        path RAW/unencoded, which is a different request on the wire -- and
        was confirmed (by a real account returning far fewer transactions
        than expected) to make Sumsub parse the date range differently, not
        just to be a cosmetic difference."""
        path_filters = ""
        if filters:
            path_filters = ";" + ";".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in filters.items())
        return self.get(f"/resources/kyt/txns/query/-{path_filters}?limit={limit}&order={order}&offset={offset}",
                         raise_on_error=False)

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


def _txn_timestamp(row: dict):
    """Best-effort sortable timestamp for a transaction: prefer the actual
    Sumsub transaction date, fall back to when this bot ingested it (always
    set) so rows with a missing/unparseable txn_created_at still sort
    sensibly instead of being silently dropped."""
    dt = _parse_dt(row)
    if dt:
        return dt
    ingested = row.get("ingested_at")
    if ingested:
        try:
            return datetime.strptime(ingested[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def most_recent_bank_rfi_timestamp(transactions: list[dict] | None = None):
    txns = transactions if transactions is not None else all_transactions()
    stamps = [s for s in (_txn_timestamp(t) for t in txns if t.get("is_bank_rfi")) if s is not None]
    return max(stamps) if stamps else None


def transactions_since_last_bank_rfi(transactions: list[dict] | None = None) -> list[dict]:
    """The candidate pool for 'what's likely to get tagged next': every
    non-bank-rfi transaction that happened after the most recent bank-rfi-
    tagged transaction. Scoped by TIME, not review status -- a "bank rfi" is
    a bank's after-the-fact request for more information about a
    transaction, so it can land on one that's already finished processing,
    not just one still sitting in a pending/open state. Falls back to every
    non-bank-rfi transaction if no bank-rfi example exists yet to anchor to."""
    txns = transactions if transactions is not None else all_transactions()
    cutoff = most_recent_bank_rfi_timestamp(txns)
    others = [t for t in txns if not t.get("is_bank_rfi")]
    if cutoff is None:
        return others
    return [t for t in others if (_txn_timestamp(t) or datetime.min) > cutoff]


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
    is_rfi = any(_normalize_tag(t) == BANK_RFI_TAG_NORMALIZED for t in tags)
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
    """Accepts EITHER kind of transaction ID and figures out which one it got:
    Sumsub's own internal ID (what webhooks hand you) or the external ID your
    own system assigned when the transaction was submitted (e.g. a
    transfer_uuid pulled from your own records/spreadsheets) -- these are two
    different ID spaces on Sumsub's side, looked up via two different
    endpoints. Tries the internal-ID lookup first (cheaper, and what webhook-
    driven ingestion always has), falls back to the external-ID lookup if
    that 404s, so callers never need to know in advance which kind they have."""
    raw = None
    try:
        raw = client.get_transaction(txn_id)
    except SumsubAPIError:
        raw = None  # not found by Sumsub's internal ID -- try the external ID next
    except Exception as e:  # noqa: BLE001 -- network hiccup, not a "not found"
        logger.warning("Network error fetching txn %s: %s", txn_id, e)
        return None

    if raw is None:
        try:
            raw = client.get_transaction_by_external_id(txn_id)
        except SumsubAPIError as e:
            logger.warning("Sumsub has no transaction matching %s (checked both its own "
                            "internal ID and your system's external ID): %s", txn_id, e)
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("Network error fetching txn %s: %s", txn_id, e)
            return None

    # Whichever lookup worked, get_txn_tags/get_txn_notes still need SUMSUB'S
    # OWN internal ID (the "id" field in the response), not necessarily the
    # ID we were originally given.
    internal_id = _dig(raw, "id", "txnId", "kytTxnId") or txn_id
    try:
        tags = client.get_txn_tags(internal_id)
    except SumsubAPIError as e:
        logger.warning("Failed to fetch tags for txn %s: %s", internal_id, e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Network error fetching tags for txn %s: %s", internal_id, e)
        return None
    try:
        notes = client.get_txn_notes(internal_id)
    except Exception as e:  # noqa: BLE001 -- notes are supplementary evidence only,
        # don't throw away an otherwise-successful fetch over a notes hiccup.
        logger.warning("Failed to fetch notes for txn %s (continuing without them): %s", internal_id, e)
        notes = []

    row = normalize_txn(raw, tags, notes)
    if not row.get("txn_id"):
        row["txn_id"] = internal_id
    upsert_transaction(row)
    return row


def import_txn_ids(client: SumsubClient, txn_ids: list[str], on_progress=None) -> dict:
    """Bulk-import by transaction ID -- a manual bridge for specific
    transactions, kept alongside run_deep_historical_backfill (which now
    covers "give me every transaction" automatically via a date-range
    query, no data.type filter -- see that function's docstring for why
    the earlier belief that Sumsub's query endpoint was Travel-Rule-only
    turned out to be wrong). This box is for cases the automatic sweep
    doesn't fit: a specific list filtered by tag in the Sumsub Dashboard,
    or IDs pulled from your own records (a spreadsheet, a Slack thread).

    ingest_single_txn() accepts EITHER Sumsub's internal ID or your own
    system's external ID for each one and figures out which it got. Every
    ID is then resolved through a real Sumsub API call
    (get_transaction/get_transaction_by_external_id + get_txn_tags +
    get_txn_notes) -- nothing here is synthetic or guessed.

    `on_progress(done, total)` is called after every ID so a caller (the
    background job below) can report live status for large lists."""
    scanned = new_rfi = failed = 0
    details = []  # per-ID diagnostics -- exactly which tags Sumsub returned for
    # each one, so a mismatch (wrong tag spelling, wrong ID, transaction not
    # found) is visible immediately instead of a silent "0 imported".
    total = len(txn_ids)
    for i, txn_id in enumerate(txn_ids):
        txn_id = (txn_id or "").strip()
        if not txn_id:
            if on_progress:
                on_progress(i + 1, total)
            continue
        row = ingest_single_txn(client, txn_id)
        if row:
            scanned += 1
            new_rfi += 1 if row["is_bank_rfi"] else 0
            details.append({"txn_id": txn_id, "found": True, "tags": row.get("tags", []),
                             "is_bank_rfi": bool(row["is_bank_rfi"])})
        else:
            failed += 1
            details.append({"txn_id": txn_id, "found": False, "tags": [], "is_bank_rfi": False})
        if on_progress:
            on_progress(i + 1, total)
    return {"strategy": "manual_id_import", "ran": scanned > 0, "scanned": scanned,
            "new_bank_rfi": new_rfi, "failed": failed, "total_ids": len(txn_ids), "details": details}


def _fmt_txn_date(dt: datetime) -> str:
    """The exact date-filter format Sumsub's transaction-query endpoint
    expects, e.g. '2026-01-31 23:59:59+0000'. Confirmed working in
    production (not just from docs) via a separate, already-functioning
    internal tool that queries this same endpoint against this same Sumsub
    account."""
    return dt.strftime("%Y-%m-%d %H:%M:%S+0000")


def _query_txn_window(client: SumsubClient, window_start: datetime, window_end: datetime, limit: int,
                       seen_ids: set) -> tuple[int, int, bool]:
    """Pages through ONE date window via offset (not the type filter --
    see backfill_from_txn_query's docstring for why). Returns
    (scanned, new_bank_rfi, had_any_items)."""
    scanned = new_rfi = offset = 0
    had_items = False
    while True:
        filters = {
            "data.txnDate__gte": _fmt_txn_date(window_start),
            "data.txnDate__lte": _fmt_txn_date(window_end),
        }
        resp = client.query_txns(filters=filters, limit=limit, order="-data.txnDate", offset=offset)
        if not resp.ok:
            break
        items = (resp.json_body or {}).get("items") or (resp.json_body or {}).get("list", {}).get("items") or []
        if not items:
            break
        had_items = True
        for item in items:
            txn_id = _dig(item, "id", "txnId", "kytTxnId")
            if not txn_id or txn_id in seen_ids:
                continue
            seen_ids.add(txn_id)
            row = ingest_single_txn(client, txn_id)
            if row:
                scanned += 1
                new_rfi += 1 if row["is_bank_rfi"] else 0
        if len(items) < limit:
            break
        offset += limit
    return scanned, new_rfi, had_items


def backfill_from_txn_query(client: SumsubClient, days_back: int = 2, limit: int = 100) -> dict:
    """The lightweight, EVERY-HOUR safety-net strategy: queries Sumsub's
    transaction-query endpoint filtered ONLY by a recent date range
    (data.txnDate__gte/__lte) -- deliberately NOT by data.type. Sumsub's
    public docs for this endpoint only show examples with
    data.type=travelRule and imply that's required, and don't mention
    offset pagination at all -- but a separate, already-working internal
    Kira Financial tool queries this exact endpoint the same way (date
    range only, plain ?offset=N paging) against this same account and gets
    transactions of every type back, not just Travel Rule. That's
    confirmed-in-production behavior, so it's trusted here over the
    ambiguous docs.

    Scoped to the last `days_back` days (default 2, comfortably overlapping
    the hourly cadence) rather than all-time, so the automatic safety net
    stays fast and cheap -- see run_deep_historical_backfill for the
    one-time/on-demand full-history version."""
    now = datetime.now(timezone.utc)
    seen_ids = set()
    scanned, new_rfi, had_items = _query_txn_window(client, now - timedelta(days=days_back), now, limit, seen_ids)
    return {"strategy": "txn_query_recent", "ran": had_items, "scanned": scanned,
            "new_bank_rfi": new_rfi, "days_back": days_back}


_backfill_history_job = {"running": False, "months_done": 0, "months_total": 0, "result": None,
                          "started_at": None, "finished_at": None}


def run_deep_historical_backfill(client: SumsubClient, months_back: int = 60, max_empty_months: int = 6,
                                  on_progress=None) -> dict:
    """One-time (or occasional, on-demand) DEEP crawl for full history --
    not part of the hourly cycle, since walking years of data every hour
    would be slow and would hammer Sumsub's API for no reason. Same proven
    date-range-only + offset-pagination approach as backfill_from_txn_query,
    just walked backward one calendar month at a time so each individual
    query window stays modest, for up to `months_back` months (default 5
    years), stopping early after `max_empty_months` consecutive empty
    months (a proxy for "before this account had any transactions").
    Trigger via POST /api/ingest/backfill-history; poll
    GET /api/ingest/backfill-history-status for progress since this can
    take a while for a large account."""
    scanned = new_rfi = empty_streak = months_walked = errored_months = 0
    seen_ids = set()
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    for i in range(months_back):
        window_start = datetime(year, month, 1, tzinfo=timezone.utc)
        window_end = (datetime(year + (1 if month == 12 else 0), (1 if month == 12 else month + 1), 1,
                                tzinfo=timezone.utc) - timedelta(seconds=1))
        try:
            m_scanned, m_new_rfi, had_items = _query_txn_window(client, window_start, window_end, 100, seen_ids)
        except Exception:
            # A transient network blip on ONE month must not abort the whole
            # multi-year crawl -- log it, count the month as "not empty" (so
            # it doesn't falsely count toward the empty-streak stop
            # condition), and keep walking backward through the rest.
            logger.exception("Deep backfill: month %04d-%02d failed, skipping", year, month)
            errored_months += 1
            months_walked += 1
            if on_progress:
                on_progress(i + 1, months_back)
            month -= 1
            if month == 0:
                month, year = 12, year - 1
            continue
        scanned += m_scanned
        new_rfi += m_new_rfi
        months_walked += 1
        empty_streak = 0 if had_items else empty_streak + 1
        if on_progress:
            on_progress(i + 1, months_back)
        if empty_streak >= max_empty_months:
            break
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return {"strategy": "deep_historical_backfill", "ran": scanned > 0, "scanned": scanned,
            "new_bank_rfi": new_rfi, "months_walked": months_walked, "errored_months": errored_months}


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
    for fn in (backfill_from_txn_query, backfill_from_applicant_walk):
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
# Tracks the most recent bulk ID import so large pastes (e.g. "every
# transaction", not just bank-rfi ones) don't have to finish inside a
# single HTTP request -- see api_import_ids / api_import_status.
_import_job = {"running": False, "done": 0, "total": 0, "result": None, "started_at": None, "finished_at": None}


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


def retrain_only():
    """Just retrain on whatever is already in the database -- no Sumsub API
    calls. This is the fast path used every time a webhook event comes in,
    so the model is current within seconds of a real transaction/tag change
    rather than waiting for the next scheduled cycle."""
    global _current_bundle
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


def run_ingest_and_retrain():
    """The SAFETY-NET job: attempts the best-effort historical backfill
    strategies (see run_full_backfill), then retrains. This is not what
    keeps the model current day-to-day anymore -- that's the webhook, which
    triggers retrain_only() directly on every event, so the bot reacts the
    moment a real transaction comes in instead of waiting on a timer. This
    job exists only to periodically sweep for anything the webhook missed
    (e.g. it wasn't registered yet, or a delivery was dropped)."""
    logger.info("Starting safety-net ingestion + retrain cycle")
    client = get_client()
    if client is None:
        logger.warning("Credentials not set yet -- skipping ingestion, will retrain on whatever's already stored.")
    else:
        try:
            result = run_full_backfill(client)
            logger.info("Ingestion result: %s", {k: v for k, v in result.items() if k != "results"})
        except Exception:
            logger.exception("Ingestion failed; will still try to retrain on existing data")
    retrain_only()


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
    logger.info(
        "Safety-net backfill+retrain scheduled every %d minute(s) -- but the model actually "
        "retrains immediately on every webhook event, not on this timer.", INGEST_INTERVAL_MINUTES,
    )


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
    cutoff = most_recent_bank_rfi_timestamp(txns)
    return {
        "total_transactions": len(txns), "total_bank_rfi": len(rfi),
        "candidate_transactions": len(transactions_since_last_bank_rfi(txns)),
        "last_bank_rfi_at": cutoff.isoformat() if cutoff else None,
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
    candidate_txns = transactions_since_last_bank_rfi(txns)
    scored = score_all_open(candidate_txns, txns, _current_bundle)
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


def _run_import_job_in_background(client: SumsubClient, raw_ids: list[str]):
    def on_progress(done, total):
        _import_job["done"] = done
        _import_job["total"] = total
    try:
        result = import_txn_ids(client, raw_ids, on_progress=on_progress)
        retrain_only()  # each ID was already fetched for real above; just retrain on the updated data now
        _import_job["result"] = result
    except Exception as e:  # noqa: BLE001 - a bad ID list must not wedge the job state
        logger.exception("Bulk import job failed")
        _import_job["result"] = {"error": str(e)}
    finally:
        _import_job["running"] = False
        _import_job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/ingest/import-ids")
async def api_import_ids(request: Request, background_tasks: BackgroundTasks):
    """Manual bridge for getting data in that Sumsub's API can't enumerate
    on its own -- whether that's specifically "bank rfi"-tagged transactions
    or literally every transaction you want monitored (see import_txn_ids
    docstring for exactly what was checked and why no API shortcut exists).
    Filter/export transaction IDs from the Sumsub Dashboard and paste them
    here (or POST {"txn_ids": [...]}) -- each one is then fetched for real
    via the Sumsub API, nothing is fabricated.

    Runs as a BACKGROUND JOB rather than blocking this request: a list of a
    few hundred or thousand IDs (e.g. "all transactions", not just the
    handful tagged bank-rfi) can take minutes to resolve one API call at a
    time, which would otherwise hit Render's/uvicorn's request timeout and
    fail the whole batch. Poll GET /api/ingest/import-status for progress."""
    body = await request.json()
    raw_ids = body.get("txn_ids")
    if raw_ids is None:
        text = body.get("text", "") or ""
        raw_ids = [x for x in re.split(r"[,\s]+", text) if x]
    raw_ids = [x for x in (raw_ids or []) if (x or "").strip()]
    client = get_client()
    if client is None:
        return JSONResponse(
            {"error": "SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY aren't configured yet -- add them and redeploy first."},
            status_code=400,
        )
    if not raw_ids:
        return JSONResponse({"error": "No transaction IDs provided."}, status_code=400)
    if _import_job["running"]:
        return JSONResponse(
            {"error": "An import is already running -- check /api/ingest/import-status and wait for it to finish."},
            status_code=409,
        )
    _import_job.update(running=True, done=0, total=len(raw_ids), result=None,
                        started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    background_tasks.add_task(_run_import_job_in_background, client, raw_ids)
    return {"status": "started", "total_ids": len(raw_ids)}


@app.get("/api/ingest/import-status")
def api_import_status():
    """Poll this while a bulk import (started via POST .../import-ids) is
    running to show live progress instead of a frozen "loading" spinner."""
    return {**_import_job, "summary": api_summary() if not _import_job["running"] else None}


def _run_backfill_history_job_in_background(client: SumsubClient, months_back: int):
    def on_progress(done, total):
        _backfill_history_job["months_done"] = done
        _backfill_history_job["months_total"] = total
    try:
        result = run_deep_historical_backfill(client, months_back=months_back, on_progress=on_progress)
        retrain_only()
        _backfill_history_job["result"] = result
    except Exception as e:  # noqa: BLE001 - must not wedge the job state
        logger.exception("Deep historical backfill job failed")
        _backfill_history_job["result"] = {"error": str(e)}
    finally:
        _backfill_history_job["running"] = False
        _backfill_history_job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/ingest/backfill-history")
def api_backfill_history(background_tasks: BackgroundTasks, months_back: int = 60):
    """Trigger a ONE-TIME deep historical crawl covering every transaction
    type (see run_deep_historical_backfill) -- not part of the automatic
    hourly cycle, which only checks a short recent window to stay fast.
    Runs in the background; poll GET /api/ingest/backfill-history-status."""
    client = get_client()
    if client is None:
        return JSONResponse(
            {"error": "SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY aren't configured yet -- add them and redeploy first."},
            status_code=400,
        )
    if _backfill_history_job["running"]:
        return JSONResponse(
            {"error": "A historical backfill is already running -- check /api/ingest/backfill-history-status."},
            status_code=409,
        )
    _backfill_history_job.update(running=True, months_done=0, months_total=months_back, result=None,
                                  started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    background_tasks.add_task(_run_backfill_history_job_in_background, client, months_back)
    return {"status": "started", "months_back": months_back}


@app.get("/api/ingest/backfill-history-status")
def api_backfill_history_status():
    """Poll this while a deep historical backfill (started via POST
    .../backfill-history) is running."""
    return {**_backfill_history_job, "summary": api_summary() if not _backfill_history_job["running"] else None}


def _process_webhook_event_in_background(event: dict):
    """Runs AFTER the HTTP response has already been sent (see
    api_sumsub_webhook) -- this is what actually fetches the transaction and
    retrains. Keeping it out of the request/response path matters: Sumsub's
    webhook delivery (and its "Test webhook" button) times out waiting for a
    response after only a few seconds, and fetching a transaction (1-3
    Sumsub API calls) plus a full model retrain can easily take longer than
    that, especially as the dataset grows. Doing this synchronously in the
    handler caused Sumsub to report "Could not get response" even though
    the server was working correctly -- it just answered too late."""
    client = get_client()
    if client is None:
        logger.warning("Webhook received but credentials aren't set yet: %s", event)
        return
    try:
        row = process_webhook_event(client, event)
        if row:
            # Retrain on EVERY real transaction event -- tagged "bank rfi"
            # or not -- so the model, the candidate pool, and the pattern
            # stats are current within seconds of Sumsub sending this event.
            # This is the bot's actual "constantly running" mechanism; the
            # hourly job is only a safety net (see run_ingest_and_retrain's
            # docstring).
            tag_note = "new bank-rfi transaction" if row.get("is_bank_rfi") else "transaction update"
            logger.info("Webhook event processed (%s): %s -- retraining now", tag_note, row["txn_id"])
            retrain_only()
    except Exception:
        logger.exception("Failed to process webhook event: %s", event)


@app.post("/api/webhooks/sumsub")
async def api_sumsub_webhook(request: Request, background_tasks: BackgroundTasks):
    """Register this URL in Sumsub Dashboard -> Settings -> Webhooks for
    Transaction Monitoring events. Responds immediately (see
    _process_webhook_event_in_background for why that matters) -- the
    actual fetch-and-retrain work happens after the response is sent."""
    event = await request.json()
    # Record receipt unconditionally (even without credentials or a
    # recognizable txn id) so the Setup panel can confirm "yes, Sumsub is
    # reaching this URL" independently of whether ingestion succeeds. This
    # is a single fast local DB write, not an API call, so it's safe to do
    # before responding.
    record_webhook_event(event.get("type", "unknown"), event.get("kytTxnId") or event.get("txnId") or "unknown", event)
    background_tasks.add_task(_process_webhook_event_in_background, event)
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
  textarea#importIdsText { width:100%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; padding:10px; border-radius:8px; border:1px solid var(--border); background: var(--page); color: var(--text-primary); resize: vertical; }
  #importStatus { color: var(--text-secondary); }
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

    <details class="section-collapse" id="deepBackfillSection">
      <summary>Run a full historical backfill (every transaction type, not just recent ones)</summary>
      <div class="details-body">
        <p class="desc">
          The automatic hourly job only checks the last couple of days (fast, cheap,
          catches anything the webhook missed recently). This button instead walks
          your account's full history month by month, pulling <strong>every</strong>
          transaction regardless of type &mdash; it queries Sumsub by date range only
          (no type filter), which is what actually returns all transactions rather
          than just Travel-Rule/crypto ones. It stops automatically once it hits
          several consecutive empty months (i.e. before your account had any
          transactions). This can take a while for a large account, so it runs in
          the background with live progress below &mdash; safe to close this panel
          and come back later.
        </p>
        <div style="margin-top:10px; display:flex; align-items:center; gap:10px;">
          <button class="refresh" id="backfillHistoryBtn">Run full historical backfill</button>
          <span id="backfillHistoryStatus"></span>
        </div>
      </div>
    </details>

    <details class="section-collapse" id="importSection">
      <summary>Import transactions by ID (bank-rfi-tagged ones, or any specific list)</summary>
      <div class="details-body">
        <p class="desc">
          For most cases the full historical backfill above or the webhook already
          covers it. This box is for pasting in specific transaction IDs directly
          &mdash; from Sumsub's Dashboard (filter by &ldquo;bank rfi&rdquo; there, copy the
          IDs it shows) or from your own records (a spreadsheet, a Slack thread).
          Either Sumsub's own transaction ID or your system's external/transfer ID
          works, the bot tries both automatically. Each one is then fetched for real
          from the Sumsub API (full details, tags, notes) &mdash; nothing here is
          guessed or made up. Large lists run as a background job so the page won't
          time out; progress updates live below.
        </p>
        <textarea id="importIdsText" rows="4" placeholder="One transaction ID per line, or comma-separated -- Sumsub's own ID or your system's external/transfer ID both work"></textarea>
        <div style="margin-top:10px; display:flex; align-items:center; gap:10px;">
          <button class="refresh" id="importBtn">Import these transactions</button>
          <span id="importStatus"></span>
        </div>
        <div id="importDetails" style="margin-top:12px;"></div>
      </div>
    </details>

    <div class="toolbar">
      <p class="hint">Click any transaction row below to see why it's flagged and which past &ldquo;bank rfi&rdquo; cases support it.</p>
      <button class="refresh" id="refreshBtn">Run ingestion + retrain now</button>
    </div>

    <div class="tiles" id="tiles"></div>

    <section>
      <h2>Predicted next bank-rfi candidates</h2>
      <p class="desc" id="predDesc">Transactions received after the most recent &ldquo;bank rfi&rdquo; tag, ranked by predicted likelihood of receiving the same tag next.</p>
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
      <div class="empty" id="predEmpty" style="display:none;">Nothing to score yet.</div>
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
      : `<strong>No webhook events received yet.</strong> This checks whether Sumsub has actually sent anything here -- it can't check Sumsub's own settings (Sumsub doesn't expose an API for that), so if you've <em>already</em> registered the URL below, this just means no matching event has fired yet, not that registration failed. If you just set it up, try clicking "Test webhook" in Sumsub's Webhook manager, or check Sumsub's own <strong>Webhook logs</strong> page to see delivery attempts and their status directly. If you haven't registered it yet, add this URL in Sumsub Dashboard &rarr; Webhook manager for Transaction Monitoring events:`)
      + `<br/><code class="copyline"><span>${s.webhook_url}</span><button onclick="copyToClipboard('${s.webhook_url}', this)">Copy</button></code>`
  });
  if (s.total_transactions === 0) {
    rows.push({icon: "warn", text: `<strong>No data yet.</strong> All data comes straight from the Sumsub API -- run "Run full historical backfill" below to pull your account's full transaction history, wait for new events to arrive via the webhook above, or use "Import transactions by ID" for a specific list.`});
  } else if (s.total_bank_rfi === 0) {
    rows.push({icon: "warn", text: `<strong>${s.total_transactions.toLocaleString()} transaction(s) ingested, but none tagged "${s.bank_rfi_tag}" yet.</strong> If you know some already carry that tag in Sumsub, use "Import transactions by ID" below -- filter by "${s.bank_rfi_tag}" in the Sumsub Dashboard, copy the IDs it shows, and paste them there. Sumsub's API has no way to list transactions by tag (or at all) on its own, so this is the fastest way in.`});
  } else {
    rows.push({icon: "good", text: `<strong>${s.total_transactions.toLocaleString()} transaction(s) ingested</strong>, ${s.total_bank_rfi.toLocaleString()} tagged "${s.bank_rfi_tag}" so far -- all pulled from the Sumsub API. The model retrains immediately every time a webhook event comes in (not on a timer); a safety-net sweep also runs every ${s.ingest_interval_minutes} minute(s) in case any webhook delivery was missed.`});
  }
  const allGood = s.connection_status === "ok" && s.total_transactions > 0;
  banner.style.display = "block";
  banner.innerHTML = `<h2>${allGood ? "Status" : "Setup"}</h2>${rows.map(r => `<div class="setup-row"><span class="setup-icon ${r.icon}">${r.icon === "good" ? "&#10003;" : "!"}</span><span>${r.text}</span></div>`).join("")}`;
}

function renderTiles(s) {
  const tiles = [
    {label: "Transactions scanned", value: fmt(s.total_transactions)},
    {label: "Tagged “bank rfi”", value: fmt(s.total_bank_rfi)},
    {label: "Candidates since last RFI", value: fmt(s.candidate_transactions),
     sub: s.last_bank_rfi_at ? ("since " + s.last_bank_rfi_at.slice(0, 16).replace("T", " ")) : "no bank-rfi tag seen yet"},
    {label: "Model mode", value: s.model_mode || "untrained", sub: s.model_trained_at ? ("trained " + s.model_trained_at) : "not trained yet"},
  ];
  document.getElementById("tiles").innerHTML = tiles.map(t => `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}</div>${t.sub ? `<div class="sub">${t.sub}</div>` : ""}</div>`).join("");
  if (!window._importAutoOpened) {
    window._importAutoOpened = true;
    document.getElementById("importSection").open = (s.total_bank_rfi === 0);
  }
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

function renderImportDetails(details) {
  const detailsEl = document.getElementById("importDetails");
  const rows = (details || []).map(d => {
    if (!d.found) return `<div style="margin:4px 0;"><span class="pill">${d.txn_id}</span> <span style="color:var(--status-critical);">not found -- checked both Sumsub's internal ID and your system's external ID, verify it's correct and credentials have access to it</span></div>`;
    const tagList = d.tags.length ? d.tags.map(t => `<span class="pill">${t}</span>`).join(" ") : '<em style="color:var(--text-muted);">no tags on this transaction</em>';
    const verdict = d.is_bank_rfi
      ? '<span style="color:var(--status-good); font-weight:600;">matched "bank rfi"</span>'
      : '<span style="color:var(--text-muted);">no "bank rfi" tag found</span>';
    return `<div style="margin:4px 0;"><span class="pill">${d.txn_id}</span> ${verdict} &mdash; tags: ${tagList}</div>`;
  }).join("");
  detailsEl.innerHTML = rows ? `<div class="desc" style="margin-bottom:6px;">What Sumsub actually returned for each ID:</div>${rows}` : "";
}

async function pollImportStatus(btn, statusEl) {
  while (true) {
    await new Promise(r => setTimeout(r, 1500));
    let s;
    try {
      s = await (await fetch("/api/ingest/import-status")).json();
    } catch (err) {
      statusEl.textContent = "Lost track of import progress: " + err;
      break;
    }
    if (s.running) {
      statusEl.textContent = `Importing… ${s.done} of ${s.total} transactions fetched from Sumsub so far. This can take a while for large lists -- one real API call per ID -- feel free to leave this open.`;
      continue;
    }
    // Finished (or a job someone else started already finished) -- show the result.
    if (s.result && s.result.error) {
      statusEl.textContent = "Import failed: " + s.result.error;
    } else if (s.result) {
      const r = s.result;
      statusEl.textContent = `Imported ${r.scanned} of ${r.total_ids} (${r.new_bank_rfi} tagged "bank rfi", ${r.failed} failed).`;
      renderImportDetails(r.details);
    }
    await loadAll();
    break;
  }
  btn.disabled = false;
  btn.textContent = "Import these transactions";
}

document.getElementById("importBtn").addEventListener("click", async (e) => {
  const btn = e.target;
  const statusEl = document.getElementById("importStatus");
  const detailsEl = document.getElementById("importDetails");
  const text = document.getElementById("importIdsText").value;
  btn.disabled = true;
  btn.textContent = "Importing…";
  statusEl.textContent = "";
  detailsEl.innerHTML = "";
  try {
    const resp = await fetch("/api/ingest/import-ids", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = data.error || "Import failed.";
      btn.disabled = false;
      btn.textContent = "Import these transactions";
      return;
    }
    statusEl.textContent = `Starting import of ${data.total_ids} transaction(s)…`;
    document.getElementById("importIdsText").value = "";
    await pollImportStatus(btn, statusEl);
  } catch (err) {
    statusEl.textContent = "Import failed: " + err;
    btn.disabled = false;
    btn.textContent = "Import these transactions";
  }
});

async function pollBackfillHistoryStatus(btn, statusEl) {
  while (true) {
    await new Promise(r => setTimeout(r, 2000));
    let s;
    try {
      s = await (await fetch("/api/ingest/backfill-history-status")).json();
    } catch (err) {
      statusEl.textContent = "Lost track of backfill progress: " + err;
      break;
    }
    if (s.running) {
      statusEl.textContent = `Walking history… month ${s.months_done} of up to ${s.months_total} checked so far. This can take a while -- feel free to leave this open.`;
      continue;
    }
    if (s.result && s.result.error) {
      statusEl.textContent = "Backfill failed: " + s.result.error;
    } else if (s.result) {
      const r = s.result;
      statusEl.textContent = `Done -- walked ${r.months_walked} month(s), found ${r.scanned} transaction(s) (${r.new_bank_rfi} tagged "bank rfi").`;
    }
    await loadAll();
    break;
  }
  btn.disabled = false;
  btn.textContent = "Run full historical backfill";
}

document.getElementById("backfillHistoryBtn").addEventListener("click", async (e) => {
  const btn = e.target;
  const statusEl = document.getElementById("backfillHistoryStatus");
  btn.disabled = true;
  btn.textContent = "Running…";
  statusEl.textContent = "";
  try {
    const resp = await fetch("/api/ingest/backfill-history", {method: "POST"});
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = data.error || "Backfill failed to start.";
      btn.disabled = false;
      btn.textContent = "Run full historical backfill";
      return;
    }
    statusEl.textContent = `Starting -- checking up to ${data.months_back} months of history…`;
    await pollBackfillHistoryStatus(btn, statusEl);
  } catch (err) {
    statusEl.textContent = "Backfill failed: " + err;
    btn.disabled = false;
    btn.textContent = "Run full historical backfill";
  }
});

document.getElementById("importBtn").addEventListener("click", async (e) => {
  const btn = e.target;
  const statusEl = document.getElementById("importStatus");
  const detailsEl = document.getElementById("importDetails");
  const text = document.getElementById("importIdsText").value;
  btn.disabled = true;
  btn.textContent = "Importing…";
  statusEl.textContent = "";
  detailsEl.innerHTML = "";
  try {
    const resp = await fetch("/api/ingest/import-ids", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = data.error || "Import failed.";
      btn.disabled = false;
      btn.textContent = "Import these transactions";
      return;
    }
    statusEl.textContent = `Starting import of ${data.total_ids} transaction(s)…`;
    document.getElementById("importIdsText").value = "";
    await pollImportStatus(btn, statusEl);
  } catch (err) {
    statusEl.textContent = "Import failed: " + err;
    btn.disabled = false;
    btn.textContent = "Import these transactions";
  }
});

loadAll();
setInterval(loadAll, 60000);
</script>
</body>
</html>
"""
