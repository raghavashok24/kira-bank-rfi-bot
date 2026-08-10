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
import pickle
import re
import sqlite3
import statistics
import threading
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

# Optional, but strongly recommended: the secret key shown in Sumsub
# Dashboard -> Webhook manager when you create/edit a webhook (auto-
# generated, or one you set yourself). Same rule as SUMSUB_APP_TOKEN /
# SUMSUB_SECRET_KEY -- put it in Render's Environment tab, never in chat,
# a ticket, or committed code. Without this set, ANY POST to
# /api/webhooks/sumsub is accepted at face value (no way to tell a real
# Sumsub event from someone else hitting the public URL) -- see
# verify_webhook_signature and docs.sumsub.com/docs/webhook-manager.
SUMSUB_WEBHOOK_SECRET_KEY = os.environ.get("SUMSUB_WEBHOOK_SECRET_KEY", "")
# Maps the algorithm name Sumsub sends in the X-Payload-Digest-Alg header to
# the matching hashlib constructor. HMAC_SHA1_HEX is Sumsub's legacy/
# deprecated option -- supported here for compatibility with older webhook
# configs, but HMAC_SHA256_HEX (the default for new webhooks) or
# HMAC_SHA512_HEX should be preferred.
WEBHOOK_DIGEST_ALGOS = {
    "HMAC_SHA256_HEX": hashlib.sha256,
    "HMAC_SHA512_HEX": hashlib.sha512,
    "HMAC_SHA1_HEX": hashlib.sha1,
}

# Optional: post-to-Slack notifications. Entirely off (every send_slack_
# message call becomes a silent no-op) unless BOTH of these are set -- the
# bot works exactly the same without Slack configured at all. Get a bot
# token from a Slack app with the chat:write scope, installed to the
# target workspace; SLACK_CHANNEL is the channel ID or name (e.g. "#kyt-
# alerts") the bot has been invited to. Same rule as the Sumsub secrets:
# only ever set these in Render's Environment tab, never in chat, a ticket,
# or committed code.
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "")
# Risk score (0-1) at or above which a newly-changed prediction triggers a
# Slack alert -- see notify_new_high_risk_predictions. Deliberately only
# fires for predictions that are NEW or materially changed since the last
# retrain (via log_predictions' dedup), not every candidate on every
# retrain, so this can't spam the channel.
SLACK_HIGH_RISK_THRESHOLD = float(os.environ.get("SLACK_HIGH_RISK_THRESHOLD", "0.75"))


def send_slack_message(text: str) -> bool:
    """Best-effort Slack post via chat.postMessage. Deliberately never
    raises -- a Slack outage, a wrong token, or SLACK_BOT_TOKEN/
    SLACK_CHANNEL simply not being set yet must never be able to break
    ingestion, training, or the webhook response path, all of which call
    this. Returns False (and logs why) on any failure instead."""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL:
        return False
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": SLACK_CHANNEL, "text": text},
            timeout=10,
        )
        body = {}
        try:
            body = resp.json()
        except ValueError:
            pass
        if not body.get("ok"):
            logger.warning("Slack message rejected: %s", body.get("error") or f"HTTP {resp.status_code}")
            return False
        return True
    except Exception:  # noqa: BLE001 -- Slack must never be able to break the caller
        logger.exception("Failed to send Slack message")
        return False


def _normalize_tag(text: str) -> str:
    """Case-insensitive, separator-insensitive tag comparison key -- so a tag
    stored on Sumsub as "Bank RFI", " bank  rfi ", "BANK RFI", "bank-rfi", or
    "bank_rfi" all still match BANK_RFI_TAG. This is the ONLY thing that
    decides whether a fetched transaction counts as "bank rfi" -- every
    transaction the bot pulls in (via webhook or the manual ID import) is
    checked against this. Hyphens/underscores are folded to spaces (in
    addition to whitespace collapsing) because Sumsub tag labels are
    sometimes typed with different separators by different reviewers, and a
    label that reads identically to a human should still match here."""
    normalized = re.sub(r"[\s_-]+", " ", (text or "").strip().lower())
    return normalized.strip()


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
    raw_json TEXT, processed INTEGER DEFAULT 0, process_detail TEXT, signature_detail TEXT
);

CREATE TABLE IF NOT EXISTS predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP, txn_id TEXT, risk_score REAL, mode TEXT, reasons_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_log_txn ON predictions_log(txn_id);

-- Single-row snapshot of the last successfully trained ModelBundle (pickled).
-- Purpose: so a process restart (Render redeploy, free-tier spin-down/up,
-- manually reopening the dashboard after it went idle) can resume serving
-- the model it already learned INSTANTLY instead of sitting in "untrained/
-- heuristic" limbo until the next full retrain finishes. This does NOT
-- replace the `transactions` table as the source of truth -- it's a
-- convenience cache of the last fitted model, always rebuilt (and
-- overwritten here) on the very next retrain once new data arrives. See
-- save_model_state/load_model_state.
CREATE TABLE IF NOT EXISTS model_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    saved_at TEXT,
    n_transactions_at_save INTEGER,
    bundle_blob BLOB
);
"""

# Sumsub's documented reviewStatus values (docs.sumsub.com/docs/transaction-
# monitoring-webhooks) are "init", "onHold", "awaitingUser", "completed" --
# only "completed" means a transaction has a final, trustworthy verdict.
# "pending"/"queued"/None/"" are kept too as a defensive catch-all for any
# undocumented/legacy value actually observed on the wire, so an unexpected
# status fails safe (treated as "still open", i.e. excluded from training
# rather than silently trusted as a confirmed negative).
OPEN_STATUSES = ("init", "onHold", "awaitingUser", "pending", "queued", None, "")


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        for new_col in ("notes", "notes_text", "counterparty_bank_name"):
            if new_col not in cols:
                conn.execute(f"ALTER TABLE transactions ADD COLUMN {new_col} TEXT")
        wh_cols = {r["name"] for r in conn.execute("PRAGMA table_info(webhook_events)").fetchall()}
        if "process_detail" not in wh_cols:
            conn.execute("ALTER TABLE webhook_events ADD COLUMN process_detail TEXT")
        if "signature_detail" not in wh_cols:
            conn.execute("ALTER TABLE webhook_events ADD COLUMN signature_detail TEXT")


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


def distinct_raw_tags(transactions: list[dict] | None = None) -> list[dict]:
    """Every distinct raw tag label actually seen across ingested
    transactions, with how many transactions carry it and whether it's the
    one this bot is matching against (BANK_RFI_TAG_NORMALIZED). This exists
    purely as a diagnostic: if the count of bank-rfi transactions is ever
    unexpectedly 0/low despite real tagged transactions existing on Sumsub,
    this list makes a spelling/separator mismatch (e.g. "bank-rfi" vs
    "bank rfi", or a totally different label) visible immediately instead of
    requiring someone to manually pull one transaction's tags via the API to
    check."""
    txns = transactions if transactions is not None else all_transactions()
    counts: dict[str, int] = {}
    for t in txns:
        for tag in (t.get("tags") or []):
            tag = (tag or "").strip()
            if not tag:
                continue
            counts[tag] = counts.get(tag, 0) + 1
    out = [
        {"tag": tag, "count": count, "matches_bank_rfi": _normalize_tag(tag) == BANK_RFI_TAG_NORMALIZED}
        for tag, count in counts.items()
    ]
    out.sort(key=lambda r: (-r["count"], r["tag"]))
    return out


def distinct_review_statuses(transactions: list[dict] | None = None) -> list[dict]:
    """Every distinct review_status value actually extracted from ingested
    transactions, with counts and whether it currently counts as "open"
    (excluded from training as a negative example -- see OPEN_STATUSES and
    train_model's docstring) or "resolved" (eligible). This exists for the
    same reason distinct_raw_tags does: training eligibility now depends
    entirely on correctly reading this field out of Sumsub's real API
    response (confirmed nested at review.reviewStatus, per
    docs.sumsub.com/reference/get-transaction), and if that extraction is
    ever wrong for a real account's data shape, this is where it would show
    up as "every transaction has review_status=None (or some unexpected
    value)" instead of silently producing a model that never leaves
    heuristic mode with zero pattern findings."""
    txns = transactions if transactions is not None else all_transactions()
    counts: dict[str, int] = {}
    for t in txns:
        status = t.get("review_status")
        key = status if status not in (None, "") else "(none/missing)"
        counts[key] = counts.get(key, 0) + 1
    out = [{"review_status": k, "count": v, "counts_as_open": (k if k != "(none/missing)" else None) in OPEN_STATUSES}
           for k, v in counts.items()]
    out.sort(key=lambda r: -r["count"])
    return out


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


def save_model_state(bundle, n_transactions: int) -> None:
    """Persists the just-trained ModelBundle (classifier, scaler, vocab,
    pattern findings, metrics -- everything needed to serve predictions)
    into the SAME sqlite file as the transactions themselves, overwriting
    the single existing row each time. This is what lets a process restart
    resume serving the model it already learned instead of starting back at
    "untrained/heuristic" -- see load_model_state, called from
    _ensure_bundle_loaded on startup. A pickling failure here must never
    take down a successful retrain, so it's caught and logged, not raised."""
    try:
        blob = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to pickle model bundle for persistence (retrain itself still succeeded)")
        return
    try:
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO model_state (id, saved_at, n_transactions_at_save, bundle_blob)
                   VALUES (1, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       saved_at=excluded.saved_at,
                       n_transactions_at_save=excluded.n_transactions_at_save,
                       bundle_blob=excluded.bundle_blob""",
                (datetime.now(timezone.utc).isoformat(), n_transactions, blob),
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save model state to the database (retrain itself still succeeded)")


def load_model_state():
    """Restores the last-persisted ModelBundle, if any. Returns None (never
    raises) on any problem -- a missing/corrupt/incompatible snapshot just
    means the caller falls back to training fresh from the transactions
    table, which is always correct even if slower. NOTE: this snapshot
    lives in the same sqlite file as `transactions` -- on a platform with no
    persistent disk (see _db_likely_ephemeral), a real restart wipes both
    together, so this bridges in-process restarts / soft redeploys where
    the disk survives, not a full ephemeral wipe. When the disk IS wiped,
    this correctly returns None and the normal auto-backfill-then-retrain
    path (see _maybe_auto_backfill_on_empty_db) rebuilds everything from
    Sumsub, which remains the actual durable source of truth."""
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT bundle_blob, saved_at, n_transactions_at_save FROM model_state WHERE id = 1"
            ).fetchone()
        if not row or not row["bundle_blob"]:
            return None
        bundle = pickle.loads(row["bundle_blob"])
        logger.info(
            "Restored previously-trained model from the database (saved_at=%s, %d transaction(s) at save "
            "time) -- resuming from there instead of starting untrained.",
            row["saved_at"], row["n_transactions_at_save"],
        )
        return bundle
    except Exception:  # noqa: BLE001 -- any restore failure just means "train fresh instead"
        logger.exception("Failed to restore persisted model state (will train fresh instead)")
        return None


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


def record_webhook_event(event_type: str, kyt_txn_id: str, raw_json: dict, signature_detail: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO webhook_events (event_type, kyt_txn_id, raw_json, signature_detail) VALUES (?,?,?,?)",
            (event_type, kyt_txn_id, json.dumps(raw_json), signature_detail),
        )
        return cur.lastrowid


def recent_webhook_events(limit: int = 25) -> list[dict]:
    """Per-event detail (not just aggregate counts) for the dashboard's
    "recent webhook activity" diagnostic -- this is what turns "the webhook
    isn't updating" from a guess into something inspectable: for each of the
    last N requests to /api/webhooks/sumsub, exactly what type it was, what
    the signature check said (present/absent/valid/invalid), and whether
    ingestion actually succeeded or why not."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, received_at, event_type, kyt_txn_id, processed, process_detail, signature_detail "
            "FROM webhook_events ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_webhook_event_processed(row_id: int, ok: bool, detail: str = ""):
    with get_conn() as conn:
        conn.execute(
            "UPDATE webhook_events SET processed=?, process_detail=? WHERE id=?",
            (1 if ok else 0, detail, row_id),
        )


def webhook_event_count(exclude_unrecognized: bool = True) -> int:
    """Counts real Sumsub events received. By default excludes the two
    synthetic bucket labels used for requests that weren't a recognizable
    Sumsub payload at all (couldn't be parsed as JSON, or had no
    "type"/txn id and failed signature check) -- a request that DID parse
    as a real Sumsub event but failed signature verification still counts
    here (under its real event type), since Sumsub genuinely sent it; see
    webhook_event_type_breakdown's sig_failed_count for that distinction."""
    with get_conn() as conn:
        if exclude_unrecognized:
            return conn.execute(
                "SELECT COUNT(*) c FROM webhook_events WHERE event_type NOT IN ('invalid_signature', 'invalid_json')"
            ).fetchone()["c"]
        return conn.execute("SELECT COUNT(*) c FROM webhook_events").fetchone()["c"]


def _signature_failed(detail: str | None) -> bool:
    """signature_detail is a free-text explanation (see
    verify_webhook_signature), not a boolean column, so this is the one
    place that interprets it: anything other than "verified" or
    "disabled" (no secret configured, so nothing to fail) counts as a
    failure -- missing headers, unsupported algorithm, or a mismatch."""
    d = (detail or "").lower()
    return not d.startswith("signature verified") and "disabled" not in d


def webhook_event_type_breakdown(hours: int = 24 * 14) -> list[dict]:
    """Every distinct Sumsub event `type` actually received in the last
    `hours`, with counts, how many of each resulted in a real
    ingest-and-retrain, and how many failed SIGNATURE verification
    specifically. This is THE diagnostic for "the webhook isn't firing for
    every new transaction" -- and it separates two very different failure
    modes that look identical from Sumsub's side (both show as a failed
    delivery):
      - The event type simply never appears here at all -> Sumsub isn't
        sending it -- enable it in Dashboard -> Webhook manager (see
        docs.sumsub.com/docs/transaction-monitoring-webhooks). The event
        that fires when a transaction is FIRST created is a specific type
        (applicantKytTxnCreated), distinct from review-outcome events.
      - The event type DOES appear, with a real count, but sig_failed_count
        equals count (everything of that type is failing signature check)
        -> Sumsub IS sending it, but SUMSUB_WEBHOOK_SECRET_KEY in Render
        doesn't match the secret currently shown in Sumsub's Webhook
        manager for this webhook -- a code/config mismatch, not a Sumsub-
        side delivery problem."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT event_type, processed, signature_detail FROM webhook_events WHERE received_at >= ?",
            (cutoff,),
        ).fetchall()
    agg: dict[str, dict] = {}
    for r in rows:
        bucket = agg.setdefault(r["event_type"], {"event_type": r["event_type"], "count": 0, "processed_count": 0, "sig_failed_count": 0})
        bucket["count"] += 1
        bucket["processed_count"] += 1 if r["processed"] else 0
        bucket["sig_failed_count"] += 1 if _signature_failed(r["signature_detail"]) else 0
    return sorted(agg.values(), key=lambda b: -b["count"])


def delete_transaction(txn_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM transactions WHERE txn_id=?", (txn_id,))


def log_predictions(scored: list[dict]) -> list[dict]:
    """Persists a snapshot of every currently-scored candidate transaction
    into predictions_log -- so "every newly predicted transaction" leaves a
    durable, queryable record instead of being computed fresh and discarded
    on every dashboard load. Called after every retrain (see retrain_only),
    so a compliance reviewer can later answer "what did the model think
    about this transaction, and when did that change" via
    GET /api/predictions/history/{txn_id}, not just "what does it think
    right now." Skips re-inserting a row for a transaction whose score
    hasn't meaningfully changed since its last logged entry, so this table
    doesn't grow by (candidates x retrains) when most retrains don't
    actually move most scores.

    Returns the subset of `scored` that was actually newly logged (new
    candidate or materially changed score) -- callers use this (see
    notify_new_high_risk_predictions) to react only to real changes, not
    every candidate on every retrain."""
    if not scored:
        return []
    with get_conn() as conn:
        last_scores = dict(conn.execute(
            "SELECT txn_id, risk_score FROM predictions_log WHERE id IN "
            "(SELECT MAX(id) FROM predictions_log GROUP BY txn_id)"
        ).fetchall())
        rows = []
        changed = []
        for s in scored:
            prev = last_scores.get(s["txn_id"])
            if prev is not None and abs(prev - s["risk_score"]) < 0.005:
                continue
            rows.append((s["txn_id"], s["risk_score"], s.get("mode"), json.dumps(s.get("reasons") or [])))
            changed.append(s)
        if rows:
            conn.executemany(
                "INSERT INTO predictions_log (txn_id, risk_score, mode, reasons_json) VALUES (?,?,?,?)", rows,
            )
    return changed


def notify_new_high_risk_predictions(changed: list[dict]) -> None:
    """Slack alert for predictions that just newly crossed (or moved while
    already above) SLACK_HIGH_RISK_THRESHOLD -- fed only the `changed`
    subset log_predictions actually wrote, so this can't spam the channel
    with the same steady-state high scores on every single retrain. A
    no-op (checked inside send_slack_message) if Slack isn't configured."""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL:
        return
    high = [c for c in (changed or []) if (c.get("risk_score") or 0) >= SLACK_HIGH_RISK_THRESHOLD]
    if not high:
        return
    high.sort(key=lambda c: -(c.get("risk_score") or 0))
    lines = [
        f"• `{c['txn_id']}` — {c['risk_score']*100:.1f}% — "
        f"{(c.get('reasons') or ['no specific reason matched'])[0]}"
        for c in high[:10]
    ]
    more = f"\n…and {len(high) - 10} more." if len(high) > 10 else ""
    text = (
        f":rotating_light: *Bank RFI Bot -- {len(high)} transaction(s) newly scored high-risk "
        f"(≥{SLACK_HIGH_RISK_THRESHOLD*100:.0f}%) for \"bank rfi\"*\n" + "\n".join(lines) + more
    )
    send_slack_message(text)


def notify_new_bank_rfi_tag(row: dict) -> None:
    """Slack alert the moment a transaction is actually confirmed as
    "bank rfi" on Sumsub (as opposed to merely predicted) -- ground truth,
    not a model opinion. No-op if Slack isn't configured."""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL:
        return
    text = (
        f":triangular_flag_on_post: *New \"bank rfi\" transaction confirmed on Sumsub*\n"
        f"• Txn ID: `{row.get('txn_id')}`\n"
        f"• Amount: {row.get('amount')} {row.get('currency') or ''}\n"
        f"• Applicant: {row.get('applicant_id') or 'unknown'}\n"
        f"• Counterparty country: {row.get('counterparty_country') or 'unknown'}"
    )
    send_slack_message(text)


def prediction_history(txn_id: str, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT logged_at, risk_score, mode, reasons_json FROM predictions_log "
            "WHERE txn_id=? ORDER BY id DESC LIMIT ?", (txn_id, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["reasons"] = json.loads(d.pop("reasons_json") or "[]")
            out.append(d)
        return out


def recent_predictions_log(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT txn_id, logged_at, risk_score, mode FROM predictions_log ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================================
# Feature engineering -- every feature here should be explainable in one
# sentence to a compliance reviewer.
# ============================================================================

# Both ISO alpha-2 AND alpha-3 forms of each country, because Sumsub's
# confirmed real field for this (data.counterparty.residenceCountry, per
# docs.sumsub.com/reference/get-transaction) returns alpha-3 codes ("RUS",
# "BLR", "IRN"...), but alpha-2 is kept too in case any other source of this
# value (a payment-method-specific field, a differently-shaped account) ever
# supplies the shorter form -- comparison is done against whichever string
# actually shows up, uppercased, so either form matches.
HIGH_RISK_COUNTRIES = {
    "IR", "IRN", "KP", "PRK", "SY", "SYR", "AF", "AFG",
    "MM", "MMR", "VE", "VEN", "RU", "RUS", "BY", "BLR",
}
STRUCTURING_THRESHOLDS = [3000, 10000]
CATEGORICAL_FEATURES = ["currency", "txn_type", "counterparty_country"]
# Caps how many one-hot columns a single categorical feature can contribute.
# A real account's counterparty_country/currency can have dozens of distinct
# values -- left uncapped, that pushes total feature count well past what a
# few dozen bank-rfi examples can support, which is a classic setup for a
# classifier to memorize each positive's exact category combination instead
# of learning anything general (everything that doesn't match one exactly
# then scores near zero). See train_model's docstring for the full story.
MAX_CATEGORICAL_VOCAB = 12
NUMERIC_FEATURES = [
    "amount", "log_amount", "direction_out", "counterparty_high_risk", "is_round_amount",
    "near_reporting_threshold", "applicant_prior_txn_count", "applicant_txn_count_24h",
    "applicant_txn_count_7d", "applicant_avg_prior_amount", "amount_vs_applicant_avg",
    "hour_of_day", "is_weekend", "is_new_applicant",
]

# ----------------------------------------------------------------------------
# ANTI-LEAKAGE GUARANTEE (do not weaken this without re-reading the incident
# this fixed): the "bank rfi" tag -- and anything counted/derived from it,
# such as how many times an applicant's PAST transactions were tagged -- must
# never appear in NUMERIC_FEATURES or CATEGORICAL_FEATURES. Those two lists
# are the entire input (X) the classifier/heuristic ever sees. is_bank_rfi is
# the *label* (y) in train_model -- ground truth to learn FROM and to
# evaluate against, never a variable the model reads to make a live
# prediction. An earlier version violated this: it fed
# "applicant_prior_bank_rfi_count"/"applicant_prior_bank_rfi_rate" into the
# feature vector AND surfaced "this applicant already has an X% bank-rfi
# rate" as a scoring justification. That let the bot shortcut to "flag it
# because it (or this applicant) was already flagged before" instead of
# learning the actual behavioral patterns (amount, timing, velocity,
# counterparty risk, structuring) that make a transaction look like a bank
# RFI in the first place -- which also meant a first-time applicant with a
# risky profile but no tag history could never score high. Fixed by removing
# both the features and the reason string, and this comment + the assertion
# right below exist so a future change can't silently reintroduce it.
_FORBIDDEN_FEATURE_MARKERS = ("bank_rfi", "is_rfi", "_tag")
for _fname in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
    if any(marker in _fname for marker in _FORBIDDEN_FEATURE_MARKERS):
        raise RuntimeError(
            f"Anti-leakage guard tripped: feature '{_fname}' looks derived from the bank-rfi tag "
            "itself and must not be used as a model input. See the comment above NUMERIC_FEATURES."
        )
del _fname
# ----------------------------------------------------------------------------


def _parse_amount(row: dict) -> float:
    try:
        return float(row.get("amount") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_dt(row: dict):
    """Parses txn_created_at into a naive UTC datetime (naive to stay
    comparable with the rest of this file's timestamps, e.g. ingested_at
    and datetime.min). This previously only tried Z-suffixed or
    offset-less formats and truncated the input to a fixed length based on
    the FORMAT string's length (not the actual date's length) before
    matching -- neither of those lined up with Sumsub's real date format,
    confirmed from actual API/webhook payloads elsewhere in this file, e.g.
    "2025-11-17 17:26:49+0000" (space-separated, +0000 offset, no 'Z', no
    'T'). Every real transaction was silently failing to parse and falling
    back to ingested_at, which -- since the historical backfill walks
    backward from now, inserting the newest real transactions FIRST and
    the oldest LAST -- scrambles chronological order badly enough to make
    "candidates since the last bank-rfi tag" come out empty even when
    there plainly should be some."""
    ts = row.get("txn_created_at")
    if ts is None or ts == "":
        return None
    ts = str(ts).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",  # naive datetime.isoformat(), e.g. dev seed data
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    # Last resort: createdAtMs-style epoch milliseconds (or seconds) stored
    # as a numeric string in the same text column.
    try:
        num = float(ts)
        return datetime.utcfromtimestamp(num / 1000.0 if num > 1e12 else num)
    except (TypeError, ValueError):
        pass
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
    # Deliberately NOT computing/returning anything derived from prior
    # is_bank_rfi tags here (e.g. a prior bank-rfi count/rate) -- see the
    # anti-leakage comment above NUMERIC_FEATURES. This function's output is
    # the model's entire view of a transaction; it must describe behavior
    # (amounts, timing, velocity, counterparty) only, never tag history.

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

def analyze_patterns(bank_rfi_features: list[dict], other_features: list[dict], n_excluded_open: int = 0) -> dict:
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
    if len(other_features) < 3:
        # This is a DIFFERENT limitation from "not enough bank-rfi examples"
        # and needs its own message: it means every non-tagged transaction
        # ingested so far is STILL OPEN/PENDING review (see train_model's
        # docstring on why open transactions are excluded here) -- there's
        # nothing yet to statistically contrast the bank-rfi examples
        # against, even though bank-rfi examples themselves may be plenty.
        # Without this message, a heuristic score of exactly 0% for every
        # candidate looks identical to "the model is broken" when it's
        # actually "the model has nothing confirmed to compare against yet."
        open_note = (
            f" ({n_excluded_open} transaction(s) are currently open/pending review and don't count "
            "until their review actually concludes)" if n_excluded_open else ""
        )
        result["narrative"].append(
            f"Only {len(other_features)} transaction(s) with a CONCLUDED review and no \"bank rfi\" tag "
            f"exist yet{open_note} -- too few to know what \"not risky\" looks like, so risk scores will "
            "sit near 0% for everyone regardless of how many bank-rfi examples exist. This resolves itself "
            "automatically as more transactions finish review (approved/rejected/reviewed) on Sumsub -- "
            "no action needed here, just more confirmed outcomes."
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
    """Builds the one-hot vocabulary for a categorical feature, capped to
    the MAX_CATEGORICAL_VOCAB most frequent values (ties broken
    alphabetically for determinism). Values outside the cap simply don't
    match any of these columns at vectorization time -- i.e. they're
    implicitly treated as "none of the common categories" rather than
    getting their own dedicated (and, for a rare value, nearly useless)
    dimension. This bounds total feature count regardless of how many
    distinct real-world values (countries, currencies) an account has."""
    counts = Counter(str(f.get(key, "unknown")) for f in features_list)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [v for v, _ in ranked[:MAX_CATEGORICAL_VOCAB]]


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
                 pattern_findings, explain_classifier=None, metrics=None):
        self.mode = mode
        self.classifier = classifier
        self.explain_classifier = explain_classifier
        self.scaler = scaler
        self.vocab = vocab
        self.feature_names = feature_names
        self.train_features = train_features
        self.train_matrix = train_matrix
        self.pattern_findings = pattern_findings
        self.metrics = metrics or {}
        # Set right after construction in train_model, not here -- a plain
        # attribute (not a constructor arg) so it survives pickling via
        # save_model_state/load_model_state with zero extra plumbing.
        self.trained_at = None


def _cross_val_metric(base_clf_factory, X: np.ndarray, y: np.ndarray, cv_folds: int, scoring: str) -> float | None:
    """Actual measured predictive performance, not just "it's in ML mode
    now": stratified k-fold cross-validation using a FRESH classifier per
    fold (never the one fit on all the data), so this number reflects how
    well the model separates real held-out bank-rfi cases from confirmed-
    negative ones. Surfaced on every model version so "is this actually
    predicting anything" has a real answer instead of vibes. Returns None
    (not a misleading default) when there isn't enough data to fold
    reliably, so the dashboard can say "not enough data yet" instead of
    implying "no better than random".

    Called with two different `scoring` values (see train_model): "roc_auc"
    and "average_precision" (precision-recall AUC). Both are reported
    because with only a few dozen bank-rfi positives -- a genuinely rare
    event even within the already-filtered "confirmed" training set -- ROC-
    AUC alone can look deceptively strong: it rewards ranking the many true
    negatives correctly, which is the easy part when negatives vastly
    outnumber positives. PR-AUC focuses on how well the model finds the
    positives specifically (precision among what it flags, recall of what
    it catches), which is the metric that actually matters for a rare-event
    early-warning system and is far less forgiving of a model that's
    secretly just predicting "not bank-rfi" most of the time."""
    if cv_folds < 2:
        return None
    try:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
        scores = cross_val_score(base_clf_factory(), X, y, cv=skf, scoring=scoring)
        return round(float(np.mean(scores)), 4)
    except Exception:  # noqa: BLE001 -- a metric failing must never block training
        logger.exception("CV %s computation failed", scoring)
        return None


# How much of the confirmed (bank-rfi + confirmed-other) dataset, sorted
# chronologically by transaction date, is held out as a genuine unseen-in-
# training TEST set for _temporal_holdout_metrics -- as opposed to the
# random k-fold CV above, this specifically tests "does the model trained
# on older data still work on the most recent transactions", i.e. whether
# it's keeping up with trend drift rather than just memorizing history.
TEMPORAL_HOLDOUT_FRACTION = 0.2
# Below this many examples of a class in either the train or test slice,
# the split is too small to mean anything (a handful of examples can flip
# an AUC from 0.9 to 0.4 on pure noise) -- reported as "not enough data
# yet" rather than a number that looks precise but isn't.
MIN_HOLDOUT_PER_CLASS = 5


def _temporal_holdout_metrics(labeled_feats: list[dict], vocab: dict) -> dict:
    """A REAL train/test split, not just cross-validation: sorts every
    labeled (bank-rfi + confirmed-other) example by its own transaction
    timestamp, trains a classifier ONLY on the older ~80%, and evaluates it
    on the newest ~20% -- data the training process never saw in any form,
    including for fitting the scaler. This is deliberately separate from
    the deployed model (which is trained on 100% of labeled data, since
    with only a few dozen bank-rfi positives, throwing away a temporal
    slice permanently would hurt the model actually serving predictions) --
    it exists purely as an honest, periodically-recomputed answer to "would
    this approach have caught last month's bank-rfi cases using only what
    was known before them", which cross-validation (random shuffling across
    all of history) cannot answer on its own. A large gap between this and
    the CV metrics above is the clearest possible signal of trend drift --
    i.e. the patterns that predicted bank-rfi a year ago no longer look like
    the patterns predicting it now."""
    n = len(labeled_feats)
    if n < 2 * MIN_HOLDOUT_PER_CLASS:
        return {"temporal_holdout_note": f"Only {n} labeled example(s) total -- too few for a temporal holdout yet."}

    ordered = sorted(range(n), key=lambda i: labeled_feats[i].get("_created_at") or datetime.min)
    n_test = max(1, round(n * TEMPORAL_HOLDOUT_FRACTION))
    test_idx = set(ordered[-n_test:])
    train_feats = [labeled_feats[i] for i in range(n) if i not in test_idx]
    test_feats = [labeled_feats[i] for i in ordered[-n_test:]]

    y_train = np.array([1 if f["_is_bank_rfi"] else 0 for f in train_feats])
    y_test = np.array([1 if f["_is_bank_rfi"] else 0 for f in test_feats])
    n_pos_train, n_neg_train = int(y_train.sum()), int(len(y_train) - y_train.sum())
    n_pos_test, n_neg_test = int(y_test.sum()), int(len(y_test) - y_test.sum())

    if min(n_pos_train, n_neg_train, n_pos_test, n_neg_test) < MIN_HOLDOUT_PER_CLASS:
        return {
            "temporal_holdout_note": (
                f"Not enough of both classes on both sides of a chronological split yet "
                f"(train: {n_pos_train} bank-rfi / {n_neg_train} other, test: {n_pos_test} bank-rfi / "
                f"{n_neg_test} other -- need at least {MIN_HOLDOUT_PER_CLASS} of each on each side). "
                f"This resolves itself as more confirmed examples accumulate over time."
            ),
        }

    try:
        # Fresh scaler fit ONLY on the training slice -- fitting it on the
        # full labeled set (train+test) would leak test-set statistics into
        # preprocessing, which is a real (if subtle) form of the exact
        # train/test contamination this function exists to avoid.
        holdout_scaler = StandardScaler()
        X_train = _vectorize(train_feats, vocab, holdout_scaler, fit_scaler=True)
        X_test = _vectorize(test_feats, vocab, holdout_scaler, fit_scaler=False)
        holdout_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)
        holdout_clf.fit(X_train, y_train)
        probs = holdout_clf.predict_proba(X_test)[:, 1]
        return {
            "temporal_holdout_n_train": len(train_feats),
            "temporal_holdout_n_test": len(test_feats),
            "temporal_holdout_roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "temporal_holdout_pr_auc": round(float(average_precision_score(y_test, probs)), 4),
            "temporal_holdout_note": (
                f"Classifier trained ONLY on the oldest {len(train_feats)} labeled examples, evaluated on "
                f"the newest {len(test_feats)} it never saw during training or preprocessing -- a real "
                f"held-out test set, not resampled cross-validation. A much lower score here than the "
                f"cross-validated AUC above suggests the model isn't keeping up with recent trend changes."
            ),
        }
    except Exception:  # noqa: BLE001 -- a metric failing must never block training
        logger.exception("Temporal holdout evaluation failed")
        return {"temporal_holdout_note": "Temporal holdout evaluation failed (see server logs) -- ignored, training continued."}


def train_model(transactions: list[dict]) -> ModelBundle:
    """The dataset here is almost always small-n (a few dozen bank-rfi
    positives) and high-dimensional after one-hot encoding categoricals --
    a setup where a lightly-regularized logistic regression easily
    memorizes each positive's exact feature combination (helped along by
    class_weight="balanced", which amplifies the loss contribution of each
    rare positive). Once sigmoid-calibrated, that overfit decision function
    tends to produce a "cliff": very high scores for inputs that closely
    match a training positive, and near-zero for everything else, even
    other transactions that share real risk characteristics. Two things
    fight this: MAX_CATEGORICAL_VOCAB bounds how many one-hot dimensions a
    single categorical feature can add (see _vocab_for), and C is set low
    (stronger L2 regularization) rather than sklearn's default/previous
    C=2.0, which was permissive enough to let the model fit almost
    anything, including noise.

    CRITICAL label-leakage fix: transactions that are still open/pending
    review (review_status in OPEN_STATUSES) do NOT yet have a trustworthy
    label. Earlier versions of this function trained on is_bank_rfi=0/1 for
    EVERY transaction, including open ones -- which means a transaction
    sitting in the candidate pool waiting to be scored had, at the exact
    same time, already been fed into training as a confirmed "definitely
    not bank-rfi" example. That's a direct feedback loop working against
    the entire point of an early-warning system: the model was being taught
    "this exact profile is safe" about the very transactions it was
    supposed to flag as risky, right up until the moment Sumsub confirmed
    otherwise. This is almost certainly why predictions looked flat/
    uninformative despite real bank-rfi examples existing -- the model
    wasn't wrong so much as trained on a self-defeating premise. Fixed by
    building the training set only from RESOLVED transactions: every
    bank-rfi-tagged transaction (a tag is ground truth the moment it's
    applied, regardless of workflow status) plus every transaction whose
    review has actually concluded without that tag. Currently-open
    transactions are still fully eligible to be SCORED (see
    score_all_open) -- they're just excluded from teaching the model what
    "not risky" looks like until their own outcome is actually known.

    RARE-EVENT NOTE: a real account here typically has on the order of a
    few dozen confirmed bank-rfi examples against a much larger pool of
    everything else -- textbook class imbalance, not just "a small
    dataset". Three separate things account for that, not just one:
    class_weight="balanced" re-weights the loss so the rare positive class
    isn't drowned out, StratifiedKFold (both for CalibratedClassifierCV's
    internal folds and for the CV metrics below) keeps the true bank-rfi
    ratio intact in every fold instead of risking a fold with zero
    positives, and both ROC-AUC and PR-AUC (precision-recall AUC) are
    computed and reported side by side -- PR-AUC specifically because
    ROC-AUC alone can look artificially strong under imbalance (see
    _cross_val_metric's docstring). This function ALWAYS retrains on the
    complete current transaction history every time it's called (never an
    incremental/partial update on top of a stale prior model) -- for a
    dataset this size that is the statistically correct choice, not a
    limitation: a fresh batch fit on all available labeled data outperforms
    incremental updates, and it means "new transaction arrives" -> "next
    retrain sees 100% of history, including that transaction" always, with
    no risk of silently drifting from what's actually in the database. What
    IS cached across restarts is the trained bundle itself (see
    save_model_state/load_model_state) so a process restart doesn't sit
    untrained while waiting for the next retrain -- not the training
    process."""
    history_index = build_applicant_history_index(transactions)
    feats = [extract_features(t, history_index) for t in transactions]
    for f, t in zip(feats, transactions):
        f["_is_bank_rfi"] = bool(t.get("is_bank_rfi"))
        f["_is_open"] = t.get("review_status") in OPEN_STATUSES
        # Used only by _temporal_holdout_metrics to build a chronological
        # train/test split -- never fed into the feature vector itself.
        f["_created_at"] = _parse_dt(t)

    bank_rfi_feats = [f for f in feats if f["_is_bank_rfi"]]
    confirmed_other_feats = [f for f in feats if not f["_is_bank_rfi"] and not f["_is_open"]]
    excluded_open_feats = [f for f in feats if not f["_is_bank_rfi"] and f["_is_open"]]
    labeled_feats = bank_rfi_feats + confirmed_other_feats
    pattern_findings = analyze_patterns(bank_rfi_feats, confirmed_other_feats, n_excluded_open=len(excluded_open_feats))

    vocab = {cat: _vocab_for(feats, cat) for cat in CATEGORICAL_FEATURES}
    feature_names = list(NUMERIC_FEATURES) + [f"{cat}={v}" for cat in CATEGORICAL_FEATURES for v in vocab[cat]]

    base_metrics = {
        "n_bank_rfi": len(bank_rfi_feats),
        "n_confirmed_other": len(confirmed_other_feats),
        "n_excluded_open_pending": len(excluded_open_feats),
        "positive_rate_pct_in_training": (
            round(100 * len(bank_rfi_feats) / len(labeled_feats), 2) if labeled_feats else None
        ),
        "rare_event_note": (
            "Bank-rfi is treated as a rare/imbalanced event: class_weight=\"balanced\" reweights training, "
            "StratifiedKFold preserves the true ratio in every fold, and both ROC-AUC and PR-AUC (more "
            "informative under imbalance) are tracked."
        ),
    }

    if len(bank_rfi_feats) < MIN_ML_SAMPLES or len(confirmed_other_feats) < MIN_ML_SAMPLES:
        scaler = StandardScaler()
        if feats:
            _vectorize(feats, vocab, scaler, fit_scaler=True)
        train_matrix = _vectorize(bank_rfi_feats, vocab, scaler) if bank_rfi_feats else np.zeros((0, len(feature_names)))
        bundle = ModelBundle(
            mode="heuristic", classifier=None, scaler=scaler, vocab=vocab, feature_names=feature_names,
            train_features=bank_rfi_feats, train_matrix=train_matrix, pattern_findings=pattern_findings,
            metrics={
                **base_metrics,
                "reason": (
                    f"Need at least {MIN_ML_SAMPLES} bank-rfi examples AND {MIN_ML_SAMPLES} confirmed "
                    f"non-bank-rfi examples (transactions whose review has actually concluded) before "
                    f"switching to a calibrated classifier -- currently {len(bank_rfi_feats)} bank-rfi / "
                    f"{len(confirmed_other_feats)} confirmed-other ({len(excluded_open_feats)} more are "
                    f"still open/pending review and can't be used as training negatives yet)."
                ),
            },
        )
    else:
        scaler = StandardScaler()
        X_labeled = _vectorize(labeled_feats, vocab, scaler, fit_scaler=True)
        y = np.array([1 if f["_is_bank_rfi"] else 0 for f in labeled_feats])

        explain_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)
        explain_clf.fit(X_labeled, y)

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        cv_folds = max(2, min(5, n_pos // 5, n_neg // 5))
        base_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)
        clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv_folds)
        clf.fit(X_labeled, y)

        clf_factory = lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)  # noqa: E731
        cv_auc = _cross_val_metric(clf_factory, X_labeled, y, cv_folds, scoring="roc_auc")
        cv_pr_auc = _cross_val_metric(clf_factory, X_labeled, y, cv_folds, scoring="average_precision")
        temporal_metrics = _temporal_holdout_metrics(labeled_feats, vocab)

        rfi_matrix = _vectorize(bank_rfi_feats, vocab, scaler)
        bundle = ModelBundle(
            mode="ml", classifier=clf, scaler=scaler, vocab=vocab, feature_names=feature_names,
            train_features=bank_rfi_feats, train_matrix=rfi_matrix, pattern_findings=pattern_findings,
            explain_classifier=explain_clf,
            metrics={
                **base_metrics,
                "cv_folds": cv_folds,
                "cv_roc_auc": cv_auc,
                "cv_roc_auc_note": (
                    "0.5 = no better than a coin flip, 1.0 = perfect separation of held-out bank-rfi vs. "
                    "confirmed-other transactions. Computed with fresh cross-validation folds, not on data "
                    "the final model was fit on."
                ),
                "cv_pr_auc": cv_pr_auc,
                "cv_pr_auc_note": (
                    "Precision-recall AUC over the same folds -- higher weight on correctly finding the "
                    "rare bank-rfi positives specifically, less forgiving than ROC-AUC of a model that "
                    "mostly just predicts \"not bank-rfi\". 1.0 = perfect; a useful floor to compare "
                    "against is the positive rate in training (see positive_rate_pct_in_training)."
                ),
                **temporal_metrics,
            },
        )

    bundle.trained_at = datetime.now(timezone.utc).isoformat()
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
    # No "this applicant was already tagged before" reason here on purpose --
    # that would be citing the bank-rfi tag itself as justification, which is
    # exactly the leakage this file guards against (see the comment above
    # NUMERIC_FEATURES). Every reason surfaced must trace back to a
    # behavioral feature, not tag history.
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
        # CONFIRMED against docs.sumsub.com/reference/get-transaction: the
        # real field is data.counterparty.residenceCountry (ISO alpha-3,
        # e.g. "RUS", "DEU"), NOT the paymentMethod.country / bare .country
        # paths an earlier version of this guessed at without a confirmed
        # source. Those guesses are kept as fallbacks (harmless if unused,
        # in case a payment-method-specific country is ever present too),
        # but residenceCountry is checked first since it's the one
        # Sumsub's own docs confirm actually exists on this response.
        # Getting this field wrong meant counterparty_country -- and the
        # counterparty_high_risk flag derived from it -- silently sat at
        # "unknown"/0 for every real transaction, which is a real chunk of
        # the "why doesn't it flag anything" problem, not just a cosmetic
        # gap.
        "counterparty_country": _dig(
            raw, "data.counterparty.residenceCountry", "data.counterparty.paymentMethod.country",
            "data.counterparty.country", "counterparty.country",
        ),
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
                       seen_ids: set, max_pages: int = 500) -> dict:
    """Pages through ONE date window via offset (not the type filter --
    see backfill_from_txn_query's docstring for why). Returns a dict with
    scanned/new_bank_rfi/had_items/errored/error_detail/pages -- crucially,
    `had_items` and `errored` are kept SEPARATE: a failed/rate-limited
    request must never be counted the same as "this month genuinely had no
    transactions", or a temporary API hiccup gets misread as "we've reached
    the start of history" and the whole crawl stops early. That distinction
    is exactly what was missing before and is the most likely reason a
    prior run returned far fewer transactions (88) than actually exist.

    Two more differences from the earlier version, both modeled on the
    proven-working reference implementation: a short sleep between page
    requests (that code does the same, presumably to avoid tripping
    Sumsub's rate limiting over a long multi-page crawl -- this code had no
    throttling at all before, which is a very plausible way to get
    silently rate-limited partway through a large backfill), and a
    stalled-pagination guard: if a full page comes back but every item on
    it was already seen, offset isn't advancing results (undocumented
    behavior, possibly ignored by Sumsub, possibly account-specific) and
    continuing would loop forever without finding anything new."""
    scanned = new_rfi = offset = pages = 0
    had_items = False
    errored = False
    error_detail = None
    while pages < max_pages:
        filters = {
            "data.txnDate__gte": _fmt_txn_date(window_start),
            "data.txnDate__lte": _fmt_txn_date(window_end),
        }
        resp = client.query_txns(filters=filters, limit=limit, order="-data.txnDate", offset=offset)
        pages += 1
        if not resp.ok:
            errored = True
            error_detail = f"HTTP {resp.status}: {resp.raw_body[:300]}"
            logger.warning("Txn query failed for window %s..%s at offset %d -- %s",
                            window_start, window_end, offset, error_detail)
            break
        items = (resp.json_body or {}).get("items") or (resp.json_body or {}).get("list", {}).get("items") or []
        if not items:
            break
        had_items = True
        new_this_page = 0
        for item in items:
            txn_id = _dig(item, "id", "txnId", "kytTxnId")
            if not txn_id or txn_id in seen_ids:
                continue
            seen_ids.add(txn_id)
            new_this_page += 1
            row = ingest_single_txn(client, txn_id)
            if row:
                scanned += 1
                new_rfi += 1 if row["is_bank_rfi"] else 0
        if len(items) < limit:
            break
        if new_this_page == 0:
            logger.warning("Txn query stalled (page full but no new items) for window %s..%s at offset %d "
                            "-- offset pagination may not be advancing on Sumsub's side, stopping this window",
                            window_start, window_end, offset)
            break
        offset += limit
        time.sleep(0.3)  # matches the proven-working reference implementation's inter-page delay
    return {"scanned": scanned, "new_bank_rfi": new_rfi, "had_items": had_items,
            "errored": errored, "error_detail": error_detail, "pages": pages}


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
    w = _query_txn_window(client, now - timedelta(days=days_back), now, limit, seen_ids)
    return {"strategy": "txn_query_recent", "ran": w["had_items"], "scanned": w["scanned"],
            "new_bank_rfi": w["new_bank_rfi"], "days_back": days_back,
            "errored": w["errored"], "error_detail": w["error_detail"], "pages": w["pages"]}


_backfill_history_job = {"running": False, "months_done": 0, "months_total": 0, "result": None,
                          "started_at": None, "finished_at": None}
# Diagnostics for _maybe_auto_backfill_on_empty_db (defined further down, after
# get_client/_run_backfill_history_job_in_background exist) -- surfaced on
# /api/setup so the dashboard can explain *why* a backfill started on its own,
# instead of a mysterious status change.
_startup_auto_backfill = {"attempted": False, "reason": None, "months_back": None}


def run_deep_historical_backfill(client: SumsubClient, months_back: int = 84, max_empty_months: int = 18,
                                  on_progress=None) -> dict:
    """One-time (or occasional, on-demand) DEEP crawl for full history --
    not part of the hourly cycle, since walking years of data every hour
    would be slow and would hammer Sumsub's API for no reason. Same proven
    date-range-only + offset-pagination approach as backfill_from_txn_query,
    just walked backward one calendar month at a time so each individual
    query window stays modest, for up to `months_back` months (default 7
    years), stopping early after `max_empty_months` consecutive GENUINELY
    empty months (a proxy for "before this account had any transactions").

    That distinction matters: a month that ERRORED (rate-limited, timed
    out, etc.) is NOT the same as a month that Sumsub confirmed has zero
    transactions, and must not count toward the empty-streak stop --
    conflating the two was a real bug in an earlier version of this
    function, and rate-limiting is a real risk here specifically because
    this crawl can issue hundreds of requests across years of monthly
    windows. A rate-limited month is retried once (after a longer pause)
    before being logged as errored and skipped, and the empty-streak
    default was widened from 6 to 18 months since ordinary month-to-month
    volume swings (slow periods, not "before the account existed") could
    otherwise trigger a false early stop.

    Trigger via POST /api/ingest/backfill-history; poll
    GET /api/ingest/backfill-history-status for progress since this can
    take a while for a large account. Returns a per-month `months` list so
    exactly where any gap is (errored vs. genuinely empty) is visible
    afterward instead of just one aggregate number."""
    scanned = new_rfi = empty_streak = months_walked = errored_months = 0
    seen_ids = set()
    months_detail = []
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    for i in range(months_back):
        window_start = datetime(year, month, 1, tzinfo=timezone.utc)
        window_end = (datetime(year + (1 if month == 12 else 0), (1 if month == 12 else month + 1), 1,
                                tzinfo=timezone.utc) - timedelta(seconds=1))
        label = f"{year:04d}-{month:02d}"
        try:
            w = _query_txn_window(client, window_start, window_end, 100, seen_ids)
            if w["errored"]:
                # One retry after a longer backoff -- covers a transient
                # rate-limit/hiccup without abandoning the whole crawl or
                # burning through retries too fast.
                logger.warning("Deep backfill: month %s errored (%s), retrying once after backoff",
                                label, w["error_detail"])
                time.sleep(3)
                w = _query_txn_window(client, window_start, window_end, 100, seen_ids)
        except Exception as e:  # noqa: BLE001 - one bad month must not abort the whole crawl
            logger.exception("Deep backfill: month %s raised an exception, skipping", label)
            w = {"scanned": 0, "new_bank_rfi": 0, "had_items": False, "errored": True, "error_detail": str(e)}
        months_walked += 1
        scanned += w["scanned"]
        new_rfi += w["new_bank_rfi"]
        months_detail.append({"month": label, "scanned": w["scanned"], "errored": w["errored"],
                               "error_detail": w.get("error_detail")})
        if on_progress:
            on_progress(i + 1, months_back)
        if w["errored"]:
            errored_months += 1
            # Do NOT touch empty_streak -- an error is not evidence this
            # month had no transactions, so it must not count as "empty".
        else:
            empty_streak = 0 if w["had_items"] else empty_streak + 1
            if empty_streak >= max_empty_months:
                break
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return {"strategy": "deep_historical_backfill", "ran": scanned > 0, "scanned": scanned,
            "new_bank_rfi": new_rfi, "months_walked": months_walked, "errored_months": errored_months,
            "months": months_detail}


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


def _webhook_txn_id(event: dict) -> str | None:
    """The real Sumsub KYT webhook payload field names (confirmed against
    docs.sumsub.com/docs/transaction-monitoring-webhooks, which documents
    all 11 transaction-monitoring event types) are `kytTxnId` (Sumsub's own
    ID) and `kytDataTxnId` (the client-supplied external ID, present when
    you submitted the transaction with your own ID attached) -- NOT `txnId`,
    which doesn't appear in that payload shape at all. An earlier version of
    this function fell back to `event.get("txnId")`, which could never
    actually match anything and masked cases where `kytTxnId` was legitimately
    missing (e.g. an AML case event, which isn't about a single transaction)."""
    return event.get("kytTxnId") or event.get("kytDataTxnId")


# Every transaction-monitoring event type Sumsub can send (see
# docs.sumsub.com/docs/transaction-monitoring-webhooks). Listed explicitly
# (rather than just accepting anything with a kytTxnId) so a new/unexpected
# type shows up distinctly in the event-type breakdown instead of silently
# blending in -- that breakdown is the main diagnostic for "is Sumsub even
# sending me applicantKytTxnCreated events" (see webhook_event_type_breakdown).
KNOWN_KYT_EVENT_TYPES = {
    "applicantKytTxnCreated", "applicantKytTxnApproved", "applicantKytTxnRejected",
    "applicantKytTxnReviewed", "applicantKytTxnDeleted", "applicantKytOnHold",
    "applicantKytTxnAwaitingUser", "applicantKytTxnDataChanged",
    "amlCaseApproved", "amlCaseRejected", "amlCaseOnHold",
}


def verify_webhook_signature(raw_body: bytes, digest_header: str | None, alg_header: str | None) -> tuple[bool, str]:
    """Confirms a webhook POST actually came from Sumsub, using the secret
    key from Sumsub Dashboard -> Webhook manager (SUMSUB_WEBHOOK_SECRET_KEY).
    Per docs.sumsub.com/docs/webhook-manager: Sumsub computes an HMAC digest
    over the RAW request body bytes (not re-serialized JSON -- whitespace/key
    order would change the bytes and break the digest) using the secret key
    and the algorithm named in the `X-Payload-Digest-Alg` header, then sends
    the hex digest in the `x-payload-digest` header. Verification is just
    recomputing that same HMAC locally and comparing with a constant-time
    comparison (hmac.compare_digest, so response-timing can't leak the
    correct value one byte at a time).

    Returns (is_valid, detail). If SUMSUB_WEBHOOK_SECRET_KEY isn't
    configured, this ALWAYS returns (True, "..."), i.e. verification is
    opt-in but permissive by default -- so the bot keeps working out of the
    box, and the Setup panel separately and loudly flags that verification
    is off (see /api/setup's webhook_signature_verification field) rather
    than silently rejecting real events for an operator who hasn't set the
    secret yet."""
    if not SUMSUB_WEBHOOK_SECRET_KEY:
        return True, "signature verification disabled (SUMSUB_WEBHOOK_SECRET_KEY not set)"
    if not digest_header:
        return False, "missing x-payload-digest header"
    alg_name = (alg_header or "").strip().upper()
    hasher = WEBHOOK_DIGEST_ALGOS.get(alg_name)
    if hasher is None:
        return False, f"unsupported/missing X-Payload-Digest-Alg '{alg_header}' (expected one of {sorted(WEBHOOK_DIGEST_ALGOS)})"
    computed = hmac.new(SUMSUB_WEBHOOK_SECRET_KEY.encode("utf-8"), raw_body, hasher).hexdigest()
    if hmac.compare_digest(computed, digest_header.strip()):
        return True, f"signature verified ({alg_name})"
    return False, f"signature mismatch ({alg_name}) -- payload does not match SUMSUB_WEBHOOK_SECRET_KEY"


def process_webhook_event(client: SumsubClient, event: dict) -> tuple[dict | None, str]:
    """Note: the webhook route already records receipt of every event (see
    api_sumsub_webhook) before calling this -- this function is purely about
    actually fetching and storing (or removing) the transaction once
    credentials exist. Returns (row_or_None, detail_string) -- the detail
    string is persisted onto the webhook_events row so a failure/skip reason
    is visible per-event instead of just a boolean."""
    event_type = event.get("type") or "unknown"
    kyt_txn_id = _webhook_txn_id(event)

    if event_type == "applicantKytTxnDeleted":
        # Sumsub removed this transaction and its data -- mirror that here
        # instead of trying to re-fetch it (which would just 404 and look
        # like an unrelated failure) so deleted transactions don't linger
        # forever as stale training/candidate rows.
        if not kyt_txn_id:
            return None, "applicantKytTxnDeleted with no kytTxnId -- nothing to remove"
        delete_transaction(kyt_txn_id)
        return {"txn_id": kyt_txn_id, "deleted": True, "is_bank_rfi": False}, "deleted local copy"

    if not kyt_txn_id:
        return None, f"no kytTxnId/kytDataTxnId on this {event_type} event -- nothing to fetch"

    # Captured BEFORE ingest_single_txn overwrites the local row, so the
    # caller can tell "just became bank-rfi" apart from "already was
    # bank-rfi, this is just a later review-outcome event for the same
    # transaction" -- see notify_new_bank_rfi_tag, which should fire once
    # per transaction, not once per webhook event about it.
    was_bank_rfi_before = bool((get_transaction(kyt_txn_id) or {}).get("is_bank_rfi"))
    row = ingest_single_txn(client, kyt_txn_id)
    if row is None:
        return None, f"ingest_single_txn failed for {kyt_txn_id} (see server logs for the exact Sumsub error)"
    row["_newly_bank_rfi"] = bool(row.get("is_bank_rfi")) and not was_bank_rfi_before
    return row, "ok"


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
                mode=_current_bundle.mode, n_bank_rfi=_current_bundle.metrics.get("n_bank_rfi", len(_current_bundle.train_features)),
                n_other=_current_bundle.metrics.get("n_confirmed_other", 0),
                metrics={"n_training_rows": len(txns), **_current_bundle.metrics},
                feature_importances=_feature_importances(_current_bundle),
            )
            # Cache the freshly-trained bundle so a process restart before
            # the NEXT retrain can resume from here instead of sitting
            # untrained (see save_model_state/load_model_state).
            save_model_state(_current_bundle, len(txns))
            logger.info(
                "Retrained model in %s mode on %d transactions (cv_roc_auc=%s, cv_pr_auc=%s)",
                _current_bundle.mode, len(txns), _current_bundle.metrics.get("cv_roc_auc"),
                _current_bundle.metrics.get("cv_pr_auc"),
            )
            # Persist a snapshot of every currently-scored candidate so
            # "every newly predicted transaction" leaves a durable record
            # (see log_predictions) -- not just a number recomputed fresh
            # on every dashboard load and thrown away.
            try:
                candidates = transactions_since_last_bank_rfi(txns)
                scored = score_all_open(candidates, txns, _current_bundle)
                changed = log_predictions(scored)
                notify_new_high_risk_predictions(changed)
            except Exception:
                logger.exception("Failed to log predictions snapshot (retrain itself succeeded)")
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
    """Called on startup (and lazily by any endpoint that needs a model
    before the scheduler's first run completes). Tries the cached, already-
    trained bundle FIRST -- see load_model_state -- so the dashboard shows
    a working model within milliseconds of the process starting instead of
    "untrained" while a full retrain (or worse, a full re-backfill from
    Sumsub after an ephemeral-disk wipe) runs in the background. The normal
    retrain_only() path (webhook-driven or the safety-net scheduler) still
    runs exactly as before and will overwrite this with a fully current
    model the moment new data justifies it -- this only fills the gap
    between process start and that first retrain."""
    global _current_bundle
    if _current_bundle is None:
        _current_bundle = load_model_state()
        if _current_bundle is not None:
            return
        # No cached model to resume from -- if the database already has
        # data (first-ever boot against an existing DB, or the cache
        # didn't survive a restart), train immediately through the SAME
        # path retrain_only() uses rather than duplicating a partial
        # version of it here. That matters: it means this also records the
        # model version (so /api/model/history and the dashboard's "Model
        # mode" tile are correct immediately, not just _current_bundle in
        # memory), caches the result for the next restart, and logs/alerts
        # on today's already-open candidates instead of silently waiting
        # for the next webhook event or scheduled sweep to do that
        # bookkeeping.
        if all_transactions():
            retrain_only()


def _maybe_auto_backfill_on_empty_db():
    """Self-healing for the common "count went back to 0" symptom: on
    Render's Free/Starter plans (no persistent disk configured -- see
    _db_likely_ephemeral and the README's "Deploying on Render" section),
    every restart/redeploy wipes the SQLite file. That doesn't break
    ingestion -- the webhook keeps working fine -- but it silently resets
    total_bank_rfi to 0 and it STAYS 0 until either someone notices and
    clicks "Run full historical backfill" by hand, or enough brand-new
    webhook events happen to carry the bank-rfi tag on their own. Rather
    than requiring a person to notice and intervene after every deploy,
    kick off a fresh deep historical backfill automatically, once, whenever
    the transactions table comes up completely empty at startup and
    credentials are configured. Runs on a plain background thread (not
    BackgroundTasks, since there's no request/response cycle at startup to
    hang it off of) so it doesn't block the app from starting to serve
    traffic. Uses the SAME job/state (_backfill_history_job) as the manual
    "Run full historical backfill" button, so its progress is visible in
    exactly the same place."""
    global _startup_auto_backfill
    try:
        existing = len(all_transactions())
    except Exception:  # noqa: BLE001 -- DB not ready is not fatal, just skip
        logger.exception("Could not check transaction count for auto-backfill")
        return
    if existing != 0:
        _startup_auto_backfill.update(
            attempted=False, reason=f"{existing} transaction(s) already stored -- no auto-backfill needed", months_back=None,
        )
        return
    client = get_client()
    if client is None:
        _startup_auto_backfill.update(
            attempted=False, reason="Database is empty but Sumsub credentials aren't configured yet.", months_back=None,
        )
        return
    months_back = 60
    _backfill_history_job.update(running=True, months_done=0, months_total=months_back, result=None,
                                  started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    _startup_auto_backfill.update(
        attempted=True,
        reason="Database was empty at startup (most likely a redeploy on a non-persistent filesystem -- "
               "see db_likely_ephemeral) -- automatically running a fresh historical backfill now.",
        months_back=months_back,
    )
    logger.info("Transactions table empty at startup -- auto-starting a %d-month historical backfill.", months_back)
    threading.Thread(target=_run_backfill_history_job_in_background, args=(client, months_back), daemon=True).start()


def notify_startup_status() -> None:
    """One Slack message per process start -- a live, independent "yes,
    this is actually running" signal that doesn't require opening the
    Render dashboard or logs. No-op if Slack isn't configured. Deliberately
    called from inside on_startup's own try/except wrapper (see below), and
    is itself defensive about every value it reads, so a problem generating
    this message can never be the thing that takes the app down."""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL:
        logger.info("SLACK_BOT_TOKEN/SLACK_CHANNEL not set -- skipping startup Slack notification.")
        return
    try:
        txns = all_transactions()
    except Exception:  # noqa: BLE001
        txns = []
    mode = _current_bundle.mode if _current_bundle else "untrained (no model loaded yet)"
    creds_status = "configured" if (os.environ.get("SUMSUB_APP_TOKEN") and os.environ.get("SUMSUB_SECRET_KEY")) else "MISSING"
    webhook_sig = "on" if SUMSUB_WEBHOOK_SECRET_KEY else "off (SUMSUB_WEBHOOK_SECRET_KEY not set -- webhook accepts unverified POSTs)"
    text = (
        f":bank: *Bank RFI Prediction Bot started*\n"
        f"• Transactions in database: {len(txns)}\n"
        f"• Model mode: {mode}\n"
        f"• Sumsub credentials: {creds_status}\n"
        f"• Webhook signature verification: {webhook_sig}"
    )
    send_slack_message(text)


@app.on_event("startup")
def on_startup():
    """Every step here is independently wrapped in try/except: a startup
    handler that raises makes FastAPI/Uvicorn fail to start AT ALL --
    every single route, including the dashboard and /api/health -- so a
    latent bug in any ONE of these (a corrupt persisted model, a scheduler
    misconfiguration, a Sumsub network hiccup during the empty-db check)
    must never be allowed to take the whole application down with it. Each
    step logs and continues if it fails; the app coming up in a partially-
    degraded state (e.g. no cached model yet, but the webhook and dashboard
    both working) is always strictly better than the app not coming up at
    all."""
    try:
        _ensure_bundle_loaded()
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to load/train an initial model bundle at startup -- continuing without one; "
            "the next webhook event or scheduled retrain will populate it."
        )
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(run_ingest_and_retrain, "interval", minutes=INGEST_INTERVAL_MINUTES,
                           next_run_time=datetime.now(timezone.utc))
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info(
            "Safety-net backfill+retrain scheduled every %d minute(s) -- but the model actually "
            "retrains immediately on every webhook event, not on this timer.", INGEST_INTERVAL_MINUTES,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to start the background scheduler -- webhook-driven retraining still works; "
            "the hourly safety-net sweep will not run until the next restart."
        )
    try:
        _maybe_auto_backfill_on_empty_db()
    except Exception:  # noqa: BLE001
        logger.exception(
            "Auto-backfill-on-empty-db check failed at startup -- not fatal, use the dashboard's "
            "manual \"Run full historical backfill\" button instead."
        )
    try:
        notify_startup_status()
    except Exception:  # noqa: BLE001
        logger.exception("Startup Slack notification failed (non-fatal)")


def _db_likely_ephemeral() -> bool:
    """Best-effort heuristic, not a guarantee: the DB path is "likely
    ephemeral" if it lives under this app's own source checkout (the
    default, ./bank_rfi_bot.db) rather than a separately-mounted path like
    Render's /data disk. On Render's Free/Starter plans (see README), the
    app's own filesystem -- including a DB file left at the default path --
    is wiped on every restart/redeploy, which silently resets total_bank_rfi
    back to 0 even though nothing about the ingestion logic is broken. This
    is surfaced here because it is the single most common explanation for
    "the count went back to 0" after a code change ships."""
    try:
        resolved = os.path.abspath(DB_PATH)
        return resolved == os.path.abspath(os.path.join(HERE, "bank_rfi_bot.db")) or resolved.startswith(HERE + os.sep)
    except Exception:  # noqa: BLE001
        return True


@app.get("/api/setup")
def api_setup(request: Request):
    conn = check_sumsub_connection()
    txns = all_transactions()
    base = str(request.base_url).rstrip("/")
    tag_diagnostics = distinct_raw_tags(txns)
    review_status_diagnostics = distinct_review_statuses(txns)
    event_types = webhook_event_type_breakdown()
    real_event_types = [e for e in event_types if e["event_type"] not in ("invalid_signature", "invalid_json")]
    seen_real_types = {e["event_type"] for e in real_event_types}
    invalid_sig_count = sum(e["sig_failed_count"] for e in event_types)
    # A real, recognized Sumsub event type where EVERY occurrence failed
    # signature verification is the specific "wrong secret" signature --
    # distinct from a type simply never showing up at all (see
    # webhook_event_type_breakdown's docstring).
    types_fully_sig_failing = [e["event_type"] for e in real_event_types if e["count"] > 0 and e["sig_failed_count"] == e["count"]]
    rfi_count = len([t for t in txns if t["is_bank_rfi"]])
    n_confirmed_other = len([t for t in txns if not t["is_bank_rfi"] and t.get("review_status") not in OPEN_STATUSES])
    n_still_open = len(txns) - rfi_count - n_confirmed_other
    return {
        "credentials_configured": conn["status"] != "not_configured",
        "connection_status": conn["status"],
        "connection_detail": conn["detail"],
        "webhook_url": f"{base}/api/webhooks/sumsub",
        "webhook_events_received": webhook_event_count(),
        "webhook_event_types": real_event_types,
        "webhook_missing_txn_created": bool(real_event_types) and "applicantKytTxnCreated" not in seen_real_types,
        "webhook_signature_verification": "enabled" if SUMSUB_WEBHOOK_SECRET_KEY else "disabled_no_secret",
        "webhook_invalid_signature_count": invalid_sig_count,
        "webhook_types_fully_sig_failing": types_fully_sig_failing,
        "slack_configured": bool(SLACK_BOT_TOKEN and SLACK_CHANNEL),
        "slack_high_risk_threshold": SLACK_HIGH_RISK_THRESHOLD,
        "total_transactions": len(txns),
        "total_bank_rfi": rfi_count,
        "total_confirmed_other": n_confirmed_other,
        "total_still_open_pending": n_still_open,
        "bank_rfi_tag": BANK_RFI_TAG,
        "ingest_interval_minutes": INGEST_INTERVAL_MINUTES,
        "db_path": DB_PATH,
        "db_likely_ephemeral": _db_likely_ephemeral(),
        "distinct_tags_seen": tag_diagnostics,
        "distinct_review_statuses_seen": review_status_diagnostics,
        "auto_backfill": dict(_startup_auto_backfill),
        "recent_webhook_events": recent_webhook_events(15),
    }


@app.get("/api/health")
def api_health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "bank_rfi_tag": BANK_RFI_TAG}


@app.get("/api/summary")
def api_summary():
    """IMPORTANT: model_mode/metrics below are read from the LIVE in-memory
    `_current_bundle` (calling _ensure_bundle_loaded() first so it's
    populated whenever a model can exist at all), NOT from the
    `model_versions` DB table. An earlier version read only from
    `latest_model_version()`, which is only written by retrain_only() --
    but _ensure_bundle_loaded() (used on every process startup, and
    lazily by other endpoints) can populate a perfectly good, fully
    working `_current_bundle` via a restored cache or a fresh train
    WITHOUT ever calling retrain_only(). That mismatch meant the dashboard
    could show "Model mode: untrained" right after every restart -- for up
    to a full INGEST_INTERVAL_MINUTES until the next scheduled retrain --
    even though predictions were already being computed correctly from a
    working model the whole time (see api_predictions, which always used
    `_current_bundle` directly and was never actually broken). That
    "looks completely broken but isn't" gap is exactly what this fixes:
    the tile now always reflects what's actually being used to score."""
    _ensure_bundle_loaded()
    txns = all_transactions()
    rfi = [t for t in txns if t["is_bank_rfi"]]
    cutoff = most_recent_bank_rfi_timestamp(txns)
    metrics = _current_bundle.metrics if _current_bundle else {}
    return {
        "total_transactions": len(txns), "total_bank_rfi": len(rfi),
        "candidate_transactions": len(transactions_since_last_bank_rfi(txns)),
        "last_bank_rfi_at": cutoff.isoformat() if cutoff else None,
        "model_mode": _current_bundle.mode if _current_bundle else "untrained",
        "model_trained_at": _current_bundle.trained_at if _current_bundle else None,
        "model_cv_roc_auc": metrics.get("cv_roc_auc"),
        "model_cv_pr_auc": metrics.get("cv_pr_auc"),
        "model_temporal_holdout_roc_auc": metrics.get("temporal_holdout_roc_auc"),
        "model_temporal_holdout_pr_auc": metrics.get("temporal_holdout_pr_auc"),
        "model_temporal_holdout_note": metrics.get("temporal_holdout_note"),
        "model_positive_rate_pct": metrics.get("positive_rate_pct_in_training"),
        "model_n_confirmed_other": metrics.get("n_confirmed_other"),
        "model_n_excluded_open_pending": metrics.get("n_excluded_open_pending"),
        "model_heuristic_reason": metrics.get("reason"),
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
        # Timestamps for the dashboard's "time added" column: txn_created_at
        # is the transaction's own date (from Sumsub); ingested_at is when
        # THIS BOT first stored/last touched the row -- surfacing both
        # distinguishes "an old transaction we just backfilled" from "a
        # transaction that just happened", which matters for judging how
        # current a candidate list actually is.
        s["txn_created_at"] = t.get("txn_created_at")
        s["ingested_at"] = t.get("ingested_at")
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


@app.get("/api/predictions/log")
def api_predictions_log(limit: int = 50):
    """Every logged prediction snapshot across all transactions, most
    recent first -- see log_predictions (called from retrain_only). This is
    the durable record of "every newly predicted transaction," independent
    of the live /api/predictions computation."""
    return {"entries": recent_predictions_log(limit)}


@app.get("/api/predictions/history/{txn_id}")
def api_prediction_history(txn_id: str):
    """One transaction's risk score over time, across every retrain that
    produced a materially different score for it -- lets a reviewer answer
    "when did the model start flagging this" instead of only "what does it
    say right now." """
    history = prediction_history(txn_id)
    if not history:
        return JSONResponse({"error": "No logged predictions for this transaction yet."}, status_code=404)
    return {"txn_id": txn_id, "history": history}


@app.get("/api/webhooks/recent")
def api_webhooks_recent(limit: int = 25):
    """Per-event detail for the last N webhook requests -- event type,
    whether ingestion succeeded and why not if it didn't, and the signature
    verification detail. The primary diagnostic for "the webhook isn't
    updating": distinguishes Sumsub not sending an event type at all from
    Sumsub sending it but the bot rejecting/failing to process it."""
    return {"events": recent_webhook_events(limit)}


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


def _run_backfill_history_job_in_background(client: SumsubClient, months_back: int, exhaustive: bool = False):
    def on_progress(done, total):
        _backfill_history_job["months_done"] = done
        _backfill_history_job["months_total"] = total
    try:
        # exhaustive=True sets max_empty_months to months_back itself, i.e.
        # the empty-streak early-stop can never trip -- every month in the
        # range gets scanned regardless of how many consecutive empty
        # months precede it. Slower (and issues more Sumsub API calls) but
        # guarantees "every past transaction in range", which is what the
        # early-stop heuristic explicitly trades away for speed by default.
        max_empty = months_back if exhaustive else 18
        result = run_deep_historical_backfill(client, months_back=months_back, max_empty_months=max_empty, on_progress=on_progress)
        retrain_only()
        _backfill_history_job["result"] = result
    except Exception as e:  # noqa: BLE001 - must not wedge the job state
        logger.exception("Deep historical backfill job failed")
        _backfill_history_job["result"] = {"error": str(e)}
    finally:
        _backfill_history_job["running"] = False
        _backfill_history_job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/ingest/backfill-history")
def api_backfill_history(background_tasks: BackgroundTasks, months_back: int = 60, exhaustive: bool = False):
    """Trigger a ONE-TIME deep historical crawl covering every transaction
    type (see run_deep_historical_backfill) -- not part of the automatic
    hourly cycle, which only checks a short recent window to stay fast.
    Runs in the background; poll GET /api/ingest/backfill-history-status.

    By default this stops early after 18 consecutive months with zero
    transactions found (a proxy for "before the account existed"), to avoid
    burning through months of API calls for an account that's newer than
    `months_back`. Pass `exhaustive=true` to disable that early-stop and
    scan every single month in range regardless of empty streaks -- slower,
    but guarantees literally every past transaction in the window gets a
    chance to be found, for accounts with genuine multi-year quiet gaps."""
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
    background_tasks.add_task(_run_backfill_history_job_in_background, client, months_back, exhaustive)
    return {"status": "started", "months_back": months_back, "exhaustive": exhaustive}


@app.get("/api/ingest/backfill-history-status")
def api_backfill_history_status():
    """Poll this while a deep historical backfill (started via POST
    .../backfill-history) is running."""
    return {**_backfill_history_job, "summary": api_summary() if not _backfill_history_job["running"] else None}


def _process_webhook_event_in_background(event: dict, event_row_id: int):
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
        mark_webhook_event_processed(event_row_id, ok=False, detail="Sumsub credentials not configured")
        return
    try:
        row, detail = process_webhook_event(client, event)
        mark_webhook_event_processed(event_row_id, ok=row is not None, detail=detail)
        if row:
            # Retrain on EVERY real transaction event -- created, reviewed,
            # approved, rejected, or deleted -- so the model, the candidate
            # pool, and the pattern stats are current within seconds of
            # Sumsub sending this event. This is the bot's actual
            # "constantly running" mechanism; the hourly job is only a
            # safety net (see run_ingest_and_retrain's docstring).
            if row.get("deleted"):
                tag_note = "transaction deleted on Sumsub"
            elif row.get("is_bank_rfi"):
                tag_note = "new bank-rfi transaction" if row.get("_newly_bank_rfi") else "bank-rfi transaction updated"
            else:
                tag_note = "transaction update"
            logger.info("Webhook event processed (%s): %s -- retraining now", tag_note, row["txn_id"])
            if row.get("_newly_bank_rfi"):
                try:
                    notify_new_bank_rfi_tag(row)
                except Exception:  # noqa: BLE001 -- a Slack hiccup must never break webhook processing
                    logger.exception("Failed to send new-bank-rfi Slack notification (non-fatal)")
            retrain_only()
        else:
            logger.info("Webhook event not ingested: %s", detail)
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to process webhook event: %s", event)
        mark_webhook_event_processed(event_row_id, ok=False, detail=f"unhandled exception: {e}")


@app.post("/api/webhooks/sumsub")
async def api_sumsub_webhook(request: Request, background_tasks: BackgroundTasks):
    """Register this URL in Sumsub Dashboard -> Settings -> Webhooks for
    Transaction Monitoring events. IMPORTANT: Sumsub only sends the event
    types you explicitly enable when registering the webhook -- there isn't
    one single "any transaction change" event. In particular, the event
    that fires the moment a NEW transaction is first created is
    `applicantKytTxnCreated`, which is a DIFFERENT checkbox from the
    review-outcome events (approved/rejected/reviewed/on-hold). If that
    checkbox isn't enabled on Sumsub's side, brand-new transactions won't
    reach this endpoint in real time no matter what this code does -- check
    the Setup panel's event-type breakdown (GET /api/setup ->
    webhook_event_types) to see exactly which event types have actually
    arrived, and see docs.sumsub.com/docs/transaction-monitoring-webhooks
    for the full list to enable.

    Responds immediately (see _process_webhook_event_in_background for why
    that matters) -- the actual fetch-and-retrain work happens after the
    response is sent.

    Signature verification (if SUMSUB_WEBHOOK_SECRET_KEY is set -- see that
    variable's definition for how to get this key from Sumsub) happens
    BEFORE any of that: it's a cheap local HMAC computation, so it doesn't
    hurt Sumsub's response-time budget, and it means a forged/unauthenticated
    POST to this public URL is rejected outright instead of being treated as
    a real transaction event."""
    raw_body = await request.body()
    is_valid, sig_detail = verify_webhook_signature(
        raw_body, request.headers.get("x-payload-digest"), request.headers.get("x-payload-digest-alg"),
    )
    # Best-effort parse even on a failed signature check -- a MISCONFIGURED
    # secret (wrong value in Render vs. what's currently shown in Sumsub's
    # Webhook manager) still sends a real, well-formed Sumsub payload that
    # just fails the HMAC compare; recording its real event type/txn id
    # (instead of collapsing everything into one "invalid_signature" bucket
    # with no other detail) makes that failure mode distinguishable from an
    # actual forged/garbage request in the recent-events diagnostic.
    try:
        event = json.loads(raw_body)
    except ValueError:
        event = {}
    event_type = event.get("type") or ("unknown" if is_valid else "invalid_signature")
    txn_id_guess = _webhook_txn_id(event) or "unknown"

    if not is_valid:
        logger.warning("Rejected webhook POST with invalid signature: %s", sig_detail)
        record_webhook_event(event_type, txn_id_guess, event or {"raw_body_preview": raw_body[:500].decode("utf-8", "replace")},
                              signature_detail=sig_detail)
        return JSONResponse({"error": "invalid signature", "detail": sig_detail}, status_code=401)

    if not event:
        logger.warning("Webhook POST body wasn't valid JSON (signature check: %s)", sig_detail)
        record_webhook_event("invalid_json", "unknown", {"raw_body_preview": raw_body[:500].decode("utf-8", "replace")},
                              signature_detail=sig_detail)
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    # Record receipt unconditionally (even without credentials or a
    # recognizable txn id) so the Setup panel can confirm "yes, Sumsub is
    # reaching this URL" independently of whether ingestion succeeds. This
    # is a single fast local DB write, not an API call, so it's safe to do
    # before responding.
    row_id = record_webhook_event(event_type, txn_id_guess, event, signature_detail=sig_detail)
    background_tasks.add_task(_process_webhook_event_in_background, event, row_id)
    return {"received": True, "signature": sig_detail}


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
            <th data-key="txn_created_at">Txn date</th>
            <th data-key="ingested_at">Added to bot</th>
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
        <div id="rawTagsDiag" style="margin-top:16px;"></div>
        <div id="reviewStatusDiag" style="margin-top:16px;"></div>
      </div>
    </details>

    <details class="section-collapse">
      <summary>Recent webhook activity</summary>
      <div class="details-body">
        <p class="desc">The last 15 requests to the webhook endpoint, in detail -- the primary place to look when "the webhook isn't updating live." Shows exactly what type each event was, whether it passed signature verification, and whether ingestion succeeded.</p>
        <div id="recentWebhooksDiag"></div>
      </div>
    </details>

    <details class="section-collapse">
      <summary>Model &amp; ingestion history</summary>
      <div class="details-body">
        <p class="desc">Every retrain is logged for audit purposes &mdash; this is a compliance tool, so nothing here is a black box.</p>
        <table id="historyTable">
          <thead><tr><th>Trained at</th><th>Mode</th><th>Bank-rfi examples</th><th>Confirmed-other examples</th><th>Excluded (open/pending)</th><th>CV ROC-AUC</th><th>CV PR-AUC</th><th>Temporal holdout AUC</th></tr></thead>
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
  const eventTypePills = (s.webhook_event_types||[]).map(e => `<span class="pill">${e.event_type} &times;${e.count} (${e.processed_count} ingested)</span>`).join(" ");
  rows.push({
    icon: webhookRegistered ? "good" : "warn",
    text: (webhookRegistered
      ? `<strong>Webhook is live</strong> -- ${s.webhook_events_received.toLocaleString()} event(s) received so far. This is the mechanism that keeps ingesting new data straight from Sumsub with no separate data files.${eventTypePills ? `<br/><span style="color:var(--text-secondary);">Event types seen (last 14 days):</span> ${eventTypePills}` : ""}`
      : `<strong>No webhook events received yet.</strong> This checks whether Sumsub has actually sent anything here -- it can't check Sumsub's own settings (Sumsub doesn't expose an API for that), so if you've <em>already</em> registered the URL below, this just means no matching event has fired yet, not that registration failed. If you just set it up, try clicking "Test webhook" in Sumsub's Webhook manager, or check Sumsub's own <strong>Webhook logs</strong> page to see delivery attempts and their status directly. If you haven't registered it yet, add this URL in Sumsub Dashboard &rarr; Webhook manager for Transaction Monitoring events, making sure to enable EVERY event type you want tracked (Txn created, approved, rejected, reviewed, on hold, awaiting user, deleted) -- Sumsub only sends the ones you check:`)
      + `<br/><code class="copyline"><span>${s.webhook_url}</span><button onclick="copyToClipboard('${s.webhook_url}', this)">Copy</button></code>`
  });
  if (s.webhook_signature_verification === "enabled") {
    rows.push({icon: "good", text: `<strong>Webhook signature verification is on.</strong> Every incoming POST to the webhook URL is checked against SUMSUB_WEBHOOK_SECRET_KEY before being trusted -- forged/unauthenticated requests are rejected with HTTP 401 and never touch your data.${s.webhook_invalid_signature_count ? ` <strong style="color:var(--status-critical);">${s.webhook_invalid_signature_count} invalid-signature attempt(s) blocked so far</strong> -- check Sumsub's Webhook logs to confirm this matches Sumsub's own delivery attempts (mismatches here usually mean the secret in Render doesn't match the one currently shown in Sumsub's Webhook manager).` : ""}`});
  } else {
    rows.push({icon: "warn", text: `<strong>Webhook signature verification is off.</strong> Any POST to the webhook URL below is currently accepted at face value -- there's no way to tell a real Sumsub event from anyone else who finds this URL. Copy the secret key shown in Sumsub Dashboard &rarr; Webhook manager (when you create or edit the webhook) into Render's Environment tab as <code>SUMSUB_WEBHOOK_SECRET_KEY</code> and redeploy -- never paste it into chat or commit it to code, same as the API credentials. Sumsub signs each request with <code>x-payload-digest</code> / <code>X-Payload-Digest-Alg</code> headers; this bot verifies them the moment that variable is set, no other change needed.`});
  }
  if (s.slack_configured) {
    rows.push({icon: "good", text: `<strong>Slack notifications are on.</strong> A message is posted on every process start, on every new confirmed "${s.bank_rfi_tag}" transaction, and whenever a candidate newly crosses ${(s.slack_high_risk_threshold*100).toFixed(0)}% predicted risk (set via SLACK_HIGH_RISK_THRESHOLD).`});
  }
  if (s.auto_backfill && s.auto_backfill.attempted) {
    rows.push({icon: "warn", text: `<strong>Auto-recovering from an empty database.</strong> ${s.auto_backfill.reason} This can take a few minutes -- refresh this page to see progress, or check the "Run full historical backfill" panel below.`});
  }
  if (s.webhook_missing_txn_created) {
    rows.push({icon: "warn", text: `<strong>Webhook events are arriving, but never "applicantKytTxnCreated"</strong> -- the specific event Sumsub sends the moment a brand-new transaction is first created. Sumsub only sends event types you've explicitly enabled in <strong>Dashboard &rarr; Webhook manager</strong>; without this one, new transactions won't show up here in real time until some later event (review/approval/rejection) happens to fire instead. Edit the webhook in Sumsub's dashboard and make sure the "Txn created" event type is checked. Event types actually seen so far: ${(s.webhook_event_types||[]).map(e => `<span class="pill">${e.event_type} &times;${e.count}</span>`).join(" ") || "none"}.`});
  }
  if ((s.webhook_types_fully_sig_failing||[]).length) {
    rows.push({icon: "crit", text: `<strong>Sumsub IS sending ${s.webhook_types_fully_sig_failing.join(", ")}, but every single one is failing signature verification.</strong> This means the webhook connection itself works, but <code>SUMSUB_WEBHOOK_SECRET_KEY</code> in Render doesn't match the secret currently shown in Sumsub Dashboard &rarr; Webhook manager for this webhook -- a config mismatch, not a delivery problem. Copy the secret shown there again (it may have been regenerated) into Render and redeploy. Check "Recent webhook activity" below for the exact per-event detail.`});
  }
  if (s.total_still_open_pending > 0 && s.total_bank_rfi > 0 && s.total_confirmed_other < 3) {
    rows.push({icon: "warn", text: `<strong>${s.total_still_open_pending.toLocaleString()} transaction(s) are still open/pending review, and only ${s.total_confirmed_other} have a CONCLUDED review without the "${s.bank_rfi_tag}" tag.</strong> Risk scores will sit near 0% for every candidate until more transactions finish review (approved/rejected/reviewed on Sumsub) -- the model needs confirmed "not risky" examples to compare against, not just confirmed "risky" ones, and an open/pending transaction's real outcome isn't known yet so it can't be used as either. This resolves itself automatically as more reviews conclude; no configuration issue here.`});
  }
  if (s.db_likely_ephemeral && s.total_transactions > 0) {
    rows.push({icon: "warn", text: `<strong>No persistent disk detected</strong> (db path: <code>${s.db_path}</code>). On Render's Free/Starter plans this file resets on every restart/redeploy, so the counts below can drop back to 0 after a deploy even though nothing is broken -- the bot just auto-refills via a historical backfill on the next startup (see above) or as new webhook events arrive. Add a Render Disk mounted at <code>/data</code> with <code>BOT_DB_PATH=/data/bank_rfi_bot.db</code> to stop this from happening -- see the README's "Deploying on Render" section.`});
  }
  if (s.total_transactions === 0) {
    rows.push({icon: "warn", text: `<strong>No data yet.</strong> All data comes straight from the Sumsub API -- run "Run full historical backfill" below to pull your account's full transaction history, wait for new events to arrive via the webhook above, or use "Import transactions by ID" for a specific list.`});
  } else if (s.total_bank_rfi === 0) {
    rows.push({icon: "warn", text: `<strong>${s.total_transactions.toLocaleString()} transaction(s) ingested, but none tagged "${s.bank_rfi_tag}" yet.</strong> If you know some already carry that tag in Sumsub, use "Import transactions by ID" below -- filter by "${s.bank_rfi_tag}" in the Sumsub Dashboard, copy the IDs it shows, and paste them there. Sumsub's API has no way to list transactions by tag (or at all) on its own, so this is the fastest way in. Check "What distinguishes bank-rfi transactions" further down the page -- it lists every raw tag label actually seen so far, so you can immediately spot if Sumsub's real tag spelling differs from "${s.bank_rfi_tag}".`});
  } else {
    rows.push({icon: "good", text: `<strong>${s.total_transactions.toLocaleString()} transaction(s) ingested</strong>, ${s.total_bank_rfi.toLocaleString()} tagged "${s.bank_rfi_tag}" so far -- all pulled from the Sumsub API. The model retrains immediately every time a webhook event comes in (not on a timer); a safety-net sweep also runs every ${s.ingest_interval_minutes} minute(s) in case any webhook delivery was missed.`});
  }
  const allGood = s.connection_status === "ok" && s.total_transactions > 0;
  banner.style.display = "block";
  banner.innerHTML = `<h2>${allGood ? "Status" : "Setup"}</h2>${rows.map(r => `<div class="setup-row"><span class="setup-icon ${r.icon}">${r.icon === "good" ? "&#10003;" : "!"}</span><span>${r.text}</span></div>`).join("")}`;
  renderRawTagsDiag(s.distinct_tags_seen || [], s.bank_rfi_tag);
  renderReviewStatusDiag(s.distinct_review_statuses_seen || []);
  renderRecentWebhooks(s.recent_webhook_events || []);
}

function renderRawTagsDiag(tags, expectedTag) {
  const el = document.getElementById("rawTagsDiag");
  if (!el) return;
  if (!tags.length) {
    el.innerHTML = '<p class="empty">No tags seen on any ingested transaction yet.</p>';
    return;
  }
  el.innerHTML = `
    <p class="desc" style="margin-bottom:6px;"><strong>Every raw tag label seen so far</strong> (diagnostic -- compare against the expected tag "${expectedTag}" below; a mismatched spelling/separator here is the fastest way to spot why a transaction you know is tagged isn't counted as bank-rfi).</p>
    <div>${tags.map(t => `<span class="pill" style="${t.matches_bank_rfi ? 'border-color:var(--status-good);color:var(--status-good);' : ''}">${t.tag} &times;${t.count}${t.matches_bank_rfi ? ' (matches)' : ''}</span>`).join(" ")}</div>
  `;
}

function renderReviewStatusDiag(statuses) {
  const el = document.getElementById("reviewStatusDiag");
  if (!el) return;
  if (!statuses.length) {
    el.innerHTML = '<p class="empty">No transactions ingested yet.</p>';
    return;
  }
  el.innerHTML = `
    <p class="desc" style="margin-bottom:6px;"><strong>Every review_status value actually seen so far</strong> (diagnostic -- "open" statuses are excluded from training as negative examples until they conclude; see "How the model works" in the README). "(none/missing)" here for every transaction usually means this field isn't being read correctly for your account's data shape -- worth reporting if so.</p>
    <div>${statuses.map(s => `<span class="pill" style="${s.counts_as_open ? '' : 'border-color:var(--status-good);color:var(--status-good);'}">${s.review_status} &times;${s.count} (${s.counts_as_open ? "open -- excluded" : "resolved -- counted"})</span>`).join(" ")}</div>
  `;
}

function renderRecentWebhooks(events) {
  const el = document.getElementById("recentWebhooksDiag");
  if (!el) return;
  if (!events.length) {
    el.innerHTML = '<p class="empty">No webhook requests received yet.</p>';
    return;
  }
  el.innerHTML = `
    <table id="recentWebhooksTable">
      <thead><tr><th>Received</th><th>Event type</th><th>Txn ID</th><th>Ingested?</th><th>Signature</th><th>Detail</th></tr></thead>
      <tbody>${events.map(e => `
        <tr>
          <td>${(e.received_at||"").replace("T"," ").slice(0,19)}</td>
          <td>${e.event_type}</td>
          <td>${e.kyt_txn_id}</td>
          <td>${e.processed ? "&#10003;" : "&#10005;"}</td>
          <td style="max-width:220px;">${e.signature_detail || "–"}</td>
          <td style="max-width:260px;">${e.process_detail || "–"}</td>
        </tr>`).join("")}</tbody>
    </table>
  `;
}

function renderTiles(s) {
  const aucVal = (typeof s.model_cv_roc_auc === "number") ? s.model_cv_roc_auc.toFixed(3) : "–";
  const prVal = (typeof s.model_cv_pr_auc === "number") ? s.model_cv_pr_auc.toFixed(3) : "–";
  const holdoutVal = (typeof s.model_temporal_holdout_roc_auc === "number") ? s.model_temporal_holdout_roc_auc.toFixed(3) : null;
  const modeSub = s.model_mode === "ml"
    ? (typeof s.model_cv_roc_auc === "number"
        ? `CV ROC-AUC ${aucVal} · PR-AUC ${prVal}` + (holdoutVal ? ` · temporal holdout AUC ${holdoutVal}` : "") + ` (${s.model_positive_rate_pct != null ? s.model_positive_rate_pct + "% positive" : "rare-event"})`
        : "trained " + (s.model_trained_at||""))
    : (s.model_heuristic_reason || (s.model_trained_at ? ("trained " + s.model_trained_at) : "not trained yet"));
  const tiles = [
    {label: "Transactions scanned", value: fmt(s.total_transactions)},
    {label: "Tagged “bank rfi”", value: fmt(s.total_bank_rfi)},
    {label: "Candidates since last RFI", value: fmt(s.candidate_transactions),
     sub: s.last_bank_rfi_at ? ("since " + s.last_bank_rfi_at.slice(0, 16).replace("T", " ")) : "no bank-rfi tag seen yet"},
    {label: "Model mode", value: s.model_mode || "untrained", sub: modeSub},
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
      <td><span class="risk-badge ${riskClass(r.risk_score, r._riskRank, rows.length)}">${(r.risk_score*100).toFixed(1)}%</span></td>
      <td>${r.txn_id}</td>
      <td>${fmt(r.amount)} ${r.currency||""}</td>
      <td>${r.counterparty_country || "–"}</td>
      <td>${(r.txn_created_at||"").replace("T"," ").slice(0,16) || "–"}</td>
      <td>${(r.ingested_at||"").replace("T"," ").slice(0,16) || "–"}</td>
      <td>${(r.reasons && r.reasons[0]) || "–"}</td>
    </tr>
    <tr class="expand-row" id="expand-${i}" style="display:none;"><td colspan="7">
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
  body.innerHTML = versions.length ? versions.map(v => {
    const m = v.metrics_json || {};
    const auc = (typeof m.cv_roc_auc === "number") ? m.cv_roc_auc.toFixed(3) : (v.mode === "ml" ? "–" : "n/a (heuristic)");
    const prAuc = (typeof m.cv_pr_auc === "number") ? m.cv_pr_auc.toFixed(3) : (v.mode === "ml" ? "–" : "n/a (heuristic)");
    const holdout = (typeof m.temporal_holdout_roc_auc === "number")
      ? m.temporal_holdout_roc_auc.toFixed(3) + ` (n=${fmt(m.temporal_holdout_n_test)})`
      : (m.temporal_holdout_note ? "not enough data yet" : "–");
    return `<tr><td>${v.trained_at}</td><td>${v.mode}</td><td>${v.n_bank_rfi}</td><td>${v.n_other}</td><td>${fmt(m.n_excluded_open_pending)}</td><td>${auc}</td><td>${prAuc}</td><td title="${(m.temporal_holdout_note||"").replace(/"/g,'&quot;')}">${holdout}</td></tr>`;
  }).join("") : '<tr><td colspan="8" class="empty">No trained model versions yet.</td></tr>';
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
      const errNote = r.errored_months > 0
        ? ` (${r.errored_months} month(s) errored even after a retry -- see server logs for details; run it again to fill those in)`
        : "";
      statusEl.textContent = `Done -- walked ${r.months_walked} month(s), found ${r.scanned} transaction(s) (${r.new_bank_rfi} tagged "bank rfi")${errNote}.`;
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
