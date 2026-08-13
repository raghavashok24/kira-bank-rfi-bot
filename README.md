# Bank RFI Prediction Bot

Predicts which Sumsub transactions are likely to receive a **bank RFI** (a bank's
after-the-fact Request For Information), before the bank actually sends one.
Ground truth is the `bank rfi` tag applied to transactions in Sumsub; the model
learns the behavioral patterns behind past tagged transactions and scores new
incoming transactions in real time.

Everything is one file (`server.py`): the Sumsub API client, SQLite persistence,
feature engineering, the model, ingestion, the FastAPI routes, and the embedded
dashboard.

## Where this Bot Runs

It is designed for a machine with a **persistent filesystem** — not an
ephemeral-disk PaaS. The SQLite database (`bank_rfi_bot.db`, created next to
`server.py`) is the durable store for both the transaction history and the
trained model, so:

- **Pretraining happens exactly once.** On the first run against an empty
  database, the bot exhaustively crawls **every past transaction in company
  history** (month by month, back to `FULL_HISTORY_MONTHS`, no early-stop),
  then trains the model once on all of it.
- **Restarts resume instantly** from the persisted model. No re-crawl, no
  retrain-on-boot. (The old Render free-tier deployment wiped the database on
  every restart, which forced a full re-backfill and retrain each time — that
  whole failure mode is gone by running locally.)

## Quick start

```bash
cp .env.example .env    # fill in SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY (+ Slack, webhook secret)
pip install -r requirements.txt
python server.py        # or: uvicorn server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> — the dashboard shows a live setup checklist,
pretraining progress, current predictions, pattern findings, and model history.

First run: expect the full-history backfill to take a while (it pages through
every month of transactions one Sumsub API call per transaction, with
rate-limit-friendly throttling). Progress is visible on the dashboard and at
`GET /api/ingest/backfill-history-status`.

## How new transactions get in

1. **Polling sweep (primary when running locally):** every
   `INGEST_INTERVAL_MINUTES` (default 60) the bot queries Sumsub for the last
   couple of days of transactions and adds anything new to the database.
2. **Sumsub webhook (optional, real-time):** register
   `<your-url>/api/webhooks/sumsub` in Sumsub Dashboard → Webhook manager for
   Transaction Monitoring events (make sure `applicantKytTxnCreated` is
   enabled). A locally-run bot needs a publicly reachable URL for this — e.g.
   an `ngrok`/`cloudflared` tunnel. Set `SUMSUB_WEBHOOK_SECRET_KEY` so
   incoming POSTs are HMAC-verified.

Either way, an arriving transaction is **added to the database and scored
against the current model** — it does *not* trigger a retrain.

## When the model retrains

Only when its actual training signal changes — i.e. when the set of labeled
examples changes:

- a transaction newly gains the `bank rfi` tag, or
- a previously open/pending review concludes (giving a trustworthy negative), or
- a labeled transaction is deleted on Sumsub.

This is checked cheaply (a hash over labeled transaction IDs + labels) after
every ingest; if nothing changed, nothing retrains. When a retrain does run, it
always uses the **complete transaction history** in the database.

## Slack notifications

Exactly **one** kind of message is ever posted: a **newly-ingested incoming
transaction whose predicted bank-rfi risk is at/above
`SLACK_HIGH_RISK_THRESHOLD`** (default 0.75). No startup messages, no
"tag confirmed" messages, no per-retrain digests.

Guards against spam: alerts only fire for transactions genuinely new to the
database, only when the transaction's own date is within
`RECENT_ALERT_WINDOW_DAYS` (default 7), and never while the one-time
pretraining backfill is still running — so a historical import can never flood
the channel.

Setup: a Slack app with the `chat:write` scope installed to the workspace, its
bot token in `SLACK_BOT_TOKEN`, and the target channel (which the bot has been
invited to) in `SLACK_CHANNEL`. Slack is entirely optional — leave both unset
and everything else works the same.

## The model (rare-event design)

`bank rfi` is a genuinely rare event — on the order of ~40 positives against
the company's full transaction history. The model is built around that:

- **Shallow gradient-boosted trees** (`HistGradientBoostingClassifier`,
  depth ≤ 3, `class_weight="balanced"`) — chosen over logistic regression on
  measured out-of-fold results against real data (~3x the PR-AUC), because
  RFI risk lives in feature *interactions* (large AND outbound AND
  thin-history), which a linear model cannot express. Depth stays capped so
  ~40 positives can't be memorized.
- **Scores are relative profile-match scores, not calibrated frequencies** —
  deliberate: with a ~0.6% base rate, calibrated probabilities compress every
  score into 0–2% (even confirmed RFI cases), making dashboards unreadable
  and alert thresholds meaningless. A score near 1 means "strongly matches
  the historical RFI profile".
- **A logistic regression rides along purely for explainability** (its
  regularization picked per-retrain by stratified-CV PR-AUC) — its
  coefficients power the feature-importance panel; it never produces the
  served score.
- **One-hot vocabularies capped** (`MAX_CATEGORICAL_VOCAB`) so dozens of
  country/currency values can't blow up dimensionality past what ~40
  positives support.
- **Honest evaluation**: ROC-AUC *and* PR-AUC via stratified cross-validation,
  plus a **temporal holdout** (train on the oldest 80%, test on the newest
  20%) to detect trend drift. All metrics are logged per model version and
  shown on the dashboard.
- **Trustworthy negatives only**: transactions still open/pending review are
  never used as "not bank-rfi" training examples — only concluded reviews
  count. Open transactions are still scored.
- Below `MIN_ML_SAMPLES` (15) positives+negatives, the bot serves a
  transparent heuristic based on statistically significant pattern findings
  instead of an undertrained classifier.

### Anti-leakage guarantee

The `bank rfi` tag is the **label only — never a feature**. A new incoming
transaction obviously doesn't have the tag yet, so nothing derived from tags,
reviewer notes, or review status may ever be a model input. The feature lists
(`NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in `server.py`) contain only
behavioral signals (amount, timing, velocity, counterparty risk, structuring
indicators), and a startup assertion refuses to boot if a feature name that
looks tag/note/review-derived is ever added. An earlier version leaked
`applicant_prior_bank_rfi_count`-style features into training; that is fixed
and guarded against regression.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SUMSUB_APP_TOKEN` | yes | — | Sumsub API app token |
| `SUMSUB_SECRET_KEY` | yes | — | Sumsub API secret key |
| `SUMSUB_WEBHOOK_SECRET_KEY` | recommended | off | HMAC-verifies incoming webhook POSTs |
| `SLACK_BOT_TOKEN` | optional | off | Slack bot token (`xoxb-…`, `chat:write` scope) |
| `SLACK_CHANNEL` | optional | off | Channel ID/name for at-risk alerts |
| `SLACK_HIGH_RISK_THRESHOLD` | optional | `0.75` | Risk score that triggers the alert |
| `INGEST_INTERVAL_MINUTES` | optional | `60` | Polling sweep cadence |
| `FULL_HISTORY_MONTHS` | optional | `120` | Reach of the one-time pretraining crawl |
| `RECENT_ALERT_WINDOW_DAYS` | optional | `7` | Max transaction age eligible for an alert |
| `BOT_DB_PATH` | optional | `./bank_rfi_bot.db` | SQLite location |
| `BANK_RFI_TAG` | optional | `bank rfi` | Tag treated as ground truth (matching is case/separator-insensitive) |

## Useful endpoints

- `GET /` — dashboard
- `GET /api/setup` — full setup/diagnostic state (credentials, webhook events, tag spellings seen, review statuses seen)
- `GET /api/summary` — counts, model mode, CV/holdout metrics
- `GET /api/predictions` — current risk scores for candidate transactions, with reasons and nearest past bank-rfi examples
- `GET /api/model/history` — every model version with its metrics
- `GET /api/predictions/history/{txn_id}` — how one transaction's score evolved
- `POST /api/ingest/run` — manual sweep + label-change check
- `POST /api/ingest/backfill-history?months_back=120&exhaustive=true` — manual full-history crawl
- `POST /api/ingest/import-ids` — paste specific transaction IDs (Sumsub internal or your external IDs)
- `POST /api/webhooks/sumsub` — the webhook receiver

## Data model

Everything lives in one SQLite file: `transactions` (the ever-growing history,
one row per Sumsub transaction with tags/notes/raw payload), `model_state`
(the pickled trained model, so restarts resume instantly), `model_versions`
(training audit log), `predictions_log` (score audit trail per transaction),
`webhook_events` (webhook delivery diagnostics), and `ingestion_runs`.
Back it up by copying the `.db` file.
