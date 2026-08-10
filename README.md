# Bank RFI Prediction Bot

A single self-contained backend that watches your Sumsub account, learns
what's statistically different about transactions that get tagged **"bank
rfi"**, and scores every other open transaction by how likely it is to get
that tag next — in real time, as new transactions arrive, with a
plain-English justification and links to the most similar past "bank rfi"
transactions for each score.

It is an early-warning system, not a report you run occasionally: the model
retrains the instant a webhook event comes in from Sumsub, so the risk
picture stays current within seconds of something changing on your account,
not on a schedule.

## Contents

- [What it actually does](#what-it-actually-does)
- [Repo layout](#repo-layout)
- [Quick start](#quick-start)
- [Credentials](#credentials)
- [How data gets in](#how-data-gets-in)
- [How the model works](#how-the-model-works)
- [The dashboard](#the-dashboard)
- [Deploying on Render](#deploying-on-render)
- [Environment variables reference](#environment-variables-reference)
- [Running it locally](#running-it-locally)
- [Trying it before real Sumsub data is connected](#trying-it-before-real-sumsub-data-is-connected)
- [Securing the webhook](#securing-the-webhook)
- [Security notes](#security-notes)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## What it actually does

Every transaction the bot ever learns from was fetched from the real Sumsub
API — there is no seed file, no manual list, and no synthetic data in the
real code path (the one exception, `dev_seed_synthetic_data.py`, is an
explicitly separate opt-in tool for trying the dashboard before Sumsub is
connected, covered below).

Three things happen continuously:

1. **Real-time ingestion.** A Sumsub webhook fires on every transaction
   lifecycle event (created, approved, rejected, reviewed, put on hold,
   awaiting user action, deleted). The bot fetches that transaction's
   current tags, stores it, and retrains on the spot — usually within a
   second or two of the event arriving.
2. **Pattern analysis.** Every retrain re-runs a plain statistical
   comparison (Mann-Whitney U for numeric features, rate/lift comparison for
   categorical ones) between confirmed "bank rfi" transactions and confirmed
   clean ones, and writes out a narrative in plain English — e.g.
   `counterparty_country="BY" appears 32.5x more often in bank-rfi
   transactions than overall`. This is always computed, independent of
   whether the ML model is active yet, so there's always something
   explainable on the dashboard even with very little data.
3. **Prediction.** Every open (not-yet-resolved) transaction that arrived
   after the most recent "bank rfi" tag is re-scored on every retrain: a
   0–1 risk score, the specific reasons it matched, and the 3 most similar
   real past "bank rfi" transactions by feature similarity, each with its
   Sumsub transaction ID so a reviewer can pull up the actual case.

## Repo layout

```
server.py                    Everything -- API client, DB, model, ingestion, routes, scheduler, dashboard
requirements.txt             Python dependencies
render.yaml                  Render config (optional -- see "Deploying on Render")
.env.example                 Copy to .env for local runs
.gitignore
dev_seed_synthetic_data.py   Optional: seeds fake data so you can try the dashboard before Sumsub is connected
```

That's the whole repo. Nothing lives in a subfolder, there's no Dockerfile,
no separate worker process, and no models/ directory — the trained model
lives in memory and is rebuilt from the database on every retrain, so
there's nothing on disk to go stale or need cleanup.

## Quick start

If your Render service already has `SUMSUB_APP_TOKEN` and
`SUMSUB_SECRET_KEY` set (the usual case — see
[Credentials](#credentials)), getting a working deployment is three steps:

1. Push this repo's files to GitHub, flat at the repo root.
2. On Render: build command `pip install -r requirements.txt`, start
   command `uvicorn server:app --host 0.0.0.0 --port $PORT`.
3. Once deployed, register `https://<your-render-url>/api/webhooks/sumsub`
   in **Sumsub Dashboard → Webhook manager**, enabling every event type you
   want tracked — see [How data gets in](#how-data-gets-in) for exactly
   which ones matter and why.

Then open the deployed URL. The **Setup** panel on the dashboard tells you,
in plain language and backed by live checks (not just "should be working"),
exactly what's configured, what's missing, and what to do next.

## Credentials

The app reads exactly two required environment variables, by these names:

```
SUMSUB_APP_TOKEN
SUMSUB_SECRET_KEY
```

`server.py` reads them with `os.environ.get("SUMSUB_APP_TOKEN")` /
`os.environ.get("SUMSUB_SECRET_KEY")` — never anything else, never
hardcoded, never logged, never sent to the frontend. A third, optional but
strongly recommended variable (`SUMSUB_WEBHOOK_SECRET_KEY`) authenticates
incoming webhook calls — see [Securing the webhook](#securing-the-webhook).

Confirm all of this on the dashboard's **Setup** panel once deployed — it
makes a real read-only call to Sumsub and reports "connected," "credentials
rejected," or "can't reach Sumsub," rather than asking you to trust that the
environment variables are spelled right.

**If any of these values were ever pasted somewhere other than Render's
Environment tab** (a chat, a ticket, a shared doc, committed code) — rotate
them in the Sumsub Dashboard. This applies equally to all three variables.

## How data gets in

There are three distinct paths data can enter the bot through, and it's
worth understanding which one covers which situation:

### 1. The webhook — real-time, and the one that matters day to day

Register the bot's webhook URL (shown on the dashboard and at
`GET /api/setup` → `webhook_url`) in **Sumsub Dashboard → Webhook manager**.
Sumsub sends 11 distinct transaction-monitoring event types, and — this is
the detail that trips people up — **it only sends the ones you explicitly
enable**. There is no single "anything changed" event:

| Event type | Fires when |
|---|---|
| `applicantKytTxnCreated` | A new transaction is first created |
| `applicantKytTxnApproved` | Transaction approved (green) |
| `applicantKytTxnRejected` | Transaction rejected (red) |
| `applicantKytTxnReviewed` | A previously held transaction was reviewed |
| `applicantKytOnHold` | Transaction queued for manual review |
| `applicantKytTxnAwaitingUser` | Awaiting applicant action |
| `applicantKytTxnDataChanged` | Transaction data enriched/unmasked |
| `applicantKytTxnDeleted` | Transaction and related data removed |
| `amlCaseApproved` / `amlCaseRejected` / `amlCaseOnHold` | AML case outcome |

**Make sure `applicantKytTxnCreated` is one of the enabled types.** It's a
separate checkbox from the review-outcome events, and it's the one that
tells the bot a brand-new transaction exists at all — without it, new
transactions won't show up as scoreable candidates until some later event
(a review, an approval) happens to fire instead, which from the outside
looks identical to "the webhook isn't working." The dashboard's Setup panel
shows exactly which event types have actually arrived (`GET /api/setup` →
`webhook_event_types`) and flags it explicitly if `applicantKytTxnCreated`
has never shown up, so this is diagnosable in one glance instead of
guesswork. Full event type reference:
[docs.sumsub.com/docs/transaction-monitoring-webhooks](https://docs.sumsub.com/docs/transaction-monitoring-webhooks).

Every event payload includes `kytTxnId` (Sumsub's own transaction ID, or
`kytDataTxnId` for your own external ID if you supplied one at submission
time) — the bot uses that to fetch the transaction's current tags via a
follow-up API call (tags aren't included in the webhook payload itself),
stores the result, and retrains immediately. `applicantKytTxnDeleted`
events remove the local copy instead of trying to re-fetch it.

### 2. Deep historical backfill — for transactions that existed before the bot did

New transactions from today onward are covered automatically by the
webhook. Transactions that were already sitting in your Sumsub account
*before* this bot was connected need a one-time catch-up, which the
**"Run a full historical backfill"** panel on the dashboard triggers
(`POST /api/ingest/backfill-history`). It walks your account's history
month by month (`months_back`, default 60), querying Sumsub's transaction
endpoint by date range with proper pagination and rate-limit backoff, and
runs as a background job — poll `GET /api/ingest/backfill-history-status`
for progress, since a full account history can take several minutes.

This also runs itself automatically, with no clicking required, if the
database is ever found completely empty at startup — see
[Troubleshooting](#troubleshooting) for why that happens and how you'll
know it's in progress.

A lighter version of the same query (just the last 2 days, no month
walking) also runs automatically every `INGEST_INTERVAL_MINUTES` as a
**safety net** — not the primary mechanism, just a backstop in case a
webhook delivery was ever missed or the webhook wasn't registered yet when
a transaction came through.

### 3. Manual ID import — for a specific list

Sumsub's API has no endpoint to "list every transaction tagged X." If you
need a specific set of transactions in fast (say, everything currently
tagged "bank rfi" that the historical backfill hasn't reached yet), filter
by that tag in the Sumsub Dashboard, copy the transaction IDs it shows, and
paste them into the **"Import transactions by ID"** panel
(`POST /api/ingest/import-ids`, accepting either a `txn_ids` array or
free-form pasted text). Each ID is resolved through a real Sumsub API call
— nothing is fabricated — and accepts either Sumsub's own internal ID or
your system's external ID for each transaction, trying one and falling back
to the other automatically. This also runs as a background job
(`GET /api/ingest/import-status` to poll), since a list of hundreds or
thousands of IDs takes real time to resolve one API call at a time.

## How the model works

**Two modes, and it's honest about which one it's in.** Below
`MIN_ML_SAMPLES` (default 15) confirmed "bank rfi" examples, or below that
same threshold of confirmed non-"bank rfi" examples, the bot uses a
transparent rule-based heuristic score built directly from the pattern
analysis above — a trained classifier on a handful of examples would be
overconfident and unreviewable, so it doesn't pretend to have one. Past
both thresholds, it switches to a calibrated logistic regression
(`class_weight="balanced"`, `C=0.3` for meaningful regularization, sigmoid
calibration via cross-validation) and says so — the dashboard's "Model
mode" tile always shows which one is currently active.

**Only resolved transactions teach the model what "not risky" looks
like.** This is the single most important correctness property of the
whole system, and it's worth spelling out because an earlier version of
this bot got it wrong: training used every transaction's `is_bank_rfi`
label at face value, including transactions still sitting open/pending
review. That meant a transaction currently waiting to be scored as a
candidate had, at the very same moment, already been fed into training as
a confirmed "definitely not bank-rfi" example — directly teaching the
model "this profile is safe" about the exact transactions it was supposed
to be flagging as risky. Fixed now: training uses every "bank rfi"-tagged
transaction (a tag is ground truth the instant it's applied, regardless of
workflow status) plus every transaction whose review has actually
concluded without that tag. Transactions still open/pending are fully
eligible to be *scored* — they're simply excluded from teaching the model
what "safe" looks like until their own real outcome is known. Every entry
in the model version history (`GET /api/model/history`) shows exactly how
many transactions were excluded from training for this reason on that
retrain.

**A measured predictive-performance number, not just a mode label.** Every
ML-mode retrain computes cross-validated ROC-AUC — 0.5 means no better than
a coin flip, 1.0 means perfect separation of held-out confirmed "bank rfi"
vs. confirmed-clean transactions — using fresh stratified folds and a
classifier that never saw the held-out fold during its own fit. This is
shown on the dashboard's "Model mode" tile and in the model history table,
so "is this actually predicting anything" has an answer you can point to
instead of just trusting the label.

**Every score comes with an explanation, never a bare number.** For each
open transaction: the specific statistically-significant patterns it
matches (e.g. `"amount" is significantly higher in bank-rfi transactions
(avg 13166.67 vs 1464.92 overall, p=0.0000)`), and the 3 most similar past
"bank rfi" transactions by cosine similarity in feature space, each with
its Sumsub transaction ID, tags, compliance notes, and (best-effort, since
which field holds this is unconfirmed on Sumsub's side) the issuing bank's
name — enough for a reviewer to go pull up the actual matching cases.

**Feature set.** Amount and log-amount, direction, currency, transaction
type, counterparty country (with a static high-risk-country flag),
round-amount and near-reporting-threshold flags, and applicant-history
features (prior transaction count, prior "bank rfi" rate, transaction
velocity in 24h/7d windows, amount relative to the applicant's own
historical average). Categorical features are one-hot encoded and capped
to the 12 most frequent values per feature (`MAX_CATEGORICAL_VOCAB`) so a
high-cardinality field like counterparty country can't blow up the feature
space relative to a small number of positive examples.

## The dashboard

Everything above is visible, not just computed silently. Beyond the
headline tiles (transactions scanned, tagged "bank rfi", candidates since
the last tag, model mode + AUC), the dashboard includes a handful of
diagnostics built specifically so problems are visible in one glance
instead of requiring a support back-and-forth:

- **Setup panel** — live-checks Sumsub connectivity, webhook receipt and
  event-type breakdown, signature verification status, database
  persistence, and data volume, all in plain language with specific next
  steps when something's off.
- **"Every raw tag label seen so far"** (inside "What distinguishes
  bank-rfi transactions") — every distinct tag string actually observed
  across ingested transactions, with a count and whether it matches the
  configured `BANK_RFI_TAG`. If Sumsub's real tag spelling ever differs
  from what the bot expects (a stray hyphen, a typo, a different phrase
  entirely), this is where it shows up immediately. Note that tag matching
  itself is already lenient — case-insensitive and separator-insensitive,
  so "Bank RFI", "bank-rfi", and "bank_rfi" all match — this panel is for
  catching the cases that are genuinely different, not just differently
  formatted.
- **Model & ingestion history** — every retrain logged with its mode,
  example counts (including how many were excluded as still-open/pending),
  and cross-validated AUC, for audit purposes.

## Deploying on Render

Given your Render service already has `SUMSUB_APP_TOKEN` and
`SUMSUB_SECRET_KEY` set:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`

That's it — no Dockerfile, no subfolders, no extra services. Push this
repo's files to GitHub (flat, at the repo root) and Render redeploys.

Once deployed, register
`https://<your-render-url>/api/webhooks/sumsub` in **Sumsub Dashboard →
Webhook manager**, enabling every transaction-monitoring event type you
want tracked (see [How data gets in](#how-data-gets-in) for the full list
and why `applicantKytTxnCreated` specifically matters) — that's the one
step that has to happen on Sumsub's side.

**One tradeoff worth knowing about: ephemeral storage by default.** The
SQLite database (everything the bot has learned) lives on the app's own
filesystem by default, which Render's Free/Starter plans reset on every
restart or redeploy. This is the single most common reason the "tagged
bank rfi" count appears to reset to 0 after a deploy — it's not a bug in
the ingestion logic, it's the database file itself starting over. To make
this self-healing, the app auto-detects a completely empty database at
startup and automatically kicks off a fresh historical backfill in the
background, no clicking required — the Setup panel explains when this is
happening and why. If persistence across restarts matters, add a **Disk**
(Settings → Disks) mounted at `/data`, set
`BOT_DB_PATH=/data/bank_rfi_bot.db`, and use at least the Starter plan —
`render.yaml` has this commented out and ready to uncomment. The Setup
panel also flags whether your current `BOT_DB_PATH` looks ephemeral
(`db_likely_ephemeral` in `GET /api/setup`).

### Environment variables reference

| Variable | Required? | Default | Purpose |
|---|---|---|---|
| `SUMSUB_APP_TOKEN` | Yes | — | Sumsub API app token |
| `SUMSUB_SECRET_KEY` | Yes | — | Sumsub API secret key |
| `SUMSUB_WEBHOOK_SECRET_KEY` | Strongly recommended | — (verification off) | Secret key from Sumsub Dashboard → Webhook manager — verifies each incoming webhook POST really came from Sumsub. See [Securing the webhook](#securing-the-webhook). |
| `INGEST_INTERVAL_MINUTES` | No | `60` | How often the **safety-net** backfill sweep runs — real retraining happens on every webhook event, not this timer |
| `BOT_DB_PATH` | No | `./bank_rfi_bot.db` | Where the SQLite file lives — point at a mounted disk path for persistence |
| `MIN_ML_SAMPLES` | No | `15` | Confirmed examples needed (of *both* classes) before switching from the transparent rule-based score to calibrated ML |
| `SUMSUB_BASE_URL` | No | `https://api.sumsub.com` | Override for sandbox/alternate environments |
| `BANK_RFI_TAG` | No | `bank rfi` | The tag string the bot watches for (matching is already case/separator-insensitive — see [The dashboard](#the-dashboard)) |

## Running it locally

```bash
cp .env.example .env        # fill in real SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — the Setup panel tells you what's left. Note
that Sumsub can't reach `localhost` for webhooks, so local runs only pick
up data via the periodic backfill sweep described above unless you tunnel
the port (e.g. `ngrok http 8000`) and register that URL instead.

## Trying it before real Sumsub data is connected

`dev_seed_synthetic_data.py` populates the local database with made-up
transactions (clearly marked `"synthetic": true`) so you can see the full
pipeline — patterns, model training, predictions, dashboard — working:

```bash
python dev_seed_synthetic_data.py
uvicorn server:app --host 0.0.0.0 --port 8000
```

Delete `bank_rfi_bot.db` afterwards so synthetic rows don't mix with real
data once connected to your actual Sumsub account.

## Securing the webhook

Sumsub Dashboard → Webhook manager shows a secret key when you create or
edit the webhook (auto-generated, or you can set your own). Copy that value
into Render's **Environment** tab as `SUMSUB_WEBHOOK_SECRET_KEY` — never
paste it into chat, a ticket, or committed code, same rule as the API
credentials above.

Once set, every incoming POST to `/api/webhooks/sumsub` is verified against
Sumsub's `x-payload-digest` / `X-Payload-Digest-Alg` headers before being
trusted. Sumsub computes an HMAC digest over the raw request body bytes
using your secret key; the bot recomputes the same digest locally and
compares with a constant-time comparison. All three of Sumsub's supported
algorithms are handled automatically based on what the header says
(`HMAC_SHA256_HEX` is the default for new webhooks; `HMAC_SHA512_HEX` and
the legacy `HMAC_SHA1_HEX` are also supported) — no extra configuration
needed. Anything that doesn't match — wrong secret, tampered body, missing
headers — is rejected with HTTP 401 and never touches your data.

Until `SUMSUB_WEBHOOK_SECRET_KEY` is set, the endpoint accepts any POST at
face value (fine for initial testing, since the bot still works normally),
and the Setup panel visibly flags that verification is off so this isn't a
silent gap. Once it's set, the Setup panel switches to confirming
verification is on and shows a running count of any invalid-signature
attempts blocked, which is also a useful signal if someone else discovers
the public URL.

## Security notes

- **Rotate credentials again if they're ever pasted anywhere outside
  Render's Environment tab** (chat, ticket, shared doc) — regenerate in the
  Sumsub Dashboard. This applies to `SUMSUB_WEBHOOK_SECRET_KEY` too.
- Credentials are read from environment variables only — never hardcoded,
  never logged, never sent to the frontend.
- See [Securing the webhook](#securing-the-webhook) for verifying webhook
  authenticity.
- This handles real KYC/AML data — treat the SQLite file and Render service
  access as sensitive.

## API reference

- `GET /` — the dashboard
- `GET /api/setup` — live credential check, webhook receipt + event-type
  breakdown, signature verification status, database persistence check,
  raw-tag diagnostics
- `GET /api/health` — basic liveness check
- `GET /api/summary` — headline counts, model status, cross-validated AUC
- `GET /api/patterns` — statistical findings + plain-English narrative
- `GET /api/predictions` — ranked open transactions with risk score,
  reasons, and similar past cases
- `GET /api/transactions/{id}` — one transaction's stored record
- `GET /api/model/history` — every retrain, with example counts and AUC,
  for audit purposes
- `POST /api/ingest/run` — trigger an on-demand safety-net ingestion +
  retrain cycle
- `POST /api/ingest/import-ids` — start a background bulk import of
  specific transaction IDs (`{"txn_ids": [...]}` or free-form pasted text)
- `GET /api/ingest/import-status` — poll progress of a running bulk import
- `POST /api/ingest/backfill-history?months_back=60` — start a background
  deep historical crawl of the full account history
- `GET /api/ingest/backfill-history-status` — poll progress of a running
  historical backfill
- `POST /api/webhooks/sumsub` — Sumsub webhook receiver (see
  [Securing the webhook](#securing-the-webhook))

## Troubleshooting

**The "tagged bank rfi" count dropped to 0 after a deploy.** Almost always
the ephemeral-storage issue described in
[Deploying on Render](#deploying-on-render) — check `db_likely_ephemeral`
on the Setup panel, and whether it says it's auto-recovering. Add a
persistent disk to stop this from recurring.

**New transactions aren't showing up as candidates in real time.** Check
`GET /api/setup` → `webhook_event_types` for whether
`applicantKytTxnCreated` has ever arrived — see
[How data gets in](#how-data-gets-in). If it's missing, enable it in
Sumsub's Webhook manager.

**A transaction you know is tagged isn't counted as "bank rfi."** Check
"Every raw tag label seen so far" on the dashboard (inside "What
distinguishes bank-rfi transactions") for the exact raw tag text Sumsub is
returning, and compare it against `BANK_RFI_TAG`. Matching is already
case/separator-insensitive, so this usually means the label is genuinely
different, not just differently formatted.

**Webhook events are arriving but nothing gets ingested.** Check the
Setup panel's event-type breakdown for `(N ingested)` per type — if it's
consistently 0, the transaction fetch itself is failing (check server logs
for the specific Sumsub error) rather than the webhook delivery.

**401s on the webhook after setting `SUMSUB_WEBHOOK_SECRET_KEY`.** The
value in Render must exactly match what's currently shown in Sumsub's
Webhook manager for that webhook — if you regenerated the secret on
Sumsub's side without updating Render (or vice versa), every request will
fail signature verification. The Setup panel's invalid-signature counter
will confirm attempts are arriving even while they're being rejected.
