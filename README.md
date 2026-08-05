# Bank RFI Prediction Bot

Learns from every Sumsub transaction tagged **"bank rfi"**, finds what's
statistically different about them, and scores currently-open transactions by
how likely they are to get that tag next — with a plain-English justification
and links to the most similar past "bank rfi" transactions for each score.

It retrains every hour on everything it's seen, so the model and the pattern
analysis both improve automatically as more transactions get tagged on
Sumsub — no manual retraining, no separate data files.

## One file, no subfolders, no Dockerfile

```
server.py                    Everything -- API client, DB, model, ingestion, routes, scheduler, dashboard
requirements.txt             Python dependencies
render.yaml                  Render config (optional -- see "Deploying on Render")
.env.example                 Copy to .env for local runs
.gitignore
dev_seed_synthetic_data.py   Optional: seeds fake data so you can try the dashboard before Sumsub is connected
```

That's the whole repo. Nothing lives in a subfolder, and there's no
Dockerfile — Render runs this as a plain Python app.

## 1. Credentials — already matches what's on Render

The app reads exactly two environment variables, by these names:

```
SUMSUB_APP_TOKEN
SUMSUB_SECRET_KEY
```

Since your Render service already has these two set (with your rotated
token/secret), there is nothing to rename or change here — `server.py` reads
them with `os.environ.get("SUMSUB_APP_TOKEN")` / `os.environ.get("SUMSUB_SECRET_KEY")`,
never anything else. Confirm on the dashboard's **Setup** panel once deployed
— it makes a real read-only call to Sumsub and reports "connected",
"credentials rejected," or "can't reach Sumsub."

## 2. Where the data comes from — the Sumsub API, not a local file

There is no seed file, no manual transaction-ID list, no synthetic data in
the real code path. Every transaction the bot knows about was fetched from
Sumsub. Two mechanisms do this, and it's worth being precise about which one
is guaranteed vs. best-effort, because I checked Sumsub's public API docs
directly while rebuilding this rather than assuming:

**Webhooks — confirmed, and the reliable one.** Register this app's webhook
URL (shown on the dashboard) in **Sumsub Dashboard → Settings → Webhooks**
for Transaction Monitoring events. Sumsub's real webhook payloads (verified
against their docs) look like this:

```json
{
  "type": "applicantKytTxnReviewed",
  "kytTxnId": "64a7dc05fbf57c624afcb72d",
  "applicantId": "634829375766b80001a40152",
  "reviewStatus": "completed"
}
```

`kytTxnId` is the exact ID this app needs to fetch the full transaction and
its tags (`GET /resources/kyt/txns/{id}/one` and `.../tags`). Every time
Sumsub fires one of these events, the bot immediately re-fetches that
transaction, checks whether it now carries the "bank rfi" tag, and stores it
if so. **This is what makes "the data set grows as changes are made on
Sumsub" actually true** — it's automatic and real-time, not a scheduled
guess.

**Historical backfill — best-effort, and here's the honest limitation.**
Sumsub's public API does not document a general "list/search all
transactions, optionally filtered by tag" endpoint. The only documented
transaction-query endpoint
(`docs.sumsub.com/reference/find-specific-tr-transactions`) only returns
results when filtered to `data.type=travelRule` — i.e. it's built for crypto
Travel Rule transactions specifically, not general fiat KYT transactions,
and it has no tag filter (it filters by amount/currency/direction/date, not
by tag) and no real pagination (max 100 items per call). This app still
calls it every hour as a best-effort attempt, working around the missing
pagination with a date-cursor trick, and separately tries walking applicants
to enumerate their transactions (an experimental strategy, since Sumsub
doesn't document a "list this applicant's transactions" endpoint either).
**If your Sumsub transactions aren't Travel Rule/crypto type, expect these
two strategies to return zero rows — that's Sumsub's API surface, not a bug
here.** The dashboard's ingestion history shows exactly what each strategy
found on every run, so this is never silently hidden.

**Practically, this means:** transactions that get tagged "bank rfi" from
today onward are captured automatically and reliably via the webhook.
Transactions that were already tagged "bank rfi" *before* this bot was
connected may or may not be recoverable automatically, depending on whether
your account's transactions are Travel Rule type — the ingestion history
panel will tell you within the first hour whether it found anything. If it
doesn't, the bot still starts learning from that point forward, and the
model naturally sharpens as more real tags accumulate.

## 3. The recurring, self-improving backend

A background job (APScheduler, running inside this same process — no
separate worker, no cron service to configure) fires every
`INGEST_INTERVAL_MINUTES` (**default: 60, i.e. hourly**, exactly as
specified) and:

1. Attempts both backfill strategies above (as a safety net alongside the
   webhook).
2. Retrains the model on every transaction currently in the database.
3. Re-scores every open (not-yet-tagged) transaction against the freshly
   retrained model.

The webhook receiver also triggers an immediate retrain whenever a new
"bank rfi" tag arrives, rather than waiting for the next hourly cycle, so
the model doesn't lag behind real tagging activity.

**Output, for every open transaction, ranked by predicted risk:**

- **The risk score** (0–1, calibrated once ≥15 real "bank rfi" examples
  exist; a transparent rule-based score before that, since a trained
  classifier on a handful of examples would be overconfident and
  unreviewable).
- **Why** — the specific statistically-significant patterns this
  transaction matches (e.g. "counterparty_country = 'BY' appears 34x more
  often in bank-rfi transactions than overall"), never a black-box number
  alone.
- **References to real past "bank rfi" transactions** — the 3 most similar
  past bank-rfi-tagged transactions by feature similarity, each with its
  Sumsub transaction ID, tags, compliance notes, and (best-effort, since
  which field holds this is unconfirmed) the issuing bank's name, so a
  reviewer can pull up the actual matching cases on Sumsub.

## 4. Deploying on Render — flat, plain Python, already has your API key

**Given your Render service already has `SUMSUB_APP_TOKEN` and
`SUMSUB_SECRET_KEY` set, the only thing left to check is the Start Command:**

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`

That's it — no Dockerfile, no subfolders, no extra services. Push this
repo's files to GitHub (flat, at the repo root) and Render redeploys.

Once deployed, open the service's public URL and register
`https://<your-render-url>/api/webhooks/sumsub` in **Sumsub Dashboard →
Settings → Webhooks** for Transaction Monitoring events — that's the one
step that has to happen on Sumsub's side.

**One optional tradeoff worth knowing:** by default the SQLite database
(everything the bot has learned) lives on the app's own filesystem, which
Render's Free/Starter plans reset on every restart or redeploy. That's fine
to start with — the bot just relearns from whatever the webhook has fed it
since the last restart. If persistence across restarts matters (and your
Render account's billing/payment-method issue from earlier is resolved),
add a **Disk** (Settings → Disks) mounted at `/data`, set
`BOT_DB_PATH=/data/bank_rfi_bot.db`, and use at least the Starter plan —
`render.yaml` has this commented out and ready to uncomment.

### Environment variables reference

| Variable | Required? | Default | Purpose |
|---|---|---|---|
| `SUMSUB_APP_TOKEN` | Yes | — | Sumsub API app token (already set on Render) |
| `SUMSUB_SECRET_KEY` | Yes | — | Sumsub API secret key (already set on Render) |
| `INGEST_INTERVAL_MINUTES` | No | `60` | How often the background job re-ingests + retrains (hourly by default, as specified) |
| `BOT_DB_PATH` | No | `./bank_rfi_bot.db` | Where the SQLite file lives — point at a mounted disk path for persistence |
| `MIN_ML_SAMPLES` | No | `15` | Bank-rfi examples needed before switching from the transparent rule-based score to calibrated ML |
| `SUMSUB_BASE_URL` | No | `https://api.sumsub.com` | Override for sandbox/alternate environments |
| `BANK_RFI_TAG` | No | `bank rfi` | The exact tag string the bot watches for |

## Running it locally

```bash
cp .env.example .env        # fill in real SUMSUB_APP_TOKEN / SUMSUB_SECRET_KEY
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — the Setup panel tells you what's left. Note
that Sumsub can't reach `localhost` for webhooks, so local runs only pick up
data via the hourly backfill attempts described above unless you tunnel the
port (e.g. `ngrok http 8000`) and register that URL instead.

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

## Security notes

- **Rotate credentials again if they're ever pasted anywhere outside
  Render's Environment tab** (chat, ticket, shared doc) — regenerate in the
  Sumsub Dashboard.
- Credentials are read from environment variables only — never hardcoded,
  never logged, never sent to the frontend.
- The webhook endpoint currently accepts any POST to `/api/webhooks/sumsub`.
  Sumsub's docs don't specify a payload-signature header for every
  plan/account, so before relying on this fully in production, consider IP
  allowlisting (Sumsub publishes their webhook source ranges) and/or a
  shared-secret query parameter.
- This handles real KYC/AML data — treat the SQLite file and Render service
  access as sensitive.

## API reference (once running)

- `GET /` — the dashboard
- `GET /api/setup` — live credential check, webhook receipt count, ingestion status
- `GET /api/health` — basic liveness check
- `GET /api/summary` — headline counts + model status
- `GET /api/patterns` — statistical findings + narrative
- `GET /api/predictions` — ranked open transactions with risk score, reasons, similar past cases
- `GET /api/transactions/{id}` — one transaction's stored record
- `GET /api/model/history` — every retrain, for audit purposes
- `POST /api/ingest/run` — trigger an ingestion + retrain cycle on demand
- `POST /api/webhooks/sumsub` — Sumsub webhook receiver
