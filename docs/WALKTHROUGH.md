# StatMind — how it works and why

A module-by-module walkthrough of the codebase: what each part does, why it is
built that way, and where its limits are. For the reasoning behind individual
choices (why macro F1, why random search, why not Celery) see
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

## 1. What the app does

You upload a CSV or Excel file. StatMind profiles it, suggests which column to
predict, checks for columns that give the answer away, trains and tunes models
with cross-validation, explains the winner, scores new files with it, and keeps
a record of every dataset, run and model.

| Page | What it is for |
|---|---|
| **Dashboard** | Landing page: what the app does, the four-step flow, links to each studio |
| **Statistics Studio** | Upload a file; schema, data quality score, summary statistics, charts |
| **ML Studio** | Suggest a target, check leakage, train one model or run AutoML, read the model report |
| **Prediction Studio** | Score a new file with a saved model and download the predictions |
| **MLOps Studio** | Experiments, model registry with stages, dataset versions |
| **AI Analyst** | Ask questions about the results; export an HTML report |

## 2. Project structure

```
STATMIND-AI/
├── backend/                     FastAPI application
│   ├── main.py                  app, CORS, routers, startup
│   ├── config.py                every setting, read from environment variables
│   ├── database.py              SQLAlchemy models + automatic migrations
│   ├── migrations/              Alembic migration scripts (0001-0003)
│   ├── worker.py                standalone background job worker
│   ├── manage.py                admin commands
│   ├── routes/                  HTTP endpoints, thin: validate, call a service, return
│   └── services/                all the logic
├── frontend/src/                React + Vite single-page app
├── tests/                       pytest suite (141 tests in 15 files) + sample datasets
├── docs/                        this file, design decisions, deploy guide, screenshots
├── docker-compose.yml           Postgres + API + worker + MLflow + frontend
├── start-dev.ps1                starts lakeFS + MLflow + API + app locally (Windows)
└── .github/workflows/ci.yml     tests on SQLite and Postgres, Docker builds
```

The rule the backend follows: **routes are thin, services hold the logic.** A
route checks the input and who is asking, calls one service function, and turns
`ValueError` into a 400. That keeps the ML code testable without HTTP.

## 3. Request flow

```
Browser (React)                        FastAPI backend
───────────────                        ───────────────
upload file ─── POST /api/analyze ───▶ storage.save_upload        hash the bytes, store once
                                       dataset_registry.register  one row per user + content hash
                                       analysis_engine.analyze    schema, quality, stats, charts
profile  ◀─────────────────────────────┘

Suggest target ─ POST /api/recommend-target ─▶ target_advisor
Check leakage ── POST /api/detect-leakage ───▶ leakage_detector

Run AutoML ──── POST /api/compare-models ───▶ jobs.enqueue        returns 202 + job id at once
                                                   │
poll ────────── GET /api/jobs/{id} ◀── progress ── worker thread / worker process
                                                   │
                                       ml_engine.compare_models_from_file
                                         split → cross-validate → tune → refit → test
                                         explain → save → (lakeFS commit) → (MLflow run)
                                         → model registry → experiment log
result ◀───────────────────────────────────────────┘

Predict ─────── POST /api/predict-test-data ─▶ ml_engine.predict_with_saved_model
Download ────── GET /api/artifacts/{id}/download-url ─▶ short-lived signed link
Ask ─────────── POST /api/chat-analyst ──────▶ ai_analyst (Claude, or rules)
```

Training is the only slow step, which is why it alone goes through the job queue.

## 4. Backend modules

### `config.py` — settings

**What it does.** Reads every setting from environment variables once, at
import. With nothing set you get the local defaults: SQLite, files on disk, jobs
in a thread, no login, no MLflow, no lakeFS.

**Why this way.** One place to see every switch, and the same code runs locally
and in production with only the environment changed.

**Limits.** Settings are read at import, so changing one needs a restart.

### `database.py` and `migrations/` — data model

**What it does.** Six tables: `users`, `datasets`, `experiments`, `models`
(the model registry), `jobs` and `artifacts`, plus Alembic's version table. `init_db()`
runs pending migrations at startup, so the schema is always current.

**Why this way.** Alembic migrations instead of `create_all` means an existing
database is upgraded without losing data; a test checks that a database from
before accounts existed upgrades cleanly. SQLAlchemy lets the same code use
SQLite locally and PostgreSQL in production; CI runs the tests on both.

**Limits.** Experiment and model details are stored as JSON text columns, so
they cannot be filtered efficiently in SQL.

### `services/storage.py` — where files live

**What it does.** One interface (`put_file`, `put_bytes`, `exists`,
`local_path`, `presigned_url`) with three backends: `LocalStorage`, `S3Storage`
and `LakeFSStorage`. `save_upload` streams an upload to disk while hashing it,
enforces the size limit, and stores it under its SHA-256 hash.

**Why this way.**
- *Content-addressed names* mean identical uploads are stored once, and a
  user-supplied filename is never used as a path, which rules out path traversal.
- *Keys are generated by the server* (`uploads/<hash>.csv`, `models/<id>.pkl`)
  and validated before use.
- pandas and joblib need real files, so the S3 backends keep a local
  read-through cache. That is safe because uploads and models never change.

**Limits.** The cache is never evicted. Uploads are capped at 50 MB by default.

### `services/dataset_registry.py` — datasets

**What it does.** Records each upload per user with its content hash, row and
column counts, missing cells and duplicate rows. Uploading the same content
again returns the existing dataset id. With lakeFS on, the upload is committed
and the commit id is saved on the row.

**Why this way.** A unique constraint on (owner, content hash) makes
deduplication a database guarantee rather than a check that can race.

### `services/analysis_engine.py` — profiling

**What it does.** `analyze_file` produces the Statistics Studio page: schema
detection (numeric, categorical, datetime, text, boolean), a data quality score
with a letter grade, numeric and categorical summaries, and chart data (bar,
pie, line, area, histogram, scatter, correlation heatmap).

**Why this way.** Chart data is computed on the server and sampled (5,000 rows
for charts, 300 points per scatter, 8 columns for the heatmap) so a large file
still returns quickly and the browser never receives the raw data. Results are
cached in memory for the last 20 files.

**Limits.** Charts on large files are drawn from a sample. The domain insights
are template sentences filled in from the statistics, not learned.

### `services/target_advisor.py` — which column to predict

**What it does.** Scores every column 0-100 as a prediction target: identifier
and contact columns are rejected, constant and mostly-missing columns are
rejected, and points are added for outcome-like names for the chosen domain, a
manageable number of distinct values, and few missing values.

**Why this way.** It is a transparent heuristic the user can overrule with one
click. It uses the same task-type rule as training, so its advice matches what
will happen.

**Limits.** It judges names and basic statistics only. It has no idea what the
columns mean.

### `services/leakage_detector.py` — columns that give the answer away

**What it does.** Five checks: a shallow tree cross-validated on each feature
alone (flagged at ≥ 0.97); exact copies of the target; correlation ≥ 0.95 for
numeric and binary targets; targets that equal `a + b`, `a - b` or `a × b` of
two columns; and identifier or outcome-like names matched by whole word.

**Why this way.** Correlation alone misses categorical and non-linear leaks, so
the single-feature model is the main check. A feature that trips several checks
counts once towards the risk score, at its worst severity.

**Limits.** Heuristics, not proof. A genuinely strong feature can be flagged,
and a leak through time (a value recorded after the outcome) passes every check.

### `services/ml_engine.py` — training, AutoML, explanation, prediction

The largest module. In order:

- **`prepare_ml_data`** reads the file, drops ID-like columns, decides the task
  type, encodes the target, and makes the train/test split (stratified, seed 42).
- **`build_preprocessor`** builds a `ColumnTransformer` with three branches:
  numbers (median imputation, scaling), categories (most-frequent imputation,
  one-hot capped at 30 categories), and dates (`DatetimeFeatures`: year, month,
  day of week, days since the earliest training date).
- **`get_available_models`** returns the algorithm list: 7 for classification
  and 9 for regression, with class weights balanced when the target is imbalanced.
- **`train_model_from_file`** fits one pipeline.
- **`compare_models_from_file`** is AutoML: cross-validate every algorithm plus
  a feature-blind baseline, tune the top two with a random search, refit the
  winner on the whole training split, score it once on the test split.
- **`_finalize_model`** is shared by both paths: metrics, diagnostics (ROC
  curves, confusion matrix, per-class scores, or predicted-vs-actual),
  permutation importance, baseline metrics, then save, commit, log and register.
- **`predict_with_saved_model`** loads a model the user owns, checks the new
  file has the required columns, predicts, and stores the result as a download.

**Why this way.** Preprocessing is inside the pipeline, so cross-validation
cannot leak statistics between folds and the saved model applies identical
steps to new data. The test split is never used for selection.

**Limits.** In-memory pandas. Cross-validation uses at most 20,000 rows, and
SVM and KNN are skipped above 5,000. Models are pickles tied to this codebase.

### `services/jobs.py`, `job_tasks.py`, `worker.py` — background jobs

**What it does.** A job queue stored in the `jobs` table. `enqueue` inserts a
row and returns immediately. A worker claims the oldest queued job, runs it,
reports progress, and writes the result. Three modes: `thread` (inside the API,
the default), `worker` (a separate process, `python worker.py`), and `inline`
(run during the request, used by tests).

**Why this way.**
- *No double runs:* a job is claimed with one conditional
  `UPDATE ... WHERE status = 'queued'`; the database lets only one worker win.
- *Crash recovery:* running jobs heartbeat every 15 s; one silent for 120 s is
  requeued, up to 3 attempts, then failed.
- *No extra infrastructure:* the database already holds job status for the UI,
  so it doubles as the queue. No Redis or RabbitMQ, and it works on Windows.

**Limits.** One job at a time per worker, and polling rather than push. Fine
for training jobs; wrong for high-throughput work.

### `services/auth.py` — accounts

**What it does.** Argon2 password hashing, JWT access tokens, and the
`get_current_user` dependency every protected route uses. Download links carry
a separate short-lived token whose `purpose` claim stops it being used as a
login token. With `AUTH_REQUIRED` off (the local default), requests without a
token act as one built-in local user.

**Why this way.** The dependency is attached at router level in `main.py`, so a
new endpoint is protected by default; a test walks every route and checks.
Every service function takes an `owner_id` and filters by it, so one user
cannot see or load another's data. Production refuses to start with login off.

**Limits.** No password reset, roles, OAuth or rate limiting.

### `services/model_registry.py`, `experiment_tracker.py`, `artifacts.py`

**What they do.** The registry stores each model's metadata and stage
(development, staging, production, archived). The tracker stores one row per
training run with hyperparameters and metrics. Artifacts are downloadable files
(prediction CSVs, reports) owned by a user.

**Why this way.** A model is a pickle, and unpickling runs code, so
`load_model_artifact` only loads a model that is registered to the requesting
user with a server-generated storage key.

### `services/mlflow_tracking.py` — optional MLflow

**What it does.** When `MLFLOW_TRACKING_URI` is set, each trained model is also
logged to MLflow: parameters, test and baseline metrics, cross-validation
scores, JSON artifacts and the fitted pipeline as a new registry version. AutoML
logs the winner as a parent run with each candidate as a child run. Changing a
model's stage in StatMind moves the matching MLflow alias.

**Why this way.** StatMind's own tables stay the source of truth, so the app
needs no extra service to run and a tracking outage cannot fail a training job.

**Limits.** Logged models need this codebase on the Python path to load.

### `services/lakefs_versioning.py` — optional lakeFS

**What it does.** When `LAKEFS_ENDPOINT` is set, files are stored through
lakeFS's S3 gateway, and each dataset upload and trained model becomes a
commit. A model records the commit holding its training data and the commit
holding its own file.

**Why this way.** lakeFS speaks the S3 protocol, so the existing S3 code is
reused; only the commit calls are new. Commits are best effort (a failed commit
must not lose an upload) but startup is strict (no repository, no storage).

**Limits.** One branch. No branching, merging or reverting from the StatMind UI.

### `services/ai_analyst.py` — the analyst

**What it does.** Builds a JSON snapshot of what StatMind computed (metrics,
baseline, cross-validation, feature importance, leaderboard, leakage findings)
and asks Claude to answer from it. With no API key, or on any API error, the
original keyword rules answer instead and the reply says which one answered.

**Why this way.** Grounding the model in computed results keeps it from
inventing numbers, and the fallback means the feature and the tests never
depend on the network.

**Limits.** The rule-based fallback matches keywords and has only six kinds of
answer.

### `services/report_generator.py` — the HTML report

Turns the same context into one self-contained HTML file, stored as an artifact.

### Not used

`services/schema_detector.py`, `chart_recommender.py`, `domain_analyzer.py` and
`stats_engine.py` are left from an earlier version and are not imported
anywhere; `analysis_engine.py` replaced them.

## 5. Frontend (`frontend/src`)

| File | Role |
|---|---|
| `App.jsx` | App state, the page functions (Dashboard, Statistics, ML, Prediction, MLOps), the top bar |
| `MlResults.jsx` | Model report, AutoML leaderboard, leakage report, and their charts |
| `AnalystPanel.jsx` | The chat and the report button; owns its own state |
| `AuthScreen.jsx` | Login and registration, shown only when the server requires login |
| `api.js` | Every backend call: token handling, `runJob` (start a job and poll), downloads |
| `App.css`, `polish.css` | Base theme; borders, motion and type refinements |
| `spotlight.js` | One pointer listener that positions the border highlight on cards |

**How a training run reaches the screen.** `runJob` posts the form, receives a
job, and polls `GET /api/jobs/{id}` with a growing delay (0.7 s up to 3 s),
passing each response to the progress bar until the job succeeds or fails.

**Why this way.** There is no router or state library: page is one piece of
state, and results are passed down as props. Presentational components live at
module scope so React keeps their identity between renders. They used to be
defined inside `App`, which rebuilt whole pages on every state change and made
inputs lose focus while typing.

**Limits.** `App.jsx` is still about 1,700 lines and holds most state. There
are no frontend tests. Refreshing the page loses the current results (they are
still in MLOps Studio).

## 6. Tests and CI

141 tests in `tests/`, run with `pytest` from the repo root against a throwaway
data directory:

| Area | Files |
|---|---|
| Auth, isolation between users, security (path traversal, unsafe model loading) | `test_auth.py`, `test_auth_optional.py`, `test_isolation.py`, `test_security.py` |
| Jobs, storage on S3 (mocked with moto), migrations | `test_jobs.py`, `test_storage_s3.py`, `test_migrations.py` |
| AutoML, task detection, preprocessing, explainability | `test_automl.py`, `test_ml_engine.py`, `test_benchmark.py` |
| Leakage detector and target advisor | `test_leakage.py` |
| Analyst (fake Claude client), MLflow (temporary local store), lakeFS (fake API, plus one test against a real server when configured) | `test_analyst.py`, `test_mlflow.py`, `test_lakefs.py` |

CI runs the suite on SQLite and PostgreSQL, builds both Docker images, and
smoke-tests the backend container.

## 7. Results on the bundled datasets

80/20 split, seed 42, from `tests/data`:

| Dataset | Winner | 5-fold CV | Held-out test | Baseline |
|---|---|---|---|---|
| Iris | SVM (tuned) | F1 0.975 ± 0.021 | accuracy 0.967 | 0.333 |
| Breast cancer | Logistic regression (tuned) | F1 0.981 ± 0.006 | accuracy 0.974, AUC 0.996 | 0.632 |
| Diabetes | Lasso (tuned) | R² 0.481 ± 0.040 | R² 0.456 | R² −0.012 |

These are small textbook datasets: they show the pipeline is correct, not that
it is competitive on hard problems. Simple models win because the data is small
and close to linear.

## 8. Running it

```powershell
.\start-dev.ps1          # lakeFS + MLflow + API + app
.\start-dev.ps1 -Plain   # API + app only
```

or by hand: `uvicorn main:app --port 8000` in `backend/`, `npm run dev` in
`frontend/`. Tests: `pytest`. See the README for configuration and
[DEPLOY.md](DEPLOY.md) for hosting.
