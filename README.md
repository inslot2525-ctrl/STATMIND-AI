<div align="center">

```
███████╗████████╗ █████╗ ████████╗███╗   ███╗██╗███╗   ██╗██████╗      █████╗ ██╗
██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝████╗ ████║██║████╗  ██║██╔══██╗    ██╔══██╗██║
███████╗   ██║   ███████║   ██║   ██╔████╔██║██║██╔██╗ ██║██║  ██║    ███████║██║
╚════██║   ██║   ██╔══██║   ██║   ██║╚██╔╝██║██║██║╚██╗██║██║  ██║    ██╔══██║██║
███████║   ██║   ██║  ██║   ██║   ██║ ╚═╝ ██║██║██║ ╚████║██████╔╝    ██║  ██║██║
╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚═╝
```

### **Your data walked in raw. It left with a PhD.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Status](https://img.shields.io/badge/status-absolutely_cooking-22c55e?style=for-the-badge)

</div>

---

## 🧠 What is StatMind AI?

StatMind AI is a **full-stack data intelligence platform** that takes your raw CSV or Excel file and turns it into statistics, machine learning models, predictions, leakage reports, and AI-generated insights — all in one workspace.

No Jupyter notebooks. No messy scripts. No PhD required (we handle that part).

Upload data → Get intelligence. That's it.

---

## ⚡ Features That Go Crazy

### 📊 Statistics Studio
Drop a dataset and instantly get:
- Full schema detection (numeric, categorical, datetime, boolean — it figures it out)
- Data quality score with letter grade
- Numeric summary stats (mean, std, quartiles, skewness, kurtosis)
- Categorical breakdowns
- Dataset intelligence bullets generated from the actual data
- Domain-aware insights (Business / Finance / Research)
- **Auto-generated charts** — bar, pie, line, area, histogram, scatter plots, and a **correlation heatmap**

> *It doesn't just describe your data. It understands it.*

---

### 🎯 Target Advisor *(AI-Suggested Target Variable)*
Before you even touch ML, StatMind tells you **which column you should be predicting**.

- Scores every column 0–100 based on domain relevance, cardinality, missing rate, data type, and business keyword matching
- Tells you if it's a **classification** or **regression** target
- Flags identifier/contact columns that would ruin your model
- Shows reasons, warnings, and a ranked leaderboard of candidate targets
- One-click **"Use as Target"** button sends it straight to ML Studio

> *Your data has a story. StatMind finds the ending.*

---

### 🤖 ML Studio
Train real machine learning models with zero setup:

| Algorithm | Type |
|---|---|
| Random Forest | Classification / Regression |
| Extra Trees | Classification / Regression |
| Decision Tree | Classification / Regression |
| K-Nearest Neighbors | Classification / Regression |
| Support Vector Machine | Classification / Regression |
| Logistic Regression | Classification |
| Linear Regression | Regression |
| Ridge Regression | Regression |
| Lasso Regression | Regression |

Every trained model gives you:
- Accuracy / F1 / Precision / Recall / R² / MAE / RMSE (whatever applies)
- Feature importance rankings
- Auto-detected task type (classification vs regression)
- Human-readable model explanation bullets
- A **saved model ID** you can reuse forever

---

### 🏆 AutoML Compare
Can't pick an algorithm? Don't.

Run **Compare Models** and StatMind trains every compatible algorithm on your data simultaneously, ranks them on a leaderboard, and crowns the winner.

> *It's like a tournament bracket, but for algorithms.*

---

### 🔮 Prediction Studio
Load any saved model ID + a new test dataset and generate predictions instantly.

- Supports CSV and Excel test files
- Returns a preview table in the UI
- Evaluates against ground truth if labels are present
- **Downloadable predictions CSV** — ready for stakeholders, reports, or flexing

---

### 🕵️ Leakage Detector
The feature that could save your entire ML project from being a lie.

StatMind analyzes your dataset for **data leakage** — columns that are suspiciously correlated with your target in ways that would never exist in production. It flags them before you waste 3 days wondering why your model has 99.8% accuracy and fails in the real world.

> *"My model gets 100% accuracy" — you, before using this feature.*

---

### 💬 AI Data Scientist Chatbot
An embedded analyst that knows your dataset context and answers questions like:
- "What's the most important feature for predicting churn?"
- "Why does this column have so many nulls?"
- "Should I normalize this data?"

---

### 📄 Intelligence Report Generator
One click. Full HTML report. Contains:
- Dataset summary and quality analysis
- Domain insights
- ML model performance
- Recommendations

Download it. Send it. Look extremely smart in meetings.

---

## 🏗️ Architecture

```
statmind-ai/
├── backend/                  # FastAPI Python backend
│   ├── main.py               # App entry point, CORS, auth-protected routers, startup
│   ├── config.py             # All settings, from environment variables
│   ├── database.py           # SQLAlchemy models (users, datasets, jobs, ...)
│   ├── migrations/           # Alembic schema migrations (run automatically)
│   ├── worker.py             # Standalone background job worker
│   ├── manage.py             # Admin CLI: create-user, claim-legacy-data
│   ├── routes/
│   │   ├── auth_routes.py          # /api/auth/register, /login, /me
│   │   ├── analysis_routes.py      # /api/analyze, /api/datasets
│   │   ├── target_routes.py        # /api/recommend-target
│   │   ├── ml_routes.py            # /api/train-model, /api/compare-models (jobs), registries
│   │   ├── job_routes.py           # /api/jobs
│   │   ├── prediction_routes.py    # /api/predict-test-data
│   │   ├── artifact_routes.py      # /api/artifacts (signed download links)
│   │   └── intelligence_routes.py  # /api/chat-analyst, /api/detect-leakage, /api/generate-report
│   └── services/
│       ├── auth.py                 # Password hashing, JWT, current-user dependency
│       ├── jobs.py, job_tasks.py   # Database-backed job queue and its tasks
│       ├── storage.py              # Local / S3 storage, content-addressed uploads
│       ├── artifacts.py            # Owned downloadable files
│       ├── ml_engine.py            # Model training, AutoML, predictions
│       ├── dataset_registry.py, experiment_tracker.py, model_registry.py
│       └── ...                     # Statistics, charts, leakage, target advisor, reports
│
└── frontend/                 # React + Vite frontend
    └── src/
        ├── App.jsx           # All studios
        ├── AuthScreen.jsx    # Login / registration
        ├── api.js            # API client: tokens, job polling, downloads
        └── App.css           # Dark glassmorphism design system
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A dataset that deserves better than Excel pivot tables

### 1. Clone it

```bash
git clone https://github.com/yourname/statmind-ai.git
cd statmind-ai
```

### 2. Backend setup

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the server
cd backend
uvicorn main:app --reload --port 8000
```

Backend runs at → `http://localhost:8000`  
API docs → `http://localhost:8000/docs`

Locally this needs no extra services: SQLite, files on disk, and background jobs
run in a worker thread inside the API. Open the app and create an account; every
dataset, model and experiment is private to its owner.

Upgrading from a version without accounts? Your old datasets and models are kept
but have no owner yet. Create your account, then give them to it:

```bash
cd backend
python manage.py claim-legacy-data you@example.com
```

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at → `http://localhost:5173`

### 4. Run the tests

```bash
pip install -r backend/requirements-dev.txt
pytest            # from the repo root; uses a throwaway data dir, never your real DB
```

### Or: the full stack with Docker

```bash
docker compose up --build
```

Runs Postgres, the API, a separate job worker and the frontend, the same shape as production.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `STATMIND_ENV` | `development` | Set to `production` to refuse to start without `JWT_SECRET`. |
| `JWT_SECRET` | random per start | Signs login tokens and download links. **Required in production** (32+ random characters). |
| `ACCESS_TOKEN_MINUTES` | `720` | How long a login lasts. |
| `ALLOW_REGISTRATION` | `true` | Set `false` to allow only accounts created with `manage.py create-user`. |
| `DATABASE_URL` | SQLite in `STATMIND_DATA_DIR` | Any SQLAlchemy URL. `postgres://...` URLs (Railway, Heroku) are accepted as-is. Schema migrations run automatically on startup. |
| `STORAGE_BACKEND` | `local` | `local` (files under `STATMIND_DATA_DIR`) or `s3` (AWS S3, Cloudflare R2, MinIO). |
| `S3_BUCKET` / `S3_ENDPOINT_URL` / `S3_REGION` / `S3_PREFIX` | — | Bucket settings for `STORAGE_BACKEND=s3`. Leave the endpoint empty for AWS; set it for R2/MinIO. Credentials come from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. |
| `JOB_BACKEND` | `thread` | `thread`: the API also runs jobs. `worker`: the API only queues them; run `python worker.py` separately. |
| `JOB_STALE_SECONDS` / `JOB_MAX_ATTEMPTS` | `120` / `3` | A job whose worker stops responding is retried, then failed. |
| `STATMIND_DATA_DIR` | `backend/` (Docker: `/data`) | Local files: SQLite DB, uploads, models, predictions, reports, S3 cache. |
| `MAX_UPLOAD_MB` | `50` | Maximum upload size. |
| `ALLOWED_ORIGINS` | localhost dev ports | Comma-separated CORS origins. |
| `ALLOWED_ORIGIN_REGEX` | `https://.*\.vercel\.app` | CORS origin regex; narrow it to your own Vercel project. |
| `LOG_LEVEL` | `INFO` | Backend log level. |

---

## 🔌 API Reference

All endpoints except registration, login and health need an `Authorization: Bearer <token>` header.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Create an account (JSON `email`, `password`) → token |
| `POST` | `/api/auth/login` | Log in (form `username`, `password`) → token |
| `GET`  | `/api/auth/me` | The current user |
| `POST` | `/api/analyze` | Upload & profile a dataset |
| `POST` | `/api/recommend-target` | AI target variable suggestion |
| `POST` | `/api/train-model` | Start a training job → `202` + job |
| `POST` | `/api/compare-models` | Start an AutoML leaderboard job → `202` + job |
| `GET`  | `/api/jobs/{id}` | Job status, progress, and result when finished |
| `POST` | `/api/predict-test-data` | Run predictions with a saved model |
| `GET`  | `/api/artifacts/{id}/download-url` | Short-lived link to download a prediction CSV or report |
| `POST` | `/api/chat-analyst` | Ask the AI data scientist |
| `POST` | `/api/detect-leakage` | Data leakage detection |
| `POST` | `/api/generate-report` | Generate HTML intelligence report |
| `GET`  | `/api/datasets`, `/api/models/`, `/api/experiments/` | Your registries |

Full interactive docs at `/docs` (FastAPI Swagger UI; use **Authorize** to log in).

### Deploying (e.g. Railway)

1. Add a **Postgres** database; set `DATABASE_URL` on the API.
2. Create a bucket on **Cloudflare R2** (or S3); set `STORAGE_BACKEND=s3` and the `S3_*` / `AWS_*` variables.
   Without a bucket, attach a persistent volume at `/data` instead.
3. Set `STATMIND_ENV=production`, a long random `JWT_SECRET`, and `ALLOWED_ORIGIN_REGEX` for your Vercel URL.
4. For heavy training, set `JOB_BACKEND=worker` and add a second service from the same image with start command `python worker.py` and the same variables.

---

## 🧪 Supported File Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Excel 2007+ | `.xlsx` |
| Excel 97-2003 | `.xls` |

---

## 🎨 Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — async Python API framework
- [SQLAlchemy](https://www.sqlalchemy.org/) + [Alembic](https://alembic.sqlalchemy.org/) — SQLite / PostgreSQL with migrations
- [PyJWT](https://pyjwt.readthedocs.io/) + argon2 ([pwdlib](https://frankie567.github.io/pwdlib/)) — accounts and tokens
- [boto3](https://boto3.amazonaws.com/) — S3 / R2 / MinIO storage
- [scikit-learn](https://scikit-learn.org/) — ML algorithms
- [pandas](https://pandas.pydata.org/) — data manipulation
- [numpy](https://numpy.org/) — numerical computing
- [uvicorn](https://www.uvicorn.org/) — ASGI server
- [joblib](https://joblib.readthedocs.io/) — model persistence

**Frontend**
- [React 19](https://react.dev/) — UI framework
- [Vite 8](https://vitejs.dev/) — blazing fast build tool
- [Recharts](https://recharts.org/) — chart library

---

## 🗺️ Roadmap

- [x] User accounts (JWT) with per-user data isolation
- [x] PostgreSQL + migrations, S3/R2 storage, background jobs
- [ ] OAuth login (Google / GitHub) and password reset
- [ ] Time series forecasting module
- [ ] NLP / text column analysis
- [ ] Model explainability (SHAP values)
- [ ] Multi-file join and merge studio
- [ ] Export to Jupyter Notebook
- [ ] Deploy to cloud (Docker + Railway / Render)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first so we can argue about it in a civilized manner.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/something-wild`
3. Commit your changes: `git commit -m 'Add something wild'`
4. Push to the branch: `git push origin feature/something-wild`
5. Open a pull request

---

## 📜 License

MIT — do whatever you want with it. Build a startup. Win a hackathon. Impress your professor.

---

<div align="center">

**Built with unreasonable ambition and a lot of caffeine.**

*StatMind AI — because your data deserves better than a spreadsheet.*

</div>
