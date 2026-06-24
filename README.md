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
│   ├── main.py               # App entry point, CORS, router registration
│   ├── routes/
│   │   ├── analysis_routes.py      # /api/analyze
│   │   ├── target_routes.py        # /api/recommend-target
│   │   ├── ml_routes.py            # /api/train-model, /api/compare-models
│   │   ├── prediction_routes.py    # /api/predict-test-data
│   │   └── intelligence_routes.py  # /api/chat-analyst, /api/detect-leakage, /api/generate-report
│   ├── services/
│   │   ├── analysis_engine.py      # Dataset profiling orchestrator
│   │   ├── stats_engine.py         # Statistics, quality scoring
│   │   ├── schema_detector.py      # Column type detection
│   │   ├── chart_recommender.py    # Dynamic chart generation
│   │   ├── domain_analyzer.py      # Domain-specific insights
│   │   ├── target_advisor.py       # AI target variable recommendation
│   │   ├── ml_engine.py            # Model training, AutoML, predictions
│   │   ├── leakage_detector.py     # Data leakage analysis
│   │   ├── ai_analyst.py           # Chatbot / Q&A engine
│   │   ├── report_generator.py     # HTML report builder
│   │   └── file_reader.py          # CSV / Excel reader
│   ├── uploads/              # Uploaded datasets
│   ├── saved_models/         # Persisted trained models (.pkl)
│   └── predictions/          # Generated prediction CSVs
│
└── frontend/                 # React + Vite frontend
    └── src/
        ├── App.jsx           # Entire UI — all studios live here
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

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at → `http://localhost:5173`

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload & profile a dataset |
| `POST` | `/api/recommend-target` | AI target variable suggestion |
| `POST` | `/api/train-model` | Train an ML model |
| `POST` | `/api/compare-models` | AutoML leaderboard |
| `POST` | `/api/predict-test-data` | Run predictions with a saved model |
| `GET`  | `/api/download-predictions/{file}` | Download prediction CSV |
| `POST` | `/api/chat-analyst` | Ask the AI data scientist |
| `POST` | `/api/detect-leakage` | Data leakage detection |
| `POST` | `/api/generate-report` | Generate HTML intelligence report |
| `GET`  | `/api/download-report/{file}` | Download the report |

Full interactive docs at `/docs` (FastAPI Swagger UI).

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
- [scikit-learn](https://scikit-learn.org/) — ML algorithms
- [pandas](https://pandas.pydata.org/) — data manipulation
- [numpy](https://numpy.org/) — numerical computing
- [uvicorn](https://www.uvicorn.org/) — ASGI server
- [joblib](https://joblib.readthedocs.io/) — model persistence

**Frontend**
- [React 19](https://react.dev/) — UI framework
- [Vite 8](https://vitejs.dev/) — blazing fast build tool
- [Recharts](https://recharts.org/) — chart library
- [Axios](https://axios-http.com/) — HTTP client

---

## 🗺️ Roadmap

- [ ] Real authentication (JWT / OAuth)
- [ ] Database persistence for sessions and models
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
