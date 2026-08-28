"""
Benchmark leaderboard: trains all 3 datasets and prints a score table.
Where to check scores:
  - pytest output (this file) — run `pytest tests/test_benchmark.py -v -s`
  - FastAPI: GET /api/experiments/ and GET /api/models/ (MLOps Studio → Experiments tab)
  - DB: backend/statmind.db → experiments table
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.ml_engine import train_model_from_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

BENCHMARKS = [
    ("iris.csv", "target", "classification", 0.85),
    ("breast_cancer.csv", "target", "classification", 0.90),
    ("diabetes.csv", "target", "regression", 0.30),
]

def test_benchmark_leaderboard():
    print("\n" + "="*60)
    print("BENCHMARK LEADERBOARD — check these scores for correctness")
    print("="*60)
    print(f"{'Dataset':<18} {'Task':<15} {'Metric':<12} {'Score':<8} {'Threshold':<9} {'Status'}")
    print("-"*60)
    all_pass = True
    for fname, target, task, thresh in BENCHMARKS:
        path = os.path.join(DATA_DIR, fname)
        result = train_model_from_file(path, target_column=target, algorithm="random_forest", test_size=0.2)
        m = result["metrics"]
        if task == "classification":
            metric_name = "accuracy" if "accuracy" in m else "f1_score"
            score = m.get(metric_name, 0)
        else:
            metric_name = "r2_score" if "r2_score" in m else "r2"
            score = m.get(metric_name, 0)
        status = "PASS" if score > thresh else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{fname:<18} {task:<15} {metric_name:<12} {score:<8.4f} {thresh:<9} {status}")
        print(f"  full metrics: {m}")
    print("="*60)
    print("Where to check scores after this run:")
    print("  1. This pytest output (above)")
    print("  2. App -> MLOps Studio -> Experiments tab (GET /api/experiments/)")
    print("  3. curl http://localhost:8000/api/experiments/ | jq")
    print("  4. DB: sqlite3 backend/statmind.db 'SELECT id,algorithm,metrics FROM experiments;'")
    print("="*60)
    assert all_pass, "One or more benchmarks below threshold — check metrics above"
