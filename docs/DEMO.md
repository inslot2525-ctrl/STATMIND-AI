# Demo script (about 6 minutes)

A path through the app that shows every major piece once, using the three files in `demo/`:

| File | What it is |
|---|---|
| `customer_churn.csv` | 1,500 customers with a `churned` column. Train on this. |
| `customer_churn_new_customers.csv` | 200 later customers who look like the training data. |
| `customer_churn_drifted.csv` | 300 later customers after prices rose, usage fell, and most moved to monthly contracts in the North. |

The data is synthetic. Churn is driven by contract type, tenure, support tickets, late payments,
usage and price, with noise, so a good model reaches about 0.81 accuracy against a 0.69 baseline.

Start everything with `.\start-dev.ps1`, then open http://localhost:5173.

## 1. The problem (30 s)

On the **dashboard**: "Upload a spreadsheet, get a model you can check." Point at the four steps,
then press **Try the app**.

## 2. Profile the data (45 s) — Statistics Studio

Upload `customer_churn.csv`, **Run Analysis**.

- 1,500 rows, 14 columns, quality score, missing values found.
- Charts are computed on the server; the raw data never goes to the browser.

## 3. Check for leakage (45 s) — ML Studio

Choose target `churned`, press **Check Leakage**.

- Risk is Low. `customer_id` is flagged as an identifier, and it is dropped automatically in training.
- The table shows how well each column predicts churn *alone*. Nothing is near 1.0, so nothing
  gives the answer away.
- Say: "This is a heuristic. It tells me where to look; it does not prove the data is clean."

## 4. Run AutoML (90 s) — ML Studio

Press **Run AutoML** and talk over the progress bar:

- The test split is set aside first. Every algorithm is 5-fold cross-validated on the rest.
- The top two are tuned. The winner is scored **once** on the held-out split.

When it finishes:

- **Leaderboard**: mean ± spread per model, and the `baseline` row that ignores the features.
- **Test metrics**: each one with the baseline's value underneath. "0.81 accuracy is only meaningful
  next to 0.69."
- **Diagnostics**: permutation importance (contract type, usage and support tickets lead, which is
  how the data was generated), ROC curve, confusion matrix.
- **Saved model**: the MLflow version, and the lakeFS commits for the data and the model.
  "This model can be traced to the exact bytes it was trained on."

## 5. Serve it (60 s) — MLOps Studio → Models → Serve & monitor

- Press **Production**. Point at the second endpoint: `/api/predict?target=churned` now answers with
  this model, and promoting another model would switch it with no change for the caller.
- Press **Predict** on the pre-filled record: a prediction with class probabilities.
- Optional, in a terminal:

  ```powershell
  curl.exe -X POST "http://127.0.0.1:8000/api/predict?target=churned" -H "Content-Type: application/json" `
    -d '{\"records\":[{\"signup_date\":\"2024-08-15\",\"age\":25,\"gender\":\"Male\",\"region\":\"West\",\"plan\":\"Premium\",\"contract_type\":\"Monthly\",\"payment_method\":\"Credit Card\",\"tenure_months\":4,\"monthly_charges\":954.61,\"avg_weekly_usage_hours\":15.9,\"support_tickets\":2,\"late_payments\":2}]}'
  ```

## 6. Catch drift (90 s) — Prediction Studio, then back to the model

1. **Prediction Studio**: upload `customer_churn_new_customers.csv`, **Generate Predictions**.
   Back in the model's details, **Check drift** → **Stable**.
2. **Prediction Studio**: upload `customer_churn_drifted.csv`, **Generate Predictions**.
   **Check drift** again → a warning naming exactly four columns: `monthly_charges`,
   `avg_weekly_usage_hours`, `contract_type`, `region`, plus a shift in the model's own predictions.
3. **Download full Evidently report** for the per-column distributions.

Say: "Drift is a warning, not an accuracy number. I would need the real outcomes to measure
accuracy; this tells me the model is now being used on data it was not trained on."

## 7. Close (30 s)

- **MLflow** (http://127.0.0.1:5000): the AutoML run with every candidate as a child run.
- What is not there yet, said plainly: automatic retraining, SHAP per prediction, data larger than memory.

## If something goes wrong

| Problem | Fix |
|---|---|
| A page says it cannot reach the server | Re-run `.\start-dev.ps1`; it restarts only what is down |
| Check drift says too few rows | Make predictions with that model first; it needs 30 rows |
| AI Analyst says built-in rules answered | `ANTHROPIC_API_KEY` is not set; that is the fallback working |
| No MLflow or lakeFS links on the model | Those services were not running when the model was trained |
