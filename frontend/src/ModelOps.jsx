// Serving and monitoring for one model, shown in MLOps Studio's model details:
// try the prediction API with a real record, and check recent inputs for drift.

import { useEffect, useState } from "react";
import { API_BASE_URL, apiFetch, downloadArtifact, postJson } from "./api";

const STATUS_LABEL = { stable: "Stable", warning: "Warning", drift: "Drift detected" };

function formatScore(row) {
  const score = row.score < 0.0001 && row.score > 0 ? row.score.toExponential(1) : Number(row.score.toFixed(4));
  return `${row.score_kind} ${score}`;
}

export default function ModelOps({ model, stage }) {
  const modelPath = `/api/models/${encodeURIComponent(model.model_id)}`;
  const target = model.metadata?.target_column;

  const [info, setInfo] = useState(null);
  const [infoError, setInfoError] = useState("");

  const [recordText, setRecordText] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState("");
  const [predicting, setPredicting] = useState(false);

  const [window_, setWindow] = useState("latest");
  const [drift, setDrift] = useState(null);
  const [driftError, setDriftError] = useState("");
  const [driftLoading, setDriftLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);

  const loadInfo = () =>
    apiFetch(`${modelPath}/serving`)
      .then((data) => {
        setInfo(data);
        return data;
      })
      .catch((error) => setInfoError(error.message || "Could not load serving details."));

  useEffect(() => {
    loadInfo().then((data) => {
      if (data?.example_record) setRecordText(JSON.stringify(data.example_record, null, 2));
    });
    // Runs once per model: the parent gives this component a key of the model id
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const predict = async () => {
    setPredicting(true);
    setPredictError("");
    setPrediction(null);

    try {
      let record;
      try {
        record = JSON.parse(recordText);
      } catch {
        throw new Error("The record is not valid JSON.");
      }
      const records = Array.isArray(record) ? record : [record];
      setPrediction(await postJson(`${modelPath}/predict`, { records }));
      loadInfo();
    } catch (error) {
      setPredictError(error.message || "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  };

  const checkDrift = async () => {
    setDriftLoading(true);
    setDriftError("");
    setDrift(null);

    try {
      setDrift(await apiFetch(`${modelPath}/drift?window=${window_}`));
    } catch (error) {
      setDriftError(error.message || "Drift check failed.");
    } finally {
      setDriftLoading(false);
    }
  };

  const downloadReport = async () => {
    setReportLoading(true);
    setDriftError("");

    try {
      const data = await apiFetch(`${modelPath}/drift/report?window=${window_}`, { method: "POST" });
      await downloadArtifact(data.artifact_id);
    } catch (error) {
      setDriftError(error.message || "Could not build the report.");
    } finally {
      setReportLoading(false);
    }
  };

  const logged = info?.predictions_logged;
  const endpoint = `${API_BASE_URL}${modelPath}/predict`;
  const productionEndpoint = `${API_BASE_URL}/api/predict?target=${encodeURIComponent(target || "")}`;

  return (
    <>
      <h4>Prediction API</h4>

      {infoError && <p className="ops-error">{infoError}</p>}

      <div className="ops-endpoint">
        <span>POST</span>
        <code>{endpoint}</code>
      </div>

      {stage === "production" ? (
        <div className="ops-endpoint">
          <span>POST</span>
          <code>{productionEndpoint}</code>
          <em>always the production model for {target}</em>
        </div>
      ) : (
        <p className="ops-hint">
          Promote this model to production and it also answers at <code>/api/predict?target={target}</code>,
          so callers never need a model id.
        </p>
      )}

      <label className="ops-label" htmlFor="ops-record">
        Try it: one record as JSON (a real row from the training data is filled in)
      </label>
      <textarea
        id="ops-record"
        className="ops-textarea"
        value={recordText}
        onChange={(event) => setRecordText(event.target.value)}
        rows={Math.min(14, Math.max(5, recordText.split("\n").length))}
        spellCheck={false}
      />

      <div className="ops-actions">
        <button className="primary" type="button" onClick={predict} disabled={predicting || !recordText.trim()}>
          {predicting ? "Predicting…" : "Predict"}
        </button>
      </div>

      {predictError && <p className="ops-error">{predictError}</p>}

      {prediction && (
        <div className="ops-result">
          <span>Predicted {prediction.target_column}</span>
          <strong>{String(prediction.predictions[0])}</strong>
          {prediction.probabilities?.[0] && (
            <div className="ops-probabilities">
              {Object.entries(prediction.probabilities[0]).map(([label, probability]) => (
                <div className="ops-probability" key={label}>
                  <span>{label}</span>
                  <div className="ops-bar">
                    <div style={{ width: `${Math.round(probability * 100)}%` }} />
                  </div>
                  <em>{(probability * 100).toFixed(1)}%</em>
                </div>
              ))}
            </div>
          )}
          {prediction.rows > 1 && <p className="ops-hint">{prediction.rows} records scored; showing the first.</p>}
        </div>
      )}

      <h4>Drift monitoring</h4>

      <p className="ops-hint">
        {logged
          ? logged.rows === 0
            ? "No predictions logged yet. Use the API above or Prediction Studio, then check for drift."
            : `${logged.rows} predictions logged in ${logged.batches} batch${logged.batches === 1 ? "" : "es"} ` +
              `(${logged.api_rows} via the API, ${logged.file_rows} from files). ` +
              "Their inputs are compared with the data this model was trained on."
          : "Loading…"}
      </p>

      <div className="ops-actions">
        <select value={window_} onChange={(event) => setWindow(event.target.value)} aria-label="Which predictions to check">
          <option value="latest">Most recent predictions</option>
          <option value="all">All logged predictions</option>
        </select>
        <button className="primary" type="button" onClick={checkDrift} disabled={driftLoading || !logged?.rows}>
          {driftLoading ? "Checking…" : "Check drift"}
        </button>
      </div>

      {driftError && <p className="ops-error">{driftError}</p>}

      {drift && (
        <>
          <div className={`ops-verdict ops-${drift.status}`}>
            <strong>{STATUS_LABEL[drift.status]}</strong>
            <p>{drift.verdict}</p>
          </div>

          <p className="ops-hint">
            {drift.current_rows} recent rows against {drift.reference_rows} training rows.
            {drift.skipped_columns.length > 0 && ` Date columns are not checked (${drift.skipped_columns.join(", ")}).`}
          </p>

          <div className="table-wrapper">
            <table className="preview-table compact ops-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Test</th>
                  <th>Score</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {[...(drift.prediction_drift ? [{ ...drift.prediction_drift, column: "Model's predictions" }] : []), ...drift.columns].map(
                  (row) => (
                    <tr key={row.column} className={row.drifted ? "drifted" : ""}>
                      <td>{row.column}</td>
                      <td>{row.method.replace(" p_value", "")}</td>
                      <td>{formatScore(row)}</td>
                      <td>{row.drifted ? "Drifted" : "Stable"}</td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>

          <p className="ops-hint">
            Each column is tested on its own: a distance above its threshold, or a p-value below it, means the
            recent values are distributed differently from the training values. P-value thresholds are tightened
            for the number of columns checked, so a column is not flagged by chance. Drift is a warning to look,
            not a measure of accuracy.
          </p>

          <div className="ops-actions">
            <button className="ghost-btn" type="button" onClick={downloadReport} disabled={reportLoading}>
              {reportLoading ? "Building report…" : "Download full Evidently report"}
            </button>
          </div>
        </>
      )}
    </>
  );
}
