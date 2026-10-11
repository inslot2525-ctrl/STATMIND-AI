// Result views for ML Studio: the trained-model report, the AutoML leaderboard
// and the leakage report. Presentational only; App.jsx owns the data.

import {
  BarChart,
  Bar,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Categorical colors for per-class ROC curves, in fixed order. Only four are
// used: beyond that the per-class table carries the detail.
const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"];
const MAX_ROC_CURVES = SERIES_COLORS.length;

const axisTick = { fontSize: 11, fill: "#a3a3a3" };

const tooltipProps = {
  contentStyle: {
    background: "#0a0a0a",
    border: "1px solid rgba(255, 255, 255, 0.12)",
    borderRadius: 12,
    boxShadow: "0 12px 32px rgba(0, 0, 0, 0.6)",
  },
  labelStyle: { color: "#ffffff", fontWeight: 700 },
  itemStyle: { color: "#f2f2f2" },
  cursor: { fill: "rgba(255, 255, 255, 0.04)" },
};

const METRIC_LABELS = {
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  precision_weighted: "Precision (weighted)",
  recall_weighted: "Recall (weighted)",
  f1_weighted: "F1 (weighted)",
  f1_macro: "F1 (macro)",
  roc_auc: "ROC AUC",
  mae: "MAE",
  rmse: "RMSE",
  r2_score: "R²",
  r2: "R²",
};

const metricLabel = (key) => METRIC_LABELS[key] || key.replaceAll("_", " ");

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return value ?? "N/A";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return String(Number(value.toFixed(4)));
}

function ChartCard({ title, note, full, children }) {
  return (
    <div className={`chart-card${full ? " full" : ""}`}>
      <div className="chart-head">
        <h4>{title}</h4>
        {note && <span>{note}</span>}
      </div>
      {children}
    </div>
  );
}

// Each metric with the feature-blind baseline's value underneath, when there is one
function MetricTiles({ metrics, baseline }) {
  const entries = Object.entries(metrics || {}).filter(([, value]) => typeof value === "number");

  if (!entries.length) {
    return <div className="empty-small">No metrics available.</div>;
  }

  return (
    <div className="summary-grid">
      {entries.map(([key, value]) => (
        <div className="summary-card" key={key}>
          <span>{metricLabel(key)}</span>
          <strong>{formatNumber(value)}</strong>
          {typeof baseline?.[key] === "number" && (
            <em className="baseline-note">baseline {formatNumber(baseline[key])}</em>
          )}
        </div>
      ))}
    </div>
  );
}

function ImportanceChart({ items }) {
  // Longest label decides the left gutter so names are never clipped
  const longest = Math.max(...items.map((item) => item.feature.length));
  const gutter = Math.min(240, Math.max(90, longest * 7));

  return (
    <ChartCard
      title="Feature importance"
      note="Permutation · score drop when shuffled"
      full
    >
      <ResponsiveContainer width="100%" height={Math.max(220, items.length * 30 + 50)}>
        <BarChart data={items} layout="vertical" margin={{ left: 8, right: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" horizontal={false} />
          <XAxis type="number" tick={axisTick} />
          <YAxis type="category" dataKey="feature" tick={axisTick} width={gutter} interval={0} />
          <Tooltip
            {...tooltipProps}
            formatter={(value, _name, entry) => [
              `${formatNumber(value)} ± ${formatNumber(entry.payload.std)}`,
              "Importance",
            ]}
          />
          <ReferenceLine x={0} stroke="#525252" />
          <Bar dataKey="importance" fill="#e5e5e5" radius={[0, 4, 4, 0]} barSize={14} isAnimationActive={false}>
            <ErrorBar dataKey="std" direction="x" width={5} stroke="#8a8a8a" strokeWidth={1.5} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function RocChart({ curves }) {
  const shown = curves.slice(0, MAX_ROC_CURVES);
  const single = shown.length === 1;

  return (
    <ChartCard
      title="ROC curve"
      note={
        single
          ? `${shown[0].label} · AUC ${formatNumber(shown[0].auc)}`
          : curves.length > shown.length
            ? `One vs rest · first ${shown.length} of ${curves.length} classes`
            : "One vs rest"
      }
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart margin={{ left: 0, right: 16, bottom: 18 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
          <XAxis
            type="number"
            dataKey="fpr"
            domain={[0, 1]}
            tick={axisTick}
            label={{ value: "False positive rate", position: "insideBottom", offset: -10, fill: "#8a8a8a", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="tpr"
            domain={[0, 1]}
            tick={axisTick}
            label={{ value: "True positive rate", angle: -90, position: "insideLeft", offset: 16, fill: "#8a8a8a", fontSize: 11 }}
          />
          <Tooltip
            {...tooltipProps}
            cursor={{ stroke: "rgba(255, 255, 255, 0.25)", strokeDasharray: "3 3" }}
            labelFormatter={(value) => `FPR ${formatNumber(value)}`}
            formatter={(value, name) => [`TPR ${formatNumber(value)}`, name]}
          />
          {!single && <Legend verticalAlign="top" height={30} wrapperStyle={{ color: "#c2c2c2", fontSize: 12 }} />}
          {/* Chance line: a model with no skill */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke="#525252"
            strokeDasharray="4 4"
          />
          {shown.map((curve, index) => (
            <Line
              key={curve.label}
              data={curve.points}
              dataKey="tpr"
              name={single ? curve.label : `${curve.label} (AUC ${formatNumber(curve.auc)})`}
              type="linear"
              stroke={single ? "#e5e5e5" : SERIES_COLORS[index]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function ConfusionMatrix({ labels, matrix }) {
  const largest = Math.max(1, ...matrix.flat());

  return (
    <ChartCard title="Confusion matrix" note="Rows actual · columns predicted">
      <div className="table-wrapper">
        <table className="confusion-table">
          <thead>
            <tr>
              <th scope="col">Actual ↓ / Predicted →</th>
              {labels.map((label) => (
                <th scope="col" key={label}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={labels[rowIndex]}>
                <th scope="row">{labels[rowIndex]}</th>
                {row.map((count, colIndex) => (
                  <td
                    key={colIndex}
                    className={rowIndex === colIndex ? "diagonal" : ""}
                    style={{ background: `rgba(255, 255, 255, ${0.04 + 0.5 * (count / largest)})` }}
                    title={`Actual ${labels[rowIndex]}, predicted ${labels[colIndex]}: ${count}`}
                  >
                    {count}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}

function PerClassTable({ rows }) {
  return (
    <ChartCard title="Per-class scores" note="Test split" full>
      <div className="table-wrapper">
        <table className="preview-table compact">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Test rows</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{formatNumber(row.precision)}</td>
                <td>{formatNumber(row.recall)}</td>
                <td>{formatNumber(row.f1)}</td>
                <td>{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}

function PredictedVsActual({ points }) {
  const values = points.flatMap((point) => [point.actual, point.predicted]);
  const low = Math.min(...values);
  const high = Math.max(...values);

  return (
    <ChartCard title="Predicted vs actual" note="Test split · dashed line is a perfect prediction" full>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ left: 0, right: 16, bottom: 18 }}>
          <CartesianGrid stroke="#262626" />
          <XAxis
            type="number"
            dataKey="actual"
            name="Actual"
            domain={[low, high]}
            tick={axisTick}
            tickFormatter={formatNumber}
            label={{ value: "Actual", position: "insideBottom", offset: -10, fill: "#8a8a8a", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="predicted"
            name="Predicted"
            domain={[low, high]}
            tick={axisTick}
            tickFormatter={formatNumber}
            label={{ value: "Predicted", angle: -90, position: "insideLeft", offset: 16, fill: "#8a8a8a", fontSize: 11 }}
          />
          <Tooltip
            {...tooltipProps}
            cursor={{ stroke: "rgba(255, 255, 255, 0.25)", strokeDasharray: "3 3" }}
            formatter={(value, name) => [formatNumber(value), name]}
          />
          <ReferenceLine
            segment={[{ x: low, y: low }, { x: high, y: high }]}
            stroke="#737373"
            strokeDasharray="4 4"
          />
          <Scatter data={points} fill="#e5e5e5" fillOpacity={0.75} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function Notes({ items }) {
  if (!items?.length) return null;

  return (
    <ul className="note-list">
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
}

function DetailRows({ items }) {
  return (
    <div className="feature-list">
      {items.map(({ label, value }) => (
        <div className="feature-row" key={label}>
          <span>{label}</span>
          <strong>{value === null || value === undefined || value === "" ? "None" : String(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function formatParams(params) {
  const entries = Object.entries(params || {});
  if (!entries.length) return "";
  return entries.map(([name, value]) => `${name}=${value === null ? "None" : value}`).join(", ");
}

function CommitLine({ label, id, url }) {
  if (!id) return null;

  return (
    <p className="commit-line">
      {label}{" "}
      {url ? (
        <a className="text-link" href={url} target="_blank" rel="noreferrer">
          <code>{id.slice(0, 12)}</code>
        </a>
      ) : (
        <code>{id.slice(0, 12)}</code>
      )}
    </p>
  );
}

export function ModelReport({ result }) {
  const diagnostics = result.diagnostics || {};
  const crossValidation = result.cross_validation;
  const confusion = diagnostics.confusion_matrix;

  return (
    <>
      <div className="summary-grid">
        <div className="summary-card">
          <span>Task type</span>
          <strong>{result.task_type}</strong>
        </div>
        <div className="summary-card">
          <span>Algorithm</span>
          <strong>{result.algorithm}</strong>
        </div>
        <div className="summary-card">
          <span>Train / test rows</span>
          <strong>{result.train_rows} / {result.test_rows}</strong>
        </div>
        <div className="summary-card">
          <span>{crossValidation ? `${crossValidation.folds}-fold CV ${metricLabel(crossValidation.metric)}` : "Target"}</span>
          <strong>
            {crossValidation
              ? `${formatNumber(crossValidation.mean)} ± ${formatNumber(crossValidation.std)}`
              : result.target_column}
          </strong>
        </div>
      </div>

      <div className="card">
        <h3>Saved model</h3>
        <div className="result-box">
          <span>Model ID</span>
          <strong>{result.model_id}</strong>
        </div>
        {result.mlflow && (
          <div className="result-box">
            <span>MLflow registry</span>
            <strong>
              {result.mlflow.registered_model}
              {result.mlflow.version && ` · version ${result.mlflow.version}`}
            </strong>
            {result.mlflow.run_url && (
              <a className="text-link" href={result.mlflow.run_url} target="_blank" rel="noreferrer">
                Open this run in MLflow
              </a>
            )}
          </div>
        )}
        {result.data_version && (
          <div className="result-box">
            <span>Data version (lakeFS)</span>
            <strong>
              {result.data_version.repository} @ {result.data_version.branch}
            </strong>
            <CommitLine
              label="Trained on data commit"
              id={result.data_version.dataset_commit}
              url={result.data_version.dataset_commit_url}
            />
            <CommitLine
              label="Model stored in commit"
              id={result.data_version.model_commit}
              url={result.data_version.model_commit_url}
            />
          </div>
        )}
        <Notes items={result.model_explanation} />
      </div>

      <div className="card">
        <h3>Test-split metrics</h3>
        <MetricTiles metrics={result.metrics} baseline={result.baseline_metrics} />
        <p className="card-footnote">
          Baseline is a model that ignores every feature
          {result.task_type === "classification"
            ? " and always predicts the class mix of the training data."
            : " and always predicts the training average."}
        </p>
      </div>

      <div className="card">
        <h3>Diagnostics</h3>
        <div className="chart-grid">
          {!!result.feature_importance?.length && <ImportanceChart items={result.feature_importance} />}
          {!!diagnostics.roc_curves?.length && <RocChart curves={diagnostics.roc_curves} />}
          {confusion && confusion.labels.length <= 12 && (
            <ConfusionMatrix labels={confusion.labels} matrix={confusion.matrix} />
          )}
          {!!diagnostics.per_class?.length && <PerClassTable rows={diagnostics.per_class} />}
          {!!diagnostics.predicted_vs_actual?.length && (
            <PredictedVsActual points={diagnostics.predicted_vs_actual} />
          )}
        </div>
      </div>

      <div className="card">
        <h3>Training details</h3>
        <DetailRows
          items={[
            { label: "Target", value: result.target_column },
            { label: "Test size", value: result.test_size },
            { label: "Numeric features", value: result.numeric_features?.join(", ") },
            { label: "Categorical features", value: result.categorical_features?.join(", ") },
            { label: "Date features", value: result.datetime_features?.join(", ") },
            { label: "Dropped ID-like columns", value: result.dropped_high_cardinality_columns?.join(", ") },
            { label: "Hyperparameters", value: formatParams(result.hyperparameters) },
          ]}
        />
        <Notes items={result.warnings} />
      </div>
    </>
  );
}

export function Leaderboard({ result }) {
  const rows = result.leaderboard || [];

  if (!rows.length) {
    return <div className="empty-small">No comparison results available.</div>;
  }

  return (
    <div className="card">
      <h3>AutoML leaderboard</h3>

      <div className="summary-grid">
        <div className="summary-card">
          <span>Best model</span>
          <strong>{result.best_model}</strong>
        </div>
        <div className="summary-card">
          <span>Ranked by</span>
          <strong>{result.cv_folds}-fold CV {metricLabel(result.ranking_metric)}</strong>
        </div>
        <div className="summary-card">
          <span>Rows cross-validated</span>
          <strong>{result.rows_used_for_compare}</strong>
        </div>
        <div className="summary-card">
          <span>Held-out test rows</span>
          <strong>{result.test_rows}</strong>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="preview-table compact">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Algorithm</th>
              <th>CV score (mean ± std)</th>
              <th>Tuning</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr
                key={row.algorithm}
                className={row.algorithm === result.best_model ? "winner" : row.is_baseline ? "baseline" : ""}
              >
                <td>{index + 1}</td>
                <td>
                  {row.algorithm}
                  {row.algorithm === result.best_model && <span className="row-tag">Winner</span>}
                  {row.is_baseline && <span className="row-tag muted">Ignores features</span>}
                </td>
                <td>
                  {row.score === null || row.score === undefined
                    ? `Failed: ${row.error || "unknown error"}`
                    : `${formatNumber(row.cv_mean)} ± ${formatNumber(row.cv_std)}`}
                </td>
                <td>
                  {row.tuned
                    ? formatParams(row.best_params)
                    : row.tuning
                      ? "Defaults were best"
                      : "—"}
                </td>
                <td>{row.seconds === undefined ? "—" : `${row.seconds}s`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Notes items={result.warnings} />
    </div>
  );
}

const SEVERITY_CLASS = { Critical: "critical", High: "high", Medium: "medium" };

export function LeakageReport({ result }) {
  const solo = result.single_feature_scores;

  return (
    <div className="card">
      <h3>Leakage check</h3>

      <div className="summary-grid">
        <div className="summary-card">
          <span>Risk level</span>
          <strong>{result.risk_level}</strong>
        </div>
        <div className="summary-card">
          <span>Risk score</span>
          <strong>{result.leakage_risk_score} / 100</strong>
        </div>
        <div className="summary-card">
          <span>Findings</span>
          <strong>{result.findings.length}</strong>
        </div>
        <div className="summary-card">
          <span>Rows checked</span>
          <strong>{result.rows_checked}</strong>
        </div>
      </div>

      {result.findings.length === 0 ? (
        <div className="empty-small">No leakage pattern was found by these checks.</div>
      ) : (
        <div className="finding-list">
          {result.findings.map((finding, index) => (
            <div className="finding" key={index}>
              <div className="finding-head">
                <span className={`severity ${SEVERITY_CLASS[finding.severity] || "medium"}`}>
                  {finding.severity}
                </span>
                <strong>{finding.feature}</strong>
              </div>
              <p>{finding.message}</p>
              <p className="finding-evidence">{finding.evidence}</p>
              <p className="finding-evidence">{finding.recommendation}</p>
            </div>
          ))}
        </div>
      )}

      {!!solo?.top?.length && (
        <>
          <h4>How well each feature predicts the target alone</h4>
          <p className="card-footnote">
            Cross-validated {solo.metric.replaceAll("_", " ")} of a shallow tree given one feature.
            A value near 1 means the feature gives the answer away.
          </p>
          <DetailRows
            items={solo.top.map(([feature, score]) => ({ label: feature, value: formatNumber(score) }))}
          />
        </>
      )}

      <Notes items={result.warnings} />
    </div>
  );
}
