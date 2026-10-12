import { useState, useEffect, useMemo, memo, lazy, Suspense } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import "./App.css";
import "./polish.css";
import AuthScreen from "./AuthScreen";
import AnalystPanel from "./AnalystPanel";
import { installSpotlight } from "./spotlight";
import { Leaderboard, LeakageReport, ModelReport } from "./MlResults";
import ModelOps from "./ModelOps";
import { CategoricalStats, ColumnOverview, NumericStats } from "./StatsTables";
import {
  apiFetch,
  clearSession,
  downloadArtifact,
  getStoredUser,
  getToken,
  onUnauthorized,
  postForm,
  postJson,
  runJob,
} from "./api";

// Shared presentational pieces. They live at module scope so their identity is
// stable across renders; inputs rendered inside them keep focus while typing.
function Card({ title, children }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function Empty({ title, text }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function Summary({ label, value, small }) {
  return (
    <div className="summary-card">
      <span>{label}</span>
      <strong>{value ?? "N/A"}</strong>
      {small && <small>{small}</small>}
    </div>
  );
}

function displayValue(value) {
  if (Array.isArray(value)) return JSON.stringify(value);
  if (value && typeof value === "object") return JSON.stringify(value);
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value);
}

function MetricGrid({ metrics }) {
  const entries = Object.entries(metrics || {});

  if (!entries.length) {
    return <div className="empty-small">No metrics available.</div>;
  }

  return (
    <div className="summary-grid">
      {entries.map(([key, value]) => (
        <Summary
          label={key.replaceAll("_", " ")}
          value={displayValue(value)}
          key={key}
        />
      ))}
    </div>
  );
}

function PreviewTable({ rows }) {
  if (!rows?.length) {
    return <div className="empty-small">No preview rows returned.</div>;
  }

  const columns = Object.keys(rows[0] || {});

  return (
    <div className="table-wrapper">
      <table className="preview-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column) => (
                <td key={column}>{displayValue(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InsightList({ items }) {
  if (!items?.length) {
    return <div className="empty-small">No insights available.</div>;
  }

  return (
    <div className="insight-list">
      {items.map((item, index) => (
        <div className="insight-item" key={index}>
          <span>{index + 1}</span>
          <p>{item}</p>
        </div>
      ))}
    </div>
  );
}

const pieColors = [
  "#e5e5e5",
  "#a3a3a3",
  "#ffffff",
  "#737373",
  "#d4d4d4",
  "#525252",
  "#f5f5f5",
  "#404040",
];

const tooltipProps = {
  contentStyle: {
    background: "#0a0a0a",
    border: "1px solid rgba(255, 255, 255, 0.12)",
    borderRadius: 12,
    boxShadow: "0 12px 32px rgba(0, 0, 0, 0.6)",
  },
  labelStyle: { color: "#ffffff", fontWeight: 700 },
  itemStyle: { color: "#f5efe0" },
  cursor: { fill: "rgba(255, 255, 255, 0.04)" },
};

function Header({ title, subtitle, onBack }) {
  return (
    <div className="page-header">
      <div>
        <p className="eyebrow">{title}</p>
        <h2>{subtitle}</h2>
      </div>

      <button className="back-btn" onClick={onBack}>
        Back to Dashboard
      </button>
    </div>
  );
}

const BarBlock = memo(function BarBlock({ chart, label = "Bar" }) {
  const data = useMemo(() => chart.data, [chart.data]);
  return (
    <div className="chart-card">
      <div className="chart-head">
        <h4>{chart.title}</h4>
        <span>{label}</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
          <XAxis dataKey={chart.x_axis} tick={{ fontSize: 11, fill: "#a3a3a3" }} interval={0} angle={data.length > 6 ? -20 : 0} dy={data.length > 6 ? 10 : 0} height={data.length > 6 ? 60 : 30} />
          <YAxis tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <Tooltip {...tooltipProps} />
          <Bar dataKey={chart.y_axis} radius={[8, 8, 0, 0]} fill="#e5e5e5" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

const LineBlock = memo(function LineBlock({ chart }) {
  return (
    <div className="chart-card">
      <div className="chart-head">
        <h4>{chart.title}</h4>
        <span>Line</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chart.data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
          <XAxis dataKey={chart.x_axis} tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <YAxis tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <Tooltip {...tooltipProps} />
          <Line type="monotone" dataKey={chart.y_axis} stroke="#737373" strokeWidth={3} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});

const AreaBlock = memo(function AreaBlock({ chart }) {
  return (
    <div className="chart-card">
      <div className="chart-head"><h4>{chart.title}</h4><span>Area</span></div>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={chart.data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
          <XAxis dataKey={chart.x_axis} tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <YAxis tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <Tooltip {...tooltipProps} />
          <Area type="monotone" dataKey={chart.y_axis} stroke="#e5e5e5" fill="#e5e5e5" fillOpacity={0.25} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
});

const PieBlock = memo(function PieBlock({ chart }) {
  return (
    <div className="chart-card">
      <div className="chart-head"><h4>{chart.title}</h4><span>Pie</span></div>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie data={chart.data} dataKey={chart.value_key} nameKey={chart.name_key} outerRadius={100} label isAnimationActive={false}>
            {chart.data.map((_, index) => (<Cell key={index} fill={pieColors[index % pieColors.length]} />))}
          </Pie>
          <Tooltip {...tooltipProps} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
});

const HistBlock = memo(function HistBlock({ chart }) {
  return (
    <div className="chart-card">
      <div className="chart-head"><h4>{chart.title}</h4><span>Histogram</span></div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chart.data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
          <XAxis dataKey="range" tick={{ fontSize: 10, fill: "#a3a3a3" }} interval={0} angle={-15} dy={8} height={50} />
          <YAxis tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <Tooltip {...tooltipProps} />
          <Bar dataKey="count" radius={[8, 8, 0, 0]} fill="#a3a3a3" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

const ScatterBlock = memo(function ScatterBlock({ chart }) {
  return (
    <div className="chart-card">
      <div className="chart-head"><h4>{chart.title}</h4><span>Scatter</span></div>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart>
          <CartesianGrid stroke="#262626" />
          <XAxis type="number" dataKey={chart.x_axis} name={chart.x_axis} tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <YAxis type="number" dataKey={chart.y_axis} name={chart.y_axis} tick={{ fontSize: 11, fill: "#a3a3a3" }} />
          <Tooltip {...tooltipProps} cursor={{ stroke: "rgba(255, 255, 255, 0.25)", strokeDasharray: "3 3" }} />
          <Scatter data={chart.data} fill="#22c55e" isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
});

const Heatmap = memo(function Heatmap({ result }) {
  const heatmap = result?.charts?.correlation_heatmap;
  if (!heatmap?.data?.length) return <div className="empty-small">Not enough numeric columns for correlation.</div>;
  return (
    <div className="chart-card full">
      <div className="chart-head"><h4>{heatmap.title}</h4><span>Correlation • capped to 8 columns for speed</span></div>
      <div className="heatmap-grid">
        {heatmap.data.map((cell, index) => {
          const intensity = Math.min(Math.abs(cell.value), 1);
          const background = cell.value >= 0 ? `rgba(255, 255, 255, ${0.12 + intensity * 0.55})` : `rgba(224, 122, 45, ${0.12 + intensity * 0.55})`;
          return (<div className="heatmap-cell" style={{ background }} key={index}><span>{cell.y} × {cell.x}</span><strong>{cell.value}</strong></div>);
        })}
      </div>
    </div>
  );
});

function App() {
  const [user, setUser] = useState(() => (getToken() ? getStoredUser() : null));
  const [authNotice, setAuthNotice] = useState("");

  // Card borders light up under the pointer (see polish.css)
  useEffect(() => installSpotlight(), []);
  // null until the server says whether it wants a login (it does not in local development)
  const [authRequired, setAuthRequired] = useState(null);

  useEffect(() => {
    onUnauthorized(() => {
      setUser(null);
      setAuthNotice("Your session has expired. Please log in again.");
    });

    apiFetch("/api/auth/config")
      .then(async (authConfig) => {
        if (!authConfig.auth_required) setUser(await apiFetch("/api/auth/me"));
        setAuthRequired(authConfig.auth_required);
      })
      .catch(() => setAuthRequired(true));
  }, []);

  const [currentPage, setCurrentPage] = useState("dashboard");
  const goToDashboard = () => setCurrentPage("dashboard");

  const [file, setFile] = useState(null);
  const [domain, setDomain] = useState("business");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [frontendError, setFrontendError] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [algorithm, setAlgorithm] = useState("random_forest");
  const [testSize, setTestSize] = useState("0.2");
  const [taskType, setTaskType] = useState("auto");
  const [leakageResult, setLeakageResult] = useState(null);
  const [leakageLoading, setLeakageLoading] = useState(false);
  const [mlResult, setMlResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [jobProgress, setJobProgress] = useState(null);
  const [mlError, setMlError] = useState("");
  const [modelId, setModelId] = useState("");
  const [predictionFile, setPredictionFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  const [targetSuggestion, setTargetSuggestion] = useState(null);
  const [targetSuggestionLoading, setTargetSuggestionLoading] = useState(false);
  const [targetSuggestionError, setTargetSuggestionError] = useState("");
  const [datasetId, setDatasetId] = useState("");

  // MLOps Studio state
  const [mlopsTab, setMlopsTab] = useState("experiments");
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [mlopsLoading, setMlopsLoading] = useState(false);
  const [mlopsError, setMlopsError] = useState("");
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [stageTarget, setStageTarget] = useState({});
  const [stageLoading, setStageLoading] = useState({});

  const analyze = async () => {
    if (!file) {
      alert("Upload a CSV or Excel dataset first.");
      return;
    }

    setLoading(true);
    setFrontendError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("domain", domain);

    try {
      const data = await postForm("/api/analyze", formData);

      setResult(data);
      const columns = Object.keys(data.schema?.dtypes || {});
      setTargetColumn((currentTarget) => currentTarget || columns[columns.length - 1] || "");
      setDatasetId(data.dataset_id || "");
      setTargetSuggestion(null);
      setTargetSuggestionError("");
      setLeakageResult(null);
    } catch (error) {
      console.error("Frontend analysis error:", error);
      setFrontendError(error.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const datasetColumns = Object.keys(result?.schema?.dtypes || {});
  const datasetFilename = result?.filename || "";

  const algorithmOptions = [
    ["random_forest", "Random Forest"],
    ["extra_trees", "Extra Trees"],
    ["gradient_boosting", "Gradient Boosting"],
    ["decision_tree", "Decision Tree"],
    ["knn", "KNN"],
    ["svm", "SVM"],
    ["logistic_regression", "Logistic Regression"],
    ["linear_regression", "Linear Regression"],
    ["ridge", "Ridge Regression"],
    ["lasso", "Lasso Regression"],
  ];

  const callFormEndpoint = postForm;

  const buildTrainingForm = () => {
    if (!datasetFilename) {
      throw new Error("Run Statistics Studio analysis first so ML Studio can use the uploaded dataset.");
    }

    if (!targetColumn) {
      throw new Error("Choose a target column.");
    }

    const numericTestSize = Number(testSize);

    if (!Number.isFinite(numericTestSize) || numericTestSize <= 0 || numericTestSize >= 1) {
      throw new Error("Test size must be between 0 and 1. Example: 0.2");
    }

    const formData = new FormData();
    formData.append("filename", datasetFilename);
    formData.append("target_column", targetColumn);
    formData.append("test_size", String(numericTestSize));
    if (datasetId) formData.append("dataset_id", datasetId);
    if (taskType !== "auto") formData.append("task_type", taskType);

    return formData;
  };

  // Training and AutoML run as background jobs on the server; runJob polls
  // until they finish and reports progress along the way.
  const trainModel = async () => {
    setMlLoading(true);
    setMlError("");
    setJobProgress(null);

    try {
      const formData = buildTrainingForm();
      formData.append("algorithm", algorithm);

      const data = await runJob("/api/train-model", formData, setJobProgress);

      setMlResult(data);
      setModelId(data.model_id || "");
      setPredictionError("");
    } catch (error) {
      setMlError(error.message || "Model training failed.");
    } finally {
      setMlLoading(false);
      setJobProgress(null);
    }
  };

  const compareModels = async () => {
    setCompareLoading(true);
    setMlError("");
    setJobProgress(null);

    try {
      const formData = buildTrainingForm();
      const data = await runJob("/api/compare-models", formData, setJobProgress);

      setCompareResult(data);
      if (data.winner) {
        setMlResult(data.winner);
        setModelId(data.model_id || "");
        setPredictionError("");
      }
    } catch (error) {
      setMlError(error.message || "Model comparison failed.");
    } finally {
      setCompareLoading(false);
      setJobProgress(null);
    }
  };

  const checkLeakage = async () => {
    setLeakageLoading(true);
    setMlError("");

    try {
      if (!datasetFilename) {
        throw new Error("Run Statistics Studio analysis first so the leakage check can use the uploaded dataset.");
      }
      if (!targetColumn) {
        throw new Error("Choose a target column.");
      }

      const data = await postJson("/api/detect-leakage", {
        filename: datasetFilename,
        dataset_id: datasetId || null,
        target_column: targetColumn,
      });

      setLeakageResult(data);
    } catch (error) {
      setMlError(error.message || "Leakage check failed.");
    } finally {
      setLeakageLoading(false);
    }
  };

  // Everything computed so far, for the AI analyst and the report generator
  const analystContext = {
    dataset: result
      ? { rows: result.rows, columns: result.columns, domain: result.domain, filename: datasetFilename }
      : {},
    schema: result?.schema || {},
    statistics: result?.statistics || {},
    domainInsights: result?.domain_insights || [],
    targetAdvice: targetSuggestion || {},
    mlResult: mlResult || {},
    // The winner is already in mlResult
    compareResult: compareResult ? { ...compareResult, winner: undefined } : {},
    predictionResult: predictionResult || {},
    leakageResult: leakageResult || {},
  };

  // ── MLOps data fetchers ──────────────────────────────────────────

  const fetchExperiments = async () => {
    setMlopsLoading(true);
    setMlopsError("");
    try {
      const data = await apiFetch("/api/experiments/");
      setExperiments(data.experiments || []);
    } catch {
      setMlopsError("Failed to load experiments.");
    } finally {
      setMlopsLoading(false);
    }
  };

  const fetchModels = async () => {
    setMlopsLoading(true);
    setMlopsError("");
    try {
      const data = await apiFetch("/api/models/");
      setModels(data.models || []);
    } catch {
      setMlopsError("Failed to load models.");
    } finally {
      setMlopsLoading(false);
    }
  };

  const fetchDatasets = async () => {
    setMlopsLoading(true);
    setMlopsError("");
    try {
      const data = await apiFetch("/api/datasets");
      setDatasets(data.datasets || []);
    } catch {
      setMlopsError("Failed to load datasets.");
    } finally {
      setMlopsLoading(false);
    }
  };

  const transitionModelStage = async (modelId, stage) => {
    setStageLoading((prev) => ({ ...prev, [modelId]: true }));
    try {
      await apiFetch(
        `/api/models/${encodeURIComponent(modelId)}/transition?stage=${encodeURIComponent(stage)}`,
        { method: "POST" }
      );
      await fetchModels();
    } catch (error) {
      setMlopsError(error.message || "Could not change the model stage.");
    } finally {
      setStageLoading((prev) => ({ ...prev, [modelId]: false }));
    }
  };

  const loadMlopsTab = (tab) => {
    setMlopsTab(tab);
    setSelectedExperiment(null);
    setSelectedModel(null);
    if (tab === "experiments") fetchExperiments();
    if (tab === "models") fetchModels();
    if (tab === "datasets") fetchDatasets();
  };
  // ── MLOps Studio component ────────────────────────────────────────

  const MLOpsStudio = () => (
    <section>
      <Header onBack={goToDashboard} title="MLOps Studio" subtitle="Experiments, model registry, and dataset lineage." />

      {/* Tab bar */}
      <div className="mlops-tabs">
        {[
          ["experiments", "🧪 Experiments"],
          ["models", "📦 Model Registry"],
          ["datasets", "🗄️ Dataset Registry"],
        ].map(([key, label]) => (
          <button
            key={key}
            className={`mlops-tab-btn ${mlopsTab === key ? "active" : ""}`}
            onClick={() => loadMlopsTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {mlopsError && (
        <div className="warning"><p>{mlopsError}</p></div>
      )}

      {mlopsLoading && (
        <div className="empty"><h3>Loading…</h3></div>
      )}

      {/* ── Experiments Tab ── */}
      {mlopsTab === "experiments" && !mlopsLoading && (
        <>
          {!experiments.length && !mlopsError && (
            <Empty
              title="No experiments yet."
              text="Train a model in ML Studio and experiments will appear here automatically."
            />
          )}

          {selectedExperiment && (
            <div className="mlops-detail-overlay">
              <div className="mlops-detail-card">
                <div className="mlops-detail-head">
                  <h3>Experiment #{selectedExperiment.id}</h3>
                  <button className="close-btn" onClick={() => setSelectedExperiment(null)}>✕</button>
                </div>
                <div className="mlops-detail-body">
                  <div className="mlops-kv-grid">
                    <KV label="Algorithm" value={selectedExperiment.algorithm} />
                    <KV label="Timestamp" value={new Date(selectedExperiment.timestamp).toLocaleString()} />
                    <KV label="Dataset" value={selectedExperiment.dataset_info?.filename || "—"} />
                    <KV label="Rows" value={selectedExperiment.dataset_info?.rows} />
                    <KV label="Model ID" value={selectedExperiment.model_id || "—"} />
                    <KV label="Dataset ID" value={selectedExperiment.dataset_id || "—"} />
                  </div>
                  <h4>Metrics</h4>
                  <div className="mlops-metrics-grid">
                    {Object.entries(selectedExperiment.metrics || {}).map(([k, v]) => (
                      <div className="mlops-metric-chip" key={k}>
                        <span>{k.replace(/_/g, " ")}</span>
                        <strong>{typeof v === "number" ? v.toFixed(4) : v}</strong>
                      </div>
                    ))}
                  </div>
                  {Object.keys(selectedExperiment.hyperparameters || {}).length > 0 && (
                    <>
                      <h4>Hyperparameters</h4>
                      <div className="mlops-metrics-grid">
                        {Object.entries(selectedExperiment.hyperparameters).map(([k, v]) => (
                          <div className="mlops-metric-chip" key={k}>
                            <span>{k}</span>
                            <strong>{String(v)}</strong>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {experiments.length > 0 && (
            <div className="card">
              <div className="mlops-table-head">
                <span>#</span>
                <span>Algorithm</span>
                <span>Dataset</span>
                <span>Primary Metric</span>
                <span>Time</span>
                <span></span>
              </div>
              {experiments.map((exp) => {
                const metricEntries = Object.entries(exp.metrics || {});
                const primaryMetric = metricEntries.find(
                  ([k]) => k === "accuracy" || k === "f1_weighted" || k === "r2_score"
                ) || metricEntries[0];
                return (
                  <div className="mlops-table-row" key={exp.id}>
                    <span className="mlops-id-badge">#{exp.id}</span>
                    <code>{exp.algorithm}</code>
                    <span className="mlops-truncate">{exp.dataset_info?.filename || "—"}</span>
                    <span className="mlops-metric-inline">
                      {primaryMetric
                        ? <><small>{primaryMetric[0].replace(/_/g, " ")}</small> <strong>{typeof primaryMetric[1] === "number" ? primaryMetric[1].toFixed(4) : primaryMetric[1]}</strong></>
                        : "—"}
                    </span>
                    <span className="mlops-time">{new Date(exp.timestamp).toLocaleDateString()}</span>
                    <button className="mlops-view-btn" onClick={() => setSelectedExperiment(exp)}>View</button>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* ── Model Registry Tab ── */}
      {mlopsTab === "models" && !mlopsLoading && (
        <>
          {!models.length && !mlopsError && (
            <Empty
              title="No registered models yet."
              text="Train a model in ML Studio and it will auto-register here."
            />
          )}

          {selectedModel && (
            <div className="mlops-detail-overlay">
              <div className="mlops-detail-card">
                <div className="mlops-detail-head">
                  <h3>Model Details</h3>
                  <button className="close-btn" onClick={() => setSelectedModel(null)}>✕</button>
                </div>
                <div className="mlops-detail-body">
                  <div className="mlops-id-copy">
                    <code>{selectedModel.model_id}</code>
                  </div>
                  <div className="mlops-kv-grid">
                    <KV label="Algorithm" value={selectedModel.metadata?.algorithm} />
                    <KV label="Task Type" value={selectedModel.metadata?.task_type} />
                    <KV label="Target Column" value={selectedModel.metadata?.target_column} />
                    <KV label="Dataset ID" value={selectedModel.metadata?.dataset_id || "—"} />
                    <KV label="Registered" value={new Date(selectedModel.registered_at).toLocaleString()} />
                    <KV label="Stage" value={selectedModel.versions?.[0]?.stage || "development"} />
                  </div>
                  <h4>Promote Stage</h4>
                  <div className="stage-row">
                    {["development", "staging", "production", "archived"].map((s) => (
                      <button
                        key={s}
                        className={`stage-btn stage-${s} ${selectedModel.versions?.[0]?.stage === s ? "active" : ""}`}
                        disabled={stageLoading[selectedModel.model_id]}
                        onClick={async () => {
                          await transitionModelStage(selectedModel.model_id, s);
                          setSelectedModel((prev) => ({
                            ...prev,
                            versions: [{ ...prev.versions?.[0], stage: s }],
                          }));
                        }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                  {selectedModel.metadata?.metrics && (
                    <>
                      <h4>Metrics</h4>
                      <div className="mlops-metrics-grid">
                        {Object.entries(selectedModel.metadata.metrics)
                          .filter(([, v]) => typeof v === "number")
                          .map(([k, v]) => (
                            <div className="mlops-metric-chip" key={k}>
                              <span>{k.replace(/_/g, " ")}</span>
                              <strong>{v.toFixed(4)}</strong>
                            </div>
                          ))}
                      </div>
                    </>
                  )}
                  <ModelOps
                    key={selectedModel.model_id}
                    model={selectedModel}
                    stage={selectedModel.versions?.[0]?.stage || "development"}
                  />
                </div>
              </div>
            </div>
          )}

          {models.length > 0 && (
            <div className="card">
              <div className="mlops-table-head">
                <span>Model ID</span>
                <span>Algorithm</span>
                <span>Target</span>
                <span>Stage</span>
                <span>Registered</span>
                <span></span>
              </div>
              {models.map((m) => {
                const stage = m.versions?.[0]?.stage || "development";
                return (
                  <div className="mlops-table-row" key={m.model_id}>
                    <code className="mlops-truncate">{m.model_id}</code>
                    <code>{m.metadata?.algorithm || "—"}</code>
                    <span>{m.metadata?.target_column || "—"}</span>
                    <span className={`stage-pill stage-${stage}`}>{stage}</span>
                    <span className="mlops-time">{new Date(m.registered_at).toLocaleDateString()}</span>
                    <button className="mlops-view-btn" onClick={() => setSelectedModel(m)}>Serve &amp; monitor</button>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* ── Dataset Registry Tab ── */}
      {mlopsTab === "datasets" && !mlopsLoading && (
        <>
          {!datasets.length && !mlopsError && (
            <Empty
              title="No datasets registered yet."
              text="Upload and analyze a dataset in Statistics Studio to register it."
            />
          )}

          {datasets.length > 0 && (
            <div className="card">
              <div className="mlops-table-head mlops-table-datasets">
                <span>ID</span>
                <span>Filename</span>
                <span>Rows</span>
                <span>Columns</span>
                <span>Missing Cells</span>
                <span>Uploaded</span>
              </div>
              {datasets.map((d) => (
                <div className="mlops-table-row mlops-table-datasets" key={d.id}>
                  <span className="mlops-id-badge">#{d.id}</span>
                  <span className="mlops-truncate">{d.original_filename}</span>
                  <strong>{d.rows ?? "—"}</strong>
                  <strong>{d.columns ?? "—"}</strong>
                  <span className={d.missing_cells > 0 ? "text-warn" : "text-ok"}>
                    {d.missing_cells ?? "—"}
                  </span>
                  <span className="mlops-time">{new Date(d.upload_time).toLocaleDateString()}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );

  function KV({ label, value }) {
    return (
      <div className="mlops-kv">
        <span>{label}</span>
        <strong>{value ?? "—"}</strong>
      </div>
    );
  }

  const fetchTargetSuggestion = async () => {
    if (!datasetFilename) return;
    setTargetSuggestionLoading(true);
    setTargetSuggestionError("");
    setTargetSuggestion(null);
    try {
      const formData = new FormData();
      formData.append("filename", datasetFilename);
      formData.append("domain", domain);
      const data = await callFormEndpoint("/api/recommend-target", formData);
      setTargetSuggestion(data);
    } catch (error) {
      setTargetSuggestionError(error.message || "Could not fetch target suggestion.");
    } finally {
      setTargetSuggestionLoading(false);
    }
  };

  const predictTestData = async () => {
    if (!modelId.trim()) {
      setPredictionError("Train a model first or paste a saved model ID.");
      return;
    }

    if (!predictionFile) {
      setPredictionError("Upload a CSV or Excel test dataset.");
      return;
    }

    setPredictionLoading(true);
    setPredictionError("");
    setPredictionResult(null);

    try {
      const formData = new FormData();
      formData.append("model_id", modelId.trim());
      formData.append("file", predictionFile);

      const data = await callFormEndpoint("/api/predict-test-data", formData);
      setPredictionResult(data);
    } catch (error) {
      setPredictionError(error.message || "Prediction failed.");
    } finally {
      setPredictionLoading(false);
    }
  };

  const Dashboard = () => (
    <section>
      <div className="hero">
        <p className="hero-kicker">AutoML you can check</p>
        <h1 className="wordmark">STATMIND</h1>
        <p className="hero-tagline">
          From a raw spreadsheet to a tested, explained and tracked machine learning model.
        </p>
        <p className="hero-sub">
          Upload a CSV or Excel file. StatMind profiles it, warns you about columns that give the
          answer away, trains and tunes every suitable algorithm with cross-validation, and shows
          you why the winning model predicts what it does.
        </p>

        <div className="hero-actions">
          <button className="primary" onClick={() => setCurrentPage("statistics")}>
            Try the app
          </button>
          <a className="ghost-btn" href="#how-it-works">
            See how it works
          </a>
        </div>

        <ul className="hero-facts">
          <li><strong>Up to 9</strong> algorithms compared</li>
          <li><strong>5-fold</strong> cross-validation</li>
          <li><strong>Baseline</strong> beside every score</li>
          <li><strong>No code</strong> required</li>
        </ul>
      </div>

      <div className="section-head" id="how-it-works">
        <p className="eyebrow">How it works</p>
        <h2>Four steps from data to a model you can trust</h2>
      </div>

      <ol className="step-grid">
        <li className="step-card">
          <span className="step-number">1</span>
          <h3>Upload and profile</h3>
          <p>
            Drop in a dataset. You get the schema, a data quality score, summary statistics and
            charts, plus a suggestion for which column to predict.
          </p>
        </li>
        <li className="step-card">
          <span className="step-number">2</span>
          <h3>Check for leakage</h3>
          <p>
            StatMind tests how well each column predicts the target on its own, and flags copies,
            derived columns and identifiers before they inflate your scores.
          </p>
        </li>
        <li className="step-card">
          <span className="step-number">3</span>
          <h3>Run AutoML</h3>
          <p>
            Every suitable algorithm is cross-validated, the best two are tuned, and the winner is
            scored once on data it has never seen. Nothing is selected on the test set.
          </p>
        </li>
        <li className="step-card">
          <span className="step-number">4</span>
          <h3>Explain, predict, track</h3>
          <p>
            See which features drive the model, score new data from a file or through the API,
            and get warned when incoming data drifts away from what the model was trained on.
          </p>
        </li>
      </ol>

      <div className="section-head">
        <p className="eyebrow">What you get</p>
        <h2>Results that hold up to questions</h2>
      </div>

      <div className="feature-grid">
        <div className="feature-card">
          <h3>Honest evaluation</h3>
          <p>
            Cross-validated scores with their spread, a held-out test score, and a baseline that
            ignores the features, so you can see how much the model actually learned.
          </p>
        </div>
        <div className="feature-card">
          <h3>Explanations for every model</h3>
          <p>
            Permutation importance with error bars, ROC curves, a confusion matrix and per-class
            scores, for linear models and SVMs as well as trees.
          </p>
        </div>
        <div className="feature-card">
          <h3>Leakage detection</h3>
          <p>
            Finds columns that predict the target suspiciously well, including categorical and
            arithmetic leaks that a correlation check misses.
          </p>
        </div>
        <div className="feature-card">
          <h3>Sensible preprocessing</h3>
          <p>
            Missing values, categories, dates and imbalanced classes are handled inside the saved
            pipeline, so predictions on new data are treated exactly the same way.
          </p>
        </div>
        <div className="feature-card">
          <h3>Experiment tracking and lineage</h3>
          <p>
            A built-in experiment log and model registry with stages, with optional MLflow
            tracking and lakeFS versioning that ties each model to the exact data it was trained on.
          </p>
        </div>
        <div className="feature-card">
          <h3>Serving and drift monitoring</h3>
          <p>
            Every model has a prediction API, and the production stage decides which model
            answers it. Recent inputs are tested against the training data, column by column.
          </p>
        </div>
        <div className="feature-card">
          <h3>An analyst to ask</h3>
          <p>
            Ask questions about your results in plain language and export everything as a single
            report to share.
          </p>
        </div>
      </div>

      <div className="section-head">
        <p className="eyebrow">Studios</p>
        <h2>Jump straight to a workspace</h2>
      </div>

      <div className="studio-grid">
        <div className="studio-card">
          <div className="studio-tag">Statistics</div>
          <h3>Statistics Studio</h3>
          <p>Upload a dataset, see its schema, quality score, statistics and charts. Start here.</p>
          <button onClick={() => setCurrentPage("statistics")}>Open</button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">Machine Learning</div>
          <h3>ML Studio</h3>
          <p>Train a model or run cross-validated AutoML, with explanations and a leakage check.</p>
          <button onClick={() => setCurrentPage("ml")}>Open</button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">Prediction</div>
          <h3>Prediction Studio</h3>
          <p>Score a new file with a saved model and download the predictions.</p>
          <button onClick={() => setCurrentPage("prediction")}>Open</button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">MLOps</div>
          <h3>MLOps Studio</h3>
          <p>Browse experiments, promote models, call them through the API, and watch for drift.</p>
          <button
            onClick={() => {
              setCurrentPage("mlops");
              setMlopsTab("experiments");
              fetchExperiments();
            }}
          >
            Open
          </button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">Analyst</div>
          <h3>AI Analyst</h3>
          <p>Ask questions about your data and models, and export the full report.</p>
          <button onClick={() => setCurrentPage("analyst")}>Open</button>
        </div>
      </div>

      <div className="cta-band">
        <div>
          <h2>Ready to try it on your own data?</h2>
          <p>Upload a CSV or Excel file up to 50 MB. A first model takes a couple of minutes.</p>
        </div>
        <button className="primary" onClick={() => setCurrentPage("statistics")}>
          Try the app
        </button>
      </div>
    </section>
  );

  const [chartTab, setChartTab] = useState("overview");
    const chartCounts = useMemo(() => {
      if (!result?.charts) return { total: 0 };
      const c = result.charts;
      return {
        bar: (c.bar_charts?.length || 0) + (c.top_n_charts?.length || 0),
        dist: (c.pie_charts?.length || 0) + (c.histograms?.length || 0),
        trends: (c.line_charts?.length || 0) + (c.area_charts?.length || 0) + (c.scatter_plots?.length || 0),
        corr: c.correlation_heatmap ? 1 : 0,
        total:
          (c.bar_charts?.length || 0) +
          (c.pie_charts?.length || 0) +
          (c.line_charts?.length || 0) +
          (c.area_charts?.length || 0) +
          (c.histograms?.length || 0) +
          (c.scatter_plots?.length || 0) +
          (c.top_n_charts?.length || 0) +
          (c.correlation_heatmap ? 1 : 0),
      };
    }, [result]);

  const Statistics = () => {
    return (
      <section>
        <Header onBack={goToDashboard} title="Statistics Studio" subtitle="Profile datasets and generate visual intelligence — optimized for speed." />

        <div className="upload-card">
          <div className="input-group">
            <label>Upload Dataset</label>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(e) => {
                setFile(e.target.files[0]);
                setResult(null);
                setFrontendError("");
              }}
            />
            {file && <span className="file-name">{file.name}</span>}
          </div>
          <div className="input-group">
            <label>Analysis Domain</label>
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              <option value="business">Business Analytics</option>
              <option value="research">Research Analytics</option>
              <option value="finance">Finance Analytics</option>
            </select>
          </div>
          <button className="primary" onClick={analyze} disabled={loading}>
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>

        {frontendError && (
          <div className="warning">
            <h4>Frontend Error</h4>
            <p>{frontendError}</p>
            <p>Open browser console using <strong>F12 → Console</strong> and check the logs.</p>
          </div>
        )}

        {!result && !loading && <Empty title="No dataset analyzed yet." text="Upload a CSV or Excel file and run analysis. Charts are now sampled & lazy-loaded for instant display." />}

        {loading && (
          <div className="empty">
            <h3>Analyzing dataset...</h3>
            <p>Optimized engine: sampling large datasets, capped correlations, progressive chart loading. This now completes in seconds.</p>
            <div className="progress-bar-skeleton"><div className="progress-bar-fill animate" /></div>
          </div>
        )}

        {result && (
          <>
            <div className="summary-grid">
              <Summary label="Rows" value={result.rows} />
              <Summary label="Columns" value={result.columns} />
              <Summary label="Domain" value={result.domain} />
              <Summary label="Quality" value={`${result.statistics?.data_quality?.data_quality_score}/100`} small={result.statistics?.data_quality?.grade} />
            </div>
            {result.processing_time_seconds !== undefined && (
              <div className="perf-badge">
                <span>⚡ Processed in {result.processing_time_seconds}s</span>
                {result.chart_sampling_applied && <span className="perf-sampling">Sampling applied for charts (large dataset)</span>}
                <span className="perf-count">{chartCounts.total} charts ready • lazy-loaded</span>
              </div>
            )}

            <Card title="Dataset Intelligence">
              <InsightList items={result.statistics?.dataset_intelligence || []} />
            </Card>
            <Card title="Domain Insights">
              <InsightList items={result.domain_insights || []} />
            </Card>

            <Card title="Dynamic Charts">
              <div className="chart-tabs">
                {[
                  ["overview", `Overview (${Math.min(chartCounts.total, 6)})`],
                  ["distributions", `Distributions (${chartCounts.bar + chartCounts.dist})`],
                  ["trends", `Trends (${chartCounts.trends})`],
                  ["correlation", "Correlation"],
                ].map(([key, label]) => (
                  <button key={key} className={`chart-tab-btn ${chartTab === key ? "active" : ""}`} onClick={() => setChartTab(key)}>{label}</button>
                ))}
              </div>
              <div className="chart-grid">
                {chartTab === "overview" && (
                  <>
                    {result.charts?.top_n_charts?.slice(0, 1).map((chart, i) => <BarBlock key={`top-${i}`} chart={chart} label="Top-N" />)}
                    {result.charts?.bar_charts?.slice(0, 1).map((chart, i) => <BarBlock key={`bar-${i}`} chart={chart} label="Bar" />)}
                    {result.charts?.line_charts?.slice(0, 1).map((chart, i) => <LineBlock key={`line-${i}`} chart={chart} />)}
                    {result.charts?.pie_charts?.slice(0, 1).map((chart, i) => <PieBlock key={`pie-${i}`} chart={chart} />)}
                    {result.charts?.histograms?.slice(0, 1).map((chart, i) => <HistBlock key={`hist-${i}`} chart={chart} />)}
                    {result.charts?.scatter_plots?.slice(0, 1).map((chart, i) => <ScatterBlock key={`scatter-${i}`} chart={chart} />)}
                  </>
                )}
                {chartTab === "distributions" && (
                  <>
                    {result.charts?.top_n_charts?.map((chart, i) => <BarBlock key={`top-${i}`} chart={chart} label="Top-N" />)}
                    {result.charts?.bar_charts?.map((chart, i) => <BarBlock key={`bar-${i}`} chart={chart} label="Bar" />)}
                    {result.charts?.pie_charts?.map((chart, i) => <PieBlock key={`pie-${i}`} chart={chart} />)}
                    {result.charts?.histograms?.map((chart, i) => <HistBlock key={`hist-${i}`} chart={chart} />)}
                  </>
                )}
                {chartTab === "trends" && (
                  <>
                    {result.charts?.line_charts?.map((chart, i) => <LineBlock key={`line-${i}`} chart={chart} />)}
                    {result.charts?.area_charts?.map((chart, i) => <AreaBlock key={`area-${i}`} chart={chart} />)}
                    {result.charts?.scatter_plots?.map((chart, i) => <ScatterBlock key={`scatter-${i}`} chart={chart} />)}
                  </>
                )}
                {chartTab === "correlation" && <Heatmap result={result} />}
              </div>
              {chartTab !== "correlation" && result.charts?.correlation_heatmap && (
                <p className="chart-hint">Correlation heatmap moved to <button className="link-btn" onClick={() => setChartTab("correlation")}>Correlation tab</button> for faster rendering.</p>
              )}
            </Card>

            <Card title="Columns and Types">
              <ColumnOverview schema={result.schema} rows={result.rows} />
            </Card>
            <Card title="Numeric Columns">
              <NumericStats summary={result.statistics?.numeric_summary} />
            </Card>
            <Card title="Category Columns">
              <CategoricalStats summary={result.statistics?.categorical_summary} rows={result.rows} />
            </Card>
          </>
        )}
      </section>
    );
  };

  const MLStudio = () => (
    <section>
      <Header
        onBack={goToDashboard}
        title="ML Studio"
        subtitle="Train models, run cross-validated AutoML, and check for leakage."
      />

      {!result && (
        <div className="warning">
          <h4>No analyzed dataset loaded</h4>
          <p>
            Run Statistics Studio first. ML Studio uses that uploaded dataset
            and its detected columns.
          </p>
          <button className="primary" onClick={() => setCurrentPage("statistics")}>
            Open Statistics Studio
          </button>
        </div>
      )}

      {result && (
        <>
          <Card title="Training Setup">
            <div className="notice">
              <p>
                Dataset ready: <strong>{datasetFilename}</strong>
              </p>
            </div>

            {/* ── Suggested Target Banner ── */}
            <div className="suggest-target-bar">
              <button
                className="suggest-btn"
                onClick={fetchTargetSuggestion}
                disabled={targetSuggestionLoading}
              >
                {targetSuggestionLoading ? "Analyzing…" : "✦ Suggest Target Variable"}
              </button>

              {targetSuggestionError && (
                <span className="suggest-error">{targetSuggestionError}</span>
              )}
            </div>

            {targetSuggestion && (
              <div className="suggestion-panel">
                {targetSuggestion.best_target ? (
                  <>
                    <div className="suggestion-header">
                      <span className="suggestion-badge">AI Recommended</span>
                      <span className="suggestion-domain">Domain: {targetSuggestion.domain}</span>
                    </div>

                    <div className="suggestion-best">
                      <div className="suggestion-col-name">
                        {targetSuggestion.best_target.column}
                      </div>
                      <div className="suggestion-meta">
                        <span className={`priority-tag priority-${targetSuggestion.best_target.priority.toLowerCase()}`}>
                          {targetSuggestion.best_target.priority} Priority
                        </span>
                        <span className="task-tag">{targetSuggestion.best_target.task_type}</span>
                        <span className="score-tag">Score {targetSuggestion.best_target.score}/100</span>
                      </div>

                      {targetSuggestion.best_target.reasons?.length > 0 && (
                        <ul className="suggestion-reasons">
                          {targetSuggestion.best_target.reasons.map((r, i) => (
                            <li key={i}>✓ {r}</li>
                          ))}
                        </ul>
                      )}

                      {targetSuggestion.best_target.warnings?.length > 0 && (
                        <ul className="suggestion-warnings">
                          {targetSuggestion.best_target.warnings.map((w, i) => (
                            <li key={i}>⚠ {w}</li>
                          ))}
                        </ul>
                      )}

                      <button
                        className="use-target-btn"
                        onClick={() => setTargetColumn(targetSuggestion.best_target.column)}
                      >
                        Use "{targetSuggestion.best_target.column}" as Target
                      </button>
                    </div>

                    {targetSuggestion.recommended_targets?.length > 1 && (
                      <div className="suggestion-others">
                        <p className="suggestion-others-label">Other candidates</p>
                        <div className="suggestion-others-grid">
                          {targetSuggestion.recommended_targets.slice(1, 5).map((t) => (
                            <button
                              key={t.column}
                              className="other-target-btn"
                              onClick={() => setTargetColumn(t.column)}
                            >
                              <span>{t.column}</span>
                              <small>{t.task_type} · {t.score}/100</small>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {targetSuggestion.advisor_summary?.length > 0 && (
                      <div className="suggestion-summary">
                        {targetSuggestion.advisor_summary.map((s, i) => (
                          <p key={i}>{s}</p>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <p className="suggestion-empty">
                    No strong target found. Choose a column manually.
                  </p>
                )}
              </div>
            )}
            {/* ── End Suggested Target ── */}

            <div className="ml-grid">
              <div className="input-group">
                <label>Target Column</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                >
                  <option value="">Choose target</option>
                  {datasetColumns.map((column) => (
                    <option value={column} key={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label>Algorithm</label>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                >
                  {algorithmOptions.map(([value, label]) => (
                    <option value={value} key={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label>Task Type</label>
                <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                  <option value="auto">Auto-detect</option>
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
              </div>

              <div className="input-group">
                <label>Test Size</label>
                <input
                  type="number"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={testSize}
                  onChange={(e) => setTestSize(e.target.value)}
                />
              </div>
            </div>

            <div className="action-row">
              <button className="primary" onClick={trainModel} disabled={mlLoading || compareLoading}>
                {mlLoading ? "Training..." : "Train Model"}
              </button>

              <button
                className="primary green"
                onClick={compareModels}
                disabled={mlLoading || compareLoading}
              >
                {compareLoading ? "Running AutoML..." : "Run AutoML"}
              </button>

              <button className="primary" onClick={checkLeakage} disabled={leakageLoading}>
                {leakageLoading ? "Checking..." : "Check Leakage"}
              </button>
            </div>

            {jobProgress && (
              <div className="job-progress" role="status" aria-live="polite">
                <div className="job-progress-label">
                  <span>{jobProgress.status === "queued" ? "Queued" : jobProgress.message || "Working"}</span>
                  <span>{jobProgress.progress ?? 0}%</span>
                </div>
                <div className="job-progress-track">
                  <div className="job-progress-fill" style={{ width: `${jobProgress.progress ?? 0}%` }} />
                </div>
              </div>
            )}
          </Card>

          {mlError && (
            <div className="warning">
              <h4>ML Error</h4>
              <p>{mlError}</p>
            </div>
          )}

          {leakageResult && <LeakageReport result={leakageResult} />}

          {compareResult && <Leaderboard result={compareResult} />}

          {mlResult && <ModelReport result={mlResult} />}
        </>
      )}
    </section>
  );

  const PredictionStudio = () => (
    <section>
      <Header
        onBack={goToDashboard}
        title="Prediction Studio"
        subtitle="Use saved models to generate predictions on test data."
      />

      <Card title="Prediction Input">
        {mlResult && (
          <div className="notice">
            <p>
              Latest trained model loaded: <strong>{mlResult.model_id}</strong>
            </p>
          </div>
        )}

        <div className="ml-grid">
          <div className="input-group">
            <label>Model ID</label>
            <input
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="Train a model or paste a saved model ID"
            />
          </div>

          <div className="input-group">
            <label>Test Dataset</label>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(e) => {
                setPredictionFile(e.target.files[0]);
                setPredictionResult(null);
                setPredictionError("");
              }}
            />
            {predictionFile && (
              <span className="file-name">{predictionFile.name}</span>
            )}
          </div>

          <button
            className="primary"
            onClick={predictTestData}
            disabled={predictionLoading}
          >
            {predictionLoading ? "Generating..." : "Generate Predictions"}
          </button>
        </div>
      </Card>

      {predictionError && (
        <div className="warning">
          <h4>Prediction Error</h4>
          <p>{predictionError}</p>
        </div>
      )}

      {!predictionResult && !predictionLoading && (
        <Empty
          title="No predictions generated yet."
          text="Train a model, upload compatible test data, and generate predictions."
        />
      )}

      {predictionResult && (
        <>
          <div className="summary-grid">
            <Summary label="Rows Predicted" value={predictionResult.rows_predicted} />
            <Summary label="Task Type" value={predictionResult.task_type} />
            <Summary label="Target" value={predictionResult.target_column} />
            <Summary
              label="Prediction Column"
              value={predictionResult.prediction_column}
            />
          </div>

          <Card title="Prediction Output">
            <div className="result-box">
              <span>Prediction File</span>
              <strong>{predictionResult.prediction_filename}</strong>
            </div>

            <button
              type="button"
              className="download-btn"
              onClick={() =>
                downloadArtifact(predictionResult.artifact_id).catch((error) =>
                  setPredictionError(error.message || "Download failed.")
                )
              }
            >
              Download Predictions CSV
            </button>
          </Card>

          {predictionResult.external_test_metrics && (
            <Card title="External Test Metrics">
              <MetricGrid metrics={predictionResult.external_test_metrics} />
            </Card>
          )}

          <Card title="Prediction Preview">
            <PreviewTable rows={predictionResult.preview || []} />
          </Card>
        </>
      )}
    </section>
  );

  const logout = () => {
    clearSession();
    setUser(null);
    setAuthNotice("");
    setResult(null);
    setMlResult(null);
    setCompareResult(null);
    setPredictionResult(null);
    setLeakageResult(null);
    setDatasetId("");
    setModelId("");
    setCurrentPage("dashboard");
  };

  if (!user) {
    if (authRequired !== true) return null; // still asking the server, or signing in as the local user

    return (
      <AuthScreen
        notice={authNotice}
        onAuthenticated={(loggedIn) => {
          setUser(loggedIn);
          setAuthNotice("");
        }}
      />
    );
  }

  return (
    <div className="shell">
      <header className="topbar">
        <button
          type="button"
          className="brand"
          onClick={goToDashboard}
          aria-label="StatMind, go to the dashboard"
        >
          <span className="brand-icon">StatMind</span>
          <span className="brand-tagline">Data Intelligence OS</span>
        </button>

        <nav className="top-nav">
          {[
            ["dashboard", "Dashboard"],
            ["statistics", "Statistics Studio"],
            ["ml", "ML Studio"],
            ["prediction", "Prediction Studio"],
            ["mlops", "MLOps Studio"],
            ["analyst", "AI Analyst"],
          ].map(([key, label]) => (
            <button
              key={key}
              className={currentPage === key ? "active" : ""}
              onClick={() => {
              setCurrentPage(key);
              if (key === "mlops") {
                setMlopsTab("experiments");
                fetchExperiments();
              }
            }}
            >
              {label}
            </button>
          ))}
        </nav>

        {authRequired && (
          <div className="user-menu">
            <span className="user-email" title={user.email}>{user.email}</span>
            <button type="button" onClick={logout}>Log out</button>
          </div>
        )}
      </header>

      <main className="main">
        {/* Keyed by page so each page fades in when you switch to it */}
        <div className="page" key={currentPage}>
          {currentPage === "dashboard" && Dashboard()}
          {currentPage === "statistics" && Statistics()}
          {currentPage === "ml" && MLStudio()}
          {currentPage === "prediction" && PredictionStudio()}
          {currentPage === "mlops" && MLOpsStudio()}
        </div>
        {/* Stays mounted so the conversation survives switching pages */}
        <section className="page" hidden={currentPage !== "analyst"}>
          <Header onBack={goToDashboard} title="AI Analyst" subtitle="Ask questions about your data and models." />
          <AnalystPanel context={analystContext} hasDataset={!!result} />
        </section>
      </main>
    </div>
  );
}

export default App;
