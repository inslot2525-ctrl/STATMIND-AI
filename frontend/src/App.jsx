import { useState } from "react";
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

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [currentPage, setCurrentPage] = useState("dashboard");

  const [file, setFile] = useState(null);
  const [domain, setDomain] = useState("business");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [frontendError, setFrontendError] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [algorithm, setAlgorithm] = useState("random_forest");
  const [testSize, setTestSize] = useState("0.2");
  const [mlResult, setMlResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
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

  const pieColors = [
    "#f0b429",
    "#e07a2d",
    "#fbd38d",
    "#c98a2d",
    "#facc15",
    "#b45309",
    "#fde68a",
    "#a3762a",
  ];

  const tooltipProps = {
    contentStyle: {
      background: "#0d0b08",
      border: "1px solid rgba(240, 180, 41, 0.3)",
      borderRadius: 12,
      boxShadow: "0 12px 32px rgba(0, 0, 0, 0.6)",
    },
    labelStyle: { color: "#fbd38d", fontWeight: 700 },
    itemStyle: { color: "#f5efe0" },
    cursor: { fill: "rgba(240, 180, 41, 0.06)" },
  };

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
      console.log("Sending analysis request...");
      console.log("File:", file.name);
      console.log("Domain:", domain);

      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", response.status);

      const text = await response.text();
      console.log("Raw backend response:", text);

      let data;

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Backend did not return valid JSON.");
      }

      if (!response.ok) {
        throw new Error(data.detail || "Backend returned an error.");
      }

      console.log("Parsed analysis result:", data);

      setResult(data);
      const columns = Object.keys(data.schema?.dtypes || {});
      setTargetColumn((currentTarget) => currentTarget || columns[columns.length - 1] || "");
      setDatasetId(data.dataset_id || "");
      setTargetSuggestion(null);
      setTargetSuggestionError("");
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
    ["decision_tree", "Decision Tree"],
    ["knn", "KNN"],
    ["svm", "SVM"],
    ["logistic_regression", "Logistic Regression"],
    ["linear_regression", "Linear Regression"],
    ["ridge", "Ridge Regression"],
    ["lasso", "Lasso Regression"],
  ];

  const callFormEndpoint = async (endpoint, formData) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      body: formData,
    });

    const text = await response.text();
    let data;

    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("Backend did not return valid JSON.");
    }

    if (!response.ok) {
      throw new Error(data.detail || "Backend returned an error.");
    }

    return data;
  };

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

    return formData;
  };

  const trainModel = async () => {
    setMlLoading(true);
    setMlError("");

    try {
      const formData = buildTrainingForm();
      formData.append("algorithm", algorithm);

      const data = await callFormEndpoint("/api/train-model", formData);

      setMlResult(data);
      setModelId(data.model_id || "");
      setPredictionError("");
    } catch (error) {
      setMlError(error.message || "Model training failed.");
    } finally {
      setMlLoading(false);
    }
  };

  const compareModels = async () => {
    setCompareLoading(true);
    setMlError("");

    try {
      const formData = buildTrainingForm();
      const data = await callFormEndpoint("/api/compare-models", formData);

      setCompareResult(data);
    } catch (error) {
      setMlError(error.message || "Model comparison failed.");
    } finally {
      setCompareLoading(false);
    }
  };

  // ── MLOps data fetchers ──────────────────────────────────────────

  const fetchExperiments = async () => {
    setMlopsLoading(true);
    setMlopsError("");
    try {
      const res = await fetch(`${API_BASE_URL}/api/experiments/`);
      const data = await res.json();
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
      const res = await fetch(`${API_BASE_URL}/api/models/`);
      const data = await res.json();
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
      const res = await fetch(`${API_BASE_URL}/api/datasets/`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data.datasets || []);
      }
    } catch {
      setMlopsError("Failed to load datasets.");
    } finally {
      setMlopsLoading(false);
    }
  };

  const transitionModelStage = async (modelId, stage) => {
    setStageLoading((prev) => ({ ...prev, [modelId]: true }));
    try {
      const res = await fetch(
        `${API_BASE_URL}/api/models/${modelId}/transition?stage=${stage}`,
        { method: "POST" }
      );
      if (res.ok) {
        await fetchModels();
      }
    } catch {
      // ignore
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
      <Header title="MLOps Studio" subtitle="Experiments, model registry, and dataset lineage." />

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
                        {Object.entries(selectedModel.metadata.metrics).map(([k, v]) => (
                          <div className="mlops-metric-chip" key={k}>
                            <span>{k.replace(/_/g, " ")}</span>
                            <strong>{typeof v === "number" ? v.toFixed(4) : v}</strong>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
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
                    <button className="mlops-view-btn" onClick={() => setSelectedModel(m)}>View</button>
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
        <h1 className="wordmark">STATMIND</h1>
        <p className="hero-tagline">Your explainable data science command center.</p>
        <p className="hero-sub">
          Upload datasets, generate statistics, detect schema, produce charts,
          and prepare the dataset for machine learning.
        </p>
      </div>

      <div className="studio-grid">
        <div className="studio-card">
          <div className="studio-tag">Statistics</div>
          <h3>Statistics Studio</h3>
          <p>Profile datasets, detect schema, generate statistics and charts.</p>
          <button onClick={() => setCurrentPage("statistics")}>Open</button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">Machine Learning</div>
          <h3>ML Studio</h3>
          <p>Train models after the analysis flow is stable.</p>
          <button onClick={() => setCurrentPage("ml")}>Open</button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">Prediction</div>
          <h3>Prediction Studio</h3>
          <p>Use trained models on test datasets.</p>
          <button onClick={() => setCurrentPage("prediction")}>Open</button>
        </div>

        <div className="studio-card">
          <div className="studio-tag">MLOps</div>
          <h3>MLOps Studio</h3>
          <p>Track experiments, manage model registry, and view dataset lineage.</p>
          <button onClick={() => setCurrentPage("mlops")}>Open</button>
        </div>
      </div>
    </section>
  );

  const Statistics = () => (
    <section>
      <Header
        title="Statistics Studio"
        subtitle="Profile datasets and generate visual intelligence."
      />

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
          <p>
            Open browser console using <strong>F12 → Console</strong> and check
            the logs printed by the app.
          </p>
        </div>
      )}

      {!result && !loading && (
        <Empty
          title="No dataset analyzed yet."
          text="Upload a CSV or Excel file and run analysis."
        />
      )}

      {loading && (
        <div className="empty">
          <h3>Analyzing dataset...</h3>
          <p>
            Waiting for backend response. Since timeout is removed, this will
            wait until backend responds.
          </p>
        </div>
      )}

      {result && (
        <>
          <div className="summary-grid">
            <Summary label="Rows" value={result.rows} />
            <Summary label="Columns" value={result.columns} />
            <Summary label="Domain" value={result.domain} />
            <Summary
              label="Quality"
              value={`${result.statistics?.data_quality?.data_quality_score}/100`}
              small={result.statistics?.data_quality?.grade}
            />
          </div>

          <Card title="Dataset Intelligence">
            <InsightList items={result.statistics?.dataset_intelligence || []} />
          </Card>

          <Card title="Domain Insights">
            <InsightList items={result.domain_insights || []} />
          </Card>

          <Card title="Dynamic Charts">
            <div className="chart-grid">
              {result.charts?.top_n_charts?.map((chart, index) => (
                <BarBlock chart={chart} label="Top-N" key={`top-${index}`} />
              ))}

              {result.charts?.bar_charts?.map((chart, index) => (
                <BarBlock chart={chart} label="Bar" key={`bar-${index}`} />
              ))}

              {result.charts?.pie_charts?.map((chart, index) => (
                <PieBlock chart={chart} key={`pie-${index}`} />
              ))}

              {result.charts?.line_charts?.slice(0, 2).map((chart, index) => (
                <LineBlock chart={chart} key={`line-${index}`} />
              ))}

              {result.charts?.area_charts?.slice(0, 2).map((chart, index) => (
                <AreaBlock chart={chart} key={`area-${index}`} />
              ))}

              {result.charts?.histograms?.slice(0, 4).map((chart, index) => (
                <HistBlock chart={chart} key={`hist-${index}`} />
              ))}

              {result.charts?.scatter_plots?.map((chart, index) => (
                <ScatterBlock chart={chart} key={`scatter-${index}`} />
              ))}

              <Heatmap result={result} />
            </div>
          </Card>

          <Card title="Advanced Numeric Statistics">
            <pre>
              {JSON.stringify(result.statistics?.numeric_summary || {}, null, 2)}
            </pre>
          </Card>

          <Card title="Categorical Statistics">
            <pre>
              {JSON.stringify(
                result.statistics?.categorical_summary || {},
                null,
                2
              )}
            </pre>
          </Card>

          <Card title="Detected Schema">
            <pre>{JSON.stringify(result.schema || {}, null, 2)}</pre>
          </Card>
        </>
      )}
    </section>
  );

  const MLStudio = () => (
    <section>
      <Header
        title="ML Studio"
        subtitle="Train models, compare algorithms, and save models for predictions."
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
              <button className="primary" onClick={trainModel} disabled={mlLoading}>
                {mlLoading ? "Training..." : "Train Model"}
              </button>

              <button
                className="primary green"
                onClick={compareModels}
                disabled={compareLoading}
              >
                {compareLoading ? "Comparing..." : "Compare Models"}
              </button>
            </div>
          </Card>

          {mlError && (
            <div className="warning">
              <h4>ML Error</h4>
              <p>{mlError}</p>
            </div>
          )}

          {mlResult && (
            <>
              <div className="summary-grid">
                <Summary label="Task Type" value={mlResult.task_type} />
                <Summary label="Algorithm" value={mlResult.algorithm} />
                <Summary label="Rows Used" value={mlResult.rows_used} />
                <Summary label="Test Rows" value={mlResult.test_rows} />
              </div>

              <Card title="Saved Model">
                <div className="result-box">
                  <span>Model ID</span>
                  <strong>{mlResult.model_id}</strong>
                </div>
                <InsightList items={mlResult.model_explanation || []} />
              </Card>

              <Card title="Model Metrics">
                <MetricGrid metrics={mlResult.metrics} />
              </Card>

              {!!mlResult.feature_importance?.length && (
                <Card title="Feature Importance">
                  <FeatureList
                    items={mlResult.feature_importance.map((item) => ({
                      label: item.feature,
                      value: item.importance,
                    }))}
                  />
                </Card>
              )}

              <Card title="Training Details">
                <FeatureList
                  items={[
                    { label: "Target", value: mlResult.target_column },
                    { label: "Train Rows", value: mlResult.train_rows },
                    { label: "Test Size", value: mlResult.test_size },
                    {
                      label: "Numeric Features",
                      value: mlResult.numeric_features?.join(", ") || "None",
                    },
                    {
                      label: "Categorical Features",
                      value: mlResult.categorical_features?.join(", ") || "None",
                    },
                    {
                      label: "Dropped Columns",
                      value:
                        mlResult.dropped_high_cardinality_columns?.join(", ") ||
                        "None",
                    },
                  ]}
                />
              </Card>
            </>
          )}

          {compareResult && (
            <Card title="AutoML Compare">
              <div className="summary-grid">
                <Summary label="Best Model" value={compareResult.best_model} />
                <Summary label="Task Type" value={compareResult.task_type} />
                <Summary label="Metric" value={compareResult.ranking_metric} />
                <Summary
                  label="Rows Used"
                  value={compareResult.rows_used_for_compare}
                />
              </div>

              <Leaderboard rows={compareResult.leaderboard || []} />
              <InsightList items={compareResult.warnings || []} />
            </Card>
          )}
        </>
      )}
    </section>
  );

  const PredictionStudio = () => (
    <section>
      <Header
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

            <a
              className="download-btn"
              href={`${API_BASE_URL}/api/download-predictions/${predictionResult.prediction_filename}`}
            >
              Download Predictions CSV
            </a>
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

  function Header({ title, subtitle }) {
    return (
      <div className="page-header">
        <div>
          <p className="eyebrow">{title}</p>
          <h2>{subtitle}</h2>
        </div>

        <button
          className="back-btn"
          onClick={() => setCurrentPage("dashboard")}
        >
          Back to Dashboard
        </button>
      </div>
    );
  }

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

  function FeatureList({ items }) {
    if (!items?.length) {
      return <div className="empty-small">No details available.</div>;
    }

    return (
      <div className="feature-list">
        {items.map((item, index) => (
          <div className="feature-row" key={`${item.label}-${index}`}>
            <span>{item.label}</span>
            <strong>{displayValue(item.value)}</strong>
          </div>
        ))}
      </div>
    );
  }

  function Leaderboard({ rows }) {
    if (!rows?.length) {
      return <div className="empty-small">No comparison results available.</div>;
    }

    return (
      <div className="leaderboard">
        <div className="leaderboard-head">
          <span>Rank</span>
          <span>Algorithm</span>
          <span>Score</span>
          <span>Metrics</span>
        </div>

        {rows.map((row, index) => (
          <div className="leaderboard-row" key={`${row.algorithm}-${index}`}>
            <span>#{index + 1}</span>
            <code>{row.algorithm}</code>
            <strong>{row.score ?? "Failed"}</strong>
            <code>{row.error || JSON.stringify(row.metrics || {})}</code>
          </div>
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

  function BarBlock({ chart, label = "Bar" }) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h4>{chart.title}</h4>
          <span>{label}</span>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2b2416" />
            <XAxis
              dataKey={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cfc4a9" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cfc4a9" }} />
            <Tooltip {...tooltipProps} />
            <Bar dataKey={chart.y_axis} radius={[8, 8, 0, 0]} fill="#f0b429" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  function LineBlock({ chart }) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h4>{chart.title}</h4>
          <span>Line</span>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2b2416" />
            <XAxis
              dataKey={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cfc4a9" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cfc4a9" }} />
            <Tooltip {...tooltipProps} />
            <Line
              type="monotone"
              dataKey={chart.y_axis}
              stroke="#c98a2d"
              strokeWidth={3}
              dot
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  function AreaBlock({ chart }) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h4>{chart.title}</h4>
          <span>Area</span>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2b2416" />
            <XAxis
              dataKey={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cfc4a9" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cfc4a9" }} />
            <Tooltip {...tooltipProps} />
            <Area
              type="monotone"
              dataKey={chart.y_axis}
              stroke="#f0b429"
              fill="#f0b429"
              fillOpacity={0.25}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  }

  function PieBlock({ chart }) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h4>{chart.title}</h4>
          <span>Pie</span>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={chart.data}
              dataKey={chart.value_key}
              nameKey={chart.name_key}
              outerRadius={100}
              label
            >
              {chart.data.map((_, index) => (
                <Cell key={index} fill={pieColors[index % pieColors.length]} />
              ))}
            </Pie>
            <Tooltip {...tooltipProps} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  }

  function HistBlock({ chart }) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h4>{chart.title}</h4>
          <span>Histogram</span>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2b2416" />
            <XAxis
              dataKey="range"
              tick={{ fontSize: 10, fill: "#cfc4a9" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cfc4a9" }} />
            <Tooltip {...tooltipProps} />
            <Bar dataKey="count" radius={[8, 8, 0, 0]} fill="#e07a2d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  function ScatterBlock({ chart }) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h4>{chart.title}</h4>
          <span>Scatter</span>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart>
            <CartesianGrid stroke="#2b2416" />
            <XAxis
              type="number"
              dataKey={chart.x_axis}
              name={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cfc4a9" }}
            />
            <YAxis
              type="number"
              dataKey={chart.y_axis}
              name={chart.y_axis}
              tick={{ fontSize: 11, fill: "#cfc4a9" }}
            />
            <Tooltip {...tooltipProps} cursor={{ stroke: "rgba(240, 180, 41, 0.4)", strokeDasharray: "3 3" }} />
            <Scatter data={chart.data} fill="#22c55e" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  }

  function Heatmap({ result }) {
    const heatmap = result?.charts?.correlation_heatmap;

    if (!heatmap?.data?.length) return null;

    return (
      <div className="chart-card full">
        <div className="chart-head">
          <h4>{heatmap.title}</h4>
          <span>Correlation</span>
        </div>

        <div className="heatmap-grid">
          {heatmap.data.map((cell, index) => {
            const intensity = Math.min(Math.abs(cell.value), 1);

            const background =
              cell.value >= 0
                ? `rgba(240, 180, 41, ${0.12 + intensity * 0.55})`
                : `rgba(224, 122, 45, ${0.12 + intensity * 0.55})`;

            return (
              <div className="heatmap-cell" style={{ background }} key={index}>
                <span>
                  {cell.y} × {cell.x}
                </span>
                <strong>{cell.value}</strong>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">S</div>

          <div>
            <h1>StatMind AI</h1>
            <p>Data Intelligence OS</p>
          </div>
        </div>

        <nav className="side-nav">
          {[
            ["dashboard", "Dashboard"],
            ["statistics", "Statistics Studio"],
            ["ml", "ML Studio"],
            ["prediction", "Prediction Studio"],
            ["mlops", "MLOps Studio"],
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
      </aside>

      <main className="main">
        {currentPage === "dashboard" && <Dashboard />}
        {currentPage === "statistics" && <Statistics />}
        {currentPage === "ml" && <MLStudio />}
        {currentPage === "prediction" && <PredictionStudio />}
        {currentPage === "mlops" && <MLOpsStudio />}
      </main>
    </div>
  );
}

export default App;
