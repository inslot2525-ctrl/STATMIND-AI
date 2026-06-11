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
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState("dashboard");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [file, setFile] = useState(null);
  const [domain, setDomain] = useState("business");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [frontendError, setFrontendError] = useState("");

  const pieColors = [
    "#38bdf8",
    "#a855f7",
    "#ec4899",
    "#22c55e",
    "#fb923c",
    "#facc15",
    "#14b8a6",
    "#818cf8",
  ];

  const handleLogin = (e) => {
    e.preventDefault();

    if (!email.trim() || !password.trim()) {
      alert("Enter email and password.");
      return;
    }

    setIsLoggedIn(true);
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
    } catch (error) {
      console.error("Frontend analysis error:", error);
      setFrontendError(error.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const Dashboard = () => (
    <section>
      <div className="hero">
        <p className="eyebrow">Welcome to StatMind AI</p>
        <h2>Your explainable data science command center.</h2>
        <p>
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
        subtitle="Model training will be connected after Statistics Studio is stable."
      />

      <div className="empty">
        <h3>ML Studio temporarily paused</h3>
        <p>
          First we are fixing the frontend upload-analysis flow. Once this works,
          ML Studio can be reconnected safely.
        </p>
      </div>
    </section>
  );

  const PredictionStudio = () => (
    <section>
      <Header
        title="Prediction Studio"
        subtitle="Prediction flow will be reconnected after ML Studio."
      />

      <div className="empty">
        <h3>Prediction Studio temporarily paused</h3>
        <p>
          After statistics and ML are stable, prediction upload will be connected.
        </p>
      </div>
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
            <CartesianGrid strokeDasharray="3 3" stroke="#26324a" />
            <XAxis
              dataKey={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cbd5e1" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cbd5e1" }} />
            <Tooltip />
            <Bar dataKey={chart.y_axis} radius={[8, 8, 0, 0]} fill="#38bdf8" />
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
            <CartesianGrid strokeDasharray="3 3" stroke="#26324a" />
            <XAxis
              dataKey={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cbd5e1" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cbd5e1" }} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey={chart.y_axis}
              stroke="#a855f7"
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
            <CartesianGrid strokeDasharray="3 3" stroke="#26324a" />
            <XAxis
              dataKey={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cbd5e1" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cbd5e1" }} />
            <Tooltip />
            <Area
              type="monotone"
              dataKey={chart.y_axis}
              stroke="#38bdf8"
              fill="#38bdf8"
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
            <Tooltip />
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
            <CartesianGrid strokeDasharray="3 3" stroke="#26324a" />
            <XAxis
              dataKey="range"
              tick={{ fontSize: 10, fill: "#cbd5e1" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#cbd5e1" }} />
            <Tooltip />
            <Bar dataKey="count" radius={[8, 8, 0, 0]} fill="#ec4899" />
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
            <CartesianGrid stroke="#26324a" />
            <XAxis
              type="number"
              dataKey={chart.x_axis}
              name={chart.x_axis}
              tick={{ fontSize: 11, fill: "#cbd5e1" }}
            />
            <YAxis
              type="number"
              dataKey={chart.y_axis}
              name={chart.y_axis}
              tick={{ fontSize: 11, fill: "#cbd5e1" }}
            />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
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
                ? `rgba(56, 189, 248, ${0.12 + intensity * 0.55})`
                : `rgba(236, 72, 153, ${0.12 + intensity * 0.55})`;

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

  if (!isLoggedIn) {
    return (
      <div className="auth-page">
        <div className="orb orb-one"></div>
        <div className="orb orb-two"></div>

        <div className="auth-card">
          <div className="logo-block">
            <div className="logo-icon">S</div>

            <div>
              <h1>StatMind AI</h1>
              <p>Analytics + ML intelligence workspace</p>
            </div>
          </div>

          <div className="auth-copy">
            <h2>Turn raw datasets into decisions.</h2>
            <p>
              Access statistics, ML, AutoML, prediction, leakage detection, and
              an AI analyst workspace.
            </p>
          </div>

          <form className="login-form" onSubmit={handleLogin}>
            <label>Email</label>
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />

            <label>Password</label>
            <input
              type="password"
              placeholder="Enter any password for demo"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <button type="submit">Enter Workspace</button>
          </form>

          <p className="demo-note">
            Demo login only. Backend authentication can be added later.
          </p>
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
          ].map(([key, label]) => (
            <button
              key={key}
              className={currentPage === key ? "active" : ""}
              onClick={() => setCurrentPage(key)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="user-chip">
          <div className="avatar">{email.charAt(0).toUpperCase()}</div>

          <div>
            <strong>{email}</strong>
            <span>Active session</span>
          </div>
        </div>

        <button className="logout-btn" onClick={() => setIsLoggedIn(false)}>
          Logout
        </button>
      </aside>

      <main className="main">
        {currentPage === "dashboard" && <Dashboard />}
        {currentPage === "statistics" && <Statistics />}
        {currentPage === "ml" && <MLStudio />}
        {currentPage === "prediction" && <PredictionStudio />}
      </main>
    </div>
  );
}

export default App;