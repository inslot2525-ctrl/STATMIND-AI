import { useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [domain, setDomain] = useState("business");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!file) {
      alert("Please upload a CSV or Excel file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("domain", domain);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/api/analyze",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Analysis failed. Check backend terminal.");
    } finally {
      setLoading(false);
    }
  };

  const getFirstCategoricalChartData = () => {
    if (!result?.statistics?.categorical_summary) return [];

    const summaries = result.statistics.categorical_summary;
    const firstColumn = Object.keys(summaries)[0];

    if (!firstColumn) return [];

    const topValues = summaries[firstColumn].top_values;

    return Object.entries(topValues).map(([name, value]) => ({
      name,
      value
    }));
  };

  const chartData = getFirstCategoricalChartData();

  return (
    <div className="app-container">
      <h1>StatMind AI</h1>
      <p className="subtitle">
        Multi-domain statistical analyst for Business, Research, and Finance data.
      </p>

      <div className="upload-card">
        <label>Upload CSV/Excel File</label>
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <label>Select Domain</label>
        <select value={domain} onChange={(e) => setDomain(e.target.value)}>
          <option value="business">Business</option>
          <option value="research">Research</option>
          <option value="finance">Finance</option>
        </select>

        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze File"}
        </button>
      </div>

      {result && (
        <div className="dashboard">
          <h2>Analysis Dashboard</h2>

          <div className="cards">
            <div className="card">
              <h3>Rows</h3>
              <p>{result.rows}</p>
            </div>

            <div className="card">
              <h3>Columns</h3>
              <p>{result.columns}</p>
            </div>

            <div className="card">
              <h3>Domain</h3>
              <p>{result.domain}</p>
            </div>

            <div className="card">
              <h3>Data Quality</h3>
              <p>{result.statistics.data_quality.data_quality_score}/100</p>
            </div>
          </div>

          <section>
            <h3>Detected Schema</h3>
            <pre>{JSON.stringify(result.schema, null, 2)}</pre>
          </section>

          <section>
            <h3>Numeric Statistics</h3>
            <pre>{JSON.stringify(result.statistics.numeric_summary, null, 2)}</pre>
          </section>

          <section>
            <h3>Domain Insights</h3>
            <ul>
              {result.domain_insights.map((insight, index) => (
                <li key={index}>{insight}</li>
              ))}
            </ul>
          </section>

          <section>
            <h3>Chart Recommendations</h3>
            <ul>
              {result.chart_recommendations.map((chart, index) => (
                <li key={index}>
                  <strong>{chart.chart_type}</strong>: {chart.reason}
                </li>
              ))}
            </ul>
          </section>

          {chartData.length > 0 && (
            <section>
              <h3>Sample Category Chart</h3>
              <div className="chart-box">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

export default App;