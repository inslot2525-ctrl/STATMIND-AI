import os
import uuid
import html
from datetime import datetime

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


def _safe(value):
    if value is None:
        return "N/A"

    return html.escape(str(value))


def _render_list(items):
    if not items:
        return "<p class='muted'>No data available.</p>"

    html_items = ""

    for item in items:
        html_items += f"<li>{_safe(item)}</li>"

    return f"<ul>{html_items}</ul>"


def _render_dict_cards(data):
    if not data:
        return "<p class='muted'>No data available.</p>"

    cards = ""

    for key, value in data.items():
        if isinstance(value, (list, dict)):
            continue

        cards += f"""
        <div class="metric-card">
          <span>{_safe(key).replace("_", " ").title()}</span>
          <strong>{_safe(value)}</strong>
        </div>
        """

    return f"<div class='metric-grid'>{cards}</div>"


def _render_table(rows, columns=None, max_rows=12):
    if not rows:
        return "<p class='muted'>No rows available.</p>"

    rows = rows[:max_rows]

    if columns is None:
        columns = list(rows[0].keys())

    thead = "".join([f"<th>{_safe(col)}</th>" for col in columns])

    tbody = ""

    for row in rows:
        cells = ""

        for col in columns:
            cells += f"<td>{_safe(row.get(col, ''))}</td>"

        tbody += f"<tr>{cells}</tr>"

    return f"""
    <div class="table-wrapper">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>{tbody}</tbody>
      </table>
    </div>
    """


def generate_intelligence_report(context: dict):
    """
    Generates an HTML report from frontend context.

    Expected context keys:
    - dataset
    - schema
    - statistics
    - domainInsights
    - targetAdvice
    - mlResult
    - compareResult
    - predictionResult
    - leakageResult
    """

    dataset = context.get("dataset") or {}
    schema = context.get("schema") or {}
    statistics = context.get("statistics") or {}
    domain_insights = context.get("domainInsights") or context.get("domain_insights") or []
    target_advice = context.get("targetAdvice") or context.get("target_advice") or {}
    ml_result = context.get("mlResult") or context.get("ml_result") or {}
    compare_result = context.get("compareResult") or context.get("compare_result") or {}
    prediction_result = context.get("predictionResult") or context.get("prediction_result") or {}
    leakage_result = context.get("leakageResult") or context.get("leakage_result") or {}

    filename = f"statmind_report_{uuid.uuid4().hex[:8]}.html"
    file_path = os.path.join(REPORT_DIR, filename)

    best_target = target_advice.get("best_target")
    data_quality = statistics.get("data_quality") or {}
    numeric_summary = statistics.get("numeric_summary") or {}
    categorical_summary = statistics.get("categorical_summary") or {}
    dataset_intelligence = statistics.get("dataset_intelligence") or []

    ml_metrics = ml_result.get("metrics") or {}
    external_metrics = prediction_result.get("external_test_metrics") or {}
    leakage_findings = leakage_result.get("findings") or []
    prediction_preview = prediction_result.get("preview") or []

    leakage_rows = []

    for finding in leakage_findings[:15]:
        leakage_rows.append(
            {
                "severity": finding.get("severity"),
                "feature": finding.get("feature"),
                "type": finding.get("type"),
                "message": finding.get("message"),
                "recommendation": finding.get("recommendation"),
            }
        )

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>StatMind AI Intelligence Report</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #020617;
      color: #f8fafc;
    }}

    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 24px;
    }}

    .hero {{
      padding: 34px;
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(168, 85, 247, 0.18));
      border: 1px solid rgba(255, 255, 255, 0.12);
      margin-bottom: 24px;
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
    }}

    h2 {{
      margin-top: 34px;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.14);
    }}

    h3 {{
      margin-bottom: 8px;
    }}

    p, li, td {{
      color: #cbd5e1;
      line-height: 1.6;
    }}

    .muted {{
      color: #94a3b8;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin: 18px 0;
    }}

    .metric-card {{
      padding: 18px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.12);
    }}

    .metric-card span {{
      display: block;
      color: #94a3b8;
      font-size: 13px;
      margin-bottom: 8px;
    }}

    .metric-card strong {{
      font-size: 22px;
      color: #ffffff;
      word-break: break-word;
    }}

    .section {{
      padding: 24px;
      border-radius: 22px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      margin-bottom: 22px;
    }}

    .badge {{
      display: inline-block;
      padding: 7px 11px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.35);
    }}

    .risk-critical, .risk-high {{
      color: #fb7185;
    }}

    .risk-medium {{
      color: #fb923c;
    }}

    .risk-low {{
      color: #22c55e;
    }}

    .table-wrapper {{
      overflow-x: auto;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      margin-top: 14px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}

    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      vertical-align: top;
    }}

    th {{
      background: rgba(255, 255, 255, 0.08);
      color: #ffffff;
    }}

    code {{
      background: rgba(255,255,255,0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}

    @media print {{
      body {{
        background: white;
        color: black;
      }}

      .section, .hero, .metric-card {{
        background: white;
        color: black;
        border: 1px solid #ddd;
      }}

      p, li, td {{
        color: #333;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <span class="badge">StatMind AI Report</span>
      <h1>Dataset Intelligence Report</h1>
      <p>Generated on {_safe(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
      <p>This report summarizes dataset statistics, target recommendation, model results, leakage risks, and prediction outputs.</p>
    </div>

    <div class="section">
      <h2>1. Dataset Overview</h2>
      <div class="metric-grid">
        <div class="metric-card"><span>Rows</span><strong>{_safe(dataset.get("rows"))}</strong></div>
        <div class="metric-card"><span>Columns</span><strong>{_safe(dataset.get("columns"))}</strong></div>
        <div class="metric-card"><span>Domain</span><strong>{_safe(dataset.get("domain"))}</strong></div>
        <div class="metric-card"><span>Data Quality</span><strong>{_safe(data_quality.get("data_quality_score"))}/100</strong></div>
      </div>
      <h3>Dataset Intelligence</h3>
      {_render_list(dataset_intelligence)}
      <h3>Domain Insights</h3>
      {_render_list(domain_insights)}
    </div>

    <div class="section">
      <h2>2. Schema Summary</h2>
      <div class="metric-grid">
        <div class="metric-card"><span>Numeric Columns</span><strong>{len(schema.get("numeric_columns", []))}</strong></div>
        <div class="metric-card"><span>Categorical Columns</span><strong>{len(schema.get("categorical_columns", []))}</strong></div>
        <div class="metric-card"><span>Date Columns</span><strong>{len(schema.get("datetime_columns", []))}</strong></div>
        <div class="metric-card"><span>Text Columns</span><strong>{len(schema.get("text_columns", []))}</strong></div>
      </div>
    </div>

    <div class="section">
      <h2>3. Target Recommendation</h2>
      {
        f'''
        <div class="metric-grid">
          <div class="metric-card"><span>Best Target</span><strong>{_safe(best_target.get("column"))}</strong></div>
          <div class="metric-card"><span>Task Type</span><strong>{_safe(best_target.get("task_type"))}</strong></div>
          <div class="metric-card"><span>Score</span><strong>{_safe(best_target.get("score"))}/100</strong></div>
          <div class="metric-card"><span>Priority</span><strong>{_safe(best_target.get("priority"))}</strong></div>
        </div>
        <h3>Reasons</h3>
        {_render_list(best_target.get("reasons"))}
        <h3>Warnings</h3>
        {_render_list(best_target.get("warnings"))}
        '''
        if best_target else "<p class='muted'>Target Advisor was not run.</p>"
      }
    </div>

    <div class="section">
      <h2>4. ML Training Results</h2>
      <div class="metric-grid">
        <div class="metric-card"><span>Model ID</span><strong>{_safe(ml_result.get("model_id"))}</strong></div>
        <div class="metric-card"><span>Algorithm</span><strong>{_safe(ml_result.get("algorithm"))}</strong></div>
        <div class="metric-card"><span>Task Type</span><strong>{_safe(ml_result.get("task_type"))}</strong></div>
        <div class="metric-card"><span>Target</span><strong>{_safe(ml_result.get("target_column"))}</strong></div>
      </div>
      <h3>Internal Test Metrics</h3>
      {_render_dict_cards(ml_metrics)}
      <h3>Model Explanation</h3>
      {_render_list(ml_result.get("model_explanation"))}
    </div>

    <div class="section">
      <h2>5. AutoML Comparison</h2>
      <div class="metric-grid">
        <div class="metric-card"><span>Best Model</span><strong>{_safe(compare_result.get("best_model"))}</strong></div>
        <div class="metric-card"><span>Ranking Metric</span><strong>{_safe(compare_result.get("ranking_metric"))}</strong></div>
        <div class="metric-card"><span>Rows Used</span><strong>{_safe(compare_result.get("rows_used_for_compare"))}</strong></div>
        <div class="metric-card"><span>Target</span><strong>{_safe(compare_result.get("target_column"))}</strong></div>
      </div>
    </div>

    <div class="section">
      <h2>6. Leakage Risk Analysis</h2>
      <div class="metric-grid">
        <div class="metric-card"><span>Risk Level</span><strong>{_safe(leakage_result.get("risk_level"))}</strong></div>
        <div class="metric-card"><span>Risk Score</span><strong>{_safe(leakage_result.get("leakage_risk_score"))}/100</strong></div>
        <div class="metric-card"><span>Findings</span><strong>{len(leakage_findings)}</strong></div>
        <div class="metric-card"><span>Rows Checked</span><strong>{_safe(leakage_result.get("rows_checked"))}</strong></div>
      </div>
      <h3>Summary</h3>
      {_render_list(leakage_result.get("summary"))}
      <h3>Findings</h3>
      {_render_table(leakage_rows, ["severity", "feature", "type", "message", "recommendation"])}
    </div>

    <div class="section">
      <h2>7. Prediction Studio Output</h2>
      <div class="metric-grid">
        <div class="metric-card"><span>Prediction File</span><strong>{_safe(prediction_result.get("prediction_filename"))}</strong></div>
        <div class="metric-card"><span>Rows Predicted</span><strong>{_safe(prediction_result.get("rows_predicted"))}</strong></div>
        <div class="metric-card"><span>Prediction Column</span><strong>{_safe(prediction_result.get("prediction_column"))}</strong></div>
        <div class="metric-card"><span>Task Type</span><strong>{_safe(prediction_result.get("task_type"))}</strong></div>
      </div>
      <h3>External Test Metrics</h3>
      {_render_dict_cards(external_metrics)}
      <h3>Prediction Preview</h3>
      {_render_table(prediction_preview)}
    </div>

    <div class="section">
      <h2>8. Final Recommendations</h2>
      <ul>
        <li>Check leakage findings before trusting high model performance.</li>
        <li>Prefer external test metrics over only internal train-test split metrics.</li>
        <li>Remove identifier-like columns unless they represent valid predictive information.</li>
        <li>Use the report together with domain knowledge before making business, finance, or research decisions.</li>
      </ul>
    </div>
  </div>
</body>
</html>
    """

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    return {
        "report_filename": filename,
        "report_path": file_path,
        "message": "Intelligence report generated successfully.",
    }