// Readable views of the dataset profile in Statistics Studio: one table of
// columns and types, one of numeric summaries, and a breakdown per category column.

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return value.toLocaleString();
  const digits = Math.abs(value) >= 100 ? 1 : Math.abs(value) >= 1 ? 2 : 4;
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function percent(part, whole) {
  if (!whole) return "";
  const share = (part / whole) * 100;
  return share > 0 && share < 0.1 ? "<0.1%" : `${share.toFixed(share < 10 ? 1 : 0)}%`;
}

// First match wins: a text column that parses as dates is a date
const TYPE_LABELS = [
  ["date_like_columns", "Date"],
  ["numeric_columns", "Number"],
  ["categorical_columns", "Category"],
  ["datetime_columns", "Date"],
  ["text_columns", "Text or ID"],
];

export function ColumnOverview({ schema, rows }) {
  const columns = Object.keys(schema?.dtypes || {});
  if (!columns.length) return <div className="empty-small">No columns found.</div>;

  const typeOf = (column) =>
    TYPE_LABELS.find(([key]) => (schema[key] || []).includes(column))?.[1] || "Other";

  return (
    <div className="table-wrapper">
      <table className="preview-table compact profile-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Type</th>
            <th>Missing values</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((column) => {
            const missing = schema.missing_values?.[column] || 0;
            return (
              <tr key={column}>
                <td>{column}</td>
                <td>{typeOf(column)}</td>
                <td>{missing === 0 ? "None" : `${missing.toLocaleString()} (${percent(missing, rows)})`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function NumericStats({ summary }) {
  const columns = Object.entries(summary || {});
  if (!columns.length) return <div className="empty-small">This dataset has no numeric columns.</div>;

  return (
    <>
      <div className="table-wrapper">
        <table className="preview-table compact stats-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Average</th>
              <th>Median</th>
              <th>Lowest</th>
              <th>Highest</th>
              <th>Typical range</th>
              <th>Spread</th>
              <th>Values</th>
            </tr>
          </thead>
          <tbody>
            {columns.map(([column, stats]) => (
              <tr key={column}>
                <td>{column}</td>
                <td>{formatNumber(stats.mean)}</td>
                <td>{formatNumber(stats.median)}</td>
                <td>{formatNumber(stats.min)}</td>
                <td>{formatNumber(stats.max)}</td>
                <td>{formatNumber(stats.q1)} to {formatNumber(stats.q3)}</td>
                <td>{formatNumber(stats.std)}</td>
                <td>{formatNumber(stats.count)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="card-footnote stats-footnote">
        Typical range holds the middle half of the values (25th to 75th percentile). Spread is the
        standard deviation. Values counts the rows that are not missing.
      </p>
    </>
  );
}

export function CategoricalStats({ summary, rows }) {
  const columns = Object.entries(summary || {});
  if (!columns.length) return <div className="empty-small">This dataset has no category columns.</div>;

  return (
    <div className="category-grid">
      {columns.map(([column, stats]) => {
        const categories = stats.top_categories || [];
        const largest = Math.max(1, ...categories.map((item) => item.count));
        const hidden = stats.unique_values - categories.filter((item) => item.category !== "nan").length;

        return (
          <div className="category-card" key={column}>
            <div className="category-head">
              <strong>{column}</strong>
              <span>
                {stats.unique_values} {stats.unique_values === 1 ? "value" : "different values"}
              </span>
            </div>

            {categories.map((item) => (
              <div className="category-row" key={item.category}>
                <span className="category-name" title={item.category}>
                  {item.category === "nan" ? "Missing" : item.category}
                </span>
                <div className="category-bar">
                  <div style={{ width: `${(item.count / largest) * 100}%` }} />
                </div>
                <span className="category-count">
                  {item.count.toLocaleString()} <em>{percent(item.count, rows)}</em>
                </span>
              </div>
            ))}

            {hidden > 0 && <p className="category-more">and {hidden} more, each less common</p>}
          </div>
        );
      })}
    </div>
  );
}
