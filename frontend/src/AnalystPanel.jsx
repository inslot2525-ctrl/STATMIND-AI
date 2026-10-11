// AI Analyst page: a chat that answers from everything StatMind has computed
// for the current dataset, plus the one-click HTML report.

import { useEffect, useRef, useState } from "react";
import { downloadArtifact, postJson } from "./api";

const SUGGESTIONS = [
  "Can I trust this model's score?",
  "Which features matter most, and why?",
  "Is there any sign of data leakage?",
  "What should I do next to improve the model?",
];

export default function AnalystPanel({ context, hasDataset }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages, loading]);

  const ask = async (text) => {
    const asked = text.trim();
    if (!asked || loading) return;

    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((current) => [...current, { role: "user", content: asked }]);
    setQuestion("");
    setError("");
    setLoading(true);

    try {
      const data = await postJson("/api/chat-analyst", { question: asked, context, history });
      setMode(data.analyst_mode === "llm" ? `Claude · ${data.model}` : "Built-in rules");
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: (data.answer || []).join("\n"),
          note: data.analyst_mode === "llm" ? "" : data.note,
        },
      ]);
    } catch (err) {
      setError(err.message || "The analyst could not answer.");
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async () => {
    setReportLoading(true);
    setError("");
    try {
      const data = await postJson("/api/generate-report", { context });
      await downloadArtifact(data.artifact_id);
    } catch (err) {
      setError(err.message || "Report generation failed.");
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <>
      <div className="card">
        <div className="analyst-head">
          <h3>Ask about your data and models</h3>
          {mode && <span className="analyst-mode">{mode}</span>}
        </div>

        {!hasDataset && (
          <div className="notice">
            <p>
              No dataset is loaded yet. Run Statistics Studio first, then train a model, so the
              analyst has results to reason about.
            </p>
          </div>
        )}

        <div className="chat-log" aria-live="polite">
          {messages.length === 0 && (
            <div className="chat-suggestions">
              {SUGGESTIONS.map((suggestion) => (
                <button type="button" key={suggestion} onClick={() => ask(suggestion)} disabled={loading}>
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {messages.map((message, index) => (
            <div className={`chat-message ${message.role}`} key={index}>
              <span className="chat-author">{message.role === "user" ? "You" : "Analyst"}</span>
              {message.content.split("\n").map((line, lineIndex) => (
                <p key={lineIndex}>{line}</p>
              ))}
              {message.note && <p className="chat-note">{message.note}</p>}
            </div>
          ))}

          {loading && (
            <div className="chat-message assistant">
              <span className="chat-author">Analyst</span>
              <p className="chat-note">Thinking…</p>
            </div>
          )}

          <div ref={endRef} />
        </div>

        {error && (
          <div className="warning">
            <p>{error}</p>
          </div>
        )}

        <form
          className="chat-form"
          onSubmit={(event) => {
            event.preventDefault();
            ask(question);
          }}
        >
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about this dataset or model"
            aria-label="Question for the analyst"
          />
          <button className="primary" type="submit" disabled={loading || !question.trim()}>
            Ask
          </button>
        </form>
      </div>

      <div className="card">
        <h3>Intelligence report</h3>
        <p>
          One HTML document with the dataset profile, data quality, model results, AutoML
          comparison and leakage findings gathered so far.
        </p>
        <button className="primary" type="button" onClick={generateReport} disabled={reportLoading || !hasDataset}>
          {reportLoading ? "Generating…" : "Generate and download report"}
        </button>
      </div>
    </>
  );
}
