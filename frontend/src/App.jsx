import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);
    const res = await fetch("http://localhost:8001/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setHistory([
      ...history,
      {
        question,
        answer: data.answer,
        context: data.context,
      },
    ]);
    setQuestion("");
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h2>AI Tutor Chat (RAG)</h2>
      <div style={{ marginBottom: 20 }}>
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && askQuestion()}
          placeholder="Ask a question..."
          style={{ width: "80%", padding: 8 }}
        />
        <button onClick={askQuestion} disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>
      <div>
        {history.map((item, idx) => (
          <div key={idx} style={{ marginBottom: 24, padding: 16, border: "1px solid #eee", borderRadius: 8 }}>
            <div><strong>You:</strong> {item.question}</div>
            <div style={{ marginTop: 8 }}><strong>AI:</strong> {item.answer}</div>
            <div style={{ marginTop: 8, fontSize: "0.95em", color: "#555" }}>
              <strong>Context:</strong>
              <ul>
                {item.context.map((ctx, i) => (
                  <li key={i}>{ctx}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
