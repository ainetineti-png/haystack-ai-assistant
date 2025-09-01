import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [ingestProgress, setIngestProgress] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history]);

  // Poll backend for ingest status
  useEffect(() => {
    let poller;
    if (ingesting) {
      poller = setInterval(async () => {
        try {
          const res = await fetch("http://localhost:8000/ingest_status");
          const data = await res.json();
          setIngestProgress(data);
          if (data.percent >= 100) {
            setIngesting(false);
            setTimeout(() => setIngestProgress(null), 2000);
          }
        } catch (err) {
          setIngestProgress(null);
        }
      }, 500);
    }
    return () => poller && clearInterval(poller);
  }, [ingesting]);

  const askQuestion = async () => {
    if (!question.trim()) return;
    
    const userMessage = { type: "user", content: question };
    setHistory(prev => [...prev, userMessage]);
    setLoading(true);
    setQuestion("");

    try {
      // Set a longer timeout for the fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout
      
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || `Server error: ${res.status}`);
      }
      
      const data = await res.json();
      
      const aiMessage = {
        type: "ai",
        content: data.answer,
        context: data.context,
        documentsFound: data.documents_found,
        processingTime: data.processing_time,
        citations: data.citations
      };
      
      setHistory(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error during question processing:", error);
      
      let errorContent = "Sorry, I couldn't process your question. Please try again.";
      
      // Provide more specific error messages based on the error type
      if (error.name === "AbortError") {
        errorContent = "The request took too long to complete. This might be due to high server load or a complex question. Please try again or simplify your question.";
      } else if (error.message && error.message.includes("Ollama")) {
        errorContent = `Error communicating with the AI model: ${error.message}. Please ensure Ollama is running properly.`;
      } else if (!navigator.onLine) {
        errorContent = "You appear to be offline. Please check your internet connection and try again.";
      }
      
      const errorMessage = {
        type: "error",
        content: errorContent
      };
      
      setHistory(prev => [...prev, errorMessage]);
    }
    
    setLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const reloadAllDocs = async () => {
    setStatusMsg("Reloading all documents...");
    setIngesting(true);
    setIngestProgress({ total: 0, processed: 0, percent: 0 });
    try {
      const res = await fetch("http://localhost:8000/ingest");
      const data = await res.json();
      setStatusMsg(`Reloaded: ${data.documents_loaded} documents.`);
      setTimeout(() => setStatusMsg(""), 3000);
    } catch (err) {
      setStatusMsg("Error reloading documents.");
      setTimeout(() => setStatusMsg(""), 3000);
    }
  };

  const incrementalIndex = async () => {
    setStatusMsg("Incremental indexing...");
    setIngesting(true);
    setIngestProgress({ total: 0, processed: 0, percent: 0 });
    try {
      const res = await fetch("http://localhost:8000/ingest_incremental");
      const data = await res.json();
      setStatusMsg(`Indexed: ${data.new_documents} new, total: ${data.total_documents} documents.`);
      setTimeout(() => setStatusMsg(""), 3000);
    } catch (err) {
      setStatusMsg("Error with incremental indexing.");
      setTimeout(() => setStatusMsg(""), 3000);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div className="header-content">
          <h1>🤖 AI Knowledge Assistant</h1>
          <p>Ask me anything about your documents</p>
          <div style={{ marginTop: 10 }}>
            <button onClick={reloadAllDocs} className="reload-btn" disabled={ingesting}>Reload All Docs</button>
            <button onClick={incrementalIndex} className="reload-btn" style={{ marginLeft: 8 }} disabled={ingesting}>Incremental Index</button>
          </div>
          {statusMsg && <div className="status-msg">{statusMsg}</div>}
          {ingestProgress && (
            <div className="ingest-progress-bar" style={{ marginTop: 10 }}>
              <div style={{ width: "100%", background: "#eee", borderRadius: 6, height: 18 }}>
                <div style={{ width: `${ingestProgress.percent}%`, background: "#4caf50", height: 18, borderRadius: 6, transition: "width 0.3s" }}></div>
              </div>
              <div style={{ fontSize: 13, marginTop: 2 }}>
                {ingestProgress.processed} / {ingestProgress.total} files processed ({ingestProgress.percent}%)
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chat Container */}
      <div className="chat-container">
        <div className="chat-messages">
          {history.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">💡</div>
              <h2>Welcome to your AI Assistant!</h2>
              <p>I can help you find information from your knowledge base. Try asking questions like:</p>
              <div className="example-questions">
                <span className="example">• "What is Python?"</span>
                <span className="example">• "Tell me about FastAPI features"</span>
                <span className="example">• "How does modularity work?"</span>
              </div>
            </div>
          )}

          {history.map((item, idx) => (
            <div key={idx} className={`message ${item.type}`}>
              {item.type === "user" && (
                <div className="message-content">
                  <div className="message-avatar user-avatar">👤</div>
                  <div className="message-text user-text">{item.content}</div>
                </div>
              )}
              
              {item.type === "ai" && (
                <div className="message-content">
                  <div className="message-avatar ai-avatar">🤖</div>
                  <div className="message-text ai-text">
                    <div className="ai-response">{item.content}</div>
                    {item.processingTime && (
                      <div className="processing-time">
                        ⏱️ Response time: {item.processingTime}s
                      </div>
                    )}
                    {item.context && item.context.length > 0 && (
                      <div className="context-section">
                        <div className="context-header">
                          📚 Sources ({item.documentsFound} document{item.documentsFound !== 1 ? 's' : ''} found):
                        </div>
                        <div className="context-list">
                          {item.context.map((ctx, i) => (
                            <div key={i} className="context-item">
                              <div className="context-filename">
                                📄 {ctx.filename} {ctx.page ? `(page ${ctx.page})` : ''}
                              </div>
                              <div className="context-preview">
                                {ctx.content}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {item.type === "error" && (
                <div className="message-content">
                  <div className="message-avatar error-avatar">❌</div>
                  <div className="message-text error-text">
                    <div>{item.content}</div>
                    {item.content.includes("Ollama") && (
                      <div className="error-help-text">
                        <p><strong>Troubleshooting tips:</strong></p>
                        <ul>
                          <li>Ensure Ollama is running on your system</li>
                          <li>Try restarting the Ollama service</li>
                          <li>Check if the model is properly loaded in Ollama</li>
                          <li>For complex questions, try breaking them into smaller parts</li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message ai">
              <div className="message-content">
                <div className="message-avatar ai-avatar">🤖</div>
                <div className="message-text ai-text">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <span className="typing-text">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={chatEndRef} />
        </div>
      </div>

      {/* Input Section */}
      <div className="input-section">
        <div className="input-container">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question... (Press Enter to send)"
            className="question-input"
            rows="1"
            disabled={loading}
          />
          <button 
            onClick={askQuestion} 
            disabled={loading || !question.trim()}
            className="send-button"
          >
            {loading ? "⏳" : "🚀"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
