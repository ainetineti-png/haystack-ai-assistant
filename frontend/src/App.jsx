import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history]);

  const askQuestion = async () => {
    if (!question.trim()) return;
    
    const userMessage = { type: "user", content: question };
    setHistory(prev => [...prev, userMessage]);
    setLoading(true);
    setQuestion("");

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      
      const aiMessage = {
        type: "ai",
        content: data.answer,
        context: data.context,
        documentsFound: data.documents_found
      };
      
      setHistory(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        type: "error",
        content: "Sorry, I couldn't process your question. Please try again."
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
            <button onClick={reloadAllDocs} className="reload-btn">Reload All Docs</button>
            <button onClick={incrementalIndex} className="reload-btn" style={{ marginLeft: 8 }}>Incremental Index</button>
          </div>
          {statusMsg && <div className="status-msg">{statusMsg}</div>}
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
                    {item.context && item.context.length > 0 && (
                      <div className="context-section">
                        <div className="context-header">
                          📚 Sources ({item.documentsFound} document{item.documentsFound !== 1 ? 's' : ''} found):
                        </div>
                        <div className="context-list">
                          {item.context.map((ctx, i) => (
                            <div key={i} className="context-item">
                              <div className="context-filename">
                                📄 {ctx.split(':')[0]}
                              </div>
                              <div className="context-preview">
                                {ctx.substring(ctx.indexOf(':') + 1).trim()}
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
                  <div className="message-text error-text">{item.content}</div>
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
