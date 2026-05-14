import { useState, useEffect, useRef } from 'react';

const API = 'http://localhost:8000';

function formatNum(n) {
  if (n === null || n === undefined) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function VideoCard({ id, data }) {
  if (!data) {
    return (
      <div className="video-card empty">
        <span>Video {id.toUpperCase()} not loaded</span>
      </div>
    );
  }

  const barW = Math.min((data.engagement_rate / 10) * 100, 100).toFixed(1);
  return (
    <div className="video-card">
      <img className="card-thumb" src={data.thumbnail} alt="thumb" loading="lazy" />
      <div className="card-body">
        <span className={`card-badge badge-${id}`}>Video {id.toUpperCase()}</span>
        <div className="card-title">{data.title}</div>
        <div className="card-stats">
          <div className="stat">
            <div className="stat-label">Views</div>
            <div className="stat-value">{formatNum(data.views)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Likes</div>
            <div className="stat-value">{formatNum(data.likes)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Comments</div>
            <div className="stat-value">{formatNum(data.comments)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Duration</div>
            <div className="stat-value">{data.duration}</div>
          </div>
        </div>
        <div className="eng-bar-wrap">
          <div className="eng-bar-track">
            <div className="eng-bar-fill" style={{ width: `${barW}%` }}></div>
          </div>
          <span className="eng-pct">{data.engagement_rate}% eng.</span>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [status, setStatus] = useState('checking...');
  const [videos, setVideos] = useState({ a: null, b: null });
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [urlA, setUrlA] = useState('');
  const [urlB, setUrlB] = useState('');
  const [ingestStatus, setIngestStatus] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const msgIdRef = useRef(0);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    checkHealth();
    const int = setInterval(checkHealth, 8000);
    return () => clearInterval(int);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API}/health`);
      if (res.ok) setStatus('ok');
      else setStatus('offline');
    } catch {
      setStatus('offline');
    }
  };

  const handleIngest = async () => {
    if (!urlA || !urlB) return alert("Please enter both URLs");
    setIsIngesting(true);
    setIngestStatus('⏳ Fetching metadata & transcripts (concurrent)...');

    try {
      const res = await fetch(`${API}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url_a: urlA, url_b: urlB }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data));

      setVideos({ a: data.videos.a, b: data.videos.b });
      setIngestStatus('✅ Done! Chunks embedded in ChromaDB.');
      setMessages([{ id: ++msgIdRef.current, role: 'system', content: `Videos loaded! Ready to chat.` }]);
    } catch (e) {
      setIngestStatus('❌ Error: ' + e.message);
    } finally {
      setIsIngesting(false);
    }
  };

  const handleSend = async (overrideText) => {
    const text = typeof overrideText === 'string' ? overrideText : input.trim();
    if (!text || isStreaming || (!videos.a && !videos.b)) return;

    setInput('');
    setIsStreaming(true);
    setMessages((prev) => [
      ...prev,
      { id: ++msgIdRef.current, role: 'user', content: text },
      { id: ++msgIdRef.current, role: 'assistant', content: '', sources: [], streaming: true },
    ]);

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n');
        buffer = parts.pop(); // keep incomplete trailing chunk

        for (const line of parts) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            setMessages((prev) => {
              const updated = prev.map((m, idx) => {
                if (idx !== prev.length - 1) return m;
                if (msg.type === 'sources') return { ...m, sources: msg.data };
                if (msg.type === 'token') return { ...m, content: m.content + msg.data };
                if (msg.type === 'done') return { ...m, streaming: false };
                return m;
              });
              return updated;
            });
          } catch {}
        }
      }
    } catch (e) {
      setMessages((prev) =>
        prev.map((m, idx) =>
          idx === prev.length - 1
            ? { ...m, content: 'Error: ' + e.message, streaming: false }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const resetChat = async () => {
    await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: '', reset: true }),
    });
    setMessages([{ id: ++msgIdRef.current, role: 'system', content: 'Chat history cleared.' }]);
  };

  return (
    <>
      <header>
        <div className="logo">
          VidRAG <span>— Video Intelligence</span>
        </div>
        <span className="status-label">{status}</span>
        <div className={`status-dot ${status === 'ok' ? 'ok' : 'err'}`}></div>
      </header>

      <div className="workspace">
        <div className="left-panel">
          <div className="ingest-bar">
            <h2>Load Videos</h2>
            <div className="url-inputs">
              <div className="url-field">
                <span className="url-label">Video A</span>
                <input
                  type="text"
                  value={urlA}
                  onChange={(e) => setUrlA(e.target.value)}
                  placeholder="YouTube URL"
                />
              </div>
              <div className="url-field">
                <span className="url-label">Video B</span>
                <input
                  type="text"
                  value={urlB}
                  onChange={(e) => setUrlB(e.target.value)}
                  placeholder="YouTube URL"
                />
              </div>
            </div>
            <button
              className="btn-ingest"
              onClick={handleIngest}
              disabled={isIngesting}
            >
              Ingest & Embed
            </button>
            {ingestStatus && (
              <div className="ingest-progress" style={{ display: 'block' }}>
                {ingestStatus}
              </div>
            )}
          </div>

          <div className="cards-area">
            <VideoCard id="a" data={videos.a} />
            <VideoCard id="b" data={videos.b} />
          </div>
        </div>

        <div className="chat-panel">
          <div className="chat-header">
            <h2>💬 RAG Chat</h2>
            <button className="btn-reset" onClick={resetChat}>
              Clear history
            </button>
          </div>

          <div className="suggestions">
            {['Why did A get more engagement?', 'Compare hooks', 'Suggest improvements for B'].map(
              (s) => (
                <div
                  key={s}
                  className="suggestion-chip"
                  onClick={() => {
                    setInput(s);
                    handleSend(s);
                  }}
                >
                  {s}
                </div>
              )
            )}
          </div>

          <div className="messages">
            {messages.length === 0 && (
              <div className="msg system" style={{ marginTop: 'auto', marginBottom: 'auto' }}>
                <span style={{ fontSize: '2rem' }}>🎬</span>
                <p style={{ marginTop: 8 }}>Paste URLs and ingest to start chatting.</p>
              </div>
            )}
            {messages.map((m) => (
              <div key={m.id} className={`msg ${m.role}`}>
                {m.role === 'system' ? (
                  m.content
                ) : (
                  <>
                    <div className="msg-bubble">
                      {m.content}
                      {m.streaming && <span className="cursor"></span>}
                    </div>
                    {m.sources && m.sources.length > 0 && (
                      <div className="sources">
                        {m.sources.map((s, idx) => (
                          <div key={idx} className="source-chip" title={s.excerpt}>
                            Video {s.video_id} · {(s.score * 100).toFixed(0)}%
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-wrap">
            <textarea
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask anything about the videos..."
              rows="1"
            />
            <button
              className="btn-send"
              onClick={handleSend}
              disabled={isStreaming}
            >
              <svg viewBox="0 0 24 24">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
