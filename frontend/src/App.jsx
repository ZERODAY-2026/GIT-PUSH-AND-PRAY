import { useEffect, useRef, useState, useCallback } from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import {
  Attachment01Icon,
  AlertCircleIcon,
  PlusSignIcon,
} from '@hugeicons/core-free-icons';
import PromptBar from './components/PromptBar';
import './App.css';

// ─────────────────────────────────────────────────────────────
//  When Vite proxy is active we call /api directly (no host).
//  For production / standalone, point to the FastAPI server.
// ─────────────────────────────────────────────────────────────
const API_BASE = '';   // empty string → same origin via Vite proxy

const LOCAL_MODELS = [
  { key: 'gemma-2-4b',  name: 'Gemma 2 4B',  tag: 'Primary'  },
  { key: 'gemma-2-2b',  name: 'Gemma 2 2B',  tag: 'Fallback' },
  { key: 'local-model', name: 'Auto-detect',  tag: 'LM Studio' },
];

const EXAMPLE_PROMPTS = [
  { emoji: '💰', text: 'Explain the payment terms in simple English' },
  { emoji: '📋', text: 'What are the termination conditions?' },
  { emoji: '⚠️', text: 'Identify any penalties or late fees' },
  { emoji: '✅', text: 'List all tenant obligations' },
  { emoji: '🔒', text: 'Summarise the confidentiality clauses' },
  { emoji: '📅', text: 'What are the key dates and deadlines?' },
];

// Upload steps shown to the user
const UPLOAD_STEPS = [
  'Uploading document…',
  'Extracting text from PDF…',
  'Cleaning & chunking legal text…',
  'Generating embeddings…',
  'Building FAISS vector index…',
  'Document ready — you can now ask questions!',
];

/* ─────────────────────────────────────────────────────────── */
/*  Utility: parse plain text into paragraphs / bullets        */
/* ─────────────────────────────────────────────────────────── */
function RichText({ text }) {
  if (!text) return null;

  const paragraphs = text.split(/\n\n+/);
  return (
    <div className="message-text">
      {paragraphs.map((para, pi) => {
        const lines = para.split('\n');
        const isBulletBlock = lines.every(l => /^\s*[-*•]\s/.test(l) || /^\d+\.\s/.test(l));
        if (isBulletBlock && lines.length > 1) {
          return (
            <ul key={pi} style={{ paddingLeft: 0, marginBottom: 10, listStyle: 'none' }}>
              {lines.filter(Boolean).map((line, li) => (
                <li key={li} style={{ position: 'relative', paddingLeft: 16, marginBottom: 5 }}>
                  <span style={{ position: 'absolute', left: 0, color: 'var(--accent-bright)', fontWeight: 700 }}>›</span>
                  {line.replace(/^\s*[-*•]\s/, '').replace(/^\d+\.\s/, '')}
                </li>
              ))}
            </ul>
          );
        }
        // Bold **text** support
        const parts = para.split(/(\*\*[^*]+\*\*)/g);
        return (
          <p key={pi}>
            {parts.map((part, i) =>
              part.startsWith('**') && part.endsWith('**')
                ? <strong key={i}>{part.slice(2, -2)}</strong>
                : part
            )}
          </p>
        );
      })}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Source citations panel                                     */
/* ─────────────────────────────────────────────────────────── */
function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources-panel">
      <div className="sources-header">📚 Evidence Sources</div>
      <div className="sources-list">
        {sources.map((src, i) => (
          <div key={i} className="source-item">
            <span className="source-num">[{i + 1}]</span>
            <div className="source-meta">
              {src.source_file && <span className="source-tag">📄 {src.source_file}</span>}
              {src.page_start != null && (
                <span className="source-tag">
                  Page {src.page_start}{src.page_end && src.page_end !== src.page_start ? `–${src.page_end}` : ''}
                </span>
              )}
              {src.section && <span className="source-tag">§ {src.section}</span>}
              {src.clause && <span className="source-tag">Clause {src.clause}</span>}
              {src.score != null && (
                <span className="source-score">
                  {(src.score * 100).toFixed(0)}% match
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Single message bubble                                      */
/* ─────────────────────────────────────────────────────────── */
function Message({ message }) {
  const isUser = message.role === 'user';
  const [sourcesOpen, setSourcesOpen] = useState(false);

  if (isUser) {
    return (
      <div className="message-group">
        <div className="user-row">
          <div className="user-bubble">
            <span>{message.content}</span>
            {message.attachments && message.attachments.length > 0 && (
              <div className="attachments-list">
                {message.attachments.map((f, i) => (
                  <span key={i} className="attachment-badge">📎 {f}</span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Assistant
  return (
    <div className="message-group">
      <div className="assistant-row">
        <div className="bot-avatar" aria-hidden="true">⚖️</div>
        <div className="assistant-content">
          <div className="assistant-label">Legal Triage</div>

          {message.error ? (
            <div className="error-bubble">
              <span className="error-icon">⚠️</span>
              <div>
                <strong>Error: </strong>{message.error}
                {message.hint && <div className="error-hint">{message.hint}</div>}
              </div>
            </div>
          ) : (
            <>
              {message.content && <RichText text={message.content} />}

              {/* Status badges */}
              {message.status && message.status !== 'ok' && (
                <div className="status-banner">
                  <span>ℹ️</span>
                  {message.status === 'low_relevance'
                    ? 'The document may not directly address this question — results may be limited.'
                    : message.status === 'no_results'
                    ? 'No relevant passages were found for this question.'
                    : message.status}
                </div>
              )}

              {/* Sources toggle */}
              {message.sources && message.sources.length > 0 && (
                <>
                  <button
                    className="sources-toggle"
                    onClick={() => setSourcesOpen(v => !v)}
                    type="button"
                  >
                    <span>📚</span>
                    {sourcesOpen ? 'Hide' : 'Show'} {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
                    <span className="toggle-arrow">{sourcesOpen ? '▲' : '▼'}</span>
                  </button>
                  {sourcesOpen && <SourcesPanel sources={message.sources} />}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Animated upload progress                                   */
/* ─────────────────────────────────────────────────────────── */
function UploadProgress({ step }) {
  const progress = Math.round(((step + 1) / UPLOAD_STEPS.length) * 100);
  return (
    <div className="upload-progress-card">
      <div className="upload-progress-header">
        <span className="upload-icon">⚙️</span>
        <span className="upload-title">Processing Document</span>
        <span className="upload-pct">{progress}%</span>
      </div>
      <div className="upload-bar-track">
        <div className="upload-bar-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="upload-step-text">{UPLOAD_STEPS[step]}</div>
      <div className="upload-steps-list">
        {UPLOAD_STEPS.slice(0, -1).map((s, i) => (
          <div
            key={i}
            className={`upload-step ${i < step ? 'done' : i === step ? 'active' : 'pending'}`}
          >
            <span className="step-dot">
              {i < step ? '✓' : i === step ? '◉' : '○'}
            </span>
            <span>{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Thinking indicator                                         */
/* ─────────────────────────────────────────────────────────── */
function ThinkingIndicator({ text }) {
  return (
    <div className="thinking-row">
      <div className="bot-avatar" aria-hidden="true">⚖️</div>
      <div className="thinking-content">
        <div className="thinking-label">Legal Triage</div>
        <div className="thinking-dots">
          <div className="thinking-dot" />
          <div className="thinking-dot" />
          <div className="thinking-dot" />
        </div>
        {text && <div className="thinking-text">{text}</div>}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Sidebar                                                    */
/* ─────────────────────────────────────────────────────────── */
function Sidebar({ currentDocument, onNewChat, backendStatus }) {
  return (
    <aside className="sidebar" aria-label="Sidebar">
      <div className="sidebar-logo-area">
        <div className="logo-icon" aria-hidden="true">⚖️</div>
        <div className="logo-text">
          <h1>Legal Triage</h1>
          <p>Private Document Analysis</p>
        </div>
      </div>

      <div className="sidebar-body">
        <button
          id="new-analysis-btn"
          className="new-chat-btn"
          onClick={onNewChat}
          type="button"
        >
          <HugeiconsIcon icon={PlusSignIcon} size={15} strokeWidth={2.5} />
          New Analysis
        </button>

        {/* Backend status */}
        <div className="doc-card">
          <div className="doc-card-label">Backend Status</div>
          <div className="backend-status">
            <div className={`backend-dot ${backendStatus.ok ? 'online' : 'offline'}`} />
            <span>{backendStatus.ok ? 'API Online' : 'API Offline'}</span>
            {backendStatus.ok && (
              <span className="backend-sub">
                {backendStatus.index_ready ? '· Index ready' : '· No index yet'}
              </span>
            )}
          </div>
          {backendStatus.ok && backendStatus.llm_base_url && (
            <div className="lm-url">
              LM Studio: {backendStatus.llm_base_url}
            </div>
          )}
        </div>

        {/* Document card */}
        <div className="doc-card">
          <div className="doc-card-label">Active Document</div>
          {currentDocument ? (
            <div className="doc-card-file">
              <div className="doc-file-icon">📄</div>
              <div className="doc-file-info">
                <div className="doc-filename">
                  {currentDocument.filename}
                </div>
                {currentDocument.summary && (
                  <div className="doc-meta">
                    {currentDocument.summary.total_pages != null &&
                      `${currentDocument.summary.total_pages} pages`}
                    {currentDocument.summary.chunks_created != null &&
                      ` · ${currentDocument.summary.chunks_created} chunks`}
                    {currentDocument.summary.embedding_model &&
                      ` · ${currentDocument.summary.embedding_model.split('/').pop()}`}
                  </div>
                )}
                <div className="doc-status-badge">
                  <span className="status-dot" />
                  Indexed &amp; Ready
                </div>
              </div>
            </div>
          ) : (
            <div className="doc-card-empty">
              <span className="doc-card-empty-icon">📁</span>
              <span>No document uploaded.<br />Attach a PDF to get started.</span>
            </div>
          )}
        </div>
      </div>

      <div className="sidebar-footer">
        <p className="sidebar-disclaimer">
          🔒 Fully local &amp; private.<br />
          Not legal advice — for information only.
        </p>
      </div>
    </aside>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Main App                                                   */
/* ─────────────────────────────────────────────────────────── */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [uploadStep, setUploadStep] = useState(-1);     // -1 = not uploading
  const [isQuerying, setIsQuerying] = useState(false);
  const [currentDocument, setCurrentDocument] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ ok: false, index_ready: false });

  const controller = useRef(null);
  const chatBottomRef = useRef(null);
  // Store picked File objects here between pickFiles() and handleSend()
  const pendingFiles = useRef([]);

  // ── Auto-scroll ──────────────────────────────────────────
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy, uploadStep, isQuerying]);

  // ── Poll backend health on mount ─────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
          const data = await res.json();
          setBackendStatus({ ok: true, ...data });
        } else {
          setBackendStatus({ ok: false });
        }
      } catch {
        setBackendStatus({ ok: false });
      }
    };
    check();
    const timer = setInterval(check, 10_000);
    return () => clearInterval(timer);
  }, []);

  // ── File picker ──────────────────────────────────────────
  const pickFiles = useCallback(() => {
    return new Promise(resolve => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.pdf';
      input.onchange = e => {
        const files = Array.from(e.target.files || []);
        pendingFiles.current = files;
        resolve(files.length ? files.map(f => f.name) : null);
      };
      input.oncancel = () => resolve(null);
      input.click();
    });
  }, []);

  // ── Upload + process document ─────────────────────────────
  const uploadAndProcess = useCallback(async () => {
    const files = pendingFiles.current;
    if (!files.length) throw new Error('No file selected.');

    const file = files[0];
    const fd = new FormData();
    fd.append('file', file);

    // Animate through steps with a small delay each
    const stepDelay = 700;
    setUploadStep(0);
    const stepTimer = setInterval(() => {
      setUploadStep(prev => {
        const next = prev + 1;
        // Don't go past "Building FAISS" — wait for real response
        return next < UPLOAD_STEPS.length - 2 ? next : prev;
      });
    }, stepDelay);

    try {
      const res = await fetch(`${API_BASE}/api/documents/process`, {
        method: 'POST',
        body: fd,
      });

      clearInterval(stepTimer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setUploadStep(-1);
        throw new Error(err.detail || `Upload failed (HTTP ${res.status})`);
      }

      const result = await res.json();
      // result: { status, filename, summary: { source_file, total_pages, chunks_created, ... } }

      // Show "Done" step briefly
      setUploadStep(UPLOAD_STEPS.length - 1);
      await new Promise(r => setTimeout(r, 1200));
      setUploadStep(-1);

      const doc = {
        filename: result.filename,
        status: result.status,
        summary: result.summary,
      };
      setCurrentDocument(doc);

      // Refresh health (index now exists)
      try {
        const h = await fetch(`${API_BASE}/api/health`);
        if (h.ok) setBackendStatus({ ok: true, ...(await h.json()) });
      } catch { /* ignore */ }

      return doc;
    } catch (err) {
      clearInterval(stepTimer);
      setUploadStep(-1);
      throw err;
    }
  }, []);

  // ── Query backend ─────────────────────────────────────────
  const queryBackend = useCallback(async ({ question, modelKey, signal }) => {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        model: modelKey,   // passed to LocalGemmaClient
        top_k: 5,
        min_score: 0.45,   // slightly lower than default for broader retrieval
      }),
      signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Query failed (HTTP ${res.status})`);
    }
    return res.json();
    // returns { question, status, answer, sources, best_score, disclaimer? }
  }, []);

  // ── Handle send ───────────────────────────────────────────
  const handleSend = useCallback(async (text, { attachments, model }) => {
    if (!text.trim() && (!attachments || attachments.length === 0)) return;

    setBusy(true);
    controller.current = new AbortController();

    // Push user message immediately
    setMessages(prev => [
      ...prev,
      { role: 'user', content: text, attachments: attachments || [] },
    ]);

    try {
      // ── Step 1: Upload document if there are attachments ──
      if (attachments && attachments.length > 0) {
        await uploadAndProcess();
      } else if (!backendStatus.index_ready && !currentDocument) {
        // No document indexed yet — tell user
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: '',
            error: 'No document has been uploaded yet. Please attach a PDF first.',
            hint: 'Use the paperclip button in the composer to attach a legal document.',
          },
        ]);
        return;
      }

      // ── Step 2: Ask the question ──────────────────────────
      if (!text.trim()) {
        // Attachment-only — document uploaded, no question
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: 'Document processed successfully! Ask me anything about it.',
            sources: [],
            status: 'ok',
          },
        ]);
        return;
      }

      setIsQuerying(true);
      const result = await queryBackend({
        question: text.trim(),
        modelKey: model?.key || 'local-model',
        signal: controller.current.signal,
      });
      setIsQuerying(false);

      // result.status: 'ok' | 'low_relevance' | 'no_results'
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: result.answer || 'No answer returned.',
          sources: result.sources || [],
          status: result.status,
          best_score: result.best_score,
        },
      ]);
    } catch (err) {
      setIsQuerying(false);
      if (err.name === 'AbortError') return;

      // Provide helpful hints for common errors
      let hint = '';
      if (err.message.includes('LM Studio') || err.message.includes('localhost') || err.message.includes('502')) {
        hint = 'Make sure LM Studio is running with a model loaded at http://127.0.0.1:1234.';
      } else if (err.message.includes('503') || err.message.includes('not ready')) {
        hint = 'The vector index is not ready. Please upload a document first.';
      } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        hint = 'Cannot reach the backend. Run: cd backend && uvicorn src.api:app --reload';
      }

      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: '', error: err.message, hint },
      ]);
    } finally {
      setBusy(false);
      controller.current = null;
      pendingFiles.current = [];
    }
  }, [uploadAndProcess, queryBackend, backendStatus, currentDocument]);

  const handleStop = useCallback(() => {
    controller.current?.abort();
    setBusy(false);
    setIsQuerying(false);
    setUploadStep(-1);
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setCurrentDocument(null);
    setUploadStep(-1);
    setIsQuerying(false);
    pendingFiles.current = [];
  }, []);

  // Fill prompt from welcome screen chips
  const fillPrompt = useCallback(text => {
    setTimeout(() => {
      const ta = document.querySelector('.prompt-bar__input');
      if (!ta) return;
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      )?.set;
      setter?.call(ta, text);
      ta.dispatchEvent(new Event('input', { bubbles: true }));
      ta.focus();
    }, 0);
  }, []);

  const showUploadProgress = uploadStep >= 0;
  const showThinking = isQuerying && !showUploadProgress;

  return (
    <div className="app">
      <Sidebar
        currentDocument={currentDocument}
        onNewChat={handleNewChat}
        backendStatus={backendStatus}
      />

      <main className="workspace">
        {/* Top bar */}
        <div className="topbar">
          <div className="topbar-left">
            <span className="topbar-title">Legal Document Analysis</span>
            {!backendStatus.ok && (
              <span className="topbar-badge offline">⚠ Backend offline</span>
            )}
            {backendStatus.ok && !backendStatus.index_ready && !currentDocument && (
              <span className="topbar-badge warn">No document indexed</span>
            )}
          </div>
          <div className="topbar-disclaimer">
            <HugeiconsIcon icon={AlertCircleIcon} size={13} strokeWidth={2} />
            Not legal advice
          </div>
        </div>

        {/* Chat area */}
        <div className="chat-area" role="log" aria-live="polite" aria-label="Conversation">
          <div className="chat-inner">

            {/* Welcome screen */}
            {messages.length === 0 && !showUploadProgress && !showThinking && (
              <div className="welcome">
                <div className="welcome-icon" aria-hidden="true">⚖️</div>
                <h2>Welcome to Legal Triage</h2>
                <p>
                  Upload a legal document — rental agreement, employment contract, NDA, or
                  notice — and ask questions in plain English. I'll explain clauses, flag
                  concerns, and cite the exact source.
                </p>

                {!backendStatus.ok && (
                  <div className="setup-card">
                    <div className="setup-card-title">⚡ Start the backend first</div>
                    <div className="setup-steps">
                      <div className="setup-step">
                        <span className="setup-num">1</span>
                        <code>cd backend</code>
                      </div>
                      <div className="setup-step">
                        <span className="setup-num">2</span>
                        <code>uvicorn src.api:app --reload --port 8000</code>
                      </div>
                      <div className="setup-step">
                        <span className="setup-num">3</span>
                        <span>Open LM Studio → load a Gemma model → start server on port 1234</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="prompt-chips" role="list">
                  {EXAMPLE_PROMPTS.map(({ emoji, text }) => (
                    <button
                      key={text}
                      type="button"
                      className="prompt-chip"
                      onClick={() => fillPrompt(text)}
                      aria-label={`Use example: ${text}`}
                    >
                      <span className="chip-emoji" aria-hidden="true">{emoji}</span>
                      <span className="chip-text">{text}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((msg, i) => (
              <Message key={i} message={msg} />
            ))}

            {/* Upload progress */}
            {showUploadProgress && <UploadProgress step={uploadStep} />}

            {/* Thinking indicator */}
            {showThinking && <ThinkingIndicator text="Retrieving relevant clauses and generating answer…" />}

            {/* Scroll anchor */}
            <div ref={chatBottomRef} aria-hidden="true" style={{ height: 1 }} />
          </div>
        </div>

        {/* Composer */}
        <div className="composer-area">
          <div className="composer-inner">
            <PromptBar
              placeholder={
                currentDocument
                  ? `Ask about ${currentDocument.filename}…`
                  : 'Attach a PDF and ask a question…'
              }
              sources={[
                {
                  key: 'files',
                  name: 'Legal Documents',
                  description: 'Upload PDF contract or notice',
                  icon: Attachment01Icon,
                  attach: true,
                },
              ]}
              commands={[]}
              models={LOCAL_MODELS}
              defaultModel="local-model"
              efforts={['Low', 'Medium', 'High']}
              defaultEffort="Medium"
              busy={busy}
              onSend={handleSend}
              onStop={handleStop}
              onAttach={pickFiles}
              background="#141417"
              color="#f0f0f5"
              menuBackground="#1c1c21"
              sparkColor="#8b5cf6"
              sparkBoost={1.2}
              width={780}
              radius={18}
              maxRows={6}
            />
          </div>
          <p className="composer-hint">
            All processing is local &amp; private — your documents never leave your machine.
          </p>
        </div>
      </main>
    </div>
  );
}
