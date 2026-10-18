import { useEffect, useRef, useState, useCallback } from 'react';
import { HugeiconsIcon } from '@hugeicons/react';
import {
  Attachment01Icon,
  PlusSignIcon,
  Settings01Icon,
  LayoutLeftIcon,
} from '@hugeicons/core-free-icons';
import PromptBar from './components/PromptBar';
import './App.css';

// ─────────────────────────────────────────────────────────────
//  API base — empty → same-origin via Vite proxy
// ─────────────────────────────────────────────────────────────
const API_BASE = '';

// Single model label — Gemma 4 E4B
const LOCAL_MODELS = [
  { key: 'gemma-4-e4b', name: 'Gemma 4 E4B', tag: 'Local' },
];

// Upload pipeline steps
const UPLOAD_STEPS = [
  'Uploading document…',
  'Extracting text from PDF (PyMuPDF)…',
  'Conservative cleaning & semantic chunking…',
  'Generating local embeddings (bge-small-en-v1.5)…',
  'Building FAISS vector index…',
  'Document ready — you can now ask questions!',
];

/*  Theme hook                                                 */
/* ─────────────────────────────────────────────────────────── */
function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem('lt-theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch (_) {}
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('lt-theme', theme); } catch (_) {}
  }, [theme]);

  const toggle = useCallback(() =>
    setTheme(t => (t === 'dark' ? 'light' : 'dark')), []);

  return [theme, toggle];
}

/* ─────────────────────────────────────────────────────────── */
/*  SVG icons (inline, no external dep needed)                */
/* ─────────────────────────────────────────────────────────── */

function IconSun() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="4"/>
      <line x1="12" y1="2" x2="12" y2="4"/>
      <line x1="12" y1="20" x2="12" y2="22"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="2" y1="12" x2="4" y2="12"/>
      <line x1="20" y1="12" x2="22" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  );
}

function IconMoon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  );
}

function IconMenu() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round"
      aria-hidden="true"
    >
      <line x1="4" y1="7" x2="20" y2="7"/>
      <line x1="4" y1="12" x2="20" y2="12"/>
      <line x1="4" y1="17" x2="20" y2="17"/>
    </svg>
  );
}

function IconFile() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Utility: parse plain text into paragraphs / legal blocks   */
/* ─────────────────────────────────────────────────────────── */
function renderInlineWithCitations(text) {
  // Replace [SOURCE X: ...] or [SOURCE X] with citation tags
  const parts = text.split(/(\[SOURCE\s*\d+[^\]]*\])/g);
  return parts.map((part, i) => {
    if (/^\[SOURCE\s*\d+[^\]]*\]$/.test(part)) {
      const label = part.replace(/^\[SOURCE\s*(\d+)[^\]]*\]$/, 'Source $1');
      return (
        <span key={i} className="citation-tag" title={part}>
          📎 {label}
        </span>
      );
    }
    // Bold **text**
    const subParts = part.split(/(\*\*[^*]+\*\*)/g);
    return subParts.map((sub, j) =>
      sub.startsWith('**') && sub.endsWith('**') ? (
        <strong key={`${i}-${j}`}>{sub.slice(2, -2)}</strong>
      ) : (
        sub
      )
    );
  });
}

function RichText({ text }) {
  if (!text) return null;

  const paragraphs = text.split(/\n\n+/);
  return (
    <div className="message-text">
      {paragraphs.map((para, pi) => {
        const trimmed = para.trim();

        // 1. Legal disclaimer box
        if (trimmed.toLowerCase().includes('not legal advice; preliminary document triage only')) {
          return (
            <div key={pi} className="triage-disclaimer-box">
              <span>⚖️</span>
              <span>{trimmed}</span>
            </div>
          );
        }

        // 2. Structured section headers
        if (trimmed.startsWith('### ')) {
          const title = trimmed.replace(/^###\s*/, '');
          let headerClass = 'explicit';
          let icon = '📌';

          if (/explicit/i.test(title)) {
            headerClass = 'explicit';
            icon = '📌';
          } else if (/interpret|concern|risk/i.test(title)) {
            headerClass = 'risk';
            icon = '⚠️';
          } else if (/jurisdiction|enforceab/i.test(title)) {
            headerClass = 'jurisdiction';
            icon = '🏛️';
          } else if (/next step|escalat|recommend/i.test(title)) {
            headerClass = 'next-steps';
            icon = '💡';
          } else if (/silent|silence/i.test(title)) {
            headerClass = 'silence';
            icon = '❓';
          }

          return (
            <div key={pi} className={`legal-section-header ${headerClass}`}>
              <span>{icon}</span>
              <span>{title}</span>
            </div>
          );
        }

        // 3. Bullet blocks
        const lines = para.split('\n');
        const isBulletBlock =
          lines.length > 1 &&
          lines.every(l => /^\s*[-*•]\s/.test(l) || /^\d+\.\s/.test(l));

        if (isBulletBlock) {
          return (
            <ul key={pi}>
              {lines.filter(Boolean).map((line, li) => (
                <li key={li} style={{ position: 'relative', paddingLeft: 16, marginBottom: 6 }}>
                  <span style={{ position: 'absolute', left: 0, color: 'var(--accent-bright)', fontWeight: 700 }}>›</span>
                  {renderInlineWithCitations(line.replace(/^\s*[-*•]\s/, '').replace(/^\d+\.\s/, ''))}
                </li>
              ))}
            </ul>
          );
        }

        // 4. Standard paragraph
        return (
          <p key={pi}>
            {renderInlineWithCitations(para)}
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
      <div className="sources-header">Evidence Sources</div>
      <div className="sources-list">
        {sources.map((src, i) => (
          <div key={i} className="source-item">
            <span className="source-num">[{i + 1}]</span>
            <div className="source-meta">
              {src.source_file && <span className="source-tag">{src.source_file}</span>}
              {src.page_start != null && (
                <span className="source-tag">
                  Page {src.page_start}
                  {src.page_end && src.page_end !== src.page_start ? `–${src.page_end}` : ''}
                </span>
              )}
              {src.section && <span className="source-tag">§ {src.section}</span>}
              {src.clause  && <span className="source-tag">Clause {src.clause}</span>}
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
                  <span key={i} className="attachment-badge">
                    <IconFile />
                    {f}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="message-group">
      <div className="assistant-row">
        <div className="bot-avatar" aria-hidden="true">JG</div>
        <div className="assistant-content">
          <div className="assistant-label">LegalTriage</div>

          {message.error ? (
            <div className="error-bubble">
              <span className="error-icon" aria-hidden="true">!</span>
              <div>
                <strong>Error: </strong>{message.error}
                {message.hint && <div className="error-hint">{message.hint}</div>}
              </div>
            </div>
          ) : (
            <>
              {message.content && <RichText text={message.content} />}

              {message.status && message.status !== 'ok' && (
                <div className="status-banner">
                  {message.status === 'low_relevance'
                    ? 'The document may not directly address this question — results may be limited.'
                    : message.status === 'no_results'
                    ? 'No relevant passages were found for this question.'
                    : message.status}
                </div>
              )}

              {message.sources && message.sources.length > 0 && (
                <>
                  <button
                    className="sources-toggle"
                    onClick={() => setSourcesOpen(v => !v)}
                    type="button"
                    aria-expanded={sourcesOpen}
                  >
                    {sourcesOpen ? 'Hide' : 'Show'}{' '}
                    {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
                    <span className="toggle-arrow" aria-hidden="true">
                      {sourcesOpen ? '▲' : '▼'}
                    </span>
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
/*  Upload progress card                                       */
/* ─────────────────────────────────────────────────────────── */
function UploadProgress({ step }) {
  const progress = Math.round(((step + 1) / UPLOAD_STEPS.length) * 100);
  return (
    <div className="upload-progress-card">
      <div className="upload-progress-header">
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
              {i < step ? '✓' : i === step ? '·' : '○'}
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
function ThinkingIndicator() {
  return (
    <div className="thinking-row">
      <div className="bot-avatar" aria-hidden="true">JG</div>
      <div className="thinking-content">
        <div className="thinking-label">JurisGuide</div>
        <div className="thinking-dots" aria-label="Thinking" role="status">
          <div className="thinking-dot" />
          <div className="thinking-dot" />
          <div className="thinking-dot" />
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Sidebar                                                    */
/* ─────────────────────────────────────────────────────────── */
function Sidebar({ currentDocument, onNewChat, collapsed }) {
  return (
    <aside
      className={`sidebar${collapsed ? ' collapsed' : ''}`}
      aria-label="Sidebar"
      aria-hidden={collapsed}
    >
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="brand-text">
          <span className="brand-name">JurisGuide</span>
          <span className="brand-sub">Document Analysis</span>
        </div>
      </div>

      {/* Body */}
      <div className="sidebar-body">
        <button
          id="new-analysis-btn"
          className="new-chat-btn"
          onClick={onNewChat}
          type="button"
        >
          <HugeiconsIcon icon={PlusSignIcon} size={14} strokeWidth={2.5} />
          New Analysis
        </button>

        {/* Active document */}
        <div className="sidebar-section-label">Active Document</div>
        <div className="doc-card">
          {currentDocument ? (
            <div className="doc-card-file">
              <div className="doc-file-info">
                <div className="doc-filename" title={currentDocument.filename}>
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
              No document uploaded.
              <br />Attach a PDF to get started.
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <button className="sidebar-footer-btn" type="button" aria-label="Settings">
          <HugeiconsIcon icon={Settings01Icon} size={14} strokeWidth={1.8} />
          Settings
        </button>
      </div>
    </aside>
  );
}

/* ─────────────────────────────────────────────────────────── */
/*  Main App                                                    */
/* ─────────────────────────────────────────────────────────── */
export default function App() {
  const [messages, setMessages]           = useState([]);
  const [busy, setBusy]                   = useState(false);
  const [uploadStep, setUploadStep]       = useState(-1);
  const [isQuerying, setIsQuerying]       = useState(false);
  const [currentDocument, setCurrentDocument] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [theme, toggleTheme]              = useTheme();

  const controller    = useRef(null);
  const chatBottomRef = useRef(null);
  const pendingFiles  = useRef([]);

  // Resolve PromptBar colors from current CSS vars
  const pbColors = useCallback(() => {
    const s = getComputedStyle(document.documentElement);
    return {
      bg:   s.getPropertyValue('--pb-bg-runtime').trim()   || '#16161b',
      ink:  s.getPropertyValue('--pb-ink-runtime').trim()  || '#f0f0f4',
      menu: s.getPropertyValue('--pb-menu-runtime').trim() || '#1e1e24',
    };
  }, []);

  // Re-render PromptBar colors when theme changes
  const [pbTheme, setPbTheme] = useState(pbColors);
  useEffect(() => {
    // Small delay so CSS vars update first
    const id = setTimeout(() => setPbTheme(pbColors()), 10);
    return () => clearTimeout(id);
  }, [theme, pbColors]);

  // Auto-scroll
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy, uploadStep, isQuerying]);

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

  // ── Upload + process ─────────────────────────────────────
  const uploadAndProcess = useCallback(async () => {
    const files = pendingFiles.current;
    if (!files.length) throw new Error('No file selected.');

    const file = files[0];
    const fd = new FormData();
    fd.append('file', file);

    const stepDelay = 700;
    setUploadStep(0);
    const stepTimer = setInterval(() => {
      setUploadStep(prev => {
        const next = prev + 1;
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
      setUploadStep(UPLOAD_STEPS.length - 1);
      await new Promise(r => setTimeout(r, 1200));
      setUploadStep(-1);

      const doc = {
        filename: result.filename,
        status: result.status,
        summary: result.summary,
      };
      setCurrentDocument(doc);
      return doc;
    } catch (err) {
      clearInterval(stepTimer);
      setUploadStep(-1);
      throw err;
    }
  }, []);

  // ── Query backend ─────────────────────────────────────────
  const queryBackend = useCallback(async ({ question, signal }) => {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        model: 'gemma-4-e4b',
        top_k: 5,
        min_score: 0.45,
      }),
      signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Query failed (HTTP ${res.status})`);
    }
    return res.json();
  }, []);

  // ── Handle send ───────────────────────────────────────────
  const handleSend = useCallback(async (text, { attachments }) => {
    if (!text.trim() && (!attachments || attachments.length === 0)) return;

    setBusy(true);
    controller.current = new AbortController();

    setMessages(prev => [
      ...prev,
      { role: 'user', content: text, attachments: attachments || [] },
    ]);

    try {
      // Step 1: upload if attachment present
      if (attachments && attachments.length > 0) {
        await uploadAndProcess();
      } else if (!currentDocument) {
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

      // Step 2: query
      if (!text.trim()) {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: 'Document processed successfully. Ask me anything about it.',
            sources: [],
            status: 'ok',
          },
        ]);
        return;
      }

      setIsQuerying(true);
      const result = await queryBackend({
        question: text.trim(),
        signal: controller.current.signal,
      });
      setIsQuerying(false);

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
  }, [uploadAndProcess, queryBackend, currentDocument]);

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

  const showUploadProgress = uploadStep >= 0;
  const showThinking       = isQuerying && !showUploadProgress;
  const isEmpty            = messages.length === 0 && !showUploadProgress && !showThinking;

  return (
    <div className="app">
      <Sidebar
        currentDocument={currentDocument}
        onNewChat={handleNewChat}
        collapsed={sidebarCollapsed}
      />

      <main className="workspace">
        {/* ── Top bar ── */}
        <div className="topbar" role="banner">
          <div className="topbar-left">
            <button
              id="sidebar-toggle-btn"
              className="topbar-menu-btn"
              onClick={() => setSidebarCollapsed(v => !v)}
              type="button"
              aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              aria-expanded={!sidebarCollapsed}
            >
              <IconMenu />
            </button>
            <span className="topbar-title">Document Workspace</span>
            <span className="model-chip" aria-label="Model: Gemma 4 E4B">
              <span className="model-chip-dot" aria-hidden="true" />
              Gemma 4 E4B
            </span>
          </div>

          <div className="topbar-right">
            <button
              id="theme-toggle-btn"
              className="theme-toggle"
              onClick={toggleTheme}
              type="button"
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? <IconSun /> : <IconMoon />}
            </button>
          </div>
        </div>

        {/* ── Chat area ── */}
        <div
          className="chat-area"
          role="log"
          aria-live="polite"
          aria-label="Conversation"
        >
          <div className="chat-inner">

            {/* Empty state */}
            {isEmpty && (
              <div className="empty-state">
                <h1 className="empty-heading">
                  Understand your legal documents.
                </h1>
                <p className="empty-sub">
                  Upload a document to explore clauses, understand obligations,
                  and identify important deadlines.
                </p>
                <button
                  id="upload-pdf-btn"
                  className="empty-upload-btn"
                  type="button"
                  onClick={() => pickFiles().then(names => {
                    if (names) {
                      pendingFiles.current = pendingFiles.current.length
                        ? pendingFiles.current
                        : [];
                      handleSend('', { attachments: names });
                    }
                  })}
                >
                  <HugeiconsIcon icon={Attachment01Icon} size={15} strokeWidth={2} />
                  Upload a document
                </button>
              </div>
            )}

            {/* Messages */}
            {messages.map((msg, i) => (
              <Message key={i} message={msg} />
            ))}

            {/* Upload progress */}
            {showUploadProgress && <UploadProgress step={uploadStep} />}

            {/* Thinking indicator */}
            {showThinking && <ThinkingIndicator />}

            {/* Scroll anchor */}
            <div ref={chatBottomRef} aria-hidden="true" style={{ height: 1 }} />
          </div>
        </div>

        {/* ── Composer ── */}
        <div className="composer-area">
          <div className="composer-inner">
            <PromptBar
              placeholder={
                currentDocument
                  ? `Ask about ${currentDocument.filename}…`
                  : 'Ask a question about your document…'
              }
              sources={[
                {
                  key: 'files',
                  name: 'Upload a document',
                  description: 'Choose a file from your device',
                  icon: Attachment01Icon,
                  attach: true,
                },
              ]}
              commands={[]}
              models={LOCAL_MODELS}
              defaultModel="gemma-4-e4b"
              efforts={[]}
              busy={busy}
              onSend={handleSend}
              onStop={handleStop}
              onAttach={pickFiles}
              background={pbTheme.bg}
              color={pbTheme.ink}
              menuBackground={pbTheme.menu}
              sparkColor={theme === 'dark' ? '#5b6aff' : '#4c5df0'}
              sparkBoost={0}
              width={740}
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
