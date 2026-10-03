"use client";

import { DragEvent, FormEvent, useEffect, useState } from "react";

type Protocol = {
  name: string;
  category?: string;
  version?: string;
  updated?: string;
  pages: number;
  filename?: string;
};

type Source = { protocol: string; page: number; text: string };
type ChatMessage = { id: string; role: "user" | "assistant"; text: string; mode?: "grounded" | "conversation" | "catalog" | "domain" | "error"; sources?: Source[]; knowledgeSources?: { id: string; source: string; kind: string }[] };
type AssistantState = "idle" | "loading" | "success" | "empty" | "error";
type UploadState = "idle" | "loading" | "success" | "error";

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const apiKey = process.env.NEXT_PUBLIC_API_KEY;
const fallbackProtocols: Protocol[] = [
  { name: "Sample Preparation Protocol", pages: 8, category: "Sample handling" },
  { name: "RNA Extraction SOP", pages: 12, category: "Nucleic acid extraction" },
  { name: "Cell Culture Maintenance", pages: 6, category: "Cell culture" },
];

function apiHeaders(json = false): HeadersInit {
  return { ...(json ? { "Content-Type": "application/json" } : {}), ...(apiKey ? { "X-API-Key": apiKey } : {}) };
}

export default function Home() {
  const [activeView, setActiveView] = useState("Workspace");
  const [protocols, setProtocols] = useState<Protocol[]>(fallbackProtocols);
  const [selectedProtocol, setSelectedProtocol] = useState(fallbackProtocols[0].name);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [answerMessage, setAnswerMessage] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [assistantState, setAssistantState] = useState<AssistantState>("idle");
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [uploadName, setUploadName] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [apiState, setApiState] = useState<"checking" | "online" | "offline">("checking");
  const [viewingSource, setViewingSource] = useState<Source | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  async function loadProtocols() {
    const response = await fetch(`${apiUrl}/protocols`);
    if (!response.ok) throw new Error("Protocol catalog could not be loaded.");
    const result = await response.json();
    const nextProtocols: Protocol[] = result.protocols ?? [];
    setProtocols(nextProtocols);
    setSelectedProtocol((current) => current && nextProtocols.some((item) => item.name === current) ? current : nextProtocols[0]?.name ?? "");
  }

  useEffect(() => {
    Promise.all([fetch(`${apiUrl}/health`), fetch(`${apiUrl}/protocols`)])
      .then(async ([healthResponse, protocolsResponse]) => {
        if (!healthResponse.ok) throw new Error("API unavailable");
        if (!protocolsResponse.ok) throw new Error("Protocol catalog unavailable");
        const result = await protocolsResponse.json();
        const nextProtocols: Protocol[] = result.protocols ?? [];
        setProtocols(nextProtocols);
        setSelectedProtocol(nextProtocols[0]?.name ?? "");
        setApiState("online");
      })
      .catch(() => setApiState("offline"));
  }, []);

  async function uploadProtocol(file: File | undefined) {
    if (!file) return;
    setUploadName(file.name);
    setUploadState("loading");
    setUploadMessage("Uploading and indexing source...");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${apiUrl}/protocols/upload`, { method: "POST", headers: apiHeaders(), body: formData });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "The file could not be indexed.");
      await loadProtocols();
      setApiState("online");
      setUploadState("success");
      setUploadMessage(`${result.protocols} protocol${result.protocols === 1 ? "" : "s"} indexed · ${result.chunks} source sections saved`);
    } catch (error) {
      setUploadState("error");
      setUploadMessage(error instanceof Error ? error.message : "The file could not be indexed.");
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    void uploadProtocol(event.dataTransfer.files[0]);
  }

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || assistantState === "loading") return;
    setQuestion("");
    setChatMessages((messages) => [...messages, { id: `${Date.now()}-user`, role: "user", text: cleanQuestion }]);
    setAssistantState("loading");
    setAnswer(null);
    setAnswerMessage("");
    setSources([]);
    try {
      const response = await fetch(`${apiUrl}/query`, { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ question: cleanQuestion, protocol: selectedProtocol || undefined }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "The assistant could not answer.");
      if (!result.answer) {
        setAssistantState("empty");
        setAnswerMessage(result.message ?? "No supporting evidence was found in this protocol.");
        setChatMessages((messages) => [...messages, { id: `${Date.now()}-assistant`, role: "assistant", text: result.message ?? "No supporting evidence was found in this protocol.", mode: "error" }]);
      } else {
        setAssistantState("success");
        setAnswer(result.answer);
        setSources(result.sources ?? []);
        setChatMessages((messages) => [...messages, { id: `${Date.now()}-assistant`, role: "assistant", text: result.answer, mode: result.mode === "conversation" ? "conversation" : result.mode === "catalog" ? "catalog" : result.mode === "domain" ? "domain" : "grounded", sources: result.sources ?? [], knowledgeSources: result.knowledge_sources ?? [] }]);
      }
    } catch (error) {
      setAssistantState("error");
      setAnswerMessage(error instanceof Error ? error.message : "The assistant could not be reached.");
      setChatMessages((messages) => [...messages, { id: `${Date.now()}-assistant`, role: "assistant", text: error instanceof Error ? error.message : "The assistant could not be reached.", mode: "error" }]);
    }
  }

  async function openSource(source: Source) {
    try {
      const response = await fetch(`${apiUrl}/protocols/${encodeURIComponent(source.protocol)}/pages/${source.page}`);
      if (!response.ok) throw new Error("Source page could not be opened.");
      setViewingSource(await response.json());
    } catch (error) {
      setUploadState("error");
      setUploadMessage(error instanceof Error ? error.message : "Source page could not be opened.");
    }
  }

  const totalPages = protocols.reduce((total, protocol) => total + protocol.pages, 0);
  const largestProtocol = Math.max(...protocols.map((protocol) => protocol.pages), 1);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">P</span><span>Protocol<br /><strong>Atlas</strong></span></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Workspace navigation">
          {["Workspace", "Protocols", "Collections"].map((item) => <button key={item} className={activeView === item ? "nav-item active" : "nav-item"} onClick={() => setActiveView(item)}><span className="nav-icon">{item === "Workspace" ? "⌂" : item === "Protocols" ? "▤" : "◫"}</span>{item}</button>)}
        </nav>
        <div className="sidebar-bottom">
          <div className={apiState === "online" ? "index-status" : "index-status warning"}><span className="status-dot" />{apiState === "online" ? "Index service online" : apiState === "checking" ? "Connecting to index" : "Index service offline"}<div className="status-detail">{protocols.length} protocols · {totalPages} pages</div></div>
          <button className="nav-item"><span className="nav-icon">⚙</span>Settings</button>
          <div className="user-chip"><span className="avatar">AK</span><span>Alex Kim<small>Research lead</small></span><span className="more">•••</span></div>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar"><div className="crumb">{activeView} <span>/</span> <strong>Protocol library</strong></div><div className="top-actions"><span className={apiState === "online" ? "api-badge online" : "api-badge"}><i />{apiState === "online" ? "API connected" : apiState === "checking" ? "Connecting" : "API offline"}</span><button className="icon-button" aria-label="Search">⌕</button><span className="top-divider" /><span className="help">?</span></div></header>
        <section className="content-wrap">
          <div className="page-intro"><div><p className="eyebrow">PROTOCOL LIBRARY</p><h1>Turn protocols into<br /><em>working knowledge.</em></h1><p className="intro-copy">Upload source documents, then ask precise questions with every answer anchored to the original text.</p></div><div className="upload-wrap"><label className={uploadState === "loading" ? "upload-button disabled" : "upload-button"}><span>＋</span> Add protocol<input type="file" accept=".pdf,.json,application/json,application/pdf" disabled={uploadState === "loading"} onChange={(event) => { const file = event.target.files?.[0]; event.currentTarget.value = ""; void uploadProtocol(file); }} /></label>{uploadName && <span className="upload-name">{uploadName}</span>}{uploadMessage && <span className={`upload-status ${uploadState}`}>{uploadState === "success" ? "✓ " : uploadState === "error" ? "! " : ""}{uploadMessage}</span>}</div></div>
          <div className="metrics"><div><span className="metric-label">INDEXED PROTOCOLS</span><strong>{String(protocols.length).padStart(2, "0")}</strong><span className="metric-note">Persistent catalog</span></div><div><span className="metric-label">SOURCE PAGES</span><strong>{totalPages}</strong><span className="metric-note">Available to query</span></div><div><span className="metric-label">INDEX HEALTH</span><strong className={apiState === "online" ? "health" : "health muted"}><i />{apiState === "online" ? "Ready" : "Offline"}</strong><span className="metric-note">SQLite source index</span></div></div>
          <section className="chart-panel"><div className="section-heading"><div><p className="eyebrow">LIBRARY OVERVIEW</p><h2>Pages by protocol</h2></div><span className="source-count">Select a bar to change scope</span></div>{protocols.length ? <div className="bar-chart">{protocols.map((protocol) => <button type="button" className={selectedProtocol === protocol.name ? "chart-bar selected" : "chart-bar"} key={protocol.name} onClick={() => { setSelectedProtocol(protocol.name); setAnswer(null); setSources([]); setAssistantState("idle"); }}><span className="bar-value">{protocol.pages}</span><span className="bar-track"><i style={{ height: `${Math.max(20, (protocol.pages / largestProtocol) * 100)}%` }} /></span><strong>{protocol.name.replace(" Protocol", "").replace(" Maintenance", "")}</strong></button>)}</div> : <div className="empty-library">No protocols are indexed yet. Upload a PDF or JSON source to begin.</div>}</section>
          <div className="workspace-grid">
            <section className="panel protocol-panel"><div className="panel-heading"><div><p className="eyebrow">YOUR SOURCES</p><h2>Indexed protocols</h2></div><span className="source-count">{protocols.length} loaded</span></div><div className="protocol-list">{protocols.map((protocol) => <button className={selectedProtocol === protocol.name ? "protocol-row selected" : "protocol-row"} key={protocol.name} onClick={() => setSelectedProtocol(protocol.name)}><span className="file-icon">{protocol.filename?.endsWith(".json") ? "JSON" : "PDF"}</span><span className="protocol-main"><strong>{protocol.name}</strong><small>{protocol.pages} pages · {protocol.category ?? "Uploaded source"}</small></span><span className="row-status"><i />Indexed</span><span className="row-arrow">›</span></button>)}</div><div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}><span className="drop-icon">↓</span><span><strong>Drop a PDF or JSON here</strong><small>or browse from your computer</small></span></div></section>
            <section className="panel ask-panel"><div className="panel-heading"><div><p className="eyebrow">ASK YOUR SOURCES</p><h2>Protocol assistant</h2><small className="selected-label">Scope: {selectedProtocol || "All indexed protocols"}</small></div><span className="mode-pill"><i />Hugging Face</span></div><div className="chat-history" aria-live="polite">{chatMessages.length === 0 && <div className="chat-history-empty">Your questions and answers will stay here.</div>}{chatMessages.map((message) => <div className={`chat-message ${message.role} ${message.mode ?? ""}`} key={message.id}><span className="chat-role">{message.role === "user" ? "YOU" : message.mode === "conversation" ? "CHAT" : message.mode === "catalog" ? "CATALOG" : message.mode === "error" ? "STATUS" : "ATLAS"}</span><p>{message.text}</p>{message.sources?.length ? <div className="chat-citations">{message.sources.map((source) => <button type="button" key={`${source.protocol}-${source.page}`} onClick={() => void openSource(source)}>p. {source.page}</button>)}</div> : null}</div>)}</div><div className={`answer-area ${assistantState}`}>{assistantState === "loading" && <div className="empty-answer"><span className="spinner" /><p>Searching the selected protocol...</p><small>Checking indexed source sections</small></div>}{assistantState === "idle" && chatMessages.length === 0 && <div className="empty-answer"><span className="spark">✦</span><p>Ask a question about your<br />selected protocol.</p><small>Answers appear only when source evidence matches.</small></div>}{(assistantState === "empty" || assistantState === "error") && chatMessages.length === 0 && <div className="empty-answer error-answer"><span className="state-symbol">{assistantState === "empty" ? "∅" : "!"}</span><p>{answerMessage}</p><small>Try another question or check the API service.</small></div>}{assistantState === "success" && chatMessages.length === 0 && answer && <div className="answer"><p>{answer}</p></div>}</div><form className="question-form" onSubmit={askQuestion}><input value={question} onChange={(event) => setQuestion(event.target.value)} disabled={assistantState === "loading"} placeholder="Ask a question about your protocols..." /><button type="submit" aria-label="Ask question" disabled={assistantState === "loading"}>{assistantState === "loading" ? "…" : "→"}</button></form><div className="suggestions"><button type="button" onClick={() => setQuestion("What are the centrifuge settings?")} disabled={assistantState === "loading"}>Centrifuge settings <span>↗</span></button><button type="button" onClick={() => setQuestion("What materials do I need?")} disabled={assistantState === "loading"}>Required materials <span>↗</span></button></div></section>
          </div>
          <section className="sources-section"><div className="section-heading"><div><p className="eyebrow">SOURCE TRACE</p><h2>Relevant excerpts</h2></div><span className="source-count">{sources.length} cited</span></div>{sources.length ? <div className="source-grid">{sources.map((source) => <article className="source-card" key={`${source.protocol}-${source.page}`}><div className="source-card-top"><span className="source-type">SOURCE</span><span>p. {source.page}</span></div><h3>{source.protocol}</h3><p>{source.text}</p><button type="button" onClick={() => void openSource(source)}>View source <span>↗</span></button></article>)}</div> : <div className="empty-sources">Submit a question to see the exact source sections used in the answer.</div>}</section>
        </section>
      </main>
      {viewingSource && <div className="source-modal-backdrop" role="presentation" onClick={() => setViewingSource(null)}><section className="source-modal" role="dialog" aria-modal="true" aria-labelledby="source-title" onClick={(event) => event.stopPropagation()}><div className="source-modal-header"><div><p className="eyebrow">SOURCE PAGE {viewingSource.page}</p><h2 id="source-title">{viewingSource.protocol}</h2></div><button className="close-button" onClick={() => setViewingSource(null)} aria-label="Close source">×</button></div><p className="source-modal-text">{viewingSource.text}</p><button className="modal-action" onClick={() => setViewingSource(null)}>Close source</button></section></div>}
    </div>
  );
}
