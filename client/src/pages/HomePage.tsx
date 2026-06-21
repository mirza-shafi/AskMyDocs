import { Link } from 'react-router-dom';

/** Landing page with hero section and feature highlights. */
export function HomePage() {
  return (
    <div className="home-page">
      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">Powered by Groq Llama-3.3-70B + pgvector</div>
        <h1 className="hero-title">
          Ask Your Documents
          <span className="hero-accent"> Anything</span>
        </h1>
        <p className="hero-subtitle">
          Upload PDFs or text files. Get precise, cited answers in seconds using
          hybrid semantic search, parent-child chunking, and Llama-3.3-70B.
        </p>
        <Link to="/chat" id="hero-cta-btn" className="btn btn-primary btn-lg">
          Start Asking →
        </Link>
      </section>

      {/* Feature grid */}
      <section className="features">
        <div className="feature-card">
          <div className="feature-icon">🔍</div>
          <h3>Hybrid Search</h3>
          <p>Combines vector similarity and keyword search via Reciprocal Rank Fusion for maximum recall.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">⚡</div>
          <h3>Cross-Encoder Reranking</h3>
          <p>ms-marco-MiniLM reranks top candidates for 6× precision improvement before LLM generation.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📎</div>
          <h3>Citation Enforcement</h3>
          <p>Every factual claim is cited with [SOURCE_ID] — no hallucinations, fully traceable answers.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Ragas Evaluation</h3>
          <p>Built-in faithfulness and relevance scoring. CI gate fails builds that regress quality.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🧩</div>
          <h3>Parent-Child Chunking</h3>
          <p>Retrieves highly precise child chunks and expands to larger parent contexts for the LLM.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🧠</div>
          <h3>Semantic Chunking</h3>
          <p>Intelligently groups text by semantic boundaries rather than rigid character limits.</p>
        </div>
      </section>

      {/* Pipeline diagram */}
      <section className="pipeline">
        <h2>The RAG Pipeline</h2>
        <div className="pipeline-steps">
          {['Upload Doc', 'Chunk + Embed', 'Hybrid Search', 'Rerank', 'Cite + Answer'].map((step, i) => (
            <div key={step} className="pipeline-step">
              <div className="step-number">{i + 1}</div>
              <div className="step-label">{step}</div>
              {i < 4 && <div className="step-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
