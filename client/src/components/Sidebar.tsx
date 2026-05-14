import { useEffect } from 'react';
import { useChatStore } from '../store/chatStore';
import { useUpload } from '../hooks/useUpload';
import { UploadPanel } from './UploadPanel';

/** Left sidebar: upload panel + list of ingested documents. */
export function Sidebar() {
  const { documents, activeDocId, setActiveDocId, isLoadingDocs } = useChatStore();
  const { remove, refreshDocuments } = useUpload();

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2 className="sidebar-title">
          <span className="sidebar-logo">📚</span>
          AskMyDocs
        </h2>
      </div>

      <UploadPanel />

      <div className="sidebar-section">
        <div className="sidebar-section-header">
          <span>Documents</span>
          {isLoadingDocs && <span className="sidebar-spinner" />}
        </div>

        {documents.length === 0 ? (
          <p className="sidebar-empty">No documents yet. Upload one above.</p>
        ) : (
          <ul className="doc-list">
            {/* "All documents" filter option */}
            <li
              id="doc-filter-all"
              className={`doc-item ${activeDocId === null ? 'doc-item-active' : ''}`}
              onClick={() => setActiveDocId(null)}
            >
              <span className="doc-icon">🌐</span>
              <span className="doc-name">All documents</span>
              <span className="doc-count">{documents.reduce((s, d) => s + d.chunk_count, 0)}</span>
            </li>

            {documents.map((doc) => (
              <li
                key={doc.doc_id}
                id={`doc-item-${doc.doc_id}`}
                className={`doc-item ${activeDocId === doc.doc_id ? 'doc-item-active' : ''}`}
                onClick={() => setActiveDocId(doc.doc_id)}
              >
                <span className="doc-icon">📄</span>
                <span className="doc-name" title={doc.source_name}>
                  {doc.source_name}
                </span>
                <span className="doc-count">{doc.chunk_count}</span>
                <button
                  id={`doc-delete-${doc.doc_id}`}
                  className="doc-delete-btn"
                  onClick={(e) => { e.stopPropagation(); void remove(doc.doc_id); }}
                  aria-label={`Delete ${doc.source_name}`}
                  title="Delete document"
                >
                  🗑
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
