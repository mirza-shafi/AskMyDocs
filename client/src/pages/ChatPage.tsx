import { ChatWindow } from '../components/ChatWindow';
import { Sidebar } from '../components/Sidebar';
import { useChatStore } from '../store/chatStore';

/** Main chat interface page: sidebar + chat window. */
export function ChatPage() {
  const { activeDocId, documents, clearMessages } = useChatStore();
  const activeDoc = documents.find((d) => d.doc_id === activeDocId);

  return (
    <div className="chat-page">
      <Sidebar />

      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-info">
            <h1 className="chat-header-title">
              {activeDoc ? `📄 ${activeDoc.source_name}` : '🌐 All Documents'}
            </h1>
            {activeDoc && (
              <span className="chat-header-meta">
                {activeDoc.chunk_count} chunks indexed
              </span>
            )}
          </div>
          <button
            id="clear-chat-btn"
            className="btn btn-ghost btn-sm"
            onClick={clearMessages}
            title="Clear conversation"
          >
            Clear chat
          </button>
        </header>

        <ChatWindow />
      </main>
    </div>
  );
}
