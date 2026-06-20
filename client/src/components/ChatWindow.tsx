import { useEffect, useRef, useState } from 'react';
import { useChat } from '../hooks/useChat';
import { useChatStore } from '../store/chatStore';
import { MessageBubble } from './MessageBubble';

/** Full chat window: message thread + input bar. */
export function ChatWindow() {
  const { messages, isLoading, activeDocId, documents } = useChatStore();
  const { sendMessage } = useChat();
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeDoc = documents.find((d) => d.doc_id === activeDocId);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = input.trim();
    if (!q) return;
    setInput('');
    void sendMessage(q);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return (
    <div className="chat-window">
      {/* Message thread */}
      <div className="message-list">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">💬</div>
            <h3>
              {activeDoc
                ? `Ask anything about "${activeDoc.source_name}"`
                : 'Select a document from the sidebar, then ask a question'}
            </h3>
            <p>
              {activeDoc
                ? `${activeDoc.chunk_count} chunks indexed and ready.`
                : 'Upload a PDF or text file, then click it to select it.'}
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="message-bubble message-assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-body">
              <div className="typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Active document indicator + input bar */}
      <div className="chat-input-wrapper">
        {/* Scope pill — shows exactly which doc will be queried */}
        <div className={`query-scope-bar ${activeDoc ? 'scope-doc' : 'scope-all'}`}>
          {activeDoc ? (
            <>
              <span className="scope-icon">📄</span>
              <span className="scope-label">
                Asking about: <strong>{activeDoc.source_name}</strong>
              </span>
            </>
          ) : (
            <>
              <span className="scope-icon">🌐</span>
              <span className="scope-label">
                Asking across <strong>all documents</strong> — select one from the sidebar to focus
              </span>
            </>
          )}
        </div>

        <form className="chat-input-bar" onSubmit={handleSubmit}>
          <textarea
            id="chat-input"
            className="chat-input"
            placeholder={
              activeDoc
                ? `Ask a question about ${activeDoc.source_name}…`
                : 'Select a document first, then ask a question…'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={1}
          />
          <button
            id="chat-send-btn"
            type="submit"
            className="chat-send-btn"
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
          >
            {isLoading ? '⏳' : '➤'}
          </button>
        </form>
      </div>
    </div>
  );
}
