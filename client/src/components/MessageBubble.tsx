import type { Message } from '../types';
import { CitationBadge } from './CitationBadge';

interface MessageBubbleProps {
  message: Message;
}

/** Renders a single chat message bubble with optional source citations. */
export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  // Replace [S1], [S2] etc. with inline citation badges
  const renderAnswer = (content: string, sources: Message['sources']) => {
    if (!sources?.length) return <p>{content}</p>;

    const parts = content.split(/(\[S\d+\])/g);
    return (
      <p>
        {parts.map((part, i) => {
          const match = part.match(/\[S(\d+)\]/);
          if (match) {
            const idx = parseInt(match[1], 10) - 1;
            const source = sources[idx];
            return source ? (
              <CitationBadge key={i} index={idx + 1} source={source} />
            ) : (
              <span key={i} className="citation-unknown">{part}</span>
            );
          }
          return <span key={i}>{part}</span>;
        })}
      </p>
    );
  };

  return (
    <div className={`message-bubble ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-avatar">{isUser ? '👤' : '🤖'}</div>
      <div className="message-body">
        <div className="message-content">
          {isUser
            ? <p>{message.content}</p>
            : renderAnswer(message.content, message.sources)
          }
        </div>

        {/* Source cards (only for assistant messages with sources) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="source-cards">
            {message.sources.map((source, i) => (
              <div key={source.chunk_id} className="source-card">
                <span className="source-badge">S{i + 1}</span>
                <span className="source-name">{source.source_name}</span>
                <span className="source-score">
                  {(source.rerank_score * 100).toFixed(0)}%
                </span>
                <p className="source-preview">{source.content.slice(0, 160)}…</p>
              </div>
            ))}
          </div>
        )}

        {message.latency_ms && (
          <span className="message-latency">{message.latency_ms.toFixed(0)}ms</span>
        )}
      </div>
    </div>
  );
}
