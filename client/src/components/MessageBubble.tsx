import type { ReactNode } from 'react';
import type { Message } from '../types';
import { CitationBadge } from './CitationBadge';

interface MessageBubbleProps {
  message: Message;
}

type Sources = Message['sources'];

/** Convert a raw cross-encoder logit into a 0-100 relevance percentage.
 *  Cross-encoder scores are unbounded logits (often negative), so a plain
 *  ×100 produces nonsensical values like -1063%. A sigmoid maps them to a
 *  meaningful 0-1 relevance probability. */
function relevancePct(score: number): number {
  return Math.round((1 / (1 + Math.exp(-score))) * 100);
}

/** Render a single line of text: resolve [Sn] citations and **bold** spans. */
function renderInline(text: string, sources: Sources, keyPrefix: string) {
  const tokens = text.split(/(\[S\d+\]|\*\*[^*]+\*\*)/g);
  return tokens.map((tok, i) => {
    const key = `${keyPrefix}-${i}`;
    const cite = tok.match(/^\[S(\d+)\]$/);
    if (cite) {
      const idx = parseInt(cite[1], 10) - 1;
      const source = sources?.[idx];
      return source ? (
        <CitationBadge key={key} index={idx + 1} source={source} />
      ) : (
        <span key={key} className="citation-unknown">{tok}</span>
      );
    }
    const bold = tok.match(/^\*\*([^*]+)\*\*$/);
    if (bold) return <strong key={key}>{bold[1]}</strong>;
    if (!tok) return null;
    return <span key={key}>{tok}</span>;
  });
}

/** Render an assistant answer as readable blocks: paragraphs + bullet lists. */
function renderAnswer(content: string, sources: Sources) {
  const lines = content.split('\n');
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (!bullets.length) return;
    const items = bullets;
    bullets = [];
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="answer-list">
        {items.map((item, i) => (
          <li key={i}>{renderInline(item, sources, `li-${blocks.length}-${i}`)}</li>
        ))}
      </ul>,
    );
  };

  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) {
      flushBullets();
      return;
    }
    const bullet = line.match(/^[*\-•]\s+(.*)$/);
    if (bullet) {
      bullets.push(bullet[1]);
    } else {
      flushBullets();
      blocks.push(
        <p key={`p-${blocks.length}`}>{renderInline(line, sources, `p-${blocks.length}`)}</p>,
      );
    }
  });
  flushBullets();

  return <>{blocks}</>;
}

/** Renders a single chat message bubble with optional source citations. */
export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

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
                  {relevancePct(source.rerank_score)}%
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
