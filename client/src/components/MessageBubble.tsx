import type { ReactNode } from 'react';
import type { Message } from '../types';

interface MessageBubbleProps {
  message: Message;
}

type Sources = Message['sources'];

/** Render a line: handle **bold** markers only. No citation badges. */
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const tokens = text.split(/(\*\*[^*]+\*\*)/g);
  return tokens.map((tok, i) => {
    const key = `${keyPrefix}-${i}`;
    const bold = tok.match(/^\*\*([^*]+)\*\*$/);
    if (bold) return <strong key={key}>{bold[1]}</strong>;
    // Strip any leftover [Sn] citation tags from the raw LLM text
    const clean = tok.replace(/\[S\d+\]/g, '').trim();
    if (!clean) return null;
    return <span key={key}>{clean}</span>;
  }).filter(Boolean) as ReactNode[];
}

/** Render assistant answer as clean paragraphs + bullet/numbered lists. */
function renderAnswer(content: string, _sources: Sources) {
  // Strip all [Sn] citation tags before rendering
  const clean = content.replace(/\[S\d+\]/g, '').replace(/\s{2,}/g, ' ');
  const lines = clean.split('\n');
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  let numbered: string[] = [];

  const flushBullets = () => {
    if (!bullets.length) return;
    const items = [...bullets]; bullets = [];
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="answer-list">
        {items.map((item, i) => (
          <li key={i}>{renderInline(item.trim(), `ul-${blocks.length}-${i}`)}</li>
        ))}
      </ul>,
    );
  };

  const flushNumbered = () => {
    if (!numbered.length) return;
    const items = [...numbered]; numbered = [];
    blocks.push(
      <ol key={`ol-${blocks.length}`} className="answer-list answer-list-numbered">
        {items.map((item, i) => (
          <li key={i}>{renderInline(item.trim(), `ol-${blocks.length}-${i}`)}</li>
        ))}
      </ol>,
    );
  };

  const flushAll = () => { flushBullets(); flushNumbered(); };

  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) { flushAll(); return; }

    const bullet = line.match(/^[*\-•]\s+(.*)$/);
    if (bullet) { flushNumbered(); bullets.push(bullet[1]); return; }

    const num = line.match(/^\d+\.\s+(.*)$/);
    if (num) { flushBullets(); numbered.push(num[1]); return; }

    flushAll();
    blocks.push(
      <p key={`p-${blocks.length}`}>
        {renderInline(line, `p-${blocks.length}`)}
      </p>,
    );
  });

  flushAll();
  return <>{blocks}</>;
}

/** Single chat message bubble — plain answer, no citation noise. */
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
        {message.latency_ms != null && (
          <span className="message-latency">{message.latency_ms.toFixed(0)}ms</span>
        )}
      </div>
    </div>
  );
}
