import type { SourceChunk } from '../types';

interface CitationBadgeProps {
  index: number;
  source: SourceChunk;
}

/** Renders a compact inline [Sn] citation pill. The full source name and a
 *  content preview are shown on hover (and in the source cards below the
 *  message), keeping the answer text itself readable. */
export function CitationBadge({ index, source }: CitationBadgeProps) {
  return (
    <sup
      className="citation-badge"
      title={`${source.source_name} — ${source.content.slice(0, 160)}…`}
    >
      S{index}
    </sup>
  );
}
