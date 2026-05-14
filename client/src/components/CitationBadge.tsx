import type { SourceChunk } from '../types';

interface CitationBadgeProps {
  index: number;
  source: SourceChunk;
}

/** Renders an inline [Sn] citation badge with a hover tooltip showing chunk preview. */
export function CitationBadge({ index, source }: CitationBadgeProps) {
  return (
    <span className="citation-badge" title={`${source.source_name} — ${source.content.slice(0, 120)}…`}>
      <span className="citation-number">S{index}</span>
      <span className="citation-source">{source.source_name}</span>
    </span>
  );
}
