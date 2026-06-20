import type { SourceChunk } from '../types';

interface CitationBadgeProps {
  index: number;
  source: SourceChunk;
}

/** Renders a compact inline [Sn] citation pill with brackets.
 *  The full source name and a content preview appear on hover.
 *  Brackets ensure adjacent citations like [S1][S2][S3] read clearly. */
export function CitationBadge({ index, source }: CitationBadgeProps) {
  return (
    <sup
      className="citation-badge"
      title={`${source.source_name} — ${source.content.slice(0, 160)}…`}
    >
      [S{index}]
    </sup>
  );
}
