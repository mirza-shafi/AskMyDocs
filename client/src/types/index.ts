/** Shared TypeScript interfaces for the AskMyDocs frontend. */

export interface SourceChunk {
  chunk_id: string;
  doc_id: string;
  source_name: string;
  chunk_index: number;
  content: string;
  rerank_score: number;
}

export interface EvalScores {
  faithfulness: number | null;
  answer_relevance: number | null;
  context_precision: number | null;
  context_recall: number | null;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: SourceChunk[];
  latency_ms: number;
  eval_scores: EvalScores | null;
}

export interface QueryRequest {
  question: string;
  doc_id?: string;
  include_eval?: boolean;
}

export interface IngestResponse {
  job_id: string;
  doc_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
}

export interface IngestStatusResponse {
  job_id: string;
  doc_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  chunks_created: number | null;
  error: string | null;
}

export interface DocumentItem {
  doc_id: string;
  source_name: string;
  chunk_count: number;
}

export interface DocumentListResponse {
  documents: DocumentItem[];
  total: number;
}

export type MessageRole = 'user' | 'assistant';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  sources?: SourceChunk[];
  latency_ms?: number;
  timestamp: Date;
}
