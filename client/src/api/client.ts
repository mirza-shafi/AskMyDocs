import axios from 'axios';
import type {
  DocumentListResponse,
  IngestResponse,
  IngestStatusResponse,
  QueryRequest,
  QueryResponse,
} from '../types';

/** Base Axios instance. VITE_API_BASE_URL is set in Vercel env vars. */
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 60_000, // 60s — RAG pipeline can take time
  // NOTE: Do NOT set a global Content-Type here.
  // For JSON requests axios defaults to application/json automatically.
  // For FormData (file uploads) axios MUST set multipart/form-data + boundary
  // by itself — a global Content-Type header overrides that and breaks uploads.
});

// ── Request interceptor ──────────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  return config;
});

// ── Response interceptor (normalise errors) ──────────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Normalise FastAPI / backend error shapes into a plain string.
    // FastAPI validation errors: { detail: [{loc, msg, type}, ...] }
    // FastAPI HTTP exceptions:   { detail: "some string" }
    // Custom backend errors:     { error: "some string" }
    const data = error.response?.data;
    let message: string;

    if (typeof data?.error === 'string') {
      message = data.error;
    } else if (typeof data?.detail === 'string') {
      message = data.detail;
    } else if (Array.isArray(data?.detail)) {
      // FastAPI validation error array — extract the human-readable messages
      message = (data.detail as Array<{ msg?: string }>)
        .map((e) => e.msg ?? JSON.stringify(e))
        .join('; ');
    } else if (error.message) {
      message = error.message;
    } else {
      message = 'An unexpected error occurred';
    }

    return Promise.reject(new Error(message));
  },
);

// ── API functions ────────────────────────────────────────────────────────────

export const queryDocuments = async (payload: QueryRequest): Promise<QueryResponse> => {
  const { data } = await api.post<QueryResponse>('/api/v1/query', payload);
  return data;
};

export const uploadDocument = async (file: File): Promise<IngestResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  // Pass the FormData directly — axios detects FormData and sets
  // multipart/form-data + the correct boundary automatically.
  // Explicitly delete the Content-Type for this request so the instance-level
  // header (if any) never clobbers the multipart boundary.
  const { data } = await api.post<IngestResponse>('/api/v1/ingest', formData, {
    headers: { 'Content-Type': undefined },
  });
  return data;
};

export const pollIngestStatus = async (jobId: string): Promise<IngestStatusResponse> => {
  const { data } = await api.get<IngestStatusResponse>(`/api/v1/ingest/${jobId}`);
  return data;
};

export const listDocuments = async (): Promise<DocumentListResponse> => {
  const { data } = await api.get<DocumentListResponse>('/api/v1/docs');
  return data;
};

export const deleteDocument = async (docId: string): Promise<void> => {
  await api.delete(`/api/v1/docs/${docId}`);
};

export const checkHealth = async (): Promise<{ status: string; version: string }> => {
  const { data } = await api.get('/health');
  return data;
};

export default api;
