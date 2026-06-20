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
  headers: { 'Content-Type': 'application/json' },
});

// ── Request interceptor ──────────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  return config;
});

// ── Response interceptor (normalise errors) ──────────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.response?.data?.detail ||
      error.message ||
      'An unexpected error occurred';
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
  // Do NOT set Content-Type manually: axios derives the correct
  // multipart/form-data header *with boundary* from the FormData instance.
  // Hard-coding it drops the boundary and breaks server-side parsing.
  const { data } = await api.post<IngestResponse>('/api/v1/ingest', formData);
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
