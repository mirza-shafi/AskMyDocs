import { create } from 'zustand';
import type { DocumentItem, Message } from '../types';

interface ChatState {
  // Messages
  messages: Message[];
  isLoading: boolean;
  error: string | null;

  // Documents
  documents: DocumentItem[];
  isLoadingDocs: boolean;

  // Selected doc filter
  activeDocId: string | null;

  // Actions
  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setDocuments: (docs: DocumentItem[]) => void;
  setLoadingDocs: (loading: boolean) => void;
  setActiveDocId: (docId: string | null) => void;
  removeDocument: (docId: string) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  error: null,
  documents: [],
  isLoadingDocs: false,
  activeDocId: null,

  addMessage: (msg) =>
    set((state) => ({
      messages: [
        ...state.messages,
        { ...msg, id: crypto.randomUUID(), timestamp: new Date() },
      ],
    })),

  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setDocuments: (docs) => set({ documents: docs || [] }),
  setLoadingDocs: (loading) => set({ isLoadingDocs: loading }),
  setActiveDocId: (docId) => set({ activeDocId: docId }),

  removeDocument: (docId) =>
    set((state) => ({
      documents: state.documents.filter((d) => d.doc_id !== docId),
      activeDocId: state.activeDocId === docId ? null : state.activeDocId,
    })),

  clearMessages: () => set({ messages: [] }),
}));
