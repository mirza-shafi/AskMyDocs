import { useCallback } from 'react';
import { queryDocuments } from '../api/client';
import { useChatStore } from '../store/chatStore';

export function useChat() {
  // Only subscribe to isLoading for re-render — activeDocId is read via
  // getState() at call time to avoid stale-closure bugs in useCallback.
  const { addMessage, setLoading, setError, isLoading } = useChatStore();

  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isLoading) return;

      // Read activeDocId at call time, NOT from a stale closure
      const activeDocId = useChatStore.getState().activeDocId;

      // Add the user message immediately
      addMessage({ role: 'user', content: question });
      setLoading(true);
      setError(null);

      try {
        const response = await queryDocuments({
          question,
          doc_id: activeDocId ?? undefined,
          include_eval: false,
        });

        addMessage({
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          latency_ms: response.latency_ms,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to get answer';
        setError(msg);
        addMessage({
          role: 'assistant',
          content: `⚠️ Error: ${msg}`,
        });
      } finally {
        setLoading(false);
      }
    },
    [addMessage, isLoading, setError, setLoading],
  );

  return { sendMessage, isLoading };
}
