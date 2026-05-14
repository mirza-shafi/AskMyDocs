import { useCallback } from 'react';
import { queryDocuments } from '../api/client';
import { useChatStore } from '../store/chatStore';

export function useChat() {
  const { activeDocId, addMessage, setLoading, setError, isLoading } = useChatStore();

  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isLoading) return;

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
    [activeDocId, addMessage, isLoading, setError, setLoading],
  );

  return { sendMessage, isLoading };
}
