import { useCallback, useState } from 'react';
import { deleteDocument, listDocuments, pollIngestStatus, uploadDocument } from '../api/client';
import { useChatStore } from '../store/chatStore';

type UploadState = 'idle' | 'uploading' | 'processing' | 'done' | 'error';

export function useUpload() {
  const { setDocuments, setLoadingDocs, removeDocument } = useChatStore();
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('');

  const refreshDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const result = await listDocuments();
      setDocuments(result.documents || []);
    } catch {
      // Silent fail on list refresh
    } finally {
      setLoadingDocs(false);
    }
  }, [setDocuments, setLoadingDocs]);

  const upload = useCallback(
    async (file: File) => {
      setUploadState('uploading');
      setUploadError(null);
      setProgress('Uploading file…');

      try {
        const { job_id } = await uploadDocument(file);
        setUploadState('processing');
        setProgress('Processing document…');

        // Poll for job completion
        const MAX_POLLS = 60;
        let transientErrors = 0;
        for (let i = 0; i < MAX_POLLS; i++) {
          await new Promise((r) => setTimeout(r, 2000)); // 2s interval

          let status;
          try {
            status = await pollIngestStatus(job_id);
          } catch {
            // A transient network blip shouldn't kill an otherwise-healthy
            // ingestion. Tolerate a few consecutive failures, then give up.
            transientErrors += 1;
            if (transientErrors >= 5) {
              throw new Error('Lost connection while processing document');
            }
            continue;
          }
          transientErrors = 0;

          if (status.status === 'completed') {
            setProgress(`✓ ${status.chunks_created ?? 0} chunks stored`);
            setUploadState('done');
            await refreshDocuments();
            setTimeout(() => setUploadState('idle'), 3000);
            return;
          }

          if (status.status === 'failed') {
            throw new Error(status.error ?? 'Ingestion failed');
          }
        }
        throw new Error('Ingestion timed out');
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        setUploadError(msg);
        setUploadState('error');
        setProgress('');
      }
    },
    [refreshDocuments],
  );

  const remove = useCallback(
    async (docId: string) => {
      try {
        await deleteDocument(docId);
        removeDocument(docId);
      } catch (err) {
        console.error('Delete failed', err);
      }
    },
    [removeDocument],
  );

  return { upload, remove, refreshDocuments, uploadState, uploadError, progress };
}
