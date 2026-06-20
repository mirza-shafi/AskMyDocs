import { useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { useUpload } from '../hooks/useUpload';

/** Drag-and-drop / click file upload panel. */
export function UploadPanel() {
  const { upload, resetUpload, uploadState, uploadError, progress } = useUpload();
  const { activeDocId, documents } = useChatStore();

  // Show which document is currently active
  const activeDoc = documents.find((d) => d.doc_id === activeDocId);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) void upload(file);
    },
    [upload],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void upload(file);
    e.target.value = '';
  };

  const isActive = uploadState === 'uploading' || uploadState === 'processing';

  return (
    <div
      className={[
        'upload-panel',
        isActive ? 'upload-active' : '',
        uploadState === 'done' ? 'upload-done' : '',
        uploadState === 'error' ? 'upload-error' : '',
        uploadState === 'idle' && activeDoc ? 'upload-has-doc' : '',
      ].filter(Boolean).join(' ')}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={(e) => e.preventDefault()}
    >
      <input
        id="file-upload-input"
        type="file"
        accept=".pdf,.txt"
        className="upload-input-hidden"
        onChange={handleFileChange}
        disabled={isActive}
      />
      <label htmlFor="file-upload-input" className="upload-label">
        {uploadState === 'idle' && (
          <>
            <span className="upload-icon">📄</span>
            <span className="upload-text">
              {activeDoc ? `Active: ${activeDoc.source_name}` : 'Drop PDF or TXT here'}
            </span>
            <span className="upload-subtext">
              {activeDoc ? 'Drop a new file to switch documents' : 'or click to browse'}
            </span>
          </>
        )}
        {isActive && (
          <>
            <span className="upload-spinner" />
            <span className="upload-text">{progress}</span>
          </>
        )}
        {uploadState === 'done' && (
          <>
            <span className="upload-icon">✅</span>
            <span className="upload-text">{progress}</span>
            <span className="upload-subtext">
              {activeDoc ? `Now chatting with: ${activeDoc.source_name}` : 'Document ready!'}
            </span>
          </>
        )}
        {uploadState === 'error' && (
          <>
            <span className="upload-icon">❌</span>
            <span className="upload-text">Upload failed</span>
            <span className="upload-subtext">{uploadError}</span>
          </>
        )}
      </label>

      {/* "Try again" button when in error state — resets without forcing a page reload */}
      {uploadState === 'error' && (
        <button
          id="upload-retry-btn"
          className="upload-retry-btn"
          onClick={(e) => {
            e.stopPropagation();
            resetUpload();
          }}
          type="button"
        >
          Try again
        </button>
      )}
    </div>
  );
}
