import React, { useRef } from 'react';

export default function UploadZone({ file, setFile, previewUrl, setPreviewUrl, onAnalyze, loading }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div style={{ marginBottom: '30px' }}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/png, image/jpeg, image/jpg"
        style={{ display: 'none' }}
      />

      <div
        className="upload-card"
        onClick={() => fileInputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {previewUrl ? (
          <div>
            <img
              src={previewUrl}
              alt="Radiograph Preview"
              style={{
                maxHeight: '260px',
                maxWidth: '100%',
                borderRadius: '12px',
                objectFit: 'contain',
                marginBottom: '10px'
              }}
            />
            <div className="upload-sub">Click or drag a new image to replace</div>
          </div>
        ) : (
          <div>
            <div className="upload-icon">🦴</div>
            <div className="upload-text">Upload Radiograph Image</div>
            <div className="upload-sub">Supports PNG, JPG, JPEG X-Ray scans</div>
          </div>
        )}
      </div>

      <button
        className="btn-primary"
        disabled={!file || loading}
        onClick={onAnalyze}
      >
        {loading ? (
          <>🚀 EXECUTING ENSEMBLE SCAN...</>
        ) : (
          <>🚀 INITIATE SCAN</>
        )}
      </button>
    </div>
  );
}
