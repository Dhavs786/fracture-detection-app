import React, { useState } from 'react';
import ControlPanel from './components/ControlPanel';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import Results from './components/Results';

export default function App() {
  const [params, setParams] = useState({
    patientName: 'John Doe',
    patientAge: '45',
    patientGender: 'Male',
    tech: 'RD-A',
    scanId: `FX-${Math.floor(100000 + Math.random() * 900000)}`,
    confidence: 0.25,
    iou: 0.45,
    clahe: true,
    invert: false,
    brightness: 1.0,
    contrast: 1.0,
    sharpness: 1.0,
    denoise: true,
    multiscale: true,
    tta: true,
  });

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('confidence', params.confidence);
      formData.append('iou', params.iou);
      formData.append('clahe', params.clahe);
      formData.append('invert', params.invert);
      formData.append('brightness', params.brightness);
      formData.append('contrast', params.contrast);
      formData.append('sharpness', params.sharpness);
      formData.append('denoise', params.denoise);
      formData.append('multiscale', params.multiscale);
      formData.append('tta', params.tta);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API analysis failed with status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error("Inference Error:", err);
      setError(err.message || "Failed to complete diagnostic scan. Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <ControlPanel params={params} setParams={setParams} />

      <main className="main-content">
        <Header />

        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#fca5a5',
            padding: '14px 20px',
            borderRadius: '12px',
            marginBottom: '20px',
            fontSize: '0.9rem'
          }}>
            ⚠️ {error}
          </div>
        )}

        <UploadZone
          file={file}
          setFile={setFile}
          previewUrl={previewUrl}
          setPreviewUrl={setPreviewUrl}
          onAnalyze={handleAnalyze}
          loading={loading}
        />

        <Results
          results={results}
          params={params}
          file={file}
        />
      </main>
    </div>
  );
}
