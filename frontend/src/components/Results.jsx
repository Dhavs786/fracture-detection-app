import React, { useState } from 'react';

export default function Results({ results, params, file }) {
  const [reportLoading, setReportLoading] = useState(false);

  if (!results) return null;

  const { detections, stats, originalImage, enhancedImage, annotatedImage } = results;
  const hasFracture = stats.zones > 0;

  const handleDownloadReport = async () => {
    if (!file) return;
    setReportLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('patient_name', params.patientName);
      formData.append('patient_age', params.patientAge);
      formData.append('patient_gender', params.patientGender);
      formData.append('tech', params.tech);
      formData.append('scan_id', params.scanId);
      formData.append('confidence', params.confidence);
      formData.append('iou', params.iou);
      formData.append('clahe', params.clahe);
      formData.append('denoise', params.denoise);
      formData.append('multiscale', params.multiscale);
      formData.append('tta', params.tta);

      const res = await fetch('/api/report', {
        method: 'POST',
        body: formData,
      });

      const htmlContent = await res.text();
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${params.scanId}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to generate report:", err);
      alert("Error generating report.");
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <div style={{ marginTop: '30px' }}>
      {/* Status Banner */}
      <div className={`status-card ${hasFracture ? 'status-danger' : 'status-safe'}`}>
        <div className="status-title">
          {hasFracture ? '⚠️ ABNORMALITY DETECTED' : '✅ NO FRACTURE DETECTED'}
        </div>
        <div className="status-sub">
          {hasFracture ? (
            <>Identified <b>{stats.zones}</b> potential fracture zone(s). Peak confidence: <b>{stats.peakConfidence}%</b>.</>
          ) : (
            <>No structural anomalies or skeletal breaks were identified above the threshold.</>
          )}
          <br />
          <small style={{ opacity: 0.7 }}>
            Ensemble: {stats.passes} inference passes in {stats.scanTime}s
          </small>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div style={{ fontSize: '1.5rem', marginBottom: '4px' }}>🎯</div>
          <div className="metric-val" style={{ color: '#fda4af' }}>{stats.zones}</div>
          <div className="metric-lbl">Zones Found</div>
        </div>

        <div className="metric-card">
          <div style={{ fontSize: '1.5rem', marginBottom: '4px' }}>📊</div>
          <div className="metric-val" style={{ color: '#fcd34d' }}>{stats.peakConfidence}%</div>
          <div className="metric-lbl">Peak Confidence</div>
        </div>

        <div className="metric-card">
          <div style={{ fontSize: '1.5rem', marginBottom: '4px' }}>🔬</div>
          <div className="metric-val" style={{ color: '#a5b4fc' }}>{stats.passes}</div>
          <div className="metric-lbl">Passes Run</div>
        </div>

        <div className="metric-card">
          <div style={{ fontSize: '1.5rem', marginBottom: '4px' }}>⚡</div>
          <div className="metric-val" style={{ color: '#67e8f9' }}>{stats.scanTime}s</div>
          <div className="metric-lbl">Scan Time</div>
        </div>
      </div>

      {/* Radiograph Scans Display Grid */}
      <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px', color: '#e2e8f0' }}>
        🩺 Radiograph Analysis Views
      </h3>
      <div className="view-grid">
        <div className="view-box">
          <img src={`data:image/jpeg;base64,${enhancedImage}`} alt="Enhanced View" />
          <div className="view-caption">1. Enhanced Preprocessed Image</div>
        </div>

        <div className="view-box">
          <img src={`data:image/jpeg;base64,${annotatedImage}`} alt="AI Diagnostic Overlay" />
          <div className="view-caption">2. AI Diagnostic Overlay</div>
        </div>
      </div>

      {/* Detection Table */}
      <div style={{ marginTop: '35px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '12px', color: '#e2e8f0' }}>
          📊 Detection Breakdown
        </h3>
        
        {detections.length > 0 ? (
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Bounding Box [x1, y1, x2, y2]</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((det, idx) => (
                <tr key={idx}>
                  <td>{idx + 1}</td>
                  <td>
                    <span style={{
                      background: 'rgba(239, 68, 68, 0.2)',
                      border: '1px solid rgba(239, 68, 68, 0.4)',
                      color: '#fca5a5',
                      padding: '2px 8px',
                      borderRadius: '12px',
                      fontSize: '0.78rem',
                      fontWeight: 600
                    }}>
                      Fracture Detected
                    </span>
                  </td>
                  <td style={{ fontWeight: 700, color: '#fcd34d' }}>{det.confidence}%</td>
                  <td style={{ fontFamily: 'monospace', color: '#94a3b8' }}>
                    [{det.box.join(', ')}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{
            background: 'rgba(15, 23, 42, 0.4)',
            padding: '20px',
            borderRadius: '12px',
            textAlign: 'center',
            color: '#94a3b8',
            fontSize: '0.9rem'
          }}>
            No detections recorded above the current threshold.
          </div>
        )}
      </div>

      {/* Report Generator Footer */}
      <div style={{
        marginTop: '35px',
        padding: '24px',
        background: 'rgba(15, 23, 42, 0.6)',
        borderRadius: '16px',
        border: '1px solid rgba(255, 255, 255, 0.05)',
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '15px'
      }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: '1rem', color: '#f8fafc' }}>
            Generate Diagnostic Case Report
          </div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            Download a standardized clinical HTML report ready for printing/PDF conversion.
          </div>
        </div>

        <button
          className="btn-emerald"
          onClick={handleDownloadReport}
          disabled={reportLoading}
        >
          {reportLoading ? '⏳ GENERATING...' : '📥 DOWNLOAD CASE REPORT'}
        </button>
      </div>
    </div>
  );
}
