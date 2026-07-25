import React from 'react';

export default function ControlPanel({ params, setParams }) {
  const handleChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">🦴</div>
        <h2 className="brand-title">Clinical Dashboard</h2>
        <div className="brand-tag">FractureVision AI v2.0</div>
      </div>

      {/* Patient Info Panel */}
      <div className="panel-card">
        <div className="panel-title">📋 Patient Metadata</div>
        
        <div className="input-group">
          <label className="input-label">Patient Name</label>
          <input
            type="text"
            className="text-input"
            value={params.patientName}
            onChange={e => handleChange('patientName', e.target.value)}
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          <div className="input-group">
            <label className="input-label">Age</label>
            <input
              type="text"
              className="text-input"
              value={params.patientAge}
              onChange={e => handleChange('patientAge', e.target.value)}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Gender</label>
            <select
              className="select-input"
              value={params.patientGender}
              onChange={e => handleChange('patientGender', e.target.value)}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>

        <div className="input-group">
          <label className="input-label">Tech Initials</label>
          <input
            type="text"
            className="text-input"
            value={params.tech}
            onChange={e => handleChange('tech', e.target.value)}
          />
        </div>

        <div className="input-group">
          <label className="input-label">Scan ID</label>
          <input
            type="text"
            className="text-input"
            value={params.scanId}
            onChange={e => handleChange('scanId', e.target.value)}
          />
        </div>
      </div>

      {/* Model Parameters */}
      <div className="panel-card">
        <div className="panel-title">⚙️ Model Parameters</div>
        
        <div className="input-group">
          <label className="input-label">
            Confidence Threshold <span className="input-val">{(params.confidence * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="1.00"
            step="0.01"
            className="range-slider"
            value={params.confidence}
            onChange={e => handleChange('confidence', parseFloat(e.target.value))}
          />
        </div>

        <div className="input-group">
          <label className="input-label">
            IoU Threshold (NMS) <span className="input-val">{params.iou.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.10"
            max="0.95"
            step="0.05"
            className="range-slider"
            value={params.iou}
            onChange={e => handleChange('iou', parseFloat(e.target.value))}
          />
        </div>
      </div>

      {/* Clinical Image Enhancers */}
      <div className="panel-card">
        <div className="panel-title">🩺 Clinical Image Enhancers</div>
        
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={params.clahe}
            onChange={e => handleChange('clahe', e.target.checked)}
          />
          CLAHE Contrast Enhancer
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={params.invert}
            onChange={e => handleChange('invert', e.target.checked)}
          />
          Invert Colors (Negative Film)
        </label>

        <div className="input-group">
          <label className="input-label">
            Brightness <span className="input-val">{params.brightness.toFixed(1)}x</span>
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            className="range-slider"
            value={params.brightness}
            onChange={e => handleChange('brightness', parseFloat(e.target.value))}
          />
        </div>

        <div className="input-group">
          <label className="input-label">
            Global Contrast <span className="input-val">{params.contrast.toFixed(1)}x</span>
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            className="range-slider"
            value={params.contrast}
            onChange={e => handleChange('contrast', parseFloat(e.target.value))}
          />
        </div>

        <div className="input-group">
          <label className="input-label">
            Sharpness <span className="input-val">{params.sharpness.toFixed(1)}x</span>
          </label>
          <input
            type="range"
            min="0.0"
            max="3.0"
            step="0.1"
            className="range-slider"
            value={params.sharpness}
            onChange={e => handleChange('sharpness', parseFloat(e.target.value))}
          />
        </div>
      </div>

      {/* AI Boosters */}
      <div className="panel-card">
        <div className="panel-title">🚀 AI Detection Boosters</div>
        
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={params.denoise}
            onChange={e => handleChange('denoise', e.target.checked)}
          />
          Bilateral Denoise + Unsharp Mask
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={params.multiscale}
            onChange={e => handleChange('multiscale', e.target.checked)}
          />
          Multi-Scale Ensemble (416-1024)
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={params.tta}
            onChange={e => handleChange('tta', e.target.checked)}
          />
          Test-Time Augmentation (Flip)
        </label>
      </div>
    </aside>
  );
}
