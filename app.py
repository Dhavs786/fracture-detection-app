import streamlit as st
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance
import PIL.ImageOps as ImageOps
import PIL.ImageFilter as ImageFilter
from pathlib import Path
import io
import cv2
import numpy as np
import pandas as pd
import datetime
import random
import base64
import time
from io import BytesIO

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Set page configuration
st.set_page_config(
    page_title="FractureVision AI", 
    page_icon="🦴", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Custom Styling: Premium Medical Dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 40%, #0f172a 70%, #0c1220 100%);
        color: #e2e8f0;
    }
    .main .block-container {
        background: linear-gradient(145deg, rgba(15,23,42,0.7), rgba(30,41,59,0.5));
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 28px;
        padding: 2.5rem 3rem !important;
        margin: 1.5rem auto; max-width: 900px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(59,130,246,0.15));
        border: 1px solid rgba(99,102,241,0.25); color: #a5b4fc;
        padding: 6px 18px; border-radius: 50px; font-size: 0.8rem;
        font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.8rem !important; font-weight: 900; text-align: center;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 6px; letter-spacing: -1.5px; line-height: 1.1;
    }
    .hero-subtitle {
        text-align: center; font-size: 1rem; color: #64748b;
        font-weight: 400; margin-top: 0; margin-bottom: 30px;
    }
    .hero-divider {
        height: 2px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
        margin: 0 auto 30px; border: none;
    }
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(145deg, rgba(99,102,241,0.04), rgba(59,130,246,0.04)) !important;
        border: 2px dashed rgba(99,102,241,0.25) !important;
        border-radius: 20px !important; padding: 2.5rem !important;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(99,102,241,0.08) !important; border-color: #6366f1 !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.15) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #3b82f6, #6366f1) !important;
        background-size: 200% 200% !important; animation: gradShift 3s ease infinite !important;
        color: white !important; font-size: 1.05rem !important; font-weight: 700 !important;
        padding: 0.85rem 2rem !important; border: none !important; border-radius: 16px !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.35) !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        width: 100% !important; letter-spacing: 0.5px !important; text-transform: uppercase !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 12px 40px rgba(99,102,241,0.5) !important;
    }
    .stButton > button:active { transform: translateY(1px) !important; }
    @keyframes gradShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    .status-card {
        padding: 28px 24px; border-radius: 20px; text-align: center;
        margin: 25px 0; animation: cardReveal 0.6s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden;
    }
    .status-card::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
    .status-danger {
        background: linear-gradient(145deg, rgba(239,68,68,0.08), rgba(153,27,27,0.18));
        border: 1px solid rgba(239,68,68,0.2); box-shadow: 0 0 40px rgba(239,68,68,0.08);
    }
    .status-danger::before { background: linear-gradient(90deg, #ef4444, #f97316); }
    .status-safe {
        background: linear-gradient(145deg, rgba(34,197,94,0.08), rgba(20,83,45,0.18));
        border: 1px solid rgba(34,197,94,0.2); box-shadow: 0 0 40px rgba(34,197,94,0.08);
    }
    .status-safe::before { background: linear-gradient(90deg, #22c55e, #06b6d4); }
    .status-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 8px; }
    .status-danger .status-title { color: #fca5a5; }
    .status-safe .status-title { color: #86efac; }
    .status-sub { font-size: 0.95rem; color: #94a3b8; line-height: 1.6; }
    .metric-row { display: flex; gap: 12px; margin: 20px 0; }
    .metric-card {
        flex: 1; background: linear-gradient(145deg, rgba(30,41,59,0.6), rgba(15,23,42,0.6));
        border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
        padding: 18px 16px; text-align: center; transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-icon { font-size: 1.6rem; margin-bottom: 4px; }
    .metric-value { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }
    .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 2px; }
    .metric-value.text-indigo { color: #a5b4fc; }
    .metric-value.text-cyan { color: #67e8f9; }
    .metric-value.text-amber { color: #fcd34d; }
    .metric-value.text-rose { color: #fda4af; }
    .image-container {
        border-radius: 16px; overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 8px; background: #0a0e1a;
    }
    .img-label { text-align:center; color:#64748b; font-size:0.78rem; font-weight:500; letter-spacing:0.5px; padding:6px 0 2px; }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f172a, #1e1b4b) !important;
        border-right: 1px solid rgba(99,102,241,0.1);
    }
    .sidebar-panel {
        background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.05);
        padding: 16px; border-radius: 14px; margin-bottom: 14px;
    }
    .sidebar-hdr { font-weight: 700; color: #a5b4fc; font-size: 0.8rem; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; background:rgba(15,23,42,0.4); border-radius:14px; padding:4px; }
    .stTabs [data-baseweb="tab"] { border-radius:10px!important; font-weight:600!important; font-size:0.85rem!important; }
    .stTabs [aria-selected="true"] { background:rgba(99,102,241,0.2)!important; border-color:transparent!important; }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        border-radius: 14px !important; font-weight: 700 !important;
    }
    .stDownloadButton > button:hover { box-shadow: 0 8px 25px rgba(16,185,129,0.3)!important; transform:translateY(-2px)!important; }
    #MainMenu, footer, header { visibility: hidden; }
    @keyframes cardReveal { 0%{opacity:0;transform:translateY(25px) scale(0.97)} 100%{opacity:1;transform:translateY(0) scale(1)} }
</style>
""", unsafe_allow_html=True)

# ----------------- Image Processing Helper Functions -----------------

def apply_clahe(pil_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization to highlight bone fractures"""
    # Convert PIL Image to Grayscale numpy array
    img_gray = np.array(pil_img.convert('L'))
    # Initialize CLAHE filter
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Apply CLAHE enhancement
    img_enhanced = clahe.apply(img_gray)
    # Convert back to RGB format for model compatibility
    return Image.fromarray(img_enhanced).convert('RGB')

def enhance_image(pil_img, brightness=1.0, contrast=1.0, sharpness=1.0, invert=False, clahe=False):
    """Pipeline for multi-stage clinical image adjustments"""
    # 1. Apply CLAHE medical contrast enhancement
    if clahe:
        pil_img = apply_clahe(pil_img)
    
    # Convert to RGB to ensure support for all operations
    pil_img = pil_img.convert('RGB')
    
    # 2. Invert colors (Negative view option, standard for X-ray examination)
    if invert:
        pil_img = ImageOps.invert(pil_img)
        
    # 3. Brightness adjustment
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness)
        
    # 4. Contrast adjustment
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
        
    # 5. Sharpness enhancement
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(sharpness)
        
    return pil_img

def advanced_preprocess(pil_img):
    """Medical-grade preprocessing: bilateral denoise + unsharp mask for bone edge enhancement"""
    img_np = np.array(pil_img.convert('RGB'))
    # Bilateral filter: smooths noise while preserving bone edges
    denoised = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    pil_out = Image.fromarray(denoised)
    # Unsharp mask: sharpens fine fracture lines
    pil_out = pil_out.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return pil_out

def _run_single_inference(model, pil_img, imgsz, conf=0.01, iou=0.45):
    """Run model on a single image at a given resolution, return list of (class_id, conf, xyxy_list)"""
    results = model(pil_img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    dets = []
    w, h = pil_img.size
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        c = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()  # already in pixel coords of input image
        # Normalize to 0-1 for fusion across scales
        dets.append((cls_id, c, [coords[0]/w, coords[1]/h, coords[2]/w, coords[3]/h]))
    return dets

def _flip_detections(dets):
    """Mirror bounding boxes horizontally (for TTA horizontal flip)"""
    flipped = []
    for cls_id, conf, box in dets:
        flipped.append((cls_id, conf, [1.0 - box[2], box[1], 1.0 - box[0], box[3]]))
    return flipped

def _iou(box_a, box_b):
    """Compute IoU between two normalized boxes [x1,y1,x2,y2]"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def weighted_box_fusion(all_dets, iou_thresh=0.5, num_passes=1):
    """Fuse detections from multiple inference passes using Weighted Box Fusion.
    Groups overlapping boxes of the same class across passes and averages their
    confidence scores weighted by how many passes found them."""
    if not all_dets:
        return []
    # Group by class
    from collections import defaultdict
    by_class = defaultdict(list)
    for cls_id, conf, box in all_dets:
        by_class[cls_id].append((conf, box))

    fused = []
    for cls_id, entries in by_class.items():
        # Sort by confidence descending
        entries.sort(key=lambda x: -x[0])
        clusters = []  # each cluster: list of (conf, box)
        for conf, box in entries:
            matched = False
            for cluster in clusters:
                # Compare against cluster representative (weighted average box)
                rep_box = cluster['box']
                if _iou(box, rep_box) >= iou_thresh:
                    cluster['members'].append((conf, box))
                    # Update representative as weighted average
                    total_conf = sum(c for c, _ in cluster['members'])
                    new_box = [0, 0, 0, 0]
                    for c, b in cluster['members']:
                        w = c / total_conf
                        for i in range(4):
                            new_box[i] += b[i] * w
                    cluster['box'] = new_box
                    matched = True
                    break
            if not matched:
                clusters.append({'box': list(box), 'members': [(conf, box)]})

        for cluster in clusters:
            members = cluster['members']
            n = len(members)
            # Boost: average conf * min(n / num_passes, 1.0) amplification
            avg_conf = sum(c for c, _ in members) / n
            max_conf = max(c for c, _ in members)
            # Boost factor: more passes that agree = much higher confidence
            # Use max conf as base, then amplify by agreement ratio
            boost = min(n / max(num_passes * 0.3, 1), 3.0)
            final_conf = min(max_conf * boost, 1.0)
            fused.append((cls_id, final_conf, cluster['box']))

    # Sort by confidence descending
    fused.sort(key=lambda x: -x[1])
    return fused

def ensemble_inference(model, pil_img, conf_thresh=0.25, iou_thresh=0.45, use_tta=True, use_multiscale=True):
    """Run multi-scale + TTA ensemble inference and fuse results with WBF.
    Returns (fused_detections, num_passes, elapsed_seconds)"""
    t0 = time.time()
    w, h = pil_img.size
    all_dets = []
    scales = [640]
    if use_multiscale:
        scales = [416, 640, 832, 1024]

    # Run at each scale on the enhanced image
    for sz in scales:
        dets = _run_single_inference(model, pil_img, imgsz=sz, conf=0.01, iou=iou_thresh)
        # Unify all classes to class 0 ("Fracture") so WBF clusters across body parts
        all_dets.extend([(0, c, b) for _, c, b in dets])

    # TTA: horizontal flip
    if use_tta:
        flipped_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        for sz in scales:
            dets = _run_single_inference(model, flipped_img, imgsz=sz, conf=0.01, iou=iou_thresh)
            flipped = _flip_detections(dets)
            all_dets.extend([(0, c, b) for _, c, b in flipped])

    num_passes = len(scales) * (2 if use_tta else 1)

    # Weighted Box Fusion
    fused = weighted_box_fusion(all_dets, iou_thresh=0.25, num_passes=num_passes)

    # Filter by user confidence threshold and convert back to pixel coords
    final = []
    for cls_id, conf, box in fused:
        if conf >= conf_thresh:
            final.append((cls_id, conf, [box[0]*w, box[1]*h, box[2]*w, box[3]*h]))

    elapsed = time.time() - t0
    return final, num_passes, elapsed

def get_image_base64(pil_img):
    """Encode PIL image to base64 for report inclusion"""
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

def generate_html_report(patient_info, detections, original_img, processed_img, result_img):
    """Generates a professional, printable diagnostic HTML report"""
    original_b64 = get_image_base64(original_img)
    processed_b64 = get_image_base64(processed_img)
    result_b64 = get_image_base64(result_img)
    
    detection_rows = ""
    if len(detections) > 0:
        for idx, det in enumerate(detections):
            detection_rows += f"""
            <tr>
                <td>{idx + 1}</td>
                <td><span class="badge badge-danger">{det['class']}</span></td>
                <td>{det['confidence']:.1f}%</td>
                <td>[{int(det['box'][0])}, {int(det['box'][1])}, {int(det['box'][2])}, {int(det['box'][3])}]</td>
            </tr>
            """
    else:
        detection_rows = """
        <tr>
            <td colspan="4" class="text-center text-muted" style="text-align: center; color: #6b7280; padding: 15px;">No abnormalities detected by the AI.</td>
        </tr>
        """
        
    status_banner = ""
    if len(detections) > 0:
        status_banner = f"""
        <div class="banner banner-danger">
            <h3>⚠️ ABNORMALITY DETECTED</h3>
            <p>Identified {len(detections)} potential fracture zone(s). Requires clinical review and correlation.</p>
        </div>
        """
    else:
        status_banner = """
        <div class="banner banner-safe">
            <h3>✅ NO FRACTURE DETECTED</h3>
            <p>No skeletal abnormalities or fractures were identified in this radiograph scan.</p>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>FractureVision AI Case Report</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #1f2937;
                line-height: 1.5;
                padding: 30px;
                background-color: #ffffff;
            }}
            .header {{
                border-bottom: 2px solid #2563eb;
                padding-bottom: 15px;
                margin-bottom: 25px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: 800;
                color: #1e3a8a;
            }}
            .report-title {{
                text-align: right;
                font-size: 13px;
                color: #4b5563;
            }}
            .section {{
                margin-bottom: 25px;
            }}
            .section-title {{
                font-size: 16px;
                font-weight: 700;
                color: #1e3a8a;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 6px;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            .info-table td {{
                padding: 8px 12px;
                border: 1px solid #e5e7eb;
                font-size: 14px;
            }}
            .info-table td.label {{
                font-weight: 600;
                background-color: #f9fafb;
                width: 20%;
                color: #374151;
            }}
            .banner {{
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 25px;
            }}
            .banner-danger {{
                background-color: #fef2f2;
                border: 1px solid #fca5a5;
                color: #991b1b;
            }}
            .banner-safe {{
                background-color: #f0fdf4;
                border: 1px solid #86efac;
                color: #166534;
            }}
            .banner h3 {{
                margin: 0 0 5px 0;
                font-size: 18px;
            }}
            .banner p {{
                margin: 0;
                font-size: 13px;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 25px;
            }}
            .image-box {{
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 8px;
                text-align: center;
                background-color: #f9fafb;
            }}
            .image-box img {{
                max-width: 100%;
                max-height: 220px;
                object-fit: contain;
                border-radius: 4px;
            }}
            .image-title {{
                font-size: 11px;
                font-weight: 600;
                color: #4b5563;
                margin-top: 6px;
            }}
            .results-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .results-table th, .results-table td {{
                padding: 8px 12px;
                border: 1px solid #e5e7eb;
                font-size: 13px;
                text-align: left;
            }}
            .results-table th {{
                background-color: #f3f4f6;
                font-weight: 600;
                color: #374151;
            }}
            .badge {{
                display: inline-block;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
                border-radius: 9999px;
            }}
            .badge-danger {{
                background-color: #fee2e2;
                color: #991b1b;
                border: 1px solid #fca5a5;
            }}
            .disclaimer {{
                font-size: 11px;
                color: #9ca3af;
                margin-top: 35px;
                border-top: 1px solid #e5e7eb;
                padding-top: 12px;
                text-align: center;
            }}
            @media print {{
                body {{ padding: 0; }}
                .no-print {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">🦴 FractureVision AI</div>
            <div class="report-title">
                <strong>CLINICAL SCAN REPORT</strong><br>
                Generated: {patient_info['date']}<br>
                Scan ID: {patient_info['scan_id']}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Patient Case Information</div>
            <table class="info-table">
                <tr>
                    <td class="label">Patient Name</td>
                    <td>{patient_info['name']}</td>
                    <td class="label">Gender</td>
                    <td>{patient_info['gender']}</td>
                </tr>
                <tr>
                    <td class="label">Age</td>
                    <td>{patient_info['age']}</td>
                    <td class="label">Radiology Tech</td>
                    <td>{patient_info['tech']}</td>
                </tr>
            </table>
        </div>

        {status_banner}

        <div class="section">
            <div class="section-title">Radiograph Scans</div>
            <div class="image-grid">
                <div class="image-box">
                    <img src="data:image/jpeg;base64,{original_b64}" />
                    <div class="image-title">1. Original Radiograph</div>
                </div>
                <div class="image-box">
                    <img src="data:image/jpeg;base64,{processed_b64}" />
                    <div class="image-title">2. Processed Image (Filters)</div>
                </div>
                <div class="image-box">
                    <img src="data:image/jpeg;base64,{result_b64}" />
                    <div class="image-title">3. AI Diagnostic Overlay</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">AI Detection Metrics</div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Classification</th>
                        <th>Confidence Score</th>
                        <th>Bounding Box Coordinates [x_min, y_min, x_max, y_max]</th>
                    </tr>
                </thead>
                <tbody>
                    {detection_rows}
                </tbody>
            </table>
        </div>

        <div class="disclaimer">
            <strong>Diagnostic Disclaimer:</strong> This report is generated by an artificial intelligence model as a clinical support tool. It is not a replacement for professional clinical evaluation. Bounding boxes represent statistical likelihood regions. Final diagnostic findings must be signed off by a certified medical professional.
        </div>
    </body>
    </html>
    """
    return html

# ----------------- UI / Sidebar & Configurations -----------------

# Clinical label dictionary — unified fracture detection mode
# The model is strong at LOCATING fractures but weak at classifying body part,
# so we group all positive classes under a single clinical label.
CLINICAL_CLASSES = {
    'elbow positive': 'Fracture Detected',
    'fingers positive': 'Fracture Detected',
    'forearm fracture': 'Fracture Detected',
    'humerus fracture': 'Fracture Detected',
    'humerus': 'Fracture Detected',
    'shoulder fracture': 'Fracture Detected',
    'wrist positive': 'Fracture Detected'
}

# Sidebar Header
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 25px; padding: 20px 0;'>
        <div style='font-size: 2.5rem; margin-bottom: 5px;'>🦴</div>
        <h2 style='color: #e2e8f0; margin: 0; font-weight: 800; font-size: 1.3rem; letter-spacing: -0.5px;'>Clinical Dashboard</h2>
        <p style='color: #64748b; font-size: 0.75rem; margin: 5px 0 0; letter-spacing: 1px; text-transform: uppercase;'>FractureVision AI</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Panel 1: Patient Data
st.sidebar.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-hdr'>📋 Patient Info</div>", unsafe_allow_html=True)
p_name = st.sidebar.text_input("Patient Name", value="John Doe")
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    p_age = st.sidebar.text_input("Age", value="45")
with col_p2:
    p_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
p_tech = st.sidebar.text_input("Radiology Tech Initials", value="RD-A")

# Auto-generate a Scan ID
if 'scan_id' not in st.session_state:
    st.session_state['scan_id'] = f"FX-{random.randint(100000, 999999)}"
scan_id = st.sidebar.text_input("Scan ID", value=st.session_state['scan_id'])
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Panel 2: Model Configuration (Speed vs Sensitivity tuning)
st.sidebar.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-hdr'>⚙️ Model Parameters</div>", unsafe_allow_html=True)
conf_val = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.00, value=0.25, step=0.01,
                             help="Minimum confidence percentage required to display a detection box.")
iou_val = st.sidebar.slider("IoU Threshold (NMS)", min_value=0.10, max_value=0.95, value=0.45, step=0.05,
                            help="Intersection over Union threshold for Non-Maximum Suppression (lower values reduce overlapping boxes).")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Panel 3: Clinical Image Enhancers
st.sidebar.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-hdr'>🩺 X-Ray Image Enhancers</div>", unsafe_allow_html=True)
clahe_active = st.sidebar.checkbox("CLAHE Contrast Enhancer", value=True,
                                   help="Contrast Limited Adaptive Histogram Equalization. Enhances local bone structures and hairline fractures.")
invert_active = st.sidebar.checkbox("Invert Colors (Negative)", value=False,
                                    help="Invert values to display standard radiograph negative film.")
brightness_val = st.sidebar.slider("Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
contrast_val = st.sidebar.slider("Global Contrast", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
sharpness_val = st.sidebar.slider("Sharpness", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Panel 4: Advanced AI Boosters
st.sidebar.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-hdr'>🚀 AI Detection Boosters</div>", unsafe_allow_html=True)
adv_preprocess = st.sidebar.checkbox("Bilateral Denoise + Unsharp Mask", value=True,
                                     help="Medical-grade noise reduction that preserves bone edges, plus edge sharpening for hairline fractures.")
use_multiscale = st.sidebar.checkbox("Multi-Scale Ensemble (416+640+1024)", value=True,
                                     help="Runs inference at 3 resolutions and fuses results. Dramatically improves detection recall and confidence.")
use_tta = st.sidebar.checkbox("Test-Time Augmentation (Flip)", value=True,
                              help="Also runs on a horizontally-flipped copy and merges detections. Catches fractures the model misses due to orientation bias.")
st.sidebar.markdown("</div>", unsafe_allow_html=True)


# ----------------- Model Loading Pipeline (Optimized) -----------------

# Auto-locate model weights (favoring ONNX for efficiency, falling back to PyTorch)
onnx_files = list(Path(".").glob("best*.onnx"))
pt_files = list(Path(".").glob("best*.pt"))

if len(onnx_files) > 0:
    model_path = str(max(onnx_files, key=lambda p: p.stat().st_mtime))
    model_format = "ONNX Runtime (Optimized)"
elif len(pt_files) > 0:
    model_path = str(max(pt_files, key=lambda p: p.stat().st_mtime))
    model_format = "PyTorch (Standard)"
else:
    model_path = "best (1).pt"
    model_format = "Fallback PyTorch"

@st.cache_resource(show_spinner="Initializing diagnostic model library...")
def load_model(path):
    if not Path(path).exists():
        return None
    try:
        # Loaded model using Ultralytics interface. For ONNX it will invoke ONNX runtime automatically.
        model = YOLO(path)
        return model
    except Exception:
        return None

model = None
if HAS_YOLO:
    model = load_model(model_path)

# Display model metadata in sidebar footer
st.sidebar.info(f"🧬 **Model:** `{Path(model_path).name}`\n🚀 **Engine:** `{model_format}`")


# ----------------- Main Interface -----------------

# Hero Header
st.markdown("<div style='text-align:center'><span class='hero-badge'>AI-Powered Diagnostics</span></div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>FractureVision AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Multi-scale ensemble detection with test-time augmentation</p>", unsafe_allow_html=True)
st.markdown("<hr class='hero-divider'>", unsafe_allow_html=True)

if not HAS_YOLO:
    st.error("Missing dependency: `ultralytics`. Please run `pip install ultralytics` in your environment.")
elif model is None:
    st.error(f"Could not load the model weights. Please make sure '{model_path}' exists in the application root.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Upload Radiograph Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load original image
        original_image = Image.open(uploaded_file)
        
        # Apply pre-processing enhancement pipeline
        enhanced_image = enhance_image(
            original_image, 
            brightness=brightness_val, 
            contrast=contrast_val, 
            sharpness=sharpness_val, 
            invert=invert_active, 
            clahe=clahe_active
        )
        
        # Apply advanced preprocessing if enabled
        if adv_preprocess:
            enhanced_image = advanced_preprocess(enhanced_image)
        
        # Display side-by-side preview comparisons
        st.markdown("### Pre-Inference Inspection")
        col_view1, col_view2 = st.columns(2)
        with col_view1:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<p class='img-label'>Original Radiograph</p>", unsafe_allow_html=True)
            
        with col_view2:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(enhanced_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<p class='img-label'>Enhanced View (Filters Active)</p>", unsafe_allow_html=True)
        
        # Trigger Scan Button
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 INITIATE SCAN")
        
        # Run Detection
        if analyze_btn or st.session_state.get('last_uploaded') == uploaded_file.name:
            st.session_state['last_uploaded'] = uploaded_file.name
            
            st.markdown("---")
            
            with st.spinner("Executing ensemble diagnostic scan (multi-scale + TTA)..."):
                try:
                    # === ENSEMBLE INFERENCE ===
                    fused_dets, num_passes, elapsed = ensemble_inference(
                        model, enhanced_image,
                        conf_thresh=conf_val, iou_thresh=iou_val,
                        use_tta=use_tta, use_multiscale=use_multiscale
                    )
                    
                    # Draw custom bounding boxes on the enhanced image for the fused detections
                    result_np = np.array(enhanced_image.convert('RGB'))
                    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                    
                    for cls_id, conf, box in fused_dets:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        color = (0, 0, 255)  # Red in BGR
                        cv2.rectangle(result_bgr, (x1, y1), (x2, y2), color, 3)
                        label = f"Fracture {conf:.0%}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(result_bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                        cv2.putText(result_bgr, label, (x1 + 3, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    res_plotted = result_bgr
                    
                except Exception as e:
                    st.error(f"⚠️ Diagnostic failure: {e}")
                    st.stop()
                
                # Show status summary with timing info
                passes_label = f"{num_passes} inference passes in {elapsed:.1f}s"
                if len(fused_dets) > 0:
                    max_confidence = max(c for _, c, _ in fused_dets)
                    st.markdown(f"""
                    <div class='status-card status-danger'>
                        <div class='status-title'>⚠️ ABNORMALITY DETECTED</div>
                        <div class='status-sub'>Identified <b>{len(fused_dets)}</b> potential fracture zone(s). Peak confidence: <b>{max_confidence:.1%}</b>.<br>
                        <small style="opacity:0.7">Ensemble: {passes_label}</small></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='status-card status-safe'>
                        <div class='status-title'>✅ NO FRACTURE DETECTED</div>
                        <div class='status-sub'>No structural anomalies identified above threshold.<br>
                        <small style="opacity:0.7">Ensemble: {passes_label}</small></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metric stat cards
                peak_conf = max((c for _, c, _ in fused_dets), default=0)
                st.markdown(f"""
                <div class='metric-row'>
                    <div class='metric-card'>
                        <div class='metric-icon'>🎯</div>
                        <div class='metric-value text-rose'>{len(fused_dets)}</div>
                        <div class='metric-label'>Zones Found</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-icon'>📊</div>
                        <div class='metric-value text-amber'>{peak_conf:.0%}</div>
                        <div class='metric-label'>Peak Confidence</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-icon'>🔬</div>
                        <div class='metric-value text-indigo'>{num_passes}</div>
                        <div class='metric-label'>Passes Run</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-icon'>⚡</div>
                        <div class='metric-value text-cyan'>{elapsed:.1f}s</div>
                        <div class='metric-label'>Scan Time</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Render results in Tabs
                tab1, tab2 = st.tabs(["🩺 Diagnostic Overlay", "📊 Clinical Metrics & Export"])
                
                with tab1:
                    st.markdown("<h4 style='color: #e2e8f0; margin-bottom: 15px;'>Diagnostic View</h4>", unsafe_allow_html=True)
                    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                    # Convert to RGB since matplotlib outputs standard channel formats
                    st.image(res_plotted, channels="BGR", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.caption("AI diagnostic boxes overlaid. Confidences represent structural match rates.")
                
                with tab2:
                    st.markdown("<h4 style='color: #e2e8f0; margin-bottom: 10px;'>Detection Breakdown</h4>", unsafe_allow_html=True)
                    
                    # Extract bounding box parameters and format table
                    detections = []
                    for cls_id, conf, box in fused_dets:
                        raw_name = model.names[cls_id]
                        clinical_name = CLINICAL_CLASSES.get(raw_name, raw_name.capitalize())
                        detections.append({
                            "class": clinical_name,
                            "confidence": conf * 100,
                            "box": box
                        })
                        
                    if len(detections) > 0:
                        # Render DataFrame table
                        df_det = pd.DataFrame(detections)
                        df_det['confidence'] = df_det['confidence'].map('{:.2f}%'.format)
                        df_det['box'] = df_det['box'].apply(lambda b: f"[{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}]")
                        df_det.columns = ["Abnormality Class", "AI Confidence Score", "Bounding Box [x_min, y_min, x_max, y_max]"]
                        st.table(df_det)
                        
                        # Plot Altair Confidence Chart
                        st.markdown("<h5 style='color: #94a3b8; margin-top: 20px; margin-bottom: 10px;'>Match Confidence Distribution</h5>", unsafe_allow_html=True)
                        chart_data = pd.DataFrame({
                            'Abnormality': [d['class'] for d in detections],
                            'Confidence (%)': [d['confidence'] for d in detections]
                        })
                        st.bar_chart(chart_data.set_index('Abnormality'))
                    else:
                        st.write("No detection points were registered above the current confidence threshold.")
                        st.info("💡 **Tip:** Try lowering the **Confidence Threshold** in the sidebar settings if you suspect a hairline fracture that is not showing up.")
                    
                    # Report Generation Form
                    st.markdown("---")
                    st.markdown("<h4 style='color: #e2e8f0;'>Generate Diagnostic Report</h4>", unsafe_allow_html=True)
                    st.write("Generate a standardized clinical summary card ready for patient file records.")
                    
                    # Store variables for report
                    patient_info = {
                        "name": p_name,
                        "age": p_age,
                        "gender": p_gender,
                        "tech": p_tech,
                        "scan_id": scan_id,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # PIL Plotted Image (Convert BGR result back to PIL image for base64 encoding)
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    pil_result_img = Image.fromarray(res_plotted_rgb)
                    
                    # Generate report HTML content
                    report_html = generate_html_report(
                        patient_info, 
                        detections, 
                        original_image, 
                        enhanced_image, 
                        pil_result_img
                    )
                    
                    # Download Report button
                    report_filename = f"report_{patient_info['scan_id']}.html"
                    st.download_button(
                        label="📥 DOWNLOAD CASE REPORT (HTML)",
                        data=report_html,
                        file_name=report_filename,
                        mime="text/html"
                    )
                    st.caption("Open the HTML report and print (Ctrl+P / Cmd+P) to save as a clean PDF document.")
