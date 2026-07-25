"""
FractureVision AI — FastAPI Backend
Serves YOLO fracture detection with ensemble inference.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import base64
import time
import datetime
import json
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage
import PIL.ImageEnhance as ImageEnhance
import PIL.ImageOps as ImageOps
import PIL.ImageFilter as ImageFilter
from ultralytics import YOLO

# ── App ──────────────────────────────────────────────
app = FastAPI(title="FractureVision AI API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Model loading ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
onnx_files = list(ROOT.glob("best*.onnx"))
pt_files = list(ROOT.glob("best*.pt"))
if onnx_files:
    MODEL_PATH = str(max(onnx_files, key=lambda p: p.stat().st_mtime))
    MODEL_FMT = "ONNX"
elif pt_files:
    MODEL_PATH = str(max(pt_files, key=lambda p: p.stat().st_mtime))
    MODEL_FMT = "PyTorch"
else:
    MODEL_PATH = str(ROOT / "best (1).pt")
    MODEL_FMT = "Fallback"

model = YOLO(MODEL_PATH)

# ── Image processing helpers ─────────────────────────
def apply_clahe(pil_img, clip=2.0):
    gray = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return PILImage.fromarray(clahe.apply(gray)).convert("RGB")

def enhance_image(img, brightness=1.0, contrast=1.0, sharpness=1.0, invert=False, clahe=False):
    if clahe:
        img = apply_clahe(img)
    img = img.convert("RGB")
    if invert:
        img = ImageOps.invert(img)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def advanced_preprocess(img):
    arr = np.array(img.convert("RGB"))
    arr = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    img = PILImage.fromarray(arr)
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# ── Ensemble inference ───────────────────────────────
def _run_single(img, imgsz, iou):
    res = model(img, imgsz=imgsz, conf=0.01, iou=iou, verbose=False)
    w, h = img.size
    dets = []
    for box in res[0].boxes:
        c = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()
        dets.append((0, c, [coords[0]/w, coords[1]/h, coords[2]/w, coords[3]/h]))
    return dets

def _flip_dets(dets):
    return [(cls, c, [1-b[2], b[1], 1-b[0], b[3]]) for cls, c, b in dets]

def _iou(a, b):
    x1, y1 = max(a[0],b[0]), max(a[1],b[1])
    x2, y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter) if (ua+ub-inter) > 0 else 0

def wbf(all_dets, iou_thresh=0.25, num_passes=1):
    if not all_dets:
        return []
    from collections import defaultdict
    by_cls = defaultdict(list)
    for cls, c, b in all_dets:
        by_cls[cls].append((c, b))
    fused = []
    for cls, entries in by_cls.items():
        entries.sort(key=lambda x: -x[0])
        clusters = []
        for conf, box in entries:
            matched = False
            for cl in clusters:
                if _iou(box, cl["box"]) >= iou_thresh:
                    cl["members"].append((conf, box))
                    tc = sum(c for c,_ in cl["members"])
                    nb = [0,0,0,0]
                    for c2, b2 in cl["members"]:
                        w2 = c2/tc
                        for i in range(4):
                            nb[i] += b2[i]*w2
                    cl["box"] = nb
                    matched = True
                    break
            if not matched:
                clusters.append({"box": list(box), "members": [(conf, box)]})
        for cl in clusters:
            n = len(cl["members"])
            mx = max(c for c,_ in cl["members"])
            boost = min(n/max(num_passes*0.3, 1), 3.0)
            fused.append((cls, min(mx*boost, 1.0), cl["box"]))
    fused.sort(key=lambda x: -x[1])
    return fused

def ensemble_inference(img, conf_thresh=0.25, iou_thresh=0.45, use_tta=True, use_multiscale=True):
    t0 = time.time()
    w, h = img.size
    all_dets = []
    scales = [416, 640, 832, 1024] if use_multiscale else [640]
    for sz in scales:
        all_dets.extend(_run_single(img, sz, iou_thresh))
    if use_tta:
        flipped = img.transpose(PILImage.FLIP_LEFT_RIGHT)
        for sz in scales:
            all_dets.extend(_flip_dets(_run_single(flipped, sz, iou_thresh)))
    num_passes = len(scales) * (2 if use_tta else 1)
    fused = wbf(all_dets, 0.25, num_passes)
    final = [(cls, c, [b[0]*w, b[1]*h, b[2]*w, b[3]*h]) for cls, c, b in fused if c >= conf_thresh]
    return final, num_passes, time.time()-t0

def pil_to_base64(img, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()

# ── API routes ───────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model": Path(MODEL_PATH).name, "engine": MODEL_FMT}

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    clahe: bool = Form(True),
    invert: bool = Form(False),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    sharpness: float = Form(1.0),
    denoise: bool = Form(True),
    multiscale: bool = Form(True),
    tta: bool = Form(True),
):
    raw = await image.read()
    pil_img = PILImage.open(BytesIO(raw))
    enhanced = enhance_image(pil_img, brightness, contrast, sharpness, invert, clahe)
    if denoise:
        enhanced = advanced_preprocess(enhanced)

    fused, passes, elapsed = ensemble_inference(enhanced, confidence, iou, tta, multiscale)

    # Draw boxes
    result_np = np.array(enhanced.convert("RGB"))
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    for _, conf, box in fused:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(result_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"Fracture {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result_bgr, (x1, y1-th-10), (x1+tw+6, y1), (0,0,255), -1)
        cv2.putText(result_bgr, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_pil = PILImage.fromarray(result_rgb)

    peak = max((c for _, c, _ in fused), default=0)
    detections = [{"confidence": round(c*100, 1), "box": [round(v) for v in b]} for _, c, b in fused]

    return {
        "detections": detections,
        "stats": {
            "zones": len(fused),
            "peakConfidence": round(peak*100, 1),
            "passes": passes,
            "scanTime": round(elapsed, 2),
        },
        "originalImage": pil_to_base64(pil_img),
        "enhancedImage": pil_to_base64(enhanced),
        "annotatedImage": pil_to_base64(result_pil),
    }

@app.post("/api/report")
async def generate_report(
    image: UploadFile = File(...),
    patient_name: str = Form("Patient"),
    patient_age: str = Form("N/A"),
    patient_gender: str = Form("N/A"),
    tech: str = Form("N/A"),
    scan_id: str = Form("FX-000000"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    clahe: bool = Form(True),
    denoise: bool = Form(True),
    multiscale: bool = Form(True),
    tta: bool = Form(True),
):
    raw = await image.read()
    pil_img = PILImage.open(BytesIO(raw))
    enhanced = enhance_image(pil_img, clahe=clahe)
    if denoise:
        enhanced = advanced_preprocess(enhanced)
    fused, _, _ = ensemble_inference(enhanced, confidence, iou, tta, multiscale)

    result_np = np.array(enhanced.convert("RGB"))
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    for _, conf, box in fused:
        x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        cv2.rectangle(result_bgr,(x1,y1),(x2,y2),(0,0,255),3)
        lbl = f"Fracture {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
        cv2.rectangle(result_bgr,(x1,y1-th-10),(x1+tw+6,y1),(0,0,255),-1)
        cv2.putText(result_bgr,lbl,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    result_pil = PILImage.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

    orig_b64 = pil_to_base64(pil_img)
    enh_b64 = pil_to_base64(enhanced)
    res_b64 = pil_to_base64(result_pil)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    det_rows = ""
    for i, (_, c, b) in enumerate(fused):
        det_rows += f"<tr><td>{i+1}</td><td>Fracture Detected</td><td>{c:.1%}</td><td>[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]</td></tr>"
    if not fused:
        det_rows = "<tr><td colspan='4' style='text-align:center;color:#6b7280;padding:15px'>No abnormalities detected.</td></tr>"
    banner = f"<div style='background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;padding:15px;border-radius:8px;text-align:center;margin-bottom:25px'><h3 style='margin:0'>⚠️ {len(fused)} FRACTURE ZONE(S) DETECTED</h3></div>" if fused else "<div style='background:#f0fdf4;border:1px solid #86efac;color:#166534;padding:15px;border-radius:8px;text-align:center;margin-bottom:25px'><h3 style='margin:0'>✅ NO FRACTURE DETECTED</h3></div>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>FractureVision AI Report</title>
    <style>body{{font-family:Helvetica,Arial,sans-serif;color:#1f2937;padding:30px;line-height:1.5}}
    .hdr{{border-bottom:2px solid #2563eb;padding-bottom:15px;margin-bottom:25px;display:flex;justify-content:space-between;align-items:center}}
    .logo{{font-size:24px;font-weight:800;color:#1e3a8a}} .rt{{text-align:right;font-size:13px;color:#4b5563}}
    .st{{font-size:16px;font-weight:700;color:#1e3a8a;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}}
    table{{width:100%;border-collapse:collapse;margin-bottom:20px}} td,th{{padding:8px 12px;border:1px solid #e5e7eb;font-size:13px;text-align:left}}
    th{{background:#f3f4f6;font-weight:600;color:#374151}} .lb{{font-weight:600;background:#f9fafb;width:20%;color:#374151}}
    .ig{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:25px}}
    .ib{{border:1px solid #e5e7eb;border-radius:8px;padding:8px;text-align:center;background:#f9fafb}}
    .ib img{{max-width:100%;max-height:220px;object-fit:contain;border-radius:4px}}
    .it{{font-size:11px;font-weight:600;color:#4b5563;margin-top:6px}}
    .disc{{font-size:11px;color:#9ca3af;margin-top:35px;border-top:1px solid #e5e7eb;padding-top:12px;text-align:center}}</style></head>
    <body><div class="hdr"><div class="logo">🦴 FractureVision AI</div><div class="rt"><strong>CLINICAL SCAN REPORT</strong><br>Generated: {now}<br>Scan ID: {scan_id}</div></div>
    <div class="st">Patient Information</div>
    <table><tr><td class="lb">Name</td><td>{patient_name}</td><td class="lb">Gender</td><td>{patient_gender}</td></tr>
    <tr><td class="lb">Age</td><td>{patient_age}</td><td class="lb">Tech</td><td>{tech}</td></tr></table>
    {banner}
    <div class="st">Radiograph Scans</div>
    <div class="ig"><div class="ib"><img src="data:image/jpeg;base64,{orig_b64}"/><div class="it">Original</div></div>
    <div class="ib"><img src="data:image/jpeg;base64,{enh_b64}"/><div class="it">Enhanced</div></div>
    <div class="ib"><img src="data:image/jpeg;base64,{res_b64}"/><div class="it">AI Overlay</div></div></div>
    <div class="st">Detection Metrics</div>
    <table><thead><tr><th>#</th><th>Classification</th><th>Confidence</th><th>Bounding Box</th></tr></thead><tbody>{det_rows}</tbody></table>
    <div class="disc"><strong>Disclaimer:</strong> AI-generated clinical support tool. Not a replacement for professional evaluation.</div></body></html>"""
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
