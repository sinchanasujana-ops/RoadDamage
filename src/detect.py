"""
Road-damage detection used by the web app and CLI.

Uses a YOLOv8 model trained on RDD2022 when available (downloaded on first
run). Filters unusable boxes (near-full-image) and falls back to a classical
OpenCV detector so new real photos still get results if YOLO finds nothing.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image
import io

from severity import severity_label

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
RDD_WEIGHTS = os.path.join(MODELS_DIR, "rdd2022_yolov8n.pt")
LOCAL_WEIGHTS = os.path.join(MODELS_DIR, "best.pt")

CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
CLASS_ALIASES = {
    "d00": "longitudinal_crack",
    "d10": "transverse_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
    "longitudinal crack": "longitudinal_crack",
    "transverse crack": "transverse_crack",
    "alligator crack": "alligator_crack",
    "alligator cracking": "alligator_crack",
}

HF_REPO = "dronefreak/rdd2022-yolov8n"
HF_FILE = "best.pt"

_model = None
_model_path = None


def decode_image(file_bytes: bytes):
    """Decode JPEG/PNG/WebP/BMP via OpenCV, then Pillow as backup."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def normalize_class(raw) -> str:
    name = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if name in CLASS_ALIASES:
        return CLASS_ALIASES[name]
    compact = name.replace("_", "")
    for key, value in CLASS_ALIASES.items():
        if key.replace("_", "").replace(" ", "") == compact:
            return value
    if name in CLASS_NAMES:
        return name
    return name or "unknown"


def _download_rdd_weights():
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        import shutil

        shutil.copyfile(path, RDD_WEIGHTS)
        return RDD_WEIGHTS
    except Exception:
        # Direct URL as a second attempt
        import urllib.request

        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"
        urllib.request.urlretrieve(url, RDD_WEIGHTS)
        return RDD_WEIGHTS


def find_weights(prefer_rdd=True) -> str:
    if prefer_rdd:
        if os.path.exists(RDD_WEIGHTS) and os.path.getsize(RDD_WEIGHTS) > 1_000_000:
            return RDD_WEIGHTS
        try:
            return _download_rdd_weights()
        except Exception as exc:
            print(f"Could not download RDD2022 weights ({exc}); using local weights if present.")
    if os.path.exists(LOCAL_WEIGHTS):
        return LOCAL_WEIGHTS
    raise FileNotFoundError(
        "No YOLO weights found. Place a trained file at models/best.pt "
        "or allow the app to download the RDD2022 model."
    )


def get_model(weights=None):
    global _model, _model_path
    from ultralytics import YOLO

    path = weights if weights and os.path.exists(weights) else find_weights()
    if _model is None or _model_path != path:
        _model = YOLO(path)
        _model_path = path
    return _model


def _area_ratio(x1, y1, x2, y2, w, h):
    return max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)


def _nms(dets, iou_thresh=0.45):
    if len(dets) <= 1:
        return dets
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets], dtype=np.float32)
    scores = np.array([d["confidence"] for d in dets], dtype=np.float32)
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thresh)
    if len(idxs) == 0:
        return dets
    keep = set(int(i) for i in np.array(idxs).reshape(-1))
    return [d for i, d in enumerate(dets) if i in keep]


def _pack(damage_type, conf, x1, y1, x2, y2, w, h):
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    sev_label, sev_score = severity_label(damage_type, x1, y1, x2, y2, w, h, conf)
    return {
        "damage_type": damage_type,
        "confidence": round(float(conf), 3),
        "severity": sev_label,
        "severity_score": sev_score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def yolo_detect(img, conf=0.18, imgsz=960, weights=None):
    model = get_model(weights)
    h, w = img.shape[:2]
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False, augment=False)[0]
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        ratio = _area_ratio(x1, y1, x2, y2, w, h)
        raw = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))
        kind = normalize_class(raw)
        # Close-up potholes can fill most of the frame; only drop near-full-image boxes.
        if ratio < 0.0004 or ratio > 0.92:
            continue
        if ratio > 0.72 and kind != "pothole":
            continue
        packed = _pack(kind, score, x1, y1, x2, y2, w, h)
        if packed:
            dets.append(packed)
    return _nms(dets)


def _classify_contour(cnt, gray, w, h):
    x, y, bw, bh = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = max(bw * bh, 1)
    fill = area / rect_area
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    img_area = w * h
    ratio = (bw * bh) / img_area
    if ratio > 0.28 or ratio < 0.002:
        return None
    patch = gray[y:y + bh, x:x + bw]
    if patch.size == 0:
        return None
    # Must be darker than the rest of the pavement
    if float(patch.mean()) > float(gray.mean()) - 12:
        return None
    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    if circularity > 0.42 and fill > 0.5 and 0.004 < ratio < 0.12:
        kind = "pothole"
    elif aspect >= 3.2 and fill < 0.4:
        kind = "longitudinal_crack" if bh >= bw else "transverse_crack"
    elif aspect < 1.6 and fill < 0.38 and ratio > 0.01:
        kind = "alligator_crack"
    else:
        return None
    conf = float(np.clip(0.32 + 0.35 * min(ratio * 10, 1.0) + 0.12 * fill, 0.32, 0.7))
    return _pack(kind, conf, x, y, x + bw, y + bh, w, h)


def cv_detect(img, max_dets=3):
    """Conservative classical detector used only when YOLO returns nothing."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    dets = []

    thr = max(35, int(np.percentile(blur, 12)))
    _, dark = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        packed = _classify_contour(cnt, gray, w, h)
        if packed:
            dets.append(packed)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(21, w // 30), 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(21, h // 30)))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel_h) + cv2.morphologyEx(
        blur, cv2.MORPH_BLACKHAT, kernel_v
    )
    _, crack = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    crack = cv2.morphologyEx(crack, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(crack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        packed = _classify_contour(cnt, gray, w, h)
        if packed:
            dets.append(packed)

    dets = _nms(dets, iou_thresh=0.35)
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets[:max_dets]


def cv_potholes(img, max_dets=3):
    """Find large bowl-shaped holes, including water-filled close-ups YOLO often misses."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 40, 120)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.dilate(edges, k, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y + bh < h * 0.22:
            continue
        ratio = (bw * bh) / float(w * h)
        if ratio < 0.02 or ratio > 0.78:
            continue
        area = cv2.contourArea(cnt)
        fill = area / max(bw * bh, 1)
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
        if fill < 0.35 or circularity < 0.18:
            continue
        packed = _pack("pothole", float(np.clip(0.4 + 0.3 * circularity, 0.4, 0.72)), x, y, x + bw, y + bh, w, h)
        if packed:
            dets.append(packed)
    return _nms(dets)[:max_dets]


def annotate(img, dets):
    out = img.copy()
    colors = {"Low": (70, 170, 70), "Medium": (20, 165, 255), "High": (50, 50, 230)}
    for d in dets:
        color = colors.get(d["severity"], (0, 200, 255))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['damage_type']} {d['confidence']:.2f} | {d['severity']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(th + 8, y1)
        cv2.rectangle(out, (x1, y_text - th - 8), (x1 + tw + 8, y_text + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_detection(img, conf=0.18, weights=None):
    h, w = img.shape[:2]
    # Larger input helps thin cracks on phone photos
    imgsz = 1280 if max(h, w) >= 800 else 960
    engine = "yolo"
    try:
        dets = yolo_detect(img, conf=conf, imgsz=imgsz, weights=weights)
        if not dets:
            dets = yolo_detect(img, conf=max(0.08, conf * 0.55), imgsz=1280, weights=weights)
    except Exception as exc:
        print(f"YOLO failed ({exc}); using OpenCV fallback.")
        dets = []
        engine = "opencv"

    extra = cv_potholes(img)
    has_pothole = any(d["damage_type"] == "pothole" for d in dets)
    if extra and not has_pothole:
        dets = _nms(dets + extra, iou_thresh=0.4)
        if engine == "none":
            engine = "opencv"
        elif extra:
            engine = engine if engine != "none" else "opencv"

    if not dets:
        dets = cv_detect(img)
        engine = "opencv" if dets else "none"

    annotated = annotate(img, dets)
    return dets, annotated, engine, (w, h)
