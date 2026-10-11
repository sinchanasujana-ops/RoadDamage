"""
Web frontend for Road Damage Detection.

Upload a photo -> YOLOv8 (RDD2022) + OpenCV fallback -> boxes + severity
-> annotated image in the browser -> append rows to outputs/csv/detections.csv
(the same file Power BI reads).

Run from the project's `src` folder:

    python app.py

Then open http://127.0.0.1:5000
"""

import base64
import csv
import os
from collections import Counter
from datetime import datetime

import cv2
from flask import Flask, Response, render_template, request, send_file

from detect import ROOT, decode_image, run_detection

CSV_PATH = os.path.join(ROOT, "outputs", "csv", "detections.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "detections")
CSV_HEADER = [
    "source_file", "damage_type", "confidence", "severity", "severity_score",
    "x1", "y1", "x2", "y2", "img_width", "img_height", "country_or_location",
    "processed_at",
]

app = Flask(__name__, template_folder="templates", static_folder="static")


def ensure_csv():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_csv(rows):
    ensure_csv()
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def load_stats():
    empty = {
        "total": 0, "images": 0, "high": 0, "avg_conf": "—",
        "by_type": [], "by_sev": [], "recent": [],
    }
    if not os.path.exists(CSV_PATH):
        return empty
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return empty
    types = Counter(r.get("damage_type", "") for r in rows)
    sevs = Counter(r.get("severity", "") for r in rows)
    files = {r.get("source_file") for r in rows}
    confs = []
    for r in rows:
        try:
            confs.append(float(r.get("confidence") or 0))
        except ValueError:
            pass
    def bars(counter, order=None):
        keys = order or [k for k, _ in counter.most_common()]
        mx = max(counter.values()) if counter else 1
        return [(k, counter.get(k, 0), round(100 * counter.get(k, 0) / mx, 1) if mx else 0) for k in keys if counter.get(k, 0)]

    return {
        "total": len(rows),
        "images": len(files),
        "high": sevs.get("High", 0),
        "avg_conf": f"{sum(confs) / len(confs):.0%}" if confs else "—",
        "by_type": bars(types),
        "by_sev": bars(sevs, ["Low", "Medium", "High"]),
        "recent": list(reversed(rows[-8:])),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    img_data = None
    rows = []
    error = None
    saved = False
    filename = ""
    engine_note = ""
    location = request.form.get("location", "My City") if request.method == "POST" else "My City"
    try:
        conf = float(request.form.get("conf", 0.18))
    except (TypeError, ValueError):
        conf = 0.18
    conf = min(0.5, max(0.1, conf))

    if request.method == "POST":
        try:
            file = request.files.get("photo")
            if not file or not file.filename:
                raise ValueError("Please choose a photo first.")
            filename = file.filename
            location = request.form.get("location", "unknown") or "unknown"
            save_csv = "save_csv" in request.form

            img = decode_image(file.read())
            if img is None:
                raise ValueError(
                    "Could not read that image. Re-save it as JPEG or PNG "
                    "(Paint → Save As JPEG) and upload again."
                )

            dets, annotated, engine, (w, h) = run_detection(img, conf=conf)
            rows = dets
            if engine == "yolo":
                engine_note = "YOLOv8 RDD2022"
            elif engine == "opencv":
                engine_note = "OpenCV fallback (YOLO found none)"
            else:
                engine_note = "No damage found"

            os.makedirs(OUT_DIR, exist_ok=True)
            safe_name = os.path.basename(filename) or "scan.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, safe_name), annotated)

            if save_csv and dets:
                now = datetime.now().isoformat(timespec="seconds")
                csv_rows = [
                    [
                        filename, d["damage_type"], d["confidence"], d["severity"],
                        d["severity_score"], d["x1"], d["y1"], d["x2"], d["y2"],
                        w, h, location, now,
                    ]
                    for d in dets
                ]
                append_csv(csv_rows)
                saved = True

            _, buf = cv2.imencode(".jpg", annotated)
            img_data = base64.b64encode(buf).decode("utf-8")
        except Exception as e:
            error = f"Error: {e}"

    return render_template(
        "index.html",
        img_data=img_data,
        rows=rows,
        error=error,
        saved=saved,
        filename=filename,
        engine_note=engine_note,
        location=location,
        conf=conf,
        stats=load_stats(),
    )


@app.route("/api/detections.csv")
def detections_csv():
    """Power BI can use Get Data → Web with this URL, then Refresh after new scans."""
    ensure_csv()
    with open(CSV_PATH, "rb") as f:
        data = f.read()
    return Response(
        data,
        mimetype="text/csv",
        headers={"Cache-Control": "no-store", "Content-Disposition": "inline; filename=detections.csv"},
    )


@app.route("/download/detections.csv")
def download_csv():
    ensure_csv()
    return send_file(CSV_PATH, as_attachment=True, download_name="detections.csv")


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    print(f"Power BI CSV: {CSV_PATH}")
    print("Power BI web source: http://127.0.0.1:5000/api/detections.csv")
    app.run(debug=True, port=5000, use_reloader=False)
