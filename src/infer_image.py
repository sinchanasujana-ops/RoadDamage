"""
Runs the trained road-damage model on a single image (or a folder of images) —
including your OWN real photos, not just dataset images — draws boxes +
severity labels, and appends structured rows to a CSV that Power BI can read.

Usage:
    python infer_image.py --source ../data/sample_images --weights ../models/rdd2022_yolov8n.pt \
        --out ../outputs/detections --csv ../outputs/csv/detections.csv

If --weights is omitted, the RDD2022 YOLOv8 model is used (downloaded on first run).
"""

import argparse
import csv
import os
from datetime import datetime

import cv2

from detect import decode_image, run_detection

CSV_HEADER = [
    "source_file", "damage_type", "confidence", "severity", "severity_score",
    "x1", "y1", "x2", "y2", "img_width", "img_height", "country_or_location",
    "processed_at",
]


def ensure_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CSV_HEADER)


def run(source, weights, out_dir, csv_path, location_tag="unknown"):
    os.makedirs(out_dir, exist_ok=True)
    ensure_csv(csv_path)

    files = []
    if os.path.isdir(source):
        for f in sorted(os.listdir(source)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                files.append(os.path.join(source, f))
    else:
        files = [source]

    rows = []
    for fpath in files:
        with open(fpath, "rb") as fh:
            img = decode_image(fh.read())
        if img is None:
            print(f"skip (unreadable): {fpath}")
            continue
        dets, annotated, engine, (w, h) = run_detection(
            img, weights=weights if os.path.exists(weights) else None
        )
        now = datetime.now().isoformat(timespec="seconds")
        for d in dets:
            rows.append([
                os.path.basename(fpath), d["damage_type"], d["confidence"], d["severity"],
                d["severity_score"], d["x1"], d["y1"], d["x2"], d["y2"], w, h,
                location_tag, now,
            ])
        out_path = os.path.join(out_dir, os.path.basename(fpath))
        cv2.imwrite(out_path, annotated)
        print(f"processed {fpath} -> {out_path} ({len(dets)} detections, {engine})")

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"\nAppended {len(rows)} detection rows -> {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image file or folder of images")
    ap.add_argument("--weights", default="../models/rdd2022_yolov8n.pt")
    ap.add_argument("--out", default="../outputs/detections")
    ap.add_argument("--csv", default="../outputs/csv/detections.csv")
    ap.add_argument("--location", default="unknown", help="tag for Power BI, e.g. city/country name")
    args = ap.parse_args()
    run(args.source, args.weights, args.out, args.csv, args.location)
