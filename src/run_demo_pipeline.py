"""
Runs the FULL pipeline (detection -> severity -> annotated images -> CSV)
right now, without needing YOLO training first, by treating the ground-truth
YOLO labels of the bundled synthetic sample images as if they were model
detections (confidence simulated).

This exists purely so you can see the whole pipeline + Power BI feed working
end-to-end today. Once you've trained a real model (train.py) on RDD2022,
switch to infer_image.py / infer_video.py with real --weights for actual
predictions on real images/video.

Usage:
    python run_demo_pipeline.py
"""

import csv
import os
import random
from datetime import datetime

import cv2

from severity import severity_label

random.seed(7)

BASE = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE, "..", "data", "sample_images")
LABELS_DIR = os.path.join(BASE, "..", "data", "annotations", "yolo_labels")
CLASSES_FILE = os.path.join(BASE, "..", "data", "annotations", "classes.txt")
OUT_DIR = os.path.join(BASE, "..", "outputs", "detections")
CSV_PATH = os.path.join(BASE, "..", "outputs", "csv", "detections.csv")

CSV_HEADER = [
    "source_file", "damage_type", "confidence", "severity", "severity_score",
    "x1", "y1", "x2", "y2", "img_width", "img_height", "country_or_location",
    "processed_at",
]


def main():
    with open(CLASSES_FILE) as f:
        class_names = [l.strip() for l in f if l.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    fake_locations = ["India", "Japan", "Czech Republic", "Norway", "United States", "China"]
    rows = []

    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(IMG_DIR, fname)
        label_path = os.path.join(LABELS_DIR, fname.replace(".jpg", ".txt"))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if not os.path.exists(label_path):
            continue
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            cls_id, xc, yc, bw, bh = line.split()
            cls_id = int(cls_id)
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            damage_type = class_names[cls_id]
            confidence = round(random.uniform(0.62, 0.97), 2)  # simulated detector confidence

            sev_label, sev_score = severity_label(damage_type, x1, y1, x2, y2, w, h, confidence)
            color = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}[sev_label]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{damage_type} {confidence} | {sev_label}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            rows.append([
                fname, damage_type, confidence, sev_label, sev_score,
                x1, y1, x2, y2, w, h, random.choice(fake_locations),
                datetime.now().isoformat(timespec="seconds"),
            ])

        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        print(f"processed {fname} -> {len(lines)} detections")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"\nDemo pipeline complete.")
    print(f"Annotated images -> {os.path.abspath(OUT_DIR)}")
    print(f"Power BI-ready CSV -> {os.path.abspath(CSV_PATH)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
