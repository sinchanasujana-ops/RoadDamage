"""
Runs the trained road-damage model on a video (e.g. your own dashcam / phone
footage of a road), sampling every Nth frame, drawing boxes + severity, saving
an annotated output video, and logging every detection to a CSV for Power BI.

Usage:
    python infer_video.py --source path/to/your_road_video.mp4 \
        --weights ../models/road_damage_run/weights/best.pt \
        --out ../outputs/detections/annotated_video.mp4 \
        --csv ../outputs/csv/detections.csv --every 5
"""

import argparse
import csv
import os
from datetime import datetime

import cv2

from severity import severity_label

CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
CSV_HEADER = [
    "source_file", "frame_number", "timestamp_sec", "damage_type", "confidence",
    "severity", "severity_score", "x1", "y1", "x2", "y2", "img_width", "img_height",
    "location_tag", "processed_at",
]


def ensure_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def run(source, weights, out_path, csv_path, every=5, location_tag="unknown"):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        raise FileNotFoundError(f"No trained weights at {weights}. Train first with train.py")

    model = YOLO(weights)
    ensure_csv(csv_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_num = 0
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num % every == 0:
            results = model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                damage_type = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                sev_label, sev_score = severity_label(damage_type, x1, y1, x2, y2, w, h, conf)

                color = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}[sev_label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{damage_type} | {sev_label}", (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                rows.append([
                    os.path.basename(source), frame_num, round(frame_num / fps, 2),
                    damage_type, round(conf, 3), sev_label, sev_score,
                    x1, y1, x2, y2, w, h, location_tag,
                    datetime.now().isoformat(timespec="seconds"),
                ])
        writer.write(frame)
        frame_num += 1

    cap.release()
    writer.release()

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Processed {frame_num} frames -> annotated video: {out_path}")
    print(f"Logged {len(rows)} detection rows -> {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to a road video file")
    ap.add_argument("--weights", default="../models/road_damage_run/weights/best.pt")
    ap.add_argument("--out", default="../outputs/detections/annotated_video.mp4")
    ap.add_argument("--csv", default="../outputs/csv/detections.csv")
    ap.add_argument("--every", type=int, default=5, help="run detection every Nth frame")
    ap.add_argument("--location", default="unknown")
    args = ap.parse_args()
    run(args.source, args.weights, args.out, args.csv, args.every, args.location)
