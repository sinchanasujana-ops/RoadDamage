"""Generates a larger (200-row) simulated dataset in the same schema as
outputs/csv/detections.csv, purely so you can start building the Power BI
dashboard immediately, before real inference results are ready."""

import csv
import random
from datetime import datetime, timedelta

random.seed(1)

DAMAGE_TYPES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
LOCATIONS = ["India", "Japan", "Czech Republic", "Norway", "United States", "China"]
HEADER = [
    "source_file", "damage_type", "confidence", "severity", "severity_score",
    "x1", "y1", "x2", "y2", "img_width", "img_height", "country_or_location",
    "processed_at",
]

rows = []
start = datetime(2026, 8, 1, 9, 0, 0)
for i in range(1, 201):
    damage_type = random.choices(DAMAGE_TYPES, weights=[3, 2, 2, 4])[0]
    confidence = round(random.uniform(0.55, 0.98), 2)
    w, h = 640, 480
    x1 = random.randint(0, 500)
    y1 = random.randint(0, 380)
    x2 = x1 + random.randint(20, 130)
    y2 = y1 + random.randint(15, 100)
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    weight = {"pothole": 1.5, "alligator_crack": 1.3, "longitudinal_crack": 1.0, "transverse_crack": 0.9}[damage_type]
    score = round(area_ratio * weight * (0.5 + 0.5 * confidence), 5)
    severity = "Low" if score <= 0.015 else ("Medium" if score <= 0.05 else "High")
    rows.append([
        f"road_img_{i:03d}.jpg", damage_type, confidence, severity, score,
        x1, y1, x2, y2, w, h, random.choice(LOCATIONS),
        (start + timedelta(minutes=i * 3)).isoformat(timespec="seconds"),
    ])

with open("sample_dashboard_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows -> sample_dashboard_data.csv")
