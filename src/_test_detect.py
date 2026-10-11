import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect import decode_image, run_detection

folder = os.path.join(os.path.dirname(__file__), "..", "test_photos")
files = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]
for name in files:
    path = os.path.join(folder, name)
    with open(path, "rb") as fh:
        img = decode_image(fh.read())
    if img is None:
        print(name, "UNREADABLE")
        continue
    dets, _ann, engine, _shape = run_detection(img, conf=0.18)
    summary = [(d["damage_type"], d["confidence"]) for d in dets]
    print(f"{name} | {engine} | n={len(dets)} | {summary}")
