"""
Severity estimation for detected road damage.

RDD2022 (like most road damage datasets) gives you damage TYPE + location
(bounding box) but not a severity label — so severity has to be derived.
This module implements a simple, explainable heuristic you can defend in a
viva/report, and is easy to extend later (e.g. swap in crack-width
measurement via contour analysis for a more rigorous version).

Heuristic used:
    severity_score = (bbox_area / image_area) * type_weight
    where type_weight reflects how dangerous/costly that damage type
    typically is (pothole > alligator > longitudinal > transverse).

    severity_score -> bucketed into Low / Medium / High via thresholds.

Feel free to tune TYPE_WEIGHTS and THRESHOLDS for your report's methodology
section.
"""

TYPE_WEIGHTS = {
    "pothole": 1.5,
    "alligator_crack": 1.3,
    "longitudinal_crack": 1.0,
    "transverse_crack": 0.9,
}

# score thresholds -> label (tune these against your own validation images)
THRESHOLDS = [
    (0.015, "Low"),
    (0.05, "Medium"),
]  # anything above the last threshold -> "High"


def bbox_area_ratio(x1, y1, x2, y2, img_w, img_h):
    box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    img_area = img_w * img_h
    return box_area / img_area if img_area > 0 else 0.0


def severity_label(damage_type, x1, y1, x2, y2, img_w, img_h, confidence=1.0):
    """
    Returns (severity_label: str, severity_score: float).
    confidence (0-1) from the detector is folded in lightly so low-confidence
    detections don't get over-weighted as severe.
    """
    weight = TYPE_WEIGHTS.get(damage_type, 1.0)
    area_ratio = bbox_area_ratio(x1, y1, x2, y2, img_w, img_h)
    score = area_ratio * weight * (0.5 + 0.5 * confidence)

    for threshold, label in THRESHOLDS:
        if score <= threshold:
            return label, round(score, 5)
    return "High", round(score, 5)


if __name__ == "__main__":
    # quick sanity check
    tests = [
        ("pothole", 100, 100, 160, 150, 640, 480, 0.9),
        ("transverse_crack", 300, 200, 330, 210, 640, 480, 0.7),
        ("alligator_crack", 50, 50, 400, 400, 640, 480, 0.95),
    ]
    for t in tests:
        label, score = severity_label(*t)
        print(f"{t[0]:20s} -> {label:6s} (score={score})")
