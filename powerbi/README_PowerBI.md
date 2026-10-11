# Power BI Dashboard — Road Damage Detection

Your model's output CSV (`outputs/csv/detections.csv`) is already shaped as a
clean tabular dataset, so Power BI just consumes it directly — no extra
conversion needed.

A ready-to-use starter file is included here too:
**`sample_dashboard_data.csv`** — a larger simulated dataset (200 rows) in
the exact same schema, so you can start building the dashboard immediately
without waiting for training/inference to finish. Swap it for the real
`outputs/csv/detections.csv` later — same columns, so nothing breaks.

## Column reference

| Column | Meaning |
|---|---|
| source_file | image/video frame the detection came from |
| frame_number / timestamp_sec | (video only) when in the clip it occurred |
| damage_type | longitudinal_crack / transverse_crack / alligator_crack / pothole |
| confidence | model's detection confidence (0–1) |
| severity | Low / Medium / High (derived, see src/severity.py) |
| severity_score | raw numeric severity score behind the label |
| x1,y1,x2,y2 | bounding box pixel coordinates |
| img_width, img_height | source image/frame dimensions |
| country_or_location / location_tag | where the image/video is from |
| processed_at | timestamp the detection was logged |

## Step-by-step: build the dashboard

1. Open Power BI Desktop → **Get Data → Text/CSV** → select the CSV.
2. In Power Query Editor, set correct types: `confidence`/`severity_score`
   as Decimal Number, `severity`/`damage_type`/`country_or_location` as Text,
   `processed_at` as Date/Time. Click **Close & Apply**.
3. Build these visuals:
   - **Clustered bar chart**: `damage_type` (axis) vs Count of detections (value) — which damage type is most common.
   - **Donut chart**: `severity` (legend) vs Count — proportion of Low/Medium/High.
   - **Filled map** (or Map visual): `country_or_location` (location) vs Count — geographic spread of damage.
   - **KPI cards**: Total detections (Count of rows), % High severity (measure below), Average confidence.
   - **Stacked bar chart**: `country_or_location` (axis) vs Count, broken down by `severity` (legend) — severity distribution per location.
   - **Line chart** (only for video-based data): `frame_number` or `timestamp_sec` (axis) vs Count of detections — damage occurrence over the video.
4. Add slicers for `damage_type`, `severity`, and `country_or_location` so the dashboard is interactive.

## Useful DAX measures

```DAX
Total Detections = COUNTROWS(detections)

High Severity % =
DIVIDE(
    CALCULATE(COUNTROWS(detections), detections[severity] = "High"),
    [Total Detections]
)

Avg Confidence = AVERAGE(detections[confidence])

Pothole Count =
CALCULATE(COUNTROWS(detections), detections[damage_type] = "pothole")
```

## Keeping it updated as you run more images/video

The web app, `infer_image.py`, and `infer_video.py` all **append** to the same
`outputs/csv/detections.csv` rather than overwriting it, so every new photo
you upload adds more rows.

**Refresh the dashboard after each scan**

1. Keep Power BI Desktop pointed at `outputs/csv/detections.csv` (Get Data → Text/CSV).
2. After you upload photos in the web app, click **Home → Refresh** in Power BI.
   New rows (location, damage type, severity, timestamp) appear immediately.
3. Optional live source while the Flask app is running: **Get Data → Web** →
   `http://127.0.0.1:5000/api/detections.csv`  
   Then Refresh uses the same live file the website writes.

The website also shows a **Live dashboard** of the same CSV so you can confirm
new scans landed before refreshing Power BI.
