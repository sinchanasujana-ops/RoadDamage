# AI-Based Road Damage Detection and Severity Analysis Using Deep Learning

End-to-end project: detect road damage (cracks/potholes) in images or video
using a YOLOv8 model trained on **RDD2022**, estimate a **severity level**
(Low/Medium/High) per detection, and feed the results into a **Power BI**
dashboard.

```
RoadDamageDetection/
├── data/
│   ├── download_rdd2022.sh          # instructions/script to fetch the real dataset
│   ├── road_damage.yaml             # YOLO dataset config
│   ├── generate_sample_images.py    # makes synthetic sample images (already run)
│   ├── sample_images/               # 10 synthetic road images (pipeline testing)
│   ├── annotations/                 # YOLO-format labels for the sample images
│   └── raw/                         # <- put real RDD2022 data here after download
├── src/
│   ├── convert_voc_to_yolo.py       # converts RDD2022 XML annotations -> YOLO format
│   ├── train.py                     # fine-tunes YOLOv8 on the dataset
│   ├── severity.py                  # severity scoring logic (Low/Medium/High)
│   ├── infer_image.py               # run trained model on real image(s)
│   ├── infer_video.py               # run trained model on real video
│   └── run_demo_pipeline.py         # runs the FULL pipeline right now, no training needed
├── outputs/
│   ├── detections/                  # annotated output images/video
│   └── csv/detections.csv           # <- THIS feeds Power BI (already generated, 18 rows)
├── powerbi/
│   ├── README_PowerBI.md            # step-by-step dashboard build guide + DAX measures
│   ├── sample_dashboard_data.csv    # 200-row starter dataset, same schema
│   └── generate_sample_dashboard_data.py
├── models/                          # trained weights land here (models/<run_name>/weights/best.pt)
├── requirements.txt
└── README.md
```

## What's already working (no setup needed)

I ran `run_demo_pipeline.py` for you — it treats the synthetic sample images'
ground-truth as if they were model detections (simulated confidence scores),
so you already have:
- 10 annotated sample images in `outputs/detections/`
- 18 rows of structured detections in `outputs/csv/detections.csv`
- A 200-row Power BI starter dataset in `powerbi/sample_dashboard_data.csv`

Open either CSV in Power BI right now and follow `powerbi/README_PowerBI.md`
to build the dashboard while the real model trains.

## Quick start (real project)

```bash
pip install -r requirements.txt
```

### 1. Get the real dataset (RDD2022)
```bash
cd data
bash download_rdd2022.sh   # prints official links; extract into data/raw/RDD2022/
```
RDD2022 = 47,420 images from 6 countries (Japan, India, Czech Republic,
Norway, USA, China), 55,000+ annotated instances of longitudinal cracks,
transverse cracks, alligator cracks, and potholes.

### 2. Convert annotations to YOLO format
```bash
cd src
python convert_voc_to_yolo.py \
  --xml_dir ../data/raw/RDD2022/Japan/train/annotations/xmls \
  --img_dir ../data/raw/RDD2022/Japan/train/images \
  --out_dir ../data/raw/RDD2022_yolo/train
# repeat per country, or write a small loop over all 6 country folders
```
Then update `data/road_damage.yaml` to point `train:`/`val:` at
`data/raw/RDD2022_yolo/{train,val}`.

### 3. Train
```bash
cd src
python train.py --data ../data/road_damage.yaml --epochs 100 --imgsz 640 --batch 16
```
Best weights: `models/road_damage_run/weights/best.pt`

### 4. Run on YOUR real photos or video
```bash
# real image(s) — a folder or a single file
python infer_image.py --source /path/to/your/photos --weights ../models/road_damage_run/weights/best.pt --location "My City"

# real video (dashcam, phone footage, etc.)
python infer_video.py --source /path/to/your_road_video.mp4 --weights ../models/road_damage_run/weights/best.pt --every 5 --location "My City"
```
Both append to `outputs/csv/detections.csv` — run them as many times as you
like on new footage; results accumulate.

**Tips for real footage:** shoot at road level (vehicle-mounted or handheld,
not aerial), in daylight, reasonably steady — matches how RDD2022 was
captured, so the model generalizes better.

### 5. Power BI
Open `outputs/csv/detections.csv` (or the starter `powerbi/sample_dashboard_data.csv`)
in Power BI Desktop → Get Data → Text/CSV. Full walkthrough with chart
suggestions and DAX measures: `powerbi/README_PowerBI.md`.

## Severity methodology (for your report)

Severity isn't provided by RDD2022, so it's derived in `src/severity.py`:
`severity_score = (bounding_box_area / image_area) * damage_type_weight * confidence_factor`,
bucketed into Low/Medium/High via tunable thresholds. Pothole and alligator
cracking are weighted higher than hairline transverse/longitudinal cracks,
reflecting real-world repair priority. This is documented in code comments
so you can cite/justify it directly in your methodology section.
