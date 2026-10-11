"""
Fine-tunes a YOLOv8 model on the road damage dataset (RDD2022, or the
synthetic sample set for a quick pipeline smoke-test).

Install first:
    pip install ultralytics --break-system-packages

Usage (smoke test on the bundled synthetic sample images):
    python train.py --data ../data/road_damage.yaml --epochs 5 --imgsz 640

Usage (real training on RDD2022, after converting annotations and
pointing road_damage.yaml at data/raw/RDD2022_yolo):
    python train.py --data ../data/road_damage.yaml --epochs 100 --imgsz 640 --batch 16
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/road_damage.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="../models")
    ap.add_argument("--name", default="road_damage_run")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(f"\nTraining complete. Best weights saved under {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
