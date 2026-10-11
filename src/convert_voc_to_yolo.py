"""
Converts RDD2022's Pascal-VOC-style XML annotations into YOLO .txt format.

RDD2022 ships each country as:
    <Country>/train/images/*.jpg
    <Country>/train/annotations/xmls/*.xml

RDD2022 damage class codes (from the official challenge) map to 4 broad
categories used throughout this project:
    D00 -> longitudinal_crack
    D10 -> transverse_crack
    D20 -> alligator_crack
    D40 -> pothole
(Other codes like D01, D11, D43, D44, D50 etc. appear in some country
subsets for finer-grained damage — extend CLASS_MAP below if you want to
keep those as separate classes instead of folding them into the 4 broad
ones.)

Usage:
    python convert_voc_to_yolo.py \
        --xml_dir data/raw/RDD2022/Japan/train/annotations/xmls \
        --img_dir data/raw/RDD2022/Japan/train/images \
        --out_dir data/raw/RDD2022_yolo/train
"""

import argparse
import os
import xml.etree.ElementTree as ET
from PIL import Image

CLASS_MAP = {
    "D00": 0, "D01": 0,             # -> longitudinal_crack
    "D10": 1, "D11": 1,             # -> transverse_crack
    "D20": 2,                       # -> alligator_crack
    "D40": 3, "D43": 3, "D44": 3,   # -> pothole
}
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def convert_one(xml_path, img_dir, out_labels_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        return False, f"missing image {filename}"

    with Image.open(img_path) as im:
        w, h = im.size

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    out_path = os.path.join(out_labels_dir, os.path.splitext(filename)[0] + ".txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return True, filename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True, help="folder of RDD2022 XML annotations")
    ap.add_argument("--img_dir", required=True, help="matching folder of images")
    ap.add_argument("--out_dir", required=True, help="output folder; will contain labels/")
    args = ap.parse_args()

    labels_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(args.xml_dir) if f.endswith(".xml")]
    ok, fail = 0, 0
    for xf in xml_files:
        success, msg = convert_one(os.path.join(args.xml_dir, xf), args.img_dir, labels_dir)
        ok += success
        fail += not success

    with open(os.path.join(args.out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(CLASS_NAMES))

    print(f"Converted {ok} annotation files -> {labels_dir}")
    if fail:
        print(f"Skipped {fail} (missing matching image)")


if __name__ == "__main__":
    main()
