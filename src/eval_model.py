from pathlib import Path
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
yaml_path = ROOT / "data" / "eval" / "eval.yaml"

model = YOLO(str(ROOT / "models" / "best.pt"))
r = model.val(data=str(yaml_path), imgsz=640, conf=0.25, plots=False, verbose=False)

cm = np.array(r.confusion_matrix.matrix)   # rows = predicted, columns = true
nc = len(model.names)
core = cm[:nc, :nc]

tp = float(np.trace(core))                 # right box, right class
wrong_class = float(core.sum() - tp)       # right box, wrong class
fp = float(cm[:nc, nc].sum())              # boxes drawn where there is no damage
fn = float(cm[nc, :nc].sum())              # real damage the model missed
total_gt = tp + wrong_class + fn

print("\nTrue positives :", int(tp))
print("Wrong class    :", int(wrong_class))
print("False alarms   :", int(fp))
print("Missed damage  :", int(fn))
print()
print("Detection accuracy       : %.1f%%  (TP / (TP + wrong class + false alarms + missed))" % (100 * tp / max(tp + wrong_class + fp + fn, 1)))
print("Damage found correctly   : %.1f%%  (TP / all real damage)" % (100 * tp / max(total_gt, 1)))
print("Classification accuracy  : %.1f%%  (right type, among damage it found)" % (100 * tp / max(tp + wrong_class, 1)))
print()
for i, name in model.names.items():
    n = core[:, i].sum() + cm[nc, i]
    if n:
        print("%-20s %.1f%%  (%d real boxes)" % (name, 100 * core[i, i] / n, int(n)))