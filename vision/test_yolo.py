"""
YOLOE-26 webcam test.

Default: looks ONLY for the waste items listed in sorter_yoloe.py (ITEM_CLASSES),
so people, shirts, laptops etc. are never reported.
    python vision/yoloe_test.py

Full vocabulary (detects EVERYTHING, ~4,585 names; useful only for exploring):
    python vision/yoloe_test.py --pf

Your own list:
    python vision/yoloe_test.py --prompts "soda can,chip bag,banana"

Options: --size n|s|m|l|x (default s), --conf 0.25, --camera 0
Press q in the video window to quit.
"""

import argparse
import time

import cv2
import torch
from ultralytics import YOLOE

from sorter_pipeline import ITEM_CLASSES   # same waste list as the station

ap = argparse.ArgumentParser()
ap.add_argument("--pf", action="store_true", help="prompt-free: detect everything")
ap.add_argument("--prompts", default=None, help="comma-separated classes (overrides the waste list)")
ap.add_argument("--size", default="s", choices=list("nsmlx"), help="model size: n (fastest) ... x (best)")
ap.add_argument("--conf", type=float, default=0.25, help="min confidence")
ap.add_argument("--camera", type=int, default=0)
args = ap.parse_args()

device = 0 if torch.cuda.is_available() else "cpu"

if args.pf:
    model = YOLOE(f"yoloe-26{args.size}-seg-pf.pt")
    print("Prompt-free mode: detecting EVERYTHING in the built-in vocabulary")
else:
    classes = ([p.strip() for p in args.prompts.split(",") if p.strip()]
               if args.prompts else list(ITEM_CLASSES))
    model = YOLOE(f"yoloe-26{args.size}-seg.pt")
    model.set_classes(classes)
    print(f"Looking ONLY for {len(classes)} items:", ", ".join(classes))
print("Running on:", torch.cuda.get_device_name(0) if device == 0 else "CPU")

cap = cv2.VideoCapture(args.camera)
t_prev, fps = time.time(), 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
    vis = r.plot()

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now
    cv2.putText(vis, f"YOLOE-26{args.size}  {fps:.1f} FPS", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if r.boxes is not None and len(r.boxes):
        found = sorted(((r.names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)),
                       key=lambda x: -x[1])
        print(", ".join(f"{n} {s:.2f}" for n, s in found[:5]))

    cv2.imshow("YOLOE-26 test", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()