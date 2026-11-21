"""
Smoke test for SAM 3: grabs ONE webcam frame, runs a text prompt, saves the result.
Run:  python test_sam3.py "water bottle"
Output: sam3_test.jpg + printed detections and timing.
"""
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

prompt = sys.argv[1] if len(sys.argv) > 1 else "person"
print("CUDA:", torch.cuda.is_available())

# 1. Grab one frame from the webcam
cap = cv2.VideoCapture(0)
for _ in range(5):          # first frames are often dark while the camera adjusts
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("Could not read from webcam (is another app using it?)")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 2. Load SAM 3 (first run downloads the weights from Hugging Face)
t = time.time()
model = build_sam3_image_model()
processor = Sam3Processor(model)
print(f"Model loaded in {time.time() - t:.1f}s")

# 3. Run it twice: the first call includes warm-up, the second shows real speed
for run in range(2):
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(Image.fromarray(rgb))
        out = processor.set_text_prompt(state=state, prompt=prompt)
    torch.cuda.synchronize()
    print(f"Run {run + 1}: {(time.time() - t) * 1000:.0f} ms")

# 4. Inspect the raw outputs so we know their exact shapes for the pipeline
for k in ("masks", "boxes", "scores"):
    v = out[k]
    print(f"{k}: type={type(v).__name__}, shape={tuple(v.shape) if hasattr(v, 'shape') else len(v)}")

masks = out["masks"].detach().float().cpu().numpy()
boxes = out["boxes"].detach().float().cpu().numpy()
scores = out["scores"].detach().float().cpu().numpy().reshape(-1)
if masks.ndim == 4:
    masks = masks[:, 0]

print(f'\nFound {len(scores)} "{prompt}" object(s)')
vis = frame.copy()
for i, s in enumerate(scores):
    x1, y1, x2, y2 = boxes[i].astype(int)
    print(f"  #{i}: score={s:.2f}  box=({x1},{y1},{x2},{y2})")
    vis[masks[i] > 0.5] = (0.5 * vis[masks[i] > 0.5] + [0, 0, 127]).astype(np.uint8)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, f"{s:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("sam3_test.jpg", vis)
print("Saved sam3_test.jpg")