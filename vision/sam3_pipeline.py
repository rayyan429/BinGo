"""
SAM 3 detection pipeline (no Depth Anything)
--------------------------------------------
Webcam -> SAM 3 (one image encode, several text prompts) -> persistence filter
       -> highest-priority active event -> one compact message per frame.

Prompts are given in PRIORITY ORDER (first = most important). Event id = position (1, 2, 3...).

Run:
    python sam3_pipeline.py --prompts "fire,person,cardboard box"
    python sam3_pipeline.py --prompts "fire,person,cardboard box" --port COM5

Messages:
    <E,id,conf>   event `id` is active (conf = %)
    <N,0,0>       nothing active

Keys in the window:  q = quit
"""

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200)]


# ---------------------------------------------------------------------------
# Camera thread: always holds ONLY the newest frame (latest-wins, no FIFO).
# ---------------------------------------------------------------------------
class LatestFrameCamera:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# SAM 3: encode the image ONCE, then ask each text prompt against it.
# The image encoder is the expensive part, so this is much faster than
# running the full model once per prompt.
# ---------------------------------------------------------------------------
def detect_all(processor, rgb, prompts, score_thresh):
    """Returns {prompt: [ {box, mask, score}, ... ]}"""
    state = processor.set_image(Image.fromarray(rgb))
    results = {}
    for p in prompts:
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)      # clear the previous prompt
        out = processor.set_text_prompt(state=state, prompt=p)

        masks = to_numpy(out["masks"])
        boxes = to_numpy(out["boxes"])
        scores = to_numpy(out["scores"]).reshape(-1)
        if masks.ndim == 4:                        # [N,1,H,W] -> [N,H,W]
            masks = masks[:, 0]

        results[p] = [
            {"box": boxes[i].astype(int), "mask": masks[i] > 0.5, "score": float(s)}
            for i, s in enumerate(scores) if s >= score_thresh
        ]
    return results


# ---------------------------------------------------------------------------
# Persistence filter: an event only counts as "active" if it was seen in
# at least `need` of the last `window` frames. Stops a single bad frame
# from triggering an alert, and a single missed frame from cancelling one.
# ---------------------------------------------------------------------------
class Persistence:
    def __init__(self, prompts, window=8, need=5):
        self.hist = {p: deque(maxlen=window) for p in prompts}
        self.need = need

    def update(self, results):
        for p, dets in results.items():
            self.hist[p].append(len(dets) > 0)

    def active(self, p):
        return sum(self.hist[p]) >= self.need


def make_message(prompts, results, persist):
    """Highest-priority active event wins (prompts are in priority order)."""
    for i, p in enumerate(prompts):
        if persist.active(p) and results[p]:
            conf = max(d["score"] for d in results[p])
            return f"<E,{i + 1},{int(conf * 100)}>"
    return "<N,0,0>"


# ---------------------------------------------------------------------------
# Overlay for the demo screen
# ---------------------------------------------------------------------------
def draw(frame, prompts, results, persist, msg, fps):
    vis = frame.copy()
    overlay = vis.copy()
    for i, p in enumerate(prompts):
        color = COLORS[i % len(COLORS)]
        for d in results[p]:
            overlay[d["mask"]] = color
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{p} {d['score']:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # Event status list (ACTIVE = passed the persistence filter)
    for i, p in enumerate(prompts):
        seen = sum(persist.hist[p])
        state = "ACTIVE" if persist.active(p) else f"{seen}/{persist.need}"
        cv2.putText(vis, f"{i + 1}. {p}: {state}", (10, 24 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i % len(COLORS)], 2)

    cv2.putText(vis, f"{fps:.1f} FPS", (vis.shape[1] - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"-> MCU: {msg}", (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="person",
                    help='comma-separated, highest priority first, e.g. "fire,person,cardboard box"')
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.5, help="min SAM 3 confidence")
    ap.add_argument("--window", type=int, default=8, help="persistence window (frames)")
    ap.add_argument("--need", type=int, default=5, help="detections needed in window")
    ap.add_argument("--port", default=None, help="serial port, e.g. COM5 (omit to just print)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected - SAM 3 requires an NVIDIA GPU.")

    print("Loading SAM 3 ...")
    processor = Sam3Processor(build_sam3_image_model())

    ser = None
    if args.port:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=0)
        time.sleep(2)

    cam = LatestFrameCamera(args.camera)
    persist = Persistence(prompts, args.window, args.need)
    t_prev, fps = time.time(), 0.0

    print("Events (priority order):", ", ".join(f"{i + 1}={p}" for i, p in enumerate(prompts)))
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # SAM 3 runs parts of its backbone in bfloat16; autocast keeps every
            # layer's inputs and weights in matching dtypes.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                results = detect_all(processor, rgb, prompts, args.score)
            persist.update(results)
            msg = make_message(prompts, results, persist)

            print(msg)
            if ser:
                ser.write(msg.encode())

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            cv2.imshow("SAM 3 pipeline", draw(frame, prompts, results, persist, msg, fps))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if ser:
            ser.write(b"<N,0,0>")
            ser.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()