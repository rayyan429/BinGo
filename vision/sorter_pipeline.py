"""
TIDAL — Recycling Guide Station (laptop side, SAM 3 only)
---------------------------------------------------------
Built on sam3_pipeline.py. The camera looks down at three zones:

    | RECYCLE bin |      SCAN ZONE      |  TRASH bin  |
    |   (left)    |  hold the item here |   (right)   |

1. IDLE        SAM 3 runs every material prompt. The top-scoring item in the SCAN
               zone gets a vote each frame.
2. DECIDE      Once one label wins `need` of the last `window` frames, it becomes
               RECYCLE or TRASH. The unsure rule: low confidence, or it also looks
               like trash -> TRASH ("when in doubt, throw it out").
3. GUIDING     The Nucleo lights the right bin. We track the item's mask and see
               which bin opening it goes into.
4. FEEDBACK    Right bin -> <R,1> (chirp), wrong bin -> <R,0> (buzz). Stats update.
5. WAIT_CLEAR  Wait for the scan zone to be empty, then back to IDLE.

Run:
    python sorter_pipeline.py                      # just print messages + overlay
    python sorter_pipeline.py --port COM5          # also drive the Nucleo
    python sorter_pipeline.py --swap-bins          # if recycle is on the RIGHT

Messages to the Nucleo (resent every 0.2 s by a background thread = heartbeat):
    <S,cat,conf,label>   cat 1 = RECYCLE, 0 = TRASH      while guiding
    <R,1> / <R,0>        correct / wrong bin             once per item
    <H>                  heartbeat when nothing is being guided
    <Z>                  reset stats (press r)

Keys in the window:  q = quit,  r = reset stats
"""

import argparse
import re
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_RECYCLE = "plastic bottle,aluminum can,glass bottle,cardboard,paper"
DEFAULT_TRASH = "chip bag,plastic wrapper,food,napkin,styrofoam"

RECYCLE, TRASH = 1, 0
CAT_NAME = {RECYCLE: "RECYCLE", TRASH: "TRASH"}
CAT_COLOR = {RECYCLE: (0, 200, 0), TRASH: (0, 0, 255)}      # BGR: green / red
ZONE_CAT = {"recycle": RECYCLE, "trash": TRASH}


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


def flatten(results, categories, frame_shape):
    """One flat list of detections, each tagged with its category and mask centroid.
    Centroid of the MASK (not the box) = where the object really is."""
    h, w = frame_shape[:2]
    dets = []
    for prompt, items in results.items():
        for d in items:
            m = d["mask"]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ys, xs = np.nonzero(m)
            if xs.size == 0:
                continue
            dets.append({
                "prompt": prompt, "cat": categories[prompt], "score": d["score"],
                "box": d["box"], "mask": m,
                "cx": xs.mean() / w, "cy": ys.mean() / h,   # 0..1 across the frame
            })
    return dets


# ---------------------------------------------------------------------------
# Zones: vertical strips across the camera frame.
# ---------------------------------------------------------------------------
class Zones:
    def __init__(self, bin_width=0.25, scan_width=0.4, swap=False):
        left, right = ("trash", "recycle") if swap else ("recycle", "trash")
        self.strips = {
            left: (0.0, bin_width),
            right: (1.0 - bin_width, 1.0),
            "scan": (0.5 - scan_width / 2, 0.5 + scan_width / 2),
        }

    def zone(self, cx):
        for name, (a, b) in self.strips.items():
            if a <= cx <= b:
                return name
        return None                                 # in the gap between zones


# ---------------------------------------------------------------------------
# Decider: votes over the last `window` frames + the unsure rule.
# ---------------------------------------------------------------------------
class Decider:
    def __init__(self, window=8, need=5, unsure=0.6, margin=0.1):
        self.votes = deque(maxlen=window)         # top scan-zone detection per frame (or None)
        self.cat_best = deque(maxlen=window)      # (best recycle score, best trash score) per frame
        self.need, self.unsure, self.margin = need, unsure, margin

    def reset(self):
        self.votes.clear()
        self.cat_best.clear()

    def update(self, scan_dets):
        if scan_dets:
            self.votes.append(max(scan_dets, key=lambda d: d["score"]))
            r = max((d["score"] for d in scan_dets if d["cat"] == RECYCLE), default=0.0)
            t = max((d["score"] for d in scan_dets if d["cat"] == TRASH), default=0.0)
            self.cat_best.append((r, t))
        else:
            self.votes.append(None)
            self.cat_best.append(None)

    def progress(self):
        """(leading label, votes) for the overlay."""
        labels = [v["prompt"] for v in self.votes if v]
        return Counter(labels).most_common(1)[0] if labels else (None, 0)

    def decision(self):
        label, count = self.progress()
        if label is None or count < self.need:
            return None

        winners = [v for v in self.votes if v and v["prompt"] == label]
        conf = float(np.mean([v["score"] for v in winners]))
        cat = winners[-1]["cat"]
        present = [c for c in self.cat_best if c]
        r_avg = float(np.mean([c[0] for c in present]))
        t_avg = float(np.mean([c[1] for c in present]))

        # Unsure rule: only ever flips RECYCLE -> TRASH. A wrong item in recycling
        # can contaminate the whole load; an extra item in trash costs very little.
        reason = ""
        if cat == RECYCLE:
            if conf < self.unsure:
                cat, reason = TRASH, "unsure"
            elif t_avg > 0 and (r_avg - t_avg) < self.margin:
                cat, reason = TRASH, "looks like trash too"

        return {"label": label, "cat": cat, "conf": conf, "reason": reason,
                "pos": (winners[-1]["cx"], winners[-1]["cy"])}


# ---------------------------------------------------------------------------
# DropTracker: follows the decided item and reports which bin it went into.
# Follows the detection NEAREST to where the item was last seen, so items that
# are already sitting in a bin don't get mistaken for the new one.
# ---------------------------------------------------------------------------
class DropTracker:
    BINS = ("recycle", "trash")

    def __init__(self, label, pos, confirm=2, lost=5, max_jump=0.3):
        self.label, self.pos = label, pos
        self.confirm, self.lost, self.max_jump = confirm, lost, max_jump
        self.recent = deque(maxlen=confirm)
        self.last_zone = "scan"
        self.missing = 0

    def update(self, dets, zones):
        """Returns 'recycle' / 'trash' once the item is in a bin, else None."""
        cands = [d for d in dets if d["prompt"] == self.label] or dets   # SAM may relabel it mid-move
        best = None
        if cands:
            near = min(cands, key=lambda d: np.hypot(d["cx"] - self.pos[0], d["cy"] - self.pos[1]))
            if np.hypot(near["cx"] - self.pos[0], near["cy"] - self.pos[1]) <= self.max_jump:
                best = near

        if best is not None:
            self.pos = (best["cx"], best["cy"])
            self.missing = 0
            z = zones.zone(best["cx"])
            if z:
                self.last_zone = z
            self.recent.append(z)
            # Seen inside the same bin opening for `confirm` frames in a row -> that's the bin.
            if z in self.BINS and len(self.recent) == self.confirm and all(r == z for r in self.recent):
                return z
        else:
            self.missing += 1
            # Item vanished (fell into the bin) and the last place we saw it was a bin opening.
            if self.missing >= self.lost and self.last_zone in self.BINS:
                return self.last_zone
        return None


# ---------------------------------------------------------------------------
# Serial link: a background thread sends the CURRENT state message every
# `period` seconds (latest-wins + heartbeat), plus one-off events like <R,1>.
# ---------------------------------------------------------------------------
class McuLink:
    def __init__(self, port=None, baud=115200, period=0.2):
        self.ser = None
        if port:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(2)                           # let the board settle after the port opens
        self.state_msg = "<H>"
        self.events = []
        self.period = period
        self.lock = threading.Lock()
        self.running = True
        self._printed = None
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, event=None, state=None):
        """Queue an event and/or change the repeating state message ATOMICALLY,
        so an old <S> can never be sent after its <R> (that would restart guiding)."""
        with self.lock:
            if event:
                self.events.append(event)
            if state:
                self.state_msg = state

    def _loop(self):
        rx = b""
        while self.running:
            with self.lock:
                out = self.events + [self.state_msg]
                self.events = []
            for m in out:
                if self.ser:
                    self.ser.write(m.encode())
                if m != self._printed:              # only print when it changes (no spam)
                    print("->", m)
                    self._printed = m
            if self.ser:
                rx += self.ser.read(256)
                while b"\n" in rx:
                    line, rx = rx.split(b"\n", 1)
                    if line.strip():
                        print("<- MCU:", line.decode(errors="ignore").strip())
            time.sleep(self.period)

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()


def clean_label(s):
    """Labels go inside <...,label>: no commas/brackets, and the LCD has 12 chars for it."""
    return re.sub(r"[^A-Za-z0-9 ]", "", s)[:12] or "item"


# ---------------------------------------------------------------------------
# Overlay for the demo screen
# ---------------------------------------------------------------------------
def draw(frame, zones, dets, state, decision, feedback, decider, stats, fps, msg):
    vis = frame.copy()
    h, w = vis.shape[:2]
    overlay = vis.copy()

    zone_color = {"recycle": CAT_COLOR[RECYCLE], "trash": CAT_COLOR[TRASH], "scan": (0, 220, 255)}
    for name, (a, b) in zones.strips.items():
        x1, x2 = int(a * w), int(b * w)
        cv2.rectangle(vis, (x1, 0), (x2, h - 1), zone_color[name], 2)
        cv2.putText(vis, name.upper(), (x1 + 6, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color[name], 2)

    for d in dets:
        color = CAT_COLOR[d["cat"]]
        overlay[d["mask"]] = color
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"{d['prompt']} {d['score']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # Status banner
    if state == "IDLE":
        label, n = decider.progress()
        text = f"Scanning... {label} {n}/{decider.need}" if label else "Hold an item in the SCAN zone"
        color = (255, 255, 255)
    elif state == "GUIDING":
        why = f" ({decision['reason']})" if decision["reason"] else ""
        text = f"{decision['label']} {decision['conf']:.0%} -> {CAT_NAME[decision['cat']]}{why}"
        color = CAT_COLOR[decision["cat"]]
    elif state == "FEEDBACK":
        ok, chosen = feedback
        text = "CORRECT BIN!" if ok else f"WRONG BIN (went in {chosen.upper()})"
        color = (0, 200, 0) if ok else (0, 0, 255)
    else:
        text, color = "Clear the scan zone...", (200, 200, 200)
    cv2.rectangle(vis, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Stats + FPS + outgoing message
    n = stats["sorted"]
    acc = 100 * stats["correct"] / n if n else 0
    rec = 100 * stats["recycled"] / n if n else 0
    cv2.putText(vis, f"Sorted {n}   Accuracy {acc:.0f}%   Recycled {rec:.0f}%", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, f"{fps:.1f} FPS", (w - 100, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, f"-> MCU: {msg}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recycle", default=DEFAULT_RECYCLE, help="comma-separated RECYCLE prompts")
    ap.add_argument("--trash", default=DEFAULT_TRASH, help="comma-separated TRASH prompts")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.4, help="min SAM 3 score to count a detection")
    ap.add_argument("--unsure", type=float, default=0.6, help="RECYCLE below this avg score -> TRASH")
    ap.add_argument("--margin", type=float, default=0.1, help="RECYCLE must beat TRASH score by this much")
    ap.add_argument("--window", type=int, default=8, help="vote window (frames)")
    ap.add_argument("--need", type=int, default=5, help="votes needed in the window")
    ap.add_argument("--bin-width", type=float, default=0.25, help="width of each bin zone (0-1)")
    ap.add_argument("--scan-width", type=float, default=0.4, help="width of the scan zone (0-1)")
    ap.add_argument("--swap-bins", action="store_true", help="recycle bin on the RIGHT")
    ap.add_argument("--guide-timeout", type=float, default=8.0, help="seconds to wait for a drop")
    ap.add_argument("--port", default=None, help="serial port, e.g. COM5 (omit to just print)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    recycle = [p.strip() for p in args.recycle.split(",") if p.strip()]
    trash = [p.strip() for p in args.trash.split(",") if p.strip()]
    prompts = recycle + trash
    categories = {**{p: RECYCLE for p in recycle}, **{p: TRASH for p in trash}}

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected - SAM 3 requires an NVIDIA GPU.")

    print("Loading SAM 3 ...")
    processor = Sam3Processor(build_sam3_image_model())

    link = McuLink(args.port, args.baud)
    cam = LatestFrameCamera(args.camera)
    zones = Zones(args.bin_width, args.scan_width, args.swap_bins)
    decider = Decider(args.window, args.need, args.unsure, args.margin)

    state, t_state = "IDLE", time.time()
    decision, tracker, feedback = None, None, None
    clear_frames = 0
    stats = {"sorted": 0, "correct": 0, "recycled": 0}
    t_prev, fps = time.time(), 0.0
    msg = "<H>"

    print("RECYCLE:", ", ".join(recycle))
    print("TRASH:  ", ", ".join(trash))
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
            dets = flatten(results, categories, frame.shape)
            scan_dets = [d for d in dets if zones.zone(d["cx"]) == "scan"]
            now = time.time()

            # ------------------------- state machine -------------------------
            if state == "IDLE":
                decider.update(scan_dets)
                decision = decider.decision()
                if decision:
                    tracker = DropTracker(decision["label"], decision["pos"])
                    msg = f"<S,{decision['cat']},{int(decision['conf'] * 100)},{clean_label(decision['label'])}>"
                    link.send(state=msg)
                    state, t_state = "GUIDING", now

            elif state == "GUIDING":
                chosen = tracker.update(dets, zones)
                if chosen:
                    ok = ZONE_CAT[chosen] == decision["cat"]
                    stats["sorted"] += 1
                    stats["correct"] += ok
                    stats["recycled"] += ok and decision["cat"] == RECYCLE
                    msg = f"<R,{int(ok)}>"
                    link.send(event=msg, state="<H>")
                    feedback = (ok, chosen)
                    state, t_state = "FEEDBACK", now
                elif now - t_state > args.guide_timeout:
                    msg = "<H>"                             # item taken away: give up quietly
                    link.send(state=msg)
                    decider.reset()
                    state, t_state = "IDLE", now

            elif state == "FEEDBACK":
                if now - t_state > 2.5:
                    state, t_state, clear_frames = "WAIT_CLEAR", now, 0

            elif state == "WAIT_CLEAR":
                clear_frames = 0 if scan_dets else clear_frames + 1
                if clear_frames >= 5:
                    decider.reset()
                    state, t_state = "IDLE", now

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            cv2.imshow("TIDAL recycling guide",
                       draw(frame, zones, dets, state, decision, feedback, decider, stats, fps, msg))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                stats = {"sorted": 0, "correct": 0, "recycled": 0}
                link.send(event="<Z>")
    finally:
        link.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()