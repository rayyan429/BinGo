"""
TIDAL — Recycling Guide Station (MAIN PROGRAM)
==============================================
Camera -> YOLOE-26 (auto-detect) -> recycle/trash decision -> USB serial -> STM32 Nucleo
-> bin LEDs + LCD + buzzer. Then the camera checks which bin the item really went in.

    | RECYCLE bin |      SCAN ZONE      |  TRASH bin  |
    |   (left)    |  hold the item here |   (right)   |

Two detection modes (nobody types prompts at runtime):
    --mode items (default)  YOLOE looks ONLY for the built-in waste list (ITEM_CLASSES).
    --mode pf               YOLOE "-pf" names anything from ~4,585 words; we keep only
                            names matching the recycle/trash keyword lists.

Pipeline per frame:
    YOLOE (labels, e.g. "water bottle", "soda can", "paper towel")
      -> items mode: look up the bin in ITEM_CLASSES
         pf mode: ignore people/furniture, trash keywords first, then recycle keywords
      -> vote on the CATEGORY over the last frames (labels flicker between synonyms)
      -> unsure rule -> guide -> drop tracking -> feedback

Run (one command does everything; it finds the Nucleo's COM port by itself):
    python vision/sorter_yoloe.py
    python vision/sorter_yoloe.py --port COM5        # name the port yourself
    python vision/sorter_yoloe.py --no-serial        # laptop only, no board
    python vision/sorter_yoloe.py --show-all         # print every label (to tune keywords)

Messages to the Nucleo (see firmware/recycle_station/recycle_station.ino):
    <S,cat,conf,label>  guiding (cat 1 = recycle, 0 = trash), resent every 0.2 s
    <R,1> / <R,0>       item went in the correct / wrong bin (once)
    <H>                 heartbeat when nothing is being guided
    <Z>                 reset stats

Keys: q = quit, r = reset stats
"""

import argparse
import re
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

RECYCLE, TRASH = 1, 0
CAT_NAME = {RECYCLE: "RECYCLE", TRASH: "TRASH"}
CAT_COLOR = {RECYCLE: (0, 200, 0), TRASH: (0, 0, 255)}      # BGR: green / red
ZONE_CAT = {"recycle": RECYCLE, "trash": TRASH}

# ---------------------------------------------------------------------------
# Label -> category rules. Edit these after watching what --show-all prints.
# TRASH is checked FIRST so "plastic bag" / "paper towel" / "coffee cup"
# don't get caught by the broader recycle words "plastic" / "paper" / "cup".
# ---------------------------------------------------------------------------
TRASH_WORDS = [
    "bag", "wrapper", "wrap", "chip", "crisp", "snack", "candy", "napkin", "tissue",
    "paper towel", "towel", "styrofoam", "foam", "straw", "coffee cup", "paper cup",
    "food", "fruit", "banana", "apple", "orange", "peel", "sandwich", "pizza",
    "cigarette", "glove", "mask", "diaper", "sponge", "receipt", "cling",
    "packet", "balloon", "gum", "wipe", "bubble wrap", "ribbon", "confetti",
    "cutlery", "lid", "cap", "fries", "pizza box", "plate",
]
RECYCLE_WORDS = [
    "bottle", "can", "tin", "aluminum", "aluminium", "soda", "beer can",
    "jar", "glass", "carton", "cardboard", "box", "paper", "newspaper",
    "magazine", "envelope", "book", "notebook", "mail", "flyer", "jug",
    "container", "tub", "plastic",
]
# Things that show up in the frame but aren't items to sort.
IGNORE_WORDS = [
    "person", "man", "woman", "boy", "girl", "child", "hand", "finger", "arm", "face",
    "head", "hair", "shirt", "sleeve", "watch", "ring", "table", "desk", "chair",
    "wall", "floor", "ceiling", "window", "door", "room", "shelf", "laptop",
    "computer", "keyboard", "monitor", "screen", "mouse", "cable", "wire", "lamp",
    "light", "trash can", "garbage can", "bin", "wastebasket", "basket",
]


def word_in(label, words):
    """Whole-word match, so 'can' doesn't match 'candle' and 'tin' doesn't match 'tinted'."""
    return any(re.search(rf"\b{re.escape(w)}s?\b", label) for w in words)


def categorize(label):
    """Prompt-free mode: RECYCLE, TRASH, or None (ignore).
    Only labels matching a recycle/trash word count; everything else (keyboards,
    walls, phones...) is ignored, so the station only reacts to waste items."""
    l = label.lower()
    if word_in(l, IGNORE_WORDS):
        return None
    if word_in(l, TRASH_WORDS):
        return TRASH
    if word_in(l, RECYCLE_WORDS):
        return RECYCLE
    return None


# ---------------------------------------------------------------------------
# "items" mode (default): YOLOE is told up front to look ONLY for these waste
# items, so it never reports keyboards, people or furniture. Nobody types
# anything at runtime; the list is built in. Edit it to match your demo items.
# ---------------------------------------------------------------------------
# Several names for the same thing (e.g. "chip bag" / "chips packet" / "crisp packet")
# are deliberate: YOLOE matches whichever wording fits best, and because the station
# votes on the BIN, not the name, synonyms in the same bin never conflict.
# Bin rules vary by city: change RECYCLE/TRASH next to any item to match yours.
ITEM_CLASSES = {
    # ======================= RECYCLE =======================
    # plastic bottles & containers
    "plastic bottle": RECYCLE, "water bottle": RECYCLE, "soda bottle": RECYCLE,
    "juice bottle": RECYCLE, "milk jug": RECYCLE, "plastic jug": RECYCLE,
    "detergent bottle": RECYCLE, "shampoo bottle": RECYCLE, "spray bottle": RECYCLE,
    "plastic container": RECYCLE, "yogurt cup": RECYCLE, "plastic tub": RECYCLE,
    # metal
    "soda can": RECYCLE, "aluminum can": RECYCLE, "beer can": RECYCLE,
    "tin can": RECYCLE, "food can": RECYCLE, "aluminum foil": RECYCLE,
    "aluminum tray": RECYCLE,
    # glass
    "glass bottle": RECYCLE, "glass jar": RECYCLE, "beer bottle": RECYCLE,
    "wine bottle": RECYCLE,
    # paper & cardboard
    "cardboard box": RECYCLE, "cardboard": RECYCLE, "shoe box": RECYCLE,
    "cereal box": RECYCLE, "tissue box": RECYCLE, "toilet paper roll": RECYCLE,
    "paper": RECYCLE, "printer paper": RECYCLE, "notebook": RECYCLE,
    "newspaper": RECYCLE, "magazine": RECYCLE, "flyer": RECYCLE,
    "envelope": RECYCLE, "paper bag": RECYCLE, "sticky note": RECYCLE,
    # cartons
    "milk carton": RECYCLE, "juice box": RECYCLE, "juice carton": RECYCLE,
    "egg carton": RECYCLE,

    # ======================== TRASH ========================
    # chip packets & wrappers (multi-layer plastic/foil: not recyclable)
    "chip bag": TRASH, "chips packet": TRASH, "crisp packet": TRASH,
    "potato chip bag": TRASH, "snack bag": TRASH, "candy wrapper": TRASH,
    "chocolate wrapper": TRASH, "snack wrapper": TRASH, "granola bar wrapper": TRASH,
    "food wrapper": TRASH,
    # soft plastic & film
    "plastic bag": TRASH, "ziploc bag": TRASH, "plastic wrap": TRASH,
    "bubble wrap": TRASH,
    # paper that's dirty / wet (fibers too short or contaminated)
    "napkin": TRASH, "paper napkin": TRASH, "tissue": TRASH, "paper towel": TRASH,
    "wet wipe": TRASH, "paper plate": TRASH, "paper cup": TRASH,
    "coffee cup": TRASH, "pizza box": TRASH, "receipt": TRASH,
    # styrofoam
    "styrofoam cup": TRASH, "styrofoam container": TRASH, "takeout container": TRASH,
    "foam tray": TRASH,
    # small plastics
    "plastic straw": TRASH, "plastic fork": TRASH, "plastic spoon": TRASH,
    "plastic knife": TRASH, "plastic cutlery": TRASH, "plastic lid": TRASH,
    "bottle cap": TRASH,
    # party items
    "balloon": TRASH, "party balloon": TRASH, "foil balloon": TRASH,
    "ribbon": TRASH, "confetti": TRASH, "party hat": TRASH,
    # personal / misc
    "face mask": TRASH, "rubber glove": TRASH, "sponge": TRASH,
    "cotton swab": TRASH, "chewing gum": TRASH, "cigarette butt": TRASH,
    "popsicle stick": TRASH, "toothpick": TRASH,
    # food
    "banana": TRASH, "banana peel": TRASH, "apple": TRASH, "apple core": TRASH,
    "orange": TRASH, "orange peel": TRASH, "sandwich": TRASH, "pizza": TRASH,
    "pizza slice": TRASH, "bread": TRASH, "french fries": TRASH, "food": TRASH,
}


# ---------------------------------------------------------------------------
# DISTRACTOR classes (accuracy trick for open-vocabulary detectors):
# YOLOE also looks for these, but every hit is IGNORED. Without them, a hand,
# face or T-shirt has no better match than e.g. "napkin" or "plastic bag" and
# gets mislabelled as waste. Giving the model a correct "home" for common
# non-waste things soaks up those false positives.
# ---------------------------------------------------------------------------
DISTRACTOR_CLASSES = [
    "person", "hand", "finger", "face", "arm", "hair",
    "t-shirt", "shirt", "hoodie", "sleeve", "jacket",
    "cell phone", "laptop", "keyboard", "computer mouse", "monitor", "cable",
    "watch", "glasses", "headphones", "pen", "keys", "wallet",
    "table", "chair", "wall", "shelf", "trash can",
]


# ---------------------------------------------------------------------------
# "coco" mode: classic closed-set YOLO (e.g. YOLOv8x-seg). It only knows the 80
# COCO classes, so only these waste-relevant ones are used. NOTE: COCO has no
# can, paper, cardboard, chip bag or wrapper classes.
# ---------------------------------------------------------------------------
COCO_CLASSES = {
    "bottle": RECYCLE, "wine glass": RECYCLE, "book": RECYCLE,
    "cup": TRASH,                                    # usually coffee cups (lined, not recyclable)
    "bowl": TRASH, "fork": TRASH, "knife": TRASH, "spoon": TRASH, "toothbrush": TRASH,
    "banana": TRASH, "apple": TRASH, "orange": TRASH, "sandwich": TRASH,
    "broccoli": TRASH, "carrot": TRASH, "hot dog": TRASH, "pizza": TRASH,
    "donut": TRASH, "cake": TRASH,
}


# ---------------------------------------------------------------------------
# Camera thread: always holds ONLY the newest frame (latest-wins, no FIFO),
# so a slow model frame never makes us process old, stale video.
# ---------------------------------------------------------------------------
class LatestFrameCamera:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {index}. Try --camera 1")
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


# ---------------------------------------------------------------------------
# YOLOE detection -> flat detection dicts
# ---------------------------------------------------------------------------
def detect(model, frame, conf, device, show_all=False, mode="items", classes=None):
    h, w = frame.shape[:2]
    r = model.predict(frame, conf=conf, device=device, classes=classes, verbose=False)[0]
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    seen = []
    for i, (c, s, box) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)):
        label = r.names[int(c)]
        if mode == "items":
            cat = ITEM_CLASSES.get(label)
        elif mode == "coco":
            cat = COCO_CLASSES.get(label)
        else:
            cat = categorize(label)
        seen.append(f"{label} {float(s):.2f}{'' if cat is not None else ' (ignored)'}")
        if cat is None:
            continue
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        if masks is not None:
            m = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m = np.zeros((h, w), bool)
            m[max(y1, 0):y2, max(x1, 0):x2] = True
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        dets.append({"prompt": label, "cat": cat, "score": float(s), "box": (x1, y1, x2, y2),
                     "mask": m, "cx": xs.mean() / w, "cy": ys.mean() / h})
    if show_all and seen:
        print("   sees:", ", ".join(seen))
    return dets


# ---------------------------------------------------------------------------
# RF-DETR (Roboflow, transformer detector). Pre-trained on the same 80 COCO
# classes as YOLOv8, so it uses COCO_CLASSES too. Returns the same dicts.
# ---------------------------------------------------------------------------
def load_rfdetr(size):
    try:
        import rfdetr
    except ImportError:
        raise SystemExit("RF-DETR not installed: pip install rfdetr")
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES as names
    except ImportError:
        from rfdetr.util.coco_classes import COCO_CLASSES as names
    cls_name = {"n": "RFDETRSegNano", "s": "RFDETRSegSmall", "m": "RFDETRSegMedium",
                "l": "RFDETRSegLarge", "x": "RFDETRSegXLarge"}[size]
    if not hasattr(rfdetr, cls_name):                 # fall back if this size isn't in your version
        print(f"{cls_name} not available, using RFDETRSegMedium")
        cls_name = "RFDETRSegMedium"
    model = getattr(rfdetr, cls_name)()               # downloads weights on first run
    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()            # faster repeated predictions
        except Exception:
            pass
    return model, names, cls_name


def detect_rfdetr(model, names, frame, conf, show_all=False):
    from PIL import Image
    h, w = frame.shape[:2]
    d = model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=conf)
    dets, seen = [], []
    for i in range(len(d.class_id)):
        cid = int(d.class_id[i])
        label = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
        s = float(d.confidence[i])
        cat = COCO_CLASSES.get(label)
        seen.append(f"{label} {s:.2f}{'' if cat is not None else ' (ignored)'}")
        if cat is None:
            continue
        x1, y1, x2, y2 = np.asarray(d.xyxy[i]).astype(int)
        if getattr(d, "mask", None) is not None:
            m = np.asarray(d.mask[i]).astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m = np.zeros((h, w), bool)
            m[max(y1, 0):y2, max(x1, 0):x2] = True
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        dets.append({"prompt": label, "cat": cat, "score": s, "box": (x1, y1, x2, y2),
                     "mask": m, "cx": xs.mean() / w, "cy": ys.mean() / h})
    if show_all and seen:
        print("   sees:", ", ".join(seen))
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
        return None


# ---------------------------------------------------------------------------
# Decider: votes on the CATEGORY (stable) rather than the label (flickers).
# ---------------------------------------------------------------------------
class Decider:
    def __init__(self, window=8, need=5, unsure=0.35):
        self.votes = deque(maxlen=window)          # top scan-zone detection per frame (or None)
        self.need, self.unsure = need, unsure

    def reset(self):
        self.votes.clear()

    def update(self, scan_dets):
        self.votes.append(max(scan_dets, key=lambda d: d["score"]) if scan_dets else None)

    def progress(self):
        cats = [v["cat"] for v in self.votes if v]
        if not cats:
            return None, 0
        cat, n = Counter(cats).most_common(1)[0]
        labels = [v["prompt"] for v in self.votes if v and v["cat"] == cat]
        return Counter(labels).most_common(1)[0][0], n

    def decision(self):
        label, count = self.progress()
        if label is None or count < self.need:
            return None
        cat = Counter(v["cat"] for v in self.votes if v).most_common(1)[0][0]
        winners = [v for v in self.votes if v and v["cat"] == cat]
        conf = float(np.mean([v["score"] for v in winners]))
        reason = ""
        if cat == RECYCLE and conf < self.unsure:  # unsure -> trash (protect the recycling stream)
            cat, reason = TRASH, "unsure"
        return {"label": label, "cat": cat, "conf": conf, "reason": reason,
                "pos": (winners[-1]["cx"], winners[-1]["cy"])}


# ---------------------------------------------------------------------------
# DropTracker: follows the decided item and reports which bin it went into.
# Follows the detection NEAREST to where the item was last seen, so items
# already sitting in a bin aren't mistaken for the new one.
# ---------------------------------------------------------------------------
class DropTracker:
    BINS = ("recycle", "trash")

    def __init__(self, pos, confirm=2, lost=5, max_jump=0.3):
        self.pos = pos
        self.confirm, self.lost, self.max_jump = confirm, lost, max_jump
        self.recent = deque(maxlen=confirm)
        self.last_zone = "scan"
        self.missing = 0

    def update(self, dets, zones):
        best = None
        if dets:
            near = min(dets, key=lambda d: np.hypot(d["cx"] - self.pos[0], d["cy"] - self.pos[1]))
            if np.hypot(near["cx"] - self.pos[0], near["cy"] - self.pos[1]) <= self.max_jump:
                best = near
        if best is not None:
            self.pos = (best["cx"], best["cy"])
            self.missing = 0
            z = zones.zone(best["cx"])
            if z:
                self.last_zone = z
            self.recent.append(z)
            if z in self.BINS and len(self.recent) == self.confirm and all(r == z for r in self.recent):
                return z
        else:
            self.missing += 1
            if self.missing >= self.lost and self.last_zone in self.BINS:
                return self.last_zone
        return None


# ---------------------------------------------------------------------------
# Serial link to the Nucleo
#   - find_stlink_port(): the Nucleo appears as "STMicroelectronics STLink Virtual COM Port"
#   - McuLink: background thread resends the CURRENT state every 0.2 s (heartbeat,
#     self-healing) plus one-off events, and prints anything the board sends back.
# ---------------------------------------------------------------------------
def find_stlink_port():
    try:
        from serial.tools import list_ports
    except ImportError:
        raise SystemExit("pyserial missing: pip install pyserial  (NOT 'serial')")
    for p in list_ports.comports():
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "stlink" in text or "st-link" in text or "stmicro" in text:
            return p.device
    return None


class McuLink:
    def __init__(self, port=None, baud=115200, period=0.2, idle_msg="<H>"):
        self.ser = None
        if port:
            import serial
            try:
                self.ser = serial.Serial(port, baud, timeout=0)   # timeout=0: reads never block
            except serial.SerialException as e:
                raise SystemExit(f"Could not open {port}: {e}\n"
                                 "Close the Arduino Serial Monitor (only one program can use the port).")
            time.sleep(2)
        self.state_msg, self.events = idle_msg, []
        self.period, self.lock, self.running, self._printed = period, threading.Lock(), True, None
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, event=None, state=None):
        """Event + state change queued ATOMICALLY so a stale <S> can't follow its <R>."""
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
                    try:
                        self.ser.write(m.encode())
                    except Exception as e:
                        print("Serial write failed (board unplugged?):", e)
                        self.ser = None
                if m != self._printed:
                    print("->", m.strip())
                    self._printed = m
            if self.ser:
                try:
                    rx += self.ser.read(256)
                except Exception:
                    rx = b""
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
    """Labels go inside <...,label>: no commas/brackets; the LCD has 12 chars for it."""
    return re.sub(r"[^A-Za-z0-9 ]", "", s)[:12] or "item"


# ---------------------------------------------------------------------------
# Message formats. Pick with --protocol:
#   framed (default) -> for firmware/recycle_station/recycle_station.ino
#       <S,1,71,soda can>   <R,1>   <H>   <Z>
#   simple           -> one plain word per line, easiest to handle on any board:
#       RECYCLE   TRASH   (repeated every 0.2 s while guiding)
#       CORRECT   WRONG   (once, after the drop)
#       IDLE              (heartbeat, nothing to guide)
#       RESET             (stats reset)
#     On the board: String s = Serial.readStringUntil('\n'); s.trim(); if (s == "RECYCLE") ...
# ---------------------------------------------------------------------------
class Proto:
    def __init__(self, kind="framed"):
        self.kind = kind

    def guide(self, cat, conf, label):
        if self.kind == "simple":
            return f"{CAT_NAME[cat]}\n"
        return f"<S,{cat},{int(conf * 100)},{clean_label(label)}>"

    def result(self, ok):
        return ("CORRECT\n" if ok else "WRONG\n") if self.kind == "simple" else f"<R,{int(ok)}>"

    def idle(self):
        return "IDLE\n" if self.kind == "simple" else "<H>"

    def reset(self):
        return "RESET\n" if self.kind == "simple" else "<Z>"


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def draw(frame, zones, dets, state, decision, feedback, decider, stats, fps, msg, board):
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
        cv2.putText(vis, f"{CAT_NAME[d['cat']]}: {d['prompt']} {d['score']:.0%}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

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

    n = stats["sorted"]
    acc = 100 * stats["correct"] / n if n else 0
    rec = 100 * stats["recycled"] / n if n else 0
    cv2.putText(vis, f"Sorted {n}   Accuracy {acc:.0f}%   Recycled {rec:.0f}%", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, f"{fps:.1f} FPS", (w - 100, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, f"{board} -> {msg.strip()}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="items", choices=["items", "pf", "coco", "rfdetr"],
                    help="items = YOLOE + waste list (default); pf = YOLOE full vocabulary, filtered; "
                         "coco = classic YOLO (e.g. YOLOv8x); rfdetr = Roboflow RF-DETR-Seg (COCO classes)")
    ap.add_argument("--weights", default="yolov8x-seg.pt", help="weights for --mode coco")
    ap.add_argument("--size", default="s", choices=list("nsmlx"), help="YOLOE size: n fastest ... x best")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25, help="min YOLOE confidence")
    ap.add_argument("--unsure", type=float, default=0.35, help="RECYCLE below this avg confidence -> TRASH")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--need", type=int, default=5)
    ap.add_argument("--bin-width", type=float, default=0.25)
    ap.add_argument("--scan-width", type=float, default=0.4)
    ap.add_argument("--swap-bins", action="store_true", help="recycle bin on the RIGHT")
    ap.add_argument("--guide-timeout", type=float, default=8.0)
    ap.add_argument("--show-all", action="store_true", help="print every label YOLOE sees")
    ap.add_argument("--no-distractors", action="store_true",
                    help="don't add the ignored person/hand/shirt/phone classes (items mode)")
    ap.add_argument("--port", default="auto", help="COMx, or 'auto' to find the Nucleo (default)")
    ap.add_argument("--no-serial", action="store_true", help="run without the board")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--protocol", default="framed", choices=["framed", "simple"],
                    help="framed = <S,1,71,can> for recycle_station.ino; simple = RECYCLE/TRASH words")
    args = ap.parse_args()
    proto = Proto(args.protocol)

    # ---- board ----
    port = None
    if not args.no_serial:
        port = find_stlink_port() if args.port == "auto" else args.port
        if port:
            print(f"Nucleo on {port} @ {args.baud} baud")
        else:
            print("No Nucleo found (is it plugged in?). Running laptop-only. Use --port COMx to force.")
    board = f"Nucleo {port}" if port else "no board"

    # ---- model ----
    device = 0 if torch.cuda.is_available() else "cpu"
    class_ids = None
    rf_names = None
    if args.mode == "rfdetr":
        model, rf_names, rf_name = load_rfdetr(args.size)
        print(f"{rf_name}: using COCO classes", ", ".join(COCO_CLASSES))
    elif args.mode == "coco":
        from ultralytics import YOLO
        model = YOLO(args.weights)                               # downloads on first run
        class_ids = [i for i, n in model.names.items() if n in COCO_CLASSES]
        print(f"{args.weights}: using COCO classes", ", ".join(model.names[i] for i in class_ids))
    elif args.mode == "items":
        model = YOLOE(f"yoloe-26{args.size}-seg.pt")             # downloads on first run
        classes = list(ITEM_CLASSES)
        if not args.no_distractors:
            classes += [c for c in DISTRACTOR_CLASSES if c not in ITEM_CLASSES]
        model.set_classes(classes)                               # waste items (+ ignored distractors)
        print(f"Looking for {len(ITEM_CLASSES)} waste items"
              + ("" if args.no_distractors else f" + {len(classes) - len(ITEM_CLASSES)} ignored distractors"))
    else:
        model = YOLOE(f"yoloe-26{args.size}-seg-pf.pt")
        print("Prompt-free: full vocabulary, filtered to recycle/trash words")
    model_name = {"coco": args.weights, "rfdetr": "RF-DETR-Seg"}.get(args.mode, f"YOLOE-26{args.size}")
    print(model_name, "on", torch.cuda.get_device_name(0) if device == 0 else "CPU (slow!)")

    link = McuLink(port, args.baud, idle_msg=proto.idle())
    print("Serial protocol:", args.protocol)
    cam = LatestFrameCamera(args.camera)
    zones = Zones(args.bin_width, args.scan_width, args.swap_bins)
    decider = Decider(args.window, args.need, args.unsure)

    state, t_state = "IDLE", time.time()
    decision, tracker, feedback = None, None, None
    clear_frames = 0
    stats = {"sorted": 0, "correct": 0, "recycled": 0}
    t_prev, fps, msg = time.time(), 0.0, proto.idle()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if args.mode == "rfdetr":
                dets = detect_rfdetr(model, rf_names, frame, args.conf, args.show_all)
            else:
                dets = detect(model, frame, args.conf, device, args.show_all, args.mode, class_ids)
            scan_dets = [d for d in dets if zones.zone(d["cx"]) == "scan"]
            now = time.time()

            # ------------------------- state machine -------------------------
            if state == "IDLE":
                decider.update(scan_dets)
                decision = decider.decision()
                if decision:
                    tracker = DropTracker(decision["pos"])
                    msg = proto.guide(decision["cat"], decision["conf"], decision["label"])
                    print(f"DECISION: {CAT_NAME[decision['cat']]}  ({decision['label']}, "
                          f"{decision['conf']:.0%}{', ' + decision['reason'] if decision['reason'] else ''})")
                    link.send(state=msg)                        # repeats every 0.2 s while guiding
                    state, t_state = "GUIDING", now

            elif state == "GUIDING":
                chosen = tracker.update(dets, zones)
                if chosen:
                    ok_bin = ZONE_CAT[chosen] == decision["cat"]
                    stats["sorted"] += 1
                    stats["correct"] += ok_bin
                    stats["recycled"] += ok_bin and decision["cat"] == RECYCLE
                    msg = proto.result(ok_bin)
                    link.send(event=msg, state=proto.idle())    # result once, then back to heartbeat
                    feedback = (ok_bin, chosen)
                    state, t_state = "FEEDBACK", now
                elif now - t_state > args.guide_timeout:
                    msg = proto.idle()                          # item taken away: give up quietly
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
                       draw(frame, zones, dets, state, decision, feedback, decider, stats, fps, msg, board))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                stats = {"sorted": 0, "correct": 0, "recycled": 0}
                link.send(event=proto.reset())
    finally:
        link.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()