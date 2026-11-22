import argparse
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

# Reuse the camera, serial link, port finder and distractor list from the main station
from sorter_yoloe import DISTRACTOR_CLASSES, LatestFrameCamera, McuLink, find_stlink_port

# ---------------------------------------------------------------------------
# Categories: tag sent to the board, text shown on screen, overlay colour (BGR)
# ---------------------------------------------------------------------------
PLASTIC, PAPER, TRASH = "PLASTIC", "PAPER", "TRASH"
DISPLAY = {PLASTIC: "Plastic&Aluminum", PAPER: "Paper&Cardboard", TRASH: "Trash"}
COLOR = {PLASTIC: (0, 200, 0), PAPER: (255, 255, 255), TRASH: (0, 0, 255)}

# ---------------------------------------------------------------------------
# Item -> category. Synonyms are deliberate: the vote is on the CATEGORY, so
# "chip bag" / "chips packet" never conflict. Bin rules vary by city; move any
# item to another category to match yours.
# ---------------------------------------------------------------------------
ITEMS = {
    # ===== PLASTIC & ALUMINUM (green) =====
    # plastic bottles & containers
    "plastic bottle": PLASTIC, "water bottle": PLASTIC, "soda bottle": PLASTIC,
    "juice bottle": PLASTIC, "milk jug": PLASTIC, "plastic jug": PLASTIC,
    "detergent bottle": PLASTIC, "shampoo bottle": PLASTIC, "spray bottle": PLASTIC,
    "plastic container": PLASTIC, "yogurt cup": PLASTIC, "plastic tub": PLASTIC,
    # aluminum & metal
    "soda can": PLASTIC, "aluminum can": PLASTIC, "beer can": PLASTIC,
    "tin can": PLASTIC, "food can": PLASTIC, "aluminum foil": PLASTIC,
    "aluminum tray": PLASTIC,
    # drink cartons usually go with containers
    "milk carton": PLASTIC, "juice box": PLASTIC, "juice carton": PLASTIC,
    # glass: no glass category, so it shares the containers bin (change if your bins differ)
    "glass bottle": PLASTIC, "glass jar": PLASTIC, "beer bottle": PLASTIC,
    "wine bottle": PLASTIC,

    # ===== PAPER & CARDBOARD (white) =====
    # paper
    "paper": PAPER, "printer paper": PAPER, "notebook": PAPER, "newspaper": PAPER,
    "magazine": PAPER, "flyer": PAPER, "envelope": PAPER, "paper bag": PAPER,
    "sticky note": PAPER, "book": PAPER,
    # cardboard (all cardboard is recycled with paper)
    "cardboard": PAPER, "cardboard box": PAPER, "brown cardboard box": PAPER,
    "corrugated cardboard": PAPER, "cardboard sheet": PAPER, "cardboard piece": PAPER,
    "shipping box": PAPER, "delivery box": PAPER, "packing box": PAPER,
    "amazon box": PAPER, "pizza box": PAPER, "shoe box": PAPER, "cereal box": PAPER,
    "cracker box": PAPER, "tissue box": PAPER, "paperboard": PAPER,
    "cardboard tube": PAPER, "toilet paper roll": PAPER, "paper towel roll": PAPER,
    "egg carton": PAPER, "cardboard sleeve": PAPER, "cardboard cup holder": PAPER,

    # ===== TRASH (red) =====
    # chip packets & wrappers (multi-layer plastic/foil)
    "chip bag": TRASH, "chips packet": TRASH, "crisp packet": TRASH,
    "potato chip bag": TRASH, "snack bag": TRASH, "candy wrapper": TRASH,
    "chocolate wrapper": TRASH, "snack wrapper": TRASH, "granola bar wrapper": TRASH,
    "food wrapper": TRASH,
    # soft plastic & film
    "plastic bag": TRASH, "ziploc bag": TRASH, "plastic wrap": TRASH, "bubble wrap": TRASH,
    # dirty / wet paper (can't be recycled)
    "napkin": TRASH, "paper napkin": TRASH, "tissue": TRASH, "paper towel": TRASH,
    "wet wipe": TRASH, "paper plate": TRASH, "paper cup": TRASH, "coffee cup": TRASH,
    "receipt": TRASH,
    # styrofoam
    "styrofoam cup": TRASH, "styrofoam container": TRASH, "takeout container": TRASH,
    "foam tray": TRASH,
    # small plastics
    "plastic straw": TRASH, "plastic fork": TRASH, "plastic spoon": TRASH,
    "plastic knife": TRASH, "plastic cutlery": TRASH, "plastic lid": TRASH,
    "bottle cap": TRASH,
    # party & misc
    "balloon": TRASH, "party balloon": TRASH, "foil balloon": TRASH, "ribbon": TRASH,
    "confetti": TRASH, "face mask": TRASH, "rubber glove": TRASH, "sponge": TRASH,
    "cotton swab": TRASH, "chewing gum": TRASH, "cigarette butt": TRASH,
    "popsicle stick": TRASH, "toothpick": TRASH,
    # food
    "banana": TRASH, "banana peel": TRASH, "apple": TRASH, "apple core": TRASH,
    "orange": TRASH, "orange peel": TRASH, "sandwich": TRASH, "pizza": TRASH,
    "pizza slice": TRASH, "bread": TRASH, "french fries": TRASH, "food": TRASH,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="s", choices=list("nsmlx"), help="YOLOE size: n fastest ... x best")
    ap.add_argument("--conf", type=float, default=0.3, help="min confidence for a mask to count")
    ap.add_argument("--smooth", type=int, default=5, help="frames in the majority vote")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", default="auto", help="COMx or 'auto'")
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--show-all", action="store_true", help="print every label YOLOE sees")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    # ---- board ----
    port = None
    if not args.no_serial:
        port = find_stlink_port() if args.port == "auto" else args.port
        print(f"Nucleo on {port}" if port else "No Nucleo found, running laptop-only")
    link = McuLink(port, args.baud, idle_msg="IDLE\n")   # resends the current tag every 0.2 s

    # ---- model ----
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLOE(f"yoloe-26{args.size}-seg.pt")
    classes = list(ITEMS) + [c for c in DISTRACTOR_CLASSES if c not in ITEMS]
    model.set_classes(classes)
    counts = Counter(ITEMS.values())
    print(f"YOLOE-26{args.size}: {counts[PLASTIC]} plastic&aluminum, {counts[PAPER]} paper, "
          f"{counts[TRASH]} trash items + distractors, on",
          torch.cuda.get_device_name(0) if device == 0 else "CPU")

    cam = LatestFrameCamera(args.camera)
    history = deque(maxlen=args.smooth)
    tag, t_prev, fps = "IDLE", time.time(), 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            h, w = frame.shape[:2]
            r = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]

            # 1. Give every waste mask a category (distractors like hands are skipped)
            vis, overlay, best, seen = frame.copy(), frame.copy(), None, []
            if r.boxes is not None and len(r.boxes):
                masks = r.masks.data.cpu().numpy() if r.masks is not None else None
                for i, (c, s, box) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)):
                    label, s = r.names[int(c)], float(s)
                    cat = ITEMS.get(label)
                    seen.append(f"{label} {s:.2f}" + ("" if cat else " (ignored)"))
                    if cat is None:
                        continue
                    if masks is not None:
                        m = cv2.resize(masks[i].astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                        overlay[m] = COLOR[cat]
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR[cat], 2)
                    cv2.putText(vis, f"{DISPLAY[cat]}: {label} {s:.0%}", (x1, max(y1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR[cat], 2)
                    if best is None or s > best[1]:
                        best = (cat, s)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
            if args.show_all and seen:
                print("   sees:", ", ".join(seen))

            # 2. Strongest mask's category -> majority vote over the last few frames
            history.append(best[0] if best else "IDLE")
            new_tag = Counter(history).most_common(1)[0][0]

            # 3. Send when it changes (McuLink keeps resending it as the heartbeat)
            if new_tag != tag:
                tag = new_tag
                link.send(state=f"{tag}\n")
                print("CIRCUIT ->", tag)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            banner = DISPLAY.get(tag, "Show an item")
            cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(vis, f"LCD: {banner}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        COLOR.get(tag, (200, 200, 200)), 2)
            cv2.putText(vis, f"{fps:.0f} FPS", (w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.imshow("TIDAL waste detector", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        link.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()