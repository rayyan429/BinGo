import argparse
import time

import serial
from serial.tools import list_ports

MESSAGES = {
    "simple": {"recycle": "RECYCLE\n", "trash": "TRASH\n", "ok": "CORRECT\n",
               "bad": "WRONG\n", "idle": "IDLE\n"},
    "framed": {"recycle": "<S,1,92,soda can>", "trash": "<S,0,81,chip bag>", "ok": "<R,1>",
               "bad": "<R,0>", "idle": "<H>"},
}


def find_stlink_port():
    """The Nucleo shows up as 'STMicroelectronics STLink Virtual COM Port (COMx)'."""
    for p in list_ports.comports():
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "stlink" in text or "st-link" in text or "stmicro" in text:
            return p.device
    return None


def send_for(ser, msg, seconds, period=0.2):
    """Resend every 0.2 s, exactly like the real program does (this is the heartbeat)."""
    print("->", msg.strip())
    end = time.time() + seconds
    while time.time() < end:
        ser.write(msg.encode())
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="e.g. COM5 (default: auto-detect the Nucleo)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--protocol", default="three", choices=["three"] + list(MESSAGES),
                    help="three = PLASTIC/PAPER/TRASH for three_bins.ino (default); "
                         "simple = RECYCLE/TRASH for simple_bins.ino; framed = LCD recycle_station.ino")
    args = ap.parse_args()
    m = MESSAGES.get(args.protocol, MESSAGES["simple"])

    port = args.port or find_stlink_port()
    if not port:
        print("No Nucleo (ST-Link) port found. Ports on this laptop:")
        for p in list_ports.comports():
            print("  ", p.device, "-", p.description)
        raise SystemExit("Plug in the Nucleo, or pass --port COMx")

    print(f"Opening {port} at {args.baud} baud ({args.protocol} protocol) ...")
    try:
        ser = serial.Serial(port, args.baud, timeout=0)
    except serial.SerialException as e:
        raise SystemExit(f"Could not open {port}: {e}\nClose the Arduino Serial Monitor and try again.")
    time.sleep(1.5)                                   # some boards reset when the port opens

    if args.protocol == "three":                     # firmware/three_bins/three_bins.ino
        for tag in ("IDLE", "PLASTIC", "PAPER", "TRASH", "IDLE"):
            send_for(ser, f"{tag}\n", 3)
        print("Stopping: in about 1.5 s the LCD should say 'Laptop offline'.")
        ser.close()
        return

    send_for(ser, m["idle"], 1)
    send_for(ser, m["recycle"], 3)
    ser.write(m["ok"].encode()); print("->", m["ok"].strip())
    send_for(ser, m["idle"], 2)
    send_for(ser, m["trash"], 3)
    ser.write(m["bad"].encode()); print("->", m["bad"].strip())
    send_for(ser, m["idle"], 2)

    print("Stopping: in about 1.5 s the board should show its offline signal.")
    ser.close()


if __name__ == "__main__":
    main()