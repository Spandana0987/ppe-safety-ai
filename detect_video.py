"""
PPE Safety Detection Pipeline — Production Grade
Fixed: video not showing, codec issues, frame consistency, syntax errors
"""

import os
import cv2
import json
import argparse
import numpy as np
import time
from ultralytics import YOLO

# ── Setup ─────────────────────────────────────────────────────────────
os.makedirs("output/frames", exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--output", type=str, default="output/processed_video.mp4")
parser.add_argument("--show", action="store_true")
parser.add_argument("--save-video", action="store_true")
args = parser.parse_args()

# ── Model ─────────────────────────────────────────────────────────────
model = YOLO("models/best.pt")

CLS = {"HELMET": 0, "PERSON": 5, "VEST": 7, "MACHINE": 8, "VEHICLE": 9}

# ── Thresholds ────────────────────────────────────────────────────────
CONF_PERSON = 0.35
CONF_PPE = 0.40
CONF_TRACK = 0.35
GHOST_RATIO = 0.30
IOU_ASSIGN = 0.25
STRIDE = 3

# ── Sequential ID ─────────────────────────────────────────────────────
id_map = {}
next_id = 1

def seq_id(tid):
    global next_id
    if tid not in id_map:
        id_map[tid] = next_id
        next_id += 1
    return id_map[tid]

# ── IoU Logic ─────────────────────────────────────────────────────────
def gear_iou(p_box, g_box, g_type):
    px1, py1, px2, py2 = p_box
    gx1, gy1, gx2, gy2 = g_box

    ph = py2 - py1
    if ph <= 0:
        return 0.0

    if g_type == "HELMET":
        zy1, zy2 = py1, py1 + ph * 0.4
    else:
        zy1, zy2 = py1 + ph * 0.3, py1 + ph * 0.8

    ix1, iy1 = max(gx1, px1), max(gy1, zy1)
    ix2, iy2 = min(gx2, px2), min(gy2, zy2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    g_area = (gx2 - gx1) * (gy2 - gy1)
    if g_area <= 0:
        return 0.0

    return inter / g_area

# ── Video Setup ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🔥 FIX: FORCE SAFE RESOLUTION
max_width = 1280
scale = min(1.0, max_width / orig_w)
W, H = int(orig_w * scale), int(orig_h * scale)

FPS = 25.0  # 🔥 FIX: stable FPS

writer = None
if args.save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

# ── Processing ────────────────────────────────────────────────────────
records = []
frame_idx = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % STRIDE != 0:
        continue

    # 🔥 FIX: Resize ALWAYS
    frame = cv2.resize(frame, (W, H))

    results = model.track(frame, persist=True, verbose=False, conf=CONF_TRACK)[0]

    if results.boxes is None or results.boxes.id is None:
        if writer:
            writer.write(frame)
            frame_count += 1
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    tids = results.boxes.id.cpu().numpy().astype(int)
    clss = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    people = []
    gear = []

    for box, tid, cls, conf in zip(boxes, tids, clss, confs):
        area = (box[2]-box[0]) * (box[3]-box[1])
        if area > (W * H * GHOST_RATIO):
            continue

        if cls == CLS["PERSON"] and conf >= CONF_PERSON:
            people.append({"sid": seq_id(tid), "box": box, "conf": conf})

        elif cls in (CLS["HELMET"], CLS["VEST"]) and conf >= CONF_PPE:
            gear.append({"cls": cls, "box": box})

    person_ppe = {p["sid"]: {"helmet": False, "vest": False} for p in people}

    for g in gear:
        g_type = "HELMET" if g["cls"] == CLS["HELMET"] else "VEST"
        for p in people:
            if gear_iou(p["box"], g["box"], g_type) > IOU_ASSIGN:
                person_ppe[p["sid"]][g_type.lower()] = True

    for p in people:
        sid = p["sid"]
        h = person_ppe[sid]["helmet"]
        v = person_ppe[sid]["vest"]
        compliant = h and v

        x1, y1, x2, y2 = map(int, p["box"])
        color = (0,255,0) if compliant else (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        cv2.putText(frame, f"W{sid}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── Record for Analytics ──────────────────────────────────────
        record = {
            "frame": frame_idx,
            "worker_id": sid,
            "p_box": [x1, y1, x2, y2],
            "conf": float(p["conf"]),
            "has_helmet": h,
            "has_vest": v
        }
        
        # Capture evidence for violations (throttle to 1 image per worker per ~30 frames)
        if not compliant and frame_idx % 30 == 0:
            img_path = f"output/frames/w{sid}_f{frame_idx}.jpg"
            # Give a robust crop boundary
            crop = frame[max(0, y1-40):min(H, y2+40), max(0, x1-40):min(W, x2+40)]
            if crop.size > 0:
                cv2.imwrite(img_path, crop)
                record["evidence"] = img_path

        records.append(record)

    if writer:
        writer.write(frame)
        frame_count += 1

    if args.show:
        cv2.imshow("PPE", frame)
        if cv2.waitKey(1) == 27:
            break

print(f"\nFrames written: {frame_count}")

# ── Cleanup ───────────────────────────────────────────────────────────
cap.release()

if writer:
    writer.release()
    time.sleep(1)

    size = os.path.getsize(args.output)
    print(f"Initial Video size: {size}")

    if size < 10000:
        raise RuntimeError("Video corrupted / too small")
        
    # Transcode to h264 for Streamlit browser playback
    temp_output = args.output.replace(".mp4", "_temp.mp4")
    os.rename(args.output, temp_output)
    print("Transcoding video to h264 for browser playback...")
    os.system(f"ffmpeg -y -i {temp_output} -vcodec libx264 -acodec aac {args.output} -hide_banner -loglevel error")
    os.remove(temp_output)
    print(f"Final Video size: {os.path.getsize(args.output)}")

cv2.destroyAllWindows()

# ── Save JSON ─────────────────────────────────────────────────────────
report_dict = {
    "source": args.source,
    "records": records
}
with open("output/output_video_data.json", "w") as f:
    json.dump(report_dict, f, indent=2)

print("✅ DONE")
