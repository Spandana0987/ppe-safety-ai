import os
from ultralytics import YOLO
import cv2
import json
import numpy as np

import argparse

# Create output directories
os.makedirs("output/frames", exist_ok=True)

# CLI Arguments
parser = argparse.ArgumentParser(description="PPE Safety Detection Pipeline")
parser.add_argument("--show", action="store_true", help="Display real-time video visualization")
args = parser.parse_args()

# Load trained model
model = YOLO("models/best.pt")

# Video path
video_path = "videos/construction_demo3.mp4"
cap = cv2.VideoCapture(video_path)

records = []
frame_count = 0
STRIDE = 3 # Process every 3rd frame for speed (10fps is sufficient for tracking)

# Class Mapping Cache
def get_class_ids(model):
    # Strict Mapping for True PPE based on Ground Truth Audit
    # Helmet=0 (White/Yellow hardhats)
    # Person=5 (Human bodies)
    # Vest=7 (Safety high-vis vests)
    return {"HELMET": 0, "PERSON": 5, "VEST": 7, "MACHINE": 8, "VEHICLE": 9}

target_ids = get_class_ids(model)

# IoU Containment Helper
def calculate_overlap(p_box, gear_box):
    px1, py1, px2, py2 = p_box
    gx1, gy1, gx2, gy2 = gear_box
    
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    
    intersection_area = iw * ih
    gear_area = (gx2 - gx1) * (gy2 - gy1)
    return intersection_area / gear_area if gear_area > 0 else 0

# Global Registry for Evidence (Best Image Selection)
worker_evidence_counts = {} # w_key -> highest confidence seen

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    if frame_count % STRIDE != 0: continue

    # EXECUTE TRACKING (ByteTrack)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.3)[0]

    if results.boxes is None or results.boxes.id is None:
        continue

    # Extract detection attributes
    boxes = results.boxes.xyxy.cpu().numpy()
    ids = results.boxes.id.cpu().numpy().astype(int)
    clss = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    people = []
    others = []

    for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
        if cls == target_ids["PERSON"]:
            people.append({"id": obj_id, "box": box, "conf": conf})
        else:
            others.append({"class": cls, "box": box, "id": obj_id, "conf": conf})

    # Associate Gear with Workers
    for p in people:
        p_id = p["id"]
        px1, py1, px2, py2 = p["box"]
        
        has_helmet = False
        has_vest = False
        
        for obj in others:
            if obj["class"] in [target_ids["HELMET"], target_ids["VEST"]]:
                if calculate_overlap(p["box"], obj["box"]) > 0.3:
                    if obj["class"] == target_ids["HELMET"]: has_helmet = True
                    if obj["class"] == target_ids["VEST"]: has_vest = True

        status = []
        if not has_helmet: status.append("no_helmet")
        if not has_vest: status.append("no_vest")
        
        # VISUALIZATION (Optional)
        if args.show:
            color = (0, 255, 0) if (has_helmet and has_vest) else (0, 0, 255)
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)
            
            label = f"ID:{p_id} | H:{'OK' if has_helmet else 'NO'} | V:{'OK' if has_vest else 'NO'}"
            cv2.putText(frame, label, (int(px1), int(py1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Log entry for later aggregation
        record = {
            "frame": frame_count,
            "worker_id": int(p_id),
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "p_box": [float(x) for x in p["box"]]
        }

        # BEST IMAGE EVIDENCE (One high-quality crop per worker-violation)
        if status:
            v_type = "multi" if len(status) > 1 else status[0]
            w_key = f"{p_id}_{v_type}"
            
            # If this detection is higher confidence than previous best for this worker
            if p["conf"] > worker_evidence_counts.get(w_key, 0.0):
                img_path = f"output/frames/worker_{p_id}_{v_type}.jpg"
                cv2.imwrite(img_path, frame[int(py1):int(py2), int(px1):int(px2)])
                record["evidence"] = img_path
                worker_evidence_counts[w_key] = p["conf"]

        records.append(record)

    # Log Machines/Vehicles
    for obj in others:
        if obj["class"] in [target_ids["MACHINE"], target_ids["VEHICLE"]]:
            records.append({
                "frame": frame_count,
                "id": int(obj["id"]),
                "class": int(obj["class"]),
                "conf": float(obj["conf"])
            })

    print(f"Processed frame {frame_count} | Targets: {len(people)}", end="\r")
    
    # DISPLAY (Optional)
    if args.show:
        cv2.imshow("PPE Safety Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nVisualization terminated by user.")
            break

cap.release()
cv2.destroyAllWindows()

# Final Data Commit
with open("output/report_data.json", "w") as f:
    json.dump(records, f, indent=4)

print(f"\nFinalized V6 Dataset: {len(records)} records saved.")
