from ultralytics import YOLO
import sys

try:
    model = YOLO('models/best.pt')
    # Use stream=True to get a generator
    results = model.predict('videos/construction_demo3.mp4', stream=True, verbose=False)
    
    print("Class ID Diagnostics (V4):")
    print("-" * 50)
    for i, res in enumerate(results):
        if i % 30 != 0: continue # Check every 30th frame
        if i > 90: break 
        print(f"\nFrame {i}:")
        for box in res.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xywh = box.xywh[0]
            w, h = int(xywh[2]), int(xywh[3])
            print(f"  Class {cls:2}: w={w:4}, h={h:4}, conf={conf:.2f}")
except Exception as e:
    print(f"Error: {e}")
