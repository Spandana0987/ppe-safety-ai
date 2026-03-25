import json
import os
import subprocess

videos = [
    "construction_demo1.mp4",
    "construction_demo2.mp4",
    "construction_demo3.mp4",
    "construction_demo4.mp4",
    "construction_demo5.mp4",
    "construction_demo6.mp4"
]

results = {}

for video in videos:
    print(f"Processing {video}...")
    source = f"videos/{video}"
    if not os.path.exists(source): 
        print(f"Skipping {video} (not found)")
        continue
        
    # Run Detection
    subprocess.run([
        "venv/bin/python3", "detect_video.py", 
        "--source", source, 
        "--save-video"
    ], capture_output=True)
    
    # Run Report
    subprocess.run([
        "venv/bin/python3", "generate_report.py"
    ], capture_output=True)
    
    # Read Metric
    try:
        with open("output/summary_report.json") as f:
            data = json.load(f)
            results[video] = data["metrics"]
    except:
        results[video] = "Error"

with open("v9_generalization_matrix.json", "w") as f:
    json.dump(results, f, indent=4)

print("V9 Generalization Matrix Complete.")
