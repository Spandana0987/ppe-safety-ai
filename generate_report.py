import json
import os
from collections import defaultdict

# Load detection log
try:
    with open("output/report_data.json") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: output/report_data.json not found. Run detect_video.py first.")
    exit(1)

# Class IDs (for reference/filters)
MACHINE = 8
VEHICLE = 9

# Storage: Track frames per worker for temporal smoothing
worker_stats = defaultdict(lambda: {
    "total_frames": 0, 
    "helmet_count": 0, 
    "vest_count": 0, 
    "evidence": []
})
machines = set()
vehicles = set()

for r in data:
    # Handle People (Worker records have 'worker_id')
    if "worker_id" in r:
        w_id = r["worker_id"]
        if w_id == -1: continue
        
        worker_stats[w_id]["total_frames"] += 1
        if r["has_helmet"]:
            worker_stats[w_id]["helmet_count"] += 1
        if r["has_vest"]:
            worker_stats[w_id]["vest_count"] += 1
            
        if "evidence" in r:
            worker_stats[w_id]["evidence"].append(r["evidence"])

    # Handle Others (id and class) - Use sets for unique counts
    elif "id" in r:
        obj_id = r["id"]
        cls = r["class"]
        if cls == MACHINE:
            machines.add(obj_id)
        elif cls == VEHICLE:
            vehicles.add(obj_id)

total_workers = len(worker_stats)
# The data loading logic will be moved inside generate_summary
DATA_FILE = "output/report_data.json"
REPORT_FILE = "output/summary_report.json"

# Class IDs (for reference/filters) - These are now handled within generate_summary
# MACHINE = 8
# VEHICLE = 9

# Storage: Track frames per worker for temporal smoothing - These are now handled within generate_summary
# worker_stats = defaultdict(lambda: {
#     "total_frames": 0,
#     "helmet_count": 0,
#     "vest_count": 0,
#     "evidence": []
# })
# machines = set()
# vehicles = set()

# The main processing loop is replaced by the generate_summary function.
# for r in data:
#     # Handle People (Worker records have 'worker_id')
#     if "worker_id" in r:
#         w_id = r["worker_id"]
#         if w_id == -1: continue
        
#         worker_stats[w_id]["total_frames"] += 1
#         if r["has_helmet"]:
#             worker_stats[w_id]["helmet_count"] += 1
#         if r["has_vest"]:
#             worker_stats[w_id]["vest_count"] += 1
            
#         if "evidence" in r:
#             worker_stats[w_id]["evidence"].append(r["evidence"])

#     # Handle Others (id and class) - Use sets for unique counts
#     elif "id" in r:
#         obj_id = r["id"]
#         cls = r["class"]
#         if cls == MACHINE:
#             machines.add(obj_id)
#         elif cls == VEHICLE:
#             vehicles.add(obj_id)

# total_workers = len(worker_stats)
# helmet_violations = 0
# vest_violations = 0
# compliant_workers = 0
# violation_details = []

# Temporal Smoothing Threshold: Confirm violation if gear presence ratio < 60%
# This makes the system more "forgiving" of momentary detection# Analysis Config
CONFIRMATION_THRESHOLD = 0.6  # PPE must be present in 60% of tracked frames
MIN_FRAMES = 30               # Ignore short-lived tracking IDs (phantoms)
MERGE_WINDOW = 60             # Frames to look for merging opportunities
MERGE_DIST = 100              # Pixel distance for centroid-based merging

def generate_summary():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # raw_stats: Initial aggregation by track_id
    raw_stats = {}
    other_counts = {8: 0, 9: 0} # 8: machine, 9: vehicle
    processed_others = set()

    for r in data:
        if "worker_id" in r:
            w_id = r["worker_id"]
            if w_id not in raw_stats:
                raw_stats[w_id] = {
                    "frames": [],
                    "helmet_hits": 0,
                    "vest_hits": 0,
                    "centroids": [],
                    "evidence": {} # v_type -> path
                }
            
            raw_stats[w_id]["frames"].append(r["frame"])
            if r.get("has_helmet"): raw_stats[w_id]["helmet_hits"] += 1
            if r.get("has_vest"): raw_stats[w_id]["vest_hits"] += 1
            
            if "p_box" in r:
                box = r["p_box"]
                centroid = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                raw_stats[w_id]["centroids"].append(centroid)
            
            if "evidence" in r:
                # Evidence is already 'best-of' from detect_video.py
                v_type = "no_helmet" if "no_helmet" in r["evidence"] else "no_vest"
                raw_stats[w_id]["evidence"][v_type] = r["evidence"]

        elif "id" in r and "class" in r:
            obj_key = (r["id"], r["class"])
            if obj_key not in processed_others:
                if r["class"] in other_counts:
                    other_counts[r["class"]] += 1
                    processed_others.add(obj_key)

    # IDENTITY HEALING: Merge fragmented tracks
    sorted_ids = sorted(raw_stats.keys(), key=lambda x: min(raw_stats[x]["frames"]))
    merged_map = {i: i for i in sorted_ids} # id -> master_id
    
    for i in range(len(sorted_ids)):
        id_a = sorted_ids[i]
        last_a = max(raw_stats[id_a]["frames"])
        cent_a = raw_stats[id_a]["centroids"][-1] if raw_stats[id_a]["centroids"] else [0,0]
        
        for j in range(i + 1, len(sorted_ids)):
            id_b = sorted_ids[j]
            first_b = min(raw_stats[id_b]["frames"])
            cent_b = raw_stats[id_b]["centroids"][0] if raw_stats[id_b]["centroids"] else [0,0]
            
            # Distance Check
            dist = ((cent_a[0]-cent_b[0])**2 + (cent_a[1]-cent_b[1])**2)**0.5
            gap = first_b - last_a
            
            if 0 < gap < MERGE_WINDOW and dist < MERGE_DIST:
                # Merge id_b into id_a
                target = merged_map[id_a]
                merged_map[id_b] = target
                # Update cent_a for next chain
                last_a = max(raw_stats[id_b]["frames"])
                cent_a = raw_stats[id_b]["centroids"][-1] if raw_stats[id_b]["centroids"] else cent_a

    # Final Aggregation by Master ID
    final_registry = {}
    for orig_id, master_id in merged_map.items():
        if master_id not in final_registry:
            final_registry[master_id] = {
                "total_frames": 0, "h_hits": 0, "v_hits": 0, "evidence": {}
            }
        
        stats = raw_stats[orig_id]
        reg = final_registry[master_id]
        reg["total_frames"] += len(stats["frames"])
        reg["h_hits"] += stats["helmet_hits"]
        reg["v_hits"] += stats["vest_hits"]
        reg["evidence"].update(stats["evidence"])

    # Pass 2: Filter and Score
    valid_workers = []
    violation_log = []
    h_vios = 0
    v_vios = 0
    
    for m_id, reg in final_registry.items():
        total = reg["total_frames"]
        if total < MIN_FRAMES: continue
        
        h_ok = (reg["h_hits"] / total) >= CONFIRMATION_THRESHOLD
        v_ok = (reg["v_hits"] / total) >= CONFIRMATION_THRESHOLD
        safe = h_ok and v_ok
        
        vios = []
        if not h_ok: 
            vios.append("No Helmet"); h_vios += 1
        if not v_ok: 
            vios.append("No Vest"); v_vios += 1
            
        valid_workers.append({
            "worker_id": m_id,
            "helmet": "✅" if h_ok else "❌",
            "vest": "✅" if v_ok else "❌",
            "status": "SAFE" if safe else "VIOLATION"
        })
        
        if not safe:
            violation_log.append({
                "worker_id": m_id,
                "violations": vios,
                "evidence": list(reg["evidence"].values())
            })

    total_count = len(valid_workers)
    safe_count = total_count - len(violation_log)
    safety_score = round((safe_count / total_count * 100), 2) if total_count > 0 else 100.0

    report = {
        "metrics": {
            "total_workers": total_count,
            "machines_detected": other_counts[8],
            "vehicles_detected": other_counts[9],
            "helmet_violations": h_vios,
            "vest_violations": v_vios,
            "compliance_rate": safety_score,
            "safety_score": safety_score
        },
        "violation_log": violation_log,
        "full_worker_list": valid_workers
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Report Generated | Valid Workers: {total_count} | Safety Score: {safety_score}%")
    print(json.dumps(report["metrics"], indent=4))

generate_summary()
