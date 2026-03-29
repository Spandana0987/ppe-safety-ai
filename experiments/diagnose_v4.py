import json
from collections import defaultdict

try:
    with open('output/report_data.json') as f:
        data = json.load(f)

    stats = defaultdict(lambda: {'frames': 0, 'h': 0, 'v': 0})
    for r in data:
        if 'worker_id' not in r: continue
        wid = r['worker_id']
        stats[wid]['frames'] += 1
        if r.get('has_helmet'): stats[wid]['h'] += 1
        if r.get('has_vest'): stats[wid]['v'] += 1

    print("Worker Diagnostics (V4):")
    print("-" * 60)
    print(f"{'Worker':<10} {'Frames':<10} {'H Ratio':<10} {'V Ratio':<10} {'Status'}")
    for wid, s in sorted(stats.items()):
        hrat = s['h']/s['frames'] if s['frames'] > 0 else 0
        vrat = s['v']/s['frames'] if s['frames'] > 0 else 0
        # Check against the 60% threshold from generate_report.py
        is_compliant = (hrat >= 0.6 and vrat >= 0.6)
        status = "✅ SAFE" if is_compliant else "❌ VIOLATION"
        print(f"{wid:<10} {s['frames']:<10} {hrat:<10.2f} {vrat:<10.2f} {status}")
except Exception as e:
    print(f"Error: {e}")
