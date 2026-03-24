import streamlit as st
import json
import os
import pandas as pd
from PIL import Image

# --- CONFIG & UI SETUP ---
st.set_page_config(page_title="PPE Safety AI: Site Monitoring", layout="wide", page_icon="👷")

# Premium Dark Theme Overrides
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.title("👷 PPE Safety Intelligence Dashboard")
st.caption("Industrial Safety Monitoring & Compliance Analytics")
st.markdown("---")

# --- LOAD DATA ---
REPORT_PATH = "output/summary_report.json"

if not os.path.exists(REPORT_PATH):
    st.error("⚠️ Summary report not found. Please run the detection pipeline first.")
    st.stop()

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

# --- SIDEBAR & FILTERS ---
st.sidebar.title("🛠️ Monitoring Controls")
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("🔎 Filter Worker View", ["All Workers", "Violations Only", "Safe Workers"])

# Bonus: Video Selection (Placeholder logic)
st.sidebar.selectbox("🎥 Source Video", ["construction_demo3.mp4", "Live Stream (RTSP)"])

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"**Site Safety Score**: {report['metrics']['safety_score']}%")
st.sidebar.progress(report['metrics']['safety_score'] / 100)

# --- TOP METRICS ---
m = report["metrics"]
c1, c2, c3, c4, c5 = st.columns(5)

# Safety Score Card
c1.metric("Site Safety Score", f"{m['safety_score']}%", delta=f"{m['safety_score']-70}%" if m['safety_score'] > 70 else "-Low")
c2.metric("Total Workers", m["total_workers"])
c3.metric("Helmet Violations", m["helmet_violations"], delta=m["helmet_violations"], delta_color="inverse")
c4.metric("Vest Violations", m["vest_violations"], delta=m["vest_violations"], delta_color="inverse")
c5.metric("Hazards Detected", m["machines_detected"] + m["vehicles_detected"])

st.markdown("---")

# --- WORKER TABLE & VIOLATION LOG ---
col_table, col_gallery = st.columns([1, 1])

with col_table:
    st.subheader("📋 Worker Compliance Registry")
    
    df = pd.DataFrame(report["full_worker_list"])
    
    # Apply filtering
    if view_mode == "Violations Only":
        df = df[df["status"] == "VIOLATION"]
    elif view_mode == "Safe Workers":
        df = df[df["status"] == "SAFE"]
        
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🚨 Recent Violation Log")
    if not report["violation_log"]:
        st.success("No active violations detected.")
    else:
        for v in report["violation_log"]:
            with st.expander(f"🔴 Worker {v['worker_id']} - {', '.join(v['violations'])}"):
                st.write(f"**Worker ID**: {v['worker_id']}")
                st.write(f"**Confirmed Violations**: {', '.join(v['violations'])}")
                st.write(f"**Evidence Samples**: {len(v['evidence'])}")

# --- EVIDENCE GALLERY ---
with col_gallery:
    st.subheader("📸 Proof-of-Violation Gallery")
    
    all_evidence = []
    for v in report["violation_log"]:
        for img_path in v["evidence"]:
            all_evidence.append((img_path, f"Worker {v['worker_id']}: {', '.join(v['violations'])}"))

    if not all_evidence:
        st.info("The gallery will populate when violations are detected.")
    else:
        # Display in 3-column grid
        cols = st.columns(3)
        for idx, (img_path, caption) in enumerate(all_evidence):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                cols[idx % 3].image(img, caption=caption, use_container_width=True)
            else:
                cols[idx % 3].warning("Image missing")

st.markdown("---")
st.caption("system status: online | ppe-association: active (IoU) | temporal-smoothing: active (60%)")
