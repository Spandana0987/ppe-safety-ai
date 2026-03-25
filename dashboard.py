"""
PPE Safety Intelligence Dashboard — Failsafe Edition
STRICT 2-PHASE FLOW: Detection -> Video -> Analysis -> Metrics
"""
import streamlit as st
import json
import os
import subprocess
import time
import pandas as pd
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Safety AI | Failsafe Monitor",
    layout="wide",
    page_icon="👷",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("<style>h1, h2, h3 { color: #58a6ff !important; }</style>", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
VIDEO_DIR   = "videos"
VIDEO_OUT   = "output/processed_video.mp4"
DATA_OUT    = "output/processed_video_data.json"
REPORT_OUT  = "output/summary_report.json"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛠️ Monitoring Controls")

available = sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mov")))
if not available: available = ["No videos found"]

selected_video = st.sidebar.selectbox("🎥 Select Source Video", available)
run_btn = st.sidebar.button("🚀 Run Site Analysis", use_container_width=True)

st.sidebar.markdown("---")
view_mode = st.sidebar.radio("🔎 Filter Registry", ["All Workers", "Violations Only", "Safe Workers"])

# ── Session State Initialization ─────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "idle"   # idle → processing → detected → analyzed
if "last_vid" not in st.session_state:
    st.session_state.last_vid = None

# Reset state if selection changes
if selected_video != st.session_state.last_vid:
    st.session_state.stage = "idle"
    st.session_state.last_vid = selected_video

# ── MAIN TITLE ───────────────────────────────────────────────────────────────
st.title("👷 PPE Safety Intelligence Dashboard")
st.caption(f"Active Monitoring: **{selected_video}**")
st.markdown("---")

# ── STEP 1: RUN LOGIC ────────────────────────────────────────────────────────
if run_btn and selected_video != "No videos found":
    st.session_state.stage = "processing"
    
    # OUTPUT ISOLATION
    import shutil
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output/frames", exist_ok=True)

    input_path = os.path.join(VIDEO_DIR, selected_video)
    
    with st.spinner("🏗️ Running neural site detection..."):
        # RUN DETECTION
        result = subprocess.run([
            "venv/bin/python3", "detect_video.py", 
            "--source", input_path, "--save-video"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Detection failed: {result.stderr}")
            st.session_state.stage = "idle"
            st.stop()
        
    st.session_state.stage = "detected"

# ── STEP 2: VIDEO RENDERING ──────────────────────────────────────────────────
if st.session_state.stage in ["detected", "analyzed"]:
    st.subheader("📺 Processed Site Intelligence Feed")
    video_path = "output/processed_video.mp4"

    # FILE STABILITY CHECK (Retry Loop)
    found = False
    for _ in range(20):
        if os.path.exists(video_path) and os.path.getsize(video_path) > 500000:
            found = True
            break
        time.sleep(0.3)
    
    if not found:
        st.error("❌ CRITICAL: Processed video not ready or corrupted.")
        st.info("Check if detect_video.py completed successfully.")
        st.stop()

    # READ AS BYTES + UNIQUE KEY (STRICT MANDATE)
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    st.video(video_bytes)
    st.success("✅ Site intelligence feed loaded successfully.")
    st.markdown("---")

# ── STEP 3: RUN ANALYSIS ─────────────────────────────────────────────────────
if st.session_state.stage == "detected":
    with st.spinner("📊 Compiling safety metrics..."):
        result = subprocess.run([
            "venv/bin/python3", "generate_report.py"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Analysis failed: {result.stderr}")
            st.stop()
            
    st.session_state.stage = "analyzed"

# ── STEP 4: SHOW ANALYSIS ────────────────────────────────────────────────────
if st.session_state.stage == "analyzed":
    report_path = "output/summary_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            data = json.load(f)
        
        m = data["metrics"]
        
        # Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Safety Score", f"{m['safety_score']}%")
        c2.metric("Total Workers", m["total_workers"])
        c3.metric("Helmet Violations", m["helmet_violations"])
        c4.metric("Vest Violations",   m["vest_violations"])
        
        st.markdown("---")
        
        # Registry
        st.subheader("📋 Compliance Registry")
        df = pd.DataFrame(data.get("full_worker_list", []))
        if not df.empty:
            if view_mode == "Violations Only": df = df[df["status"] == "VIOLATION"]
            elif view_mode == "Safe Workers": df = df[df["status"] == "SAFE"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        st.markdown("---")
        # Evidence
        st.subheader("📸 Proof-of-Violation Gallery")
        vlog = data.get("violation_log", [])
        evidence = [(img, f"Track {v['worker_id']}") for v in vlog for img in v.get("evidence", []) if os.path.exists(img)]
        if evidence:
            cols = st.columns(4)
            for idx, (img, cap) in enumerate(evidence):
                cols[idx % 4].image(img, caption=cap, use_container_width=True)
        
        st.success("Analysis Complete")
    else:
        st.error("⚠️ Safety analytics generation failed (JSON not found).")

# ── IDLE STATE (PRE-RUN) ─────────────────────────────────────────────────────
if st.session_state.stage == "idle":
    st.info("👋 Select a video and click **Run Site Analysis** to begin monitoring.")
    raw_path = os.path.join(VIDEO_DIR, selected_video) if selected_video != "No videos found" else None
    if raw_path and os.path.exists(raw_path):
        st.subheader("📹 Source Feed Preview")
        with open(raw_path, "rb") as f:
            st.video(f.read())

st.markdown("---")
st.caption("Engine: Industrial Failsafe | Status: Deterministic Loop Active")
