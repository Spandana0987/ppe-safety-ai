# 👷‍♂️ PPE Safety Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An industrial-grade Artificial Intelligence monitoring system engineered to ensure strict site safety compliance. This platform utilizes state-of-the-art computer vision to autonomously detect personnel and track the usage of mandatory Personal Protective Equipment (PPE) such as safety helmets and high-visibility vests. 

Featuring a robust, failsafe pipeline and an intuitive real-time dashboard, this solution seamlessly transforms raw video feeds into actionable safety governance metrics.

---

## ✨ Core Capabilities

- **High-Fidelity Neural Detection:** Utilizes optimized YOLO models to instantly detect workers, safety helmets, and safety vests with high accuracy under complex industrial conditions.
- **Precision PPE Attribution:** Implements advanced Intersection-over-Union (IoU) spatial logic to accurately map detected safety gear to individual workers.
- **Automated Evidence Capture:** Whenever a compliance violation is detected (e.g., missing helmet or vest), the system automatically captures timestamped photographic evidence of the incident.
- **Transcoded Intelligence Feeds:** Built-in integration with `ffmpeg` ensures output videos are effectively transcoded to H.264 format for flawless playback in modern web browsers.
- **Failsafe Streamlit Dashboard:** A deterministically structured monitoring interface ensuring a strict two-phase operational workflow: raw detection processing followed by analytics rendering.
- **Comprehensive Analytics:** Instantly generates safety scores, full worker registries, and visual violation galleries for site managers.

---

## 🏗️ System Architecture

The pipeline consists of three interconnected micro-components operating under a sequenced workflow:

1. **`detect_video.py` (The Engine):** Ingests raw video footage, applies YOLO inference, assigns unique sequential tracking IDs, and generates compliance records alongside visual crop evidence.
2. **`generate_report.py` (The Analyzer):** Compiles the raw detection JSON arrays into structured safety metrics, overall compliance scores, and formatted registry data.
3. **`dashboard.py` (The Interface):** A responsive Streamlit frontend giving site managers full visibility over video sources, processed feeds, and compiled visual reports.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- `ffmpeg` installed on your system (for video transcoding)

### 1. Clone & Environment Setup
```bash
git clone https://github.com/your-org/ppe-safety-ai.git
cd ppe-safety-ai

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python streamlit pandas Pillow
```
*(Ensure PyTorch is properly installed for your target hardware: CPU/CUDA/MPS)*

### 3. Model & Data Preparation
- Place your trained YOLO weights (e.g., `best.pt`) inside the `models/` directory.
- Place your industrial site video feeds inside the `videos/` directory.

---

## 🚀 Usage

The entire ecosystem is managed centrally via the Streamlit dashboard for ease of use.

```bash
# Start the Monitoring Interface
streamlit run dashboard.py
```

### Dashboard Workflow:
1. **Target Selection:** Select your source video feed from the sidebar dropdown.
2. **Execute Flow:** Click **"Run Site Analysis"**. The engine will isolate the workspace, run neural inference, and compile frames.
3. **Review Intelligence:** Once complete, the dashboard will asynchronously load the processed video, calculate site safety metrics, and display actionable violation evidence in the gallery.

---

## 📁 Repository Structure

```tree
ppe_safety_ai/
├── dashboard.py         # Streamlit-based monitoring portal
├── detect_video.py      # Core YOLO inference and bounding-box script
├── generate_report.py   # Analytics computation and JSON exporter
├── models/              # Directory containing trained tracking weights (.pt files)
├── videos/              # Drop raw source videos here
└── output/              # Auto-generated reports, transcoded videos, and crop evidence
```

---

## 🛡️ Contributing

We welcome structural improvements, model optimizations, and frontend enhancements. Please follow standard fork-and-pull workflows and ensure all code is strictly typed and documented.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
