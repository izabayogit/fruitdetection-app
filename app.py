
import io
import time
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="FruitVision | Fruit Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- THEME / STYLES --------------------
st.markdown("""
<style>
    .stApp { background: #f6f8fb; }
    [data-testid="stSidebar"] { background: #10251f; }
    [data-testid="stSidebar"] * { color: #f4fff8 !important; }
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #123b2b, #23845a);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.25rem; }
    .hero p { color: #e4f5eb; margin: .4rem 0 0 0; }
    .metric-card {
        background: white; border: 1px solid #e7ece9; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(0,0,0,.035);
    }
    .metric-label { color: #63736b; font-size: .85rem; }
    .metric-value { color: #153d2c; font-size: 1.8rem; font-weight: 750; }
    .section-title { color: #173e2d; font-weight: 700; font-size: 1.2rem; }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px; font-weight: 650; min-height: 2.7rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOADING --------------------
MODEL_PATH = Path(__file__).parent / "fruitdetection.pt"

@st.cache_resource(show_spinner="Loading your trained fruit detection model...")
def load_model(model_path: str):
    return YOLO(model_path)

def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

def run_detection(model, image, confidence, iou, image_size):
    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        verbose=False
    )
    return results[0]

def result_to_dataframe(result):
    rows = []
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["Fruit", "Confidence", "X1", "Y1", "X2", "Y2"])

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        rows.append({
            "Fruit": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id],
            "Confidence": round(conf * 100, 2),
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2
        })
    return pd.DataFrame(rows)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🍏 FruitVision")
    st.caption("AI-powered fruit detection")
    page = st.radio("Navigation", ["Fruit Detection", "About the System"])
    st.divider()
    st.markdown("### Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                           help="Higher values reduce weak detections.")
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05,
                    help="Controls overlap suppression between bounding boxes.")
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
    st.divider()
    st.caption("Model: fruitdetection.pt")

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
  <h1>🍎 Fruit Detection System</h1>
  <p>Upload an image and let your trained YOLO model identify fruits automatically.</p>
</div>
""", unsafe_allow_html=True)

if page == "About the System":
    st.markdown("## About this application")
    st.write(
        "FruitVision is a Streamlit interface for running inference with your trained "
        "Ultralytics YOLO fruit-detection model. It draws bounding boxes around detected "
        "objects and reports the predicted class and confidence for each detection."
    )
    st.markdown("### How to use")
    st.markdown("""
    1. Open **Fruit Detection** from the sidebar.
    2. Upload a fruit image (JPG, JPEG, or PNG).
    3. Adjust the confidence threshold if necessary.
    4. Select **Run fruit detection**.
    5. Review the annotated image, detection counts, and confidence values.
    6. Download the annotated image if you want to save it.
    """)
    st.info("The fruit categories displayed by the app are read directly from your trained model. No class names are hard-coded.")
    st.markdown("### Model file")
    st.code("fruitdetection.pt")
    st.write("Keep this file in the same folder as app.py.")
    st.stop()

# -------------------- MODEL STATUS --------------------
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Place fruitdetection.pt in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error("The model could not be loaded. Check that it is a compatible Ultralytics YOLO .pt model.")
    st.exception(e)
    st.stop()

# Read model class names dynamically
class_names = model.names
if isinstance(class_names, dict):
    class_list = [class_names[k] for k in sorted(class_names)]
else:
    class_list = list(class_names)

st.markdown('<div class="section-title">Upload an image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a clear image where fruits are visible."
)

camera_image = st.camera_input("Or take a picture using your camera")

source_image = None
source_label = None
if uploaded_file is not None:
    source_image = Image.open(uploaded_file).convert("RGB")
    source_label = uploaded_file.name
elif camera_image is not None:
    source_image = Image.open(camera_image).convert("RGB")
    source_label = "Camera capture"

if source_image is not None:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Original image")
        st.image(source_image, use_container_width=True, caption=source_label)

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("🔍 Run fruit detection", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Analyzing image with your trained model..."):
            start = time.perf_counter()
            try:
                result = run_detection(model, source_image, confidence, iou, image_size)
                elapsed = time.perf_counter() - start
                annotated = Image.fromarray(result.plot()[:, :, ::-1])
                detections = result_to_dataframe(result)

                st.session_state["fruit_last_result"] = {
                    "annotated": annotated,
                    "detections": detections,
                    "elapsed": elapsed,
                    "source": source_label,
                }
            except Exception as e:
                st.error("An error occurred while running detection.")
                st.exception(e)

# -------------------- DISPLAY LAST RESULT --------------------
saved = st.session_state.get("fruit_last_result")
if saved:
    st.divider()
    st.markdown("## Detection results")
    annotated = saved["annotated"]
    detections = saved["detections"]
    total = len(detections)
    unique = detections["Fruit"].nunique() if total else 0
    avg_conf = detections["Confidence"].mean() if total else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Total fruits detected", total)
    with m2: metric_card("Fruit classes detected", unique)
    with m3: metric_card("Average confidence", f"{avg_conf:.1f}%")
    with m4: metric_card("Inference time", f'{saved["elapsed"]:.2f}s')

    out_col, info_col = st.columns([1.35, 1], gap="large")
    with out_col:
        st.markdown("### Annotated image")
        st.image(annotated, use_container_width=True)
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        st.download_button(
            "⬇️ Download detected image",
            data=buffer.getvalue(),
            file_name="fruit_detection_result.png",
            mime="image/png",
            use_container_width=True
        )

    with info_col:
        st.markdown("### Fruit count by class")
        if total:
            counts = detections["Fruit"].value_counts().rename_axis("Fruit").reset_index(name="Count")
            st.bar_chart(counts.set_index("Fruit"))
            st.dataframe(counts, use_container_width=True, hide_index=True)
        else:
            st.warning("No fruits were detected. Try a lower confidence threshold or a clearer image.")

    st.markdown("### Individual detections")
    if total:
        st.dataframe(
            detections.style.format({"Confidence": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )
        csv_data = detections.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download detection data (CSV)",
            data=csv_data,
            file_name="fruit_detection_results.csv",
            mime="text/csv"
        )
    else:
        st.caption("There are no detection rows to display.")

st.divider()
st.caption("FruitVision • Powered by your trained YOLO model")
