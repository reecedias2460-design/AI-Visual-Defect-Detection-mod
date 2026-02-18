import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Visual Defect Inspection",
    layout="wide",
    page_icon="🔍"
)

# -------------------------------
# CUSTOM CSS (Professional UI)
# -------------------------------
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: 700;
    color: #1f2937;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
}
.metric-card {
    background-color: #f3f4f6;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="big-title">AI-Powered Visual Defect Inspection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Surface Anomaly Detection System</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# DEFECT DETECTION FUNCTION
# -------------------------------
def detect_defect(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Count edge pixels
    edge_pixels = np.sum(edges > 0)
    total_pixels = image.shape[0] * image.shape[1]

    defect_score = (edge_pixels / total_pixels) * 100

    # Find contours for highlighting
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    highlighted = image.copy()
    cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)

    return defect_score, edges, highlighted

# -------------------------------
# PROCESS IMAGE
# -------------------------------
if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    defect_score, edges, highlighted = detect_defect(image_cv)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(highlighted, caption="Detected Defect Regions", use_column_width=True)

    st.divider()

    # -------------------------------
    # METRICS PANEL
    # -------------------------------
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Defect Score (%)", f"{defect_score:.2f}")

    with col4:
        if defect_score > 2:
            st.metric("Status", "Defective")
        else:
            st.metric("Status", "Acceptable")

    with col5:
        st.metric("Resolution", f"{image_np.shape[0]} x {image_np.shape[1]}")

    st.divider()

    # Progress bar visualization
    st.subheader("Defect Severity Indicator")

    severity = min(defect_score / 5, 1.0)
    st.progress(severity)

    if defect_score > 2:
        st.error("⚠ Surface irregularities detected beyond acceptable threshold.")
    else:
        st.success("✅ Product surface within acceptable limits.")

else:
    st.info("Upload an image to begin inspection.")
