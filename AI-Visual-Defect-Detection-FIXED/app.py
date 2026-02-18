import streamlit as st
import cv2
import numpy as np
from src.predict import predict_image


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Visual Defect Inspection",
    layout="wide",
    page_icon="✨"
)

# -------------------------------
# BRIGHT PROFESSIONAL CSS
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f0f9ff;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 18px;
    color: #1e3a8a;
    margin-bottom: 20px;
}
.metric-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    text-align: center;
}
.section-box {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="main-title">AI-Powered Visual Defect Inspection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">High-Precision Surface Anomaly Detection System</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Product Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# DEFECT DETECTION
# -------------------------------
def detect_defect(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edge_pixels = np.sum(edges > 0)
    total_pixels = image.shape[0] * image.shape[1]
    defect_score = (edge_pixels / total_pixels) * 100

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = image.copy()
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 2)

    return defect_score, highlighted

# -------------------------------
# PROCESS IMAGE
# -------------------------------
if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    defect_score, highlighted = detect_defect(image_cv)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.image(image, caption="Original Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.image(highlighted, caption="Detected Defect Regions", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------
    # METRICS
    # -------------------------------
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Defect Score (%)", f"{defect_score:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "Defective" if defect_score > 2 else "Acceptable"
        st.metric("Inspection Status", status)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Resolution", f"{image_np.shape[0]} x {image_np.shape[1]}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------
    # BRIGHT SEVERITY BAR
    # -------------------------------
    st.subheader("Defect Severity Level")

    severity = min(defect_score / 5, 1.0)
    st.progress(severity)

    if defect_score > 2:
        st.error("⚠ Surface irregularities detected beyond threshold.")
    else:
        st.success("✅ Product surface meets quality standards.")

else:
    st.info("Upload an image to start inspection.")
