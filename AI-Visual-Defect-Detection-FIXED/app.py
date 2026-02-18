import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Visual Defect Detection")

st.title("🔍 AI-Powered Visual Defect Detection")
st.write("Upload a product image to detect surface defects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def detect_defect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edge_pixels = np.sum(edges > 0)

    if edge_pixels > 5000:
        return "Defective", edges
    else:
        return "Non-Defective", edges

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result, processed = detect_defect(image_cv)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(processed, caption="Edge Detection", use_column_width=True)

    if result == "Defective":
        st.error("⚠ Defect Detected")
    else:
        st.success("✅ Product Looks Good")
