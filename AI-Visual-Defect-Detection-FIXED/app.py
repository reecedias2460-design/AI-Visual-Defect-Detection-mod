import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Visual Defect Detection", layout="centered")

st.title("🔍 AI-Powered Visual Defect Detection")
st.write("Upload a product image to detect possible surface defects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def detect_defect(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Count edge pixels
    edge_count = np.sum(edges > 0)

    # Simple threshold logic
    if edge_count > 5000:
        return "Defective", edges
    else:
        return "Non-Defective", edges

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result, processed = detect_defect(image_cv)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(processed, caption="Detected Edges", use_column_width=True)

    if result == "Defective":
        st.error("⚠ Defect Detected")
    else:
        st.success("✅ Product Looks Good")
