import streamlit as st
from PIL import Image
from src.predict import predict_image

st.title("AI-Powered Visual Defect Detection (Auto Model)")

uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = predict_image(image)
    st.subheader(f"Prediction: {result}")
