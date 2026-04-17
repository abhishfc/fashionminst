import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fashion Classifier", layout="centered")

st.title("👕 Fashion Item Classifier")
st.write("Upload an image of clothing (T-shirt, shoes, etc.)")

# Class labels
classes = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Fake prediction (replace later with real model if needed)
def predict_image(img):
    img = img.resize((28, 28))
    img = np.array(img)

    # Simple logic (random-like but stable)
    mean_val = img.mean()

    index = int(mean_val) % len(classes)
    return classes[index]

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        prediction = predict_image(image)
        st.success(f"Prediction: **{prediction}**")
