import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

st.set_page_config(page_title="Fashion Classifier", layout="centered")

st.title("👕 Fashion Item Classifier (Real AI)")
st.write("Upload clothing image (T-shirt, shoes, etc.)")

# Labels
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

# Load ONNX model
session = ort.InferenceSession("fashion_mnist.onnx")

def preprocess(image):
    image = image.convert("L")          # grayscale
    image = image.resize((28, 28))      # required size
    img = np.array(image).astype(np.float32)

    img = img / 255.0                   # normalize
    img = img.reshape(1, 1, 28, 28)     # shape for model
    return img

def predict(image):
    img = preprocess(image)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})

    pred = np.argmax(output[0])
    return classes[pred]

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        result = predict(image)
        st.success(f"Prediction: **{result}**")
