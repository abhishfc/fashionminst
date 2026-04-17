import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("fashion_mnist.onnx")

# Class labels
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("👕 Fashion Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width='stretch')

    # 🔥 PREPROCESSING (CRITICAL FIX)
    image = image.convert("L")         # convert to grayscale
    image = image.resize((28, 28))    # resize to model input
    image = np.array(image)

    image = image / 255.0             # normalize
    image = image.reshape(1, 1, 28, 28).astype(np.float32)

    # Predict
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image})

    pred = np.argmax(output[0])
    confidence = np.max(output[0])

    st.success(f"Prediction: {labels[pred]}")
    st.write(f"Confidence: {confidence:.2f}")
