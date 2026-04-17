import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# ---------------- UI ----------------
st.set_page_config(page_title="Fashion Classifier", layout="centered")
st.title("👕 Fashion Item Classifier")

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 🔥 FIXED LABELS (IMPORTANT)
CLASS_NAMES = [
    "T-shirt/top",
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

# ---------------- Preprocess ----------------
def preprocess(img: Image.Image):
    img = img.convert("L")          # grayscale (important for Fashion MNIST models)
    img = img.resize((28, 28))      # must match training size

    arr = np.array(img).astype(np.float32)

    # normalize exactly like training (IMPORTANT FIX)
    arr = arr / 255.0

    # reshape: (1, 1, 28, 28) for ONNX models
    arr = arr.reshape(1, 1, 28, 28)

    return arr

# ---------------- Upload ----------------
file = st.file_uploader("Upload a fashion image", type=["png", "jpg", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess(image)

    preds = session.run([output_name], {input_name: input_data})[0]

    # FIX: correct prediction axis
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.success(f"Prediction: {CLASS_NAMES[pred_class]}")
    st.write(f"Confidence: {confidence:.2f}")
