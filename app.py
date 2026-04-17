import streamlit as st
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="Fashion Classifier")

st.title("👕 Fashion Image Classifier")
st.write("Upload an image → Predict clothing type")

# ================================
# LOAD DATA
# ================================
@st.cache_resource
def load_data():
    X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = X / 255.0
    return X[:60000], y[:60000].astype(int), X[60000:], y[60000:].astype(int)

X_train, y_train, X_test, y_test = load_data()

# ================================
# TRAIN MODEL
# ================================
@st.cache_resource
def train_model():
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=5)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ================================
# IMAGE UPLOAD
# ================================
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

def preprocess(img):
    img = img.convert("L")
    img = img.resize((28,28))
    img = np.array(img)/255.0
    return img.reshape(1, -1)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=200)

    data = preprocess(img)
    pred = model.predict(data)[0]

    st.subheader(f"Prediction: {class_names[pred]}")
