import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Fashion CNN Classifier", layout="centered")

st.title("👕 Fashion Image Classifier (CNN)")
st.write("Upload an image to classify into Bag, Shirt, Sneaker, Boot, etc.")

# ================================
# CLASS LABELS
# ================================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ================================
# LOAD DATA
# ================================
@st.cache_resource
def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data()

# ================================
# MODEL
# ================================
@st.cache_resource
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = build_model()

# ================================
# TRAIN MODEL
# ================================
if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        model.fit(train_images, train_labels, epochs=5, verbose=1)
        loss, acc = model.evaluate(test_images, test_labels, verbose=0)
        st.success(f"Model trained! Accuracy: {acc:.2f}")

# ================================
# IMAGE UPLOAD
# ================================
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png"])

def preprocess_image(image):
    image = image.convert("L")      # convert to grayscale
    image = image.resize((28, 28))  # resize
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)

    st.subheader(f"🧠 Prediction: {class_names[predicted_class]}")

    st.write("### Confidence Scores")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")

# ================================
# SAMPLE VISUALIZATION
# ================================
if st.checkbox("📊 Show Sample Predictions"):
    fig, axes = plt.subplots(1, 5, figsize=(10,3))
    for i in range(5):
        axes[i].imshow(test_images[i].reshape(28,28), cmap='gray')
        pred = np.argmax(model.predict(test_images[i:i+1]))
        axes[i].set_title(class_names[pred])
        axes[i].axis('off')
    st.pyplot(fig)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.write("Built with CNN + TensorFlow + Streamlit")
