import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page configuration
st.set_page_config(
    page_title="Fresh vs Rotten Classifier",
    page_icon="🥕",
    layout="wide"
)

# 2. Load trained ANN model (safe loading)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_ann_small.keras")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Check that the model file exists in the repo
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.write("Files in app folder:", os.listdir(BASE_DIR))
    st.stop()

# Show file size (a tiny size means it is a Git LFS pointer / bad upload)
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
if model_size_mb < 0.01:
    st.error(
        f"model_ann_small.keras is only {model_size_mb:.4f} MB. "
        "The file was not uploaded properly (maybe a Git LFS pointer). "
        "Please upload the real model file to GitHub."
    )
    st.stop()

# Load the model and show the real error inside the app if it fails
try:
    model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.write("TensorFlow version:", tf.__version__)
    st.write("Model file size (MB):", round(model_size_mb, 2))
    st.stop()

# 3. Class labels
# NOTE: verify with print(train_data.class_indices) in your training notebook.
# If it shows {'fresh': 0, 'rotten': 1}, change this to {0: "Fresh", 1: "Rotten"}
class_names = {
    0: "Rotten",
    1: "Fresh"
}

# 4. Application title
st.title("🍎 Fresh vs Rotten Fruits & Vegetables Classifier")

st.write(
    "Upload an image of a fruit or vegetable "
    "to predict whether it is Fresh or Rotten using ANN."
)

# 5. Supported images note
st.info(
    """
    📌 **Note:**

    This model is designed to classify only the following
    fruits and vegetables as Fresh or Rotten.

    **🍎 Fruits:**
    Apple, Mango, Banana, Orange, Strawberry

    **🥕 Vegetables:**
    Tomato, Potato, Cucumber, Carrot, Bell Pepper

    Please upload images belonging to these categories only.
    """
)

st.divider()


# 6. Image preprocessing
def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image to 100x100
    image = image.resize((100, 100))

    # Convert image into NumPy array
    image = np.asarray(image, dtype=np.float32)

    # Normalize pixel values between 0 and 1
    image = image / 255.0

    # Flatten image into 30000 features
    image = image.flatten()

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# 7. Prediction function
def predict_image(image):

    processed_image = preprocess_image(image)

    prediction = model(processed_image, training=False).numpy()

    probability = float(prediction[0, 0])

    predicted_class = 1 if probability >= 0.5 else 0

    predicted_label = class_names[predicted_class]

    confidence = (
        probability if predicted_class == 1
        else 1 - probability
    ) * 100

    return predicted_label, confidence, probability


# 8. Image upload
st.subheader("📸 Upload Fruit or Vegetable Images")

uploaded_files = st.file_uploader(
    "Choose image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# 9. Display images and predictions
if uploaded_files:

    st.subheader("🔍 Prediction Results")

    for uploaded_file in uploaded_files:

        try:
            image = Image.open(uploaded_file)
            predicted_label, confidence, probability = predict_image(image)
        except Exception as e:
            st.error(f"Could not process {uploaded_file.name}: {e}")
            st.divider()
            continue

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption=uploaded_file.name,
                width=250
            )

        with col2:

            st.write("**Prediction:**")

            if predicted_label == "Fresh":

                st.success("🥬 Fresh")

            else:

                st.error("🍂 Rotten")

            st.write(
                f"**Confidence:** {confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

        st.divider()

else:

    st.info(
        "Please upload a fruit or vegetable image "
        "to get a prediction."
    )


# 10. Footer
st.caption(
    "Developed using Artificial Neural Network (ANN), "
    "TensorFlow, NumPy, PIL and Streamlit."
)