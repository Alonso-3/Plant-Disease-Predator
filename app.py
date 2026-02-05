import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

# ------------------ CONFIG ------------------
IMG_SIZE = 224
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_detection.h5")
MODEL_URL = "https://drive.google.com/uc?id=13yI7LGK36ZlQoxmV4lplSViJvb3Mw1rv"

ASSET_IMAGE = os.path.join(BASE_DIR, "assets", "Disease.png")

# ------------------ CLASS LABELS ------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ------------------ LOAD MODEL (ONCE) ------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model ")
        gdown.download(
            url=MODEL_URL,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )
    return tf.keras.models.load_model(MODEL_PATH)

# ✅ CRITICAL LINE (THIS WAS MISSING)
model = load_trained_model()

# ------------------ PREDICTION FUNCTION ------------------
def predict_disease(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.sidebar.title("🌱 Plant Disease Detection")
page = st.sidebar.selectbox("Navigate", ["Home", "Disease Recognition"])

# ------------------ HOME PAGE ------------------
if page == "Home":
    st.markdown(
        "<h1 style='text-align:center;'>Plant Disease Prediction System</h1>",
        unsafe_allow_html=True
    )

    if os.path.exists(ASSET_IMAGE):
        st.image(Image.open(ASSET_IMAGE), use_container_width=True)

    st.markdown("""
    ### 🌾 About This Project
    This application uses a **deep learning CNN model** to detect plant diseases
    from leaf images and supports sustainable agriculture.
    """)

# ------------------ DISEASE RECOGNITION PAGE ------------------
elif page == "Disease Recognition":
    st.header("📷 Upload a Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing image..."):
                result_index = predict_disease(image)
                result = CLASS_NAMES[result_index]

            st.success(f"🧪 **Prediction:** {result}")

