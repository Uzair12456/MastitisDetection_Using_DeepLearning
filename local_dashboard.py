import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from PIL import Image, ImageOps
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\junai\OneDrive\Desktop\Mastitis Project\mastitis_mobilenetv2.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Mastitis Detector", 
    page_icon="🐄",
    layout="wide"
)

st.title("🐄 Mastitis Detection Dashboard")
st.markdown("---")

# --- BUILD MODEL FUNCTION ---
def build_model():
    """
    Reconstructs the exact model architecture used in training.
    This fixes version mismatch errors by building a fresh graph.
    """
    # 1. Base Model (MobileNetV2)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False

    # 2. Classification Head (Must match your notebook exactly!)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary output
    ])
    
    return model

# --- LOAD WEIGHTS ---
@st.cache_resource
def load_model_weights():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ **Model File Not Found!**")
        st.info(f"Path checked: `{MODEL_PATH}`")
        return None

    try:
        # Step 1: Build the empty structure
        model = build_model()
        
        # Step 2: Load the learned weights into it
        # We assume the saved file contains weights (h5 usually does)
        model.load_weights(MODEL_PATH)
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading weights: {e}")
        st.caption("Tip: Ensure you have 'tensorflow' installed.")
        return None

model = load_model_weights()

# --- PREDICTION LOGIC ---
def import_and_predict(image_data, model):
    # 1. Resize image
    image = ImageOps.fit(image_data, (IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
    
    # 2. Convert to numpy array
    img = np.asarray(image)
    
    # 3. Handle Channels (Grayscale -> RGB)
    if img.ndim == 2:  
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 4:  # RGBA -> RGB
        img = img[:, :, :3]
        
    # 4. Preprocess (MobileNetV2 style: -1 to 1)
    img_reshape = img[np.newaxis, ...]
    img_reshape = tf.keras.applications.mobilenet_v2.preprocess_input(img_reshape)
    
    # 5. Predict
    prediction = model.predict(img_reshape)
    return prediction

# --- SIDEBAR STATUS ---
with st.sidebar:
    st.header("System Status")
    if model:
        st.success("Model Loaded Successfully ✅")
    else:
        st.error("Model Failed ❌")

# --- MAIN LAYOUT ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Upload Image")
    file = st.file_uploader("Choose a thermal or visual image", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        # Updated parameter to fix the warning in your screenshot
        st.image(image, use_container_width=True, caption="Uploaded Udder Image")
    else:
        st.info("Waiting for image upload...")

with col_right:
    st.subheader("2. Analysis Results")
    
    if file is not None and model is not None:
        with st.spinner('Analyzing image patterns...'):
            predictions = import_and_predict(image, model)
            score = predictions[0][0] # Output is probability of Class 1 (Normal)
            
            # Notebook classes: ['mastitis', 'normal']
            # 0 = Mastitis, 1 = Normal
            prob_healthy = score
            prob_mastitis = 1.0 - score
            
            if prob_mastitis > 0.5:
                st.error("## 🚨 MASTITIS DETECTED")
            else:
                st.success("## ✅ HEALTHY UDDER")
            
            st.divider()
            
            st.write("### Confidence Breakdown")
            
            st.write(f"**Mastitis Probability:** {prob_mastitis*100:.2f}%")
            st.progress(int(prob_mastitis * 100))
            
            st.write(f"**Healthy Probability:** {prob_healthy*100:.2f}%")
            st.progress(int(prob_healthy * 100))
            
    elif file is None:
        st.write("Predictions will appear here after you upload an image.")