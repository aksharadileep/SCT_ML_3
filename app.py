"""
Cat vs Dog Classifier App - Fixed Feature Mismatch
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="wide"
)

# Title
st.title("🐱 Cat vs Dog Classifier 🐶")
st.markdown("---")

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing components"""
    try:
        # Load model, scaler, and PCA
        svm = joblib.load("models/svm_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        pca = joblib.load("models/pca.pkl") if os.path.exists("models/pca.pkl") else None
        
        # Load results
        import json
        if os.path.exists("models/results.json"):
            with open("models/results.json", 'r') as f:
                results = json.load(f)
        else:
            results = None
        
        return svm, scaler, pca, results
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def preprocess_image(image, target_size=(64, 64)):
    """Preprocess image to match training data format"""
    try:
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        # Convert RGB to BGR if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization (same as training)
        gray = cv2.equalizeHist(gray)
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        # Flatten - this gives 4096 features (64*64)
        features = gray.flatten()
        
        return features
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_image(image, svm, scaler, pca=None):
    """Make prediction on a single image"""
    # Preprocess
    features = preprocess_image(image)
    if features is None:
        return None, None
    
    # Reshape for sklearn
    features = features.reshape(1, -1)
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Apply PCA if available
    if pca is not None:
        features_scaled = pca.transform(features_scaled)
    
    # Predict
    prediction = svm.predict(features_scaled)[0]
    probability = svm.predict_proba(features_scaled)[0]
    
    return prediction, probability

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    # Load model
    svm, scaler, pca, results = load_model()
    
    if svm is not None:
        if results:
            accuracy = results.get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
            st.metric("F1-Score", f"{results.get('f1_score', 0):.4f}")
            st.metric("Precision", f"{results.get('precision', 0):.4f}")
            st.metric("Recall", f"{results.get('recall', 0):.4f}")
            
            st.markdown("---")
            st.subheader("📁 Dataset Info")
            st.write(f"**Total Images:** {results.get('total_images', 'N/A'):,}")
            st.write(f"**Cats:** {results.get('cats', 'N/A'):,}")
            st.write(f"**Dogs:** {results.get('dogs', 'N/A'):,}")
            st.write(f"**Image Size:** {results.get('image_size', 64)}x{results.get('image_size', 64)}")
            st.write(f"**Features:** {results.get('features_original', 4096):,}")
            
            if results.get('pca_used'):
                st.write(f"**PCA Features:** {results.get('features_after_pca', 'N/A'):,}")
                reduction = (1 - results.get('features_after_pca', 0) / results.get('features_original', 1)) * 100
                st.write(f"**Reduction:** {reduction:.1f}%")
            
            st.markdown("---")
            st.subheader("⚙️ Model Parameters")
            st.write(f"C: {results.get('c_value', 'N/A')}")
            st.write(f"Kernel: {results.get('kernel', 'rbf')}")
            st.write(f"Gamma: {results.get('gamma', 'scale')}")
        else:
            st.info("No results file found")
    else:
        st.error("❌ Model not loaded")
        st.info("Please train the model first using:")
        st.code("python src/train_5000.py")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Image")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if st.button("🔍 Classify", type="primary"):
            if svm is not None:
                with st.spinner("Analyzing image..."):
                    # Measure prediction time
                    start_time = time.time()
                    prediction, probability = predict_image(image, svm, scaler, pca)
                    pred_time = time.time() - start_time
                
                if prediction is not None:
                    # Display result
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    
                    # Create columns for metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        if prediction == 0:
                            st.metric("Prediction", "🐱 CAT", delta=None)
                        else:
                            st.metric("Prediction", "🐶 DOG", delta=None)
                    
                    with col_metric2:
                        confidence = probability[prediction] * 100
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    with col_metric3:
                        st.metric("Time", f"{pred_time*1000:.2f} ms")
                    
                    # Show probability bars
                    st.markdown("### Confidence Scores")
                    
                    col_prob1, col_prob2 = st.columns(2)
                    
                    with col_prob1:
                        st.markdown("**🐱 Cat**")
                        st.progress(probability[0])
                        st.write(f"{probability[0]*100:.2f}%")
                    
                    with col_prob2:
                        st.markdown("**🐶 Dog**")
                        st.progress(probability[1])
                        st.write(f"{probability[1]*100:.2f}%")
                    
                    # Add confidence interpretation
                    st.markdown("---")
                    st.subheader("🎯 Analysis")
                    if confidence > 80:
                        st.success("✅ High confidence prediction!")
                    elif confidence > 60:
                        st.info("⚠️ Moderate confidence - likely correct")
                    else:
                        st.warning("❓ Low confidence - try a clearer image")
                else:
                    st.error("Could not process image. Please try another image.")
            else:
                st.error("Model not loaded. Please train the model first.")

with col2:
    st.subheader("ℹ️ Instructions")
    st.markdown("""
    ### How to Use
    1. **Upload** an image of a cat or dog
    2. Click the **Classify** button
    3. View the **prediction** and **confidence**
    
    ### Supported Formats
    - JPG / JPEG
    - PNG
    
    ### Tips for Best Results
    - Use clear, well-lit photos
    - Single animal in frame
    - Front-facing photos work best
    - Any orientation (auto-resized)
    
    ### Model Features
    - SVM with RBF kernel
    - Trained on 64x64 grayscale images
    - Histogram equalization for contrast
    - PCA dimensionality reduction
    - Trained on 5000 images
    """)
    
    # Add test info
    st.markdown("---")
    st.subheader("📊 Model Status")
    if svm is not None:
        st.success("✅ Model loaded successfully")
        if pca is not None:
            st.info("✅ PCA dimensionality reduction enabled")
    else:
        st.error("❌ Model not loaded")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by SVM | Built with Streamlit</div>",
    unsafe_allow_html=True
)