"""
Cat vs Dog Classifier — High Confidence SVM App
Features: HOG + Color Histogram (matches train_svm.py exactly)
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import os
import json
import time
from PIL import Image

try:
    from skimage.feature import hog
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "scikit-image", "-q"], check=True)
    from skimage.feature import hog

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawsAI · Cat vs Dog",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

.hero {
    background: #0a0a0f;
    border-bottom: 1px solid #1e1e2e;
    padding: 1.6rem 3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: -0.03em;
    color: #fff;
}
.hero-logo span { color: #7c6af5; }
.hero-badge {
    background: #1a1a2e;
    border: 1px solid #2e2e4e;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.7rem;
    color: #8884b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

.left-panel { padding: 2.5rem 3rem; border-right: 1px solid #1e1e2e; }
.right-panel { padding: 2.5rem 2rem; background: #0d0d14; }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4e4a78;
    margin-bottom: 1rem;
}

[data-testid="stFileUploader"] > div {
    border: 1.5px dashed #2a2a3e !important;
    border-radius: 16px !important;
    background: #0f0f1a !important;
    padding: 2rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover { border-color: #5a52c0 !important; }
[data-testid="stFileUploader"] label { color: #6b6890 !important; }

.result-card {
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-cat { background: linear-gradient(135deg,#1a0f2e 0%,#0f0f1a 100%); border: 1px solid #3d2d6e; }
.result-dog { background: linear-gradient(135deg,#0f1a2e 0%,#0f0f1a 100%); border: 1px solid #2d5d8e; }
.result-animal { font-size: 4rem; line-height: 1; margin-bottom: 0.4rem; }
.result-label  { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; letter-spacing:-0.04em; color:#fff; margin-bottom:0.25rem; }
.result-meta   { font-size:0.82rem; color:#8884b8; margin-bottom:1.4rem; }

.conf-row { margin-bottom: 0.85rem; }
.conf-header { display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.3rem; color:#c4c0e0; }
.conf-track  { height:5px; background:#1e1e2e; border-radius:10px; overflow:hidden; }
.conf-fill-cat { height:100%; border-radius:10px; background:linear-gradient(90deg,#7c6af5,#b86af5); }
.conf-fill-dog { height:100%; border-radius:10px; background:linear-gradient(90deg,#3b82f6,#38bdf8); }

.verdict { display:inline-block; padding:0.3rem 0.9rem; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; margin-top:0.4rem; }
.verdict-high { background:#1a3a1a; color:#4ade80; border:1px solid #2d6a2d; }
.verdict-mid  { background:#2a2a10; color:#fbbf24; border:1px solid #5a4a10; }
.verdict-low  { background:#2a1010; color:#f87171; border:1px solid #5a2020; }

.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-bottom:1.5rem; }
.stat-card  { background:#111118; border:1px solid #1e1e2e; border-radius:14px; padding:1rem 1.1rem; }
.stat-val   { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#fff; letter-spacing:-0.03em; line-height:1; margin-bottom:0.25rem; }
.stat-val.accent { color:#7c6af5; }
.stat-lbl   { font-size:0.7rem; color:#4e4a78; text-transform:uppercase; letter-spacing:0.08em; font-weight:500; }

.divider { height:1px; background:#1e1e2e; margin:1.6rem 0; }

.tip-item { display:flex; align-items:flex-start; gap:0.75rem; margin-bottom:0.7rem; font-size:0.82rem; color:#6b6890; line-height:1.5; }
.tip-dot  { width:5px; height:5px; border-radius:50%; background:#3a3658; margin-top:0.45rem; flex-shrink:0; }

[data-testid="stButton"] > button {
    background: #7c6af5 !important; color:#fff !important; border:none !important;
    border-radius:12px !important; padding:0.7rem 2rem !important;
    font-family:'Syne',sans-serif !important; font-size:0.88rem !important;
    font-weight:700 !important; letter-spacing:0.04em !important;
    width:100% !important; cursor:pointer !important; transition:opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity:0.85 !important; }
[data-testid="stImage"] img { border-radius:14px !important; }

footer, #MainMenu, [data-testid="stDecoration"] { display:none !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — must be IDENTICAL to train_svm.py
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(pil_image, img_size=64):
    """
    HOG + HSV color histogram.
    Returns shape (2020,) — same as training pipeline.
    """
    # PIL → BGR numpy
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size))

    # HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feats = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    c_hist = cv2.calcHist([hsv], [0, 1], None,
                          [16, 16], [0, 180, 0, 256])
    c_hist = cv2.normalize(c_hist, c_hist).flatten()

    return np.concatenate([hog_feats, c_hist]).reshape(1, -1)  # (1, 2020)


# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        svm    = joblib.load("models/svm_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        pca    = joblib.load("models/pca.pkl") if os.path.exists("models/pca.pkl") else None
        results = None
        if os.path.exists("models/results.json"):
            with open("models/results.json") as f:
                results = json.load(f)
        return svm, scaler, pca, results
    except Exception:
        return None, None, None, None


def predict(image, svm, scaler, pca):
    feats = extract_features(image)          # (1, 2020)
    t0 = time.time()
    feats = scaler.transform(feats)          # uses scaler fitted on 2020 features
    if pca is not None:
        feats = pca.transform(feats)
    pred  = svm.predict(feats)[0]
    proba = svm.predict_proba(feats)[0]
    ms    = (time.time() - t0) * 1000
    return pred, proba, ms


# ─── Load ─────────────────────────────────────────────────────────────────────
svm, scaler, pca, results = load_model()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">Paws<span>AI</span></div>
    <div class="hero-badge">SVM · HOG Features · RBF Kernel</div>
</div>
""", unsafe_allow_html=True)

# ─── Two-column layout ────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="small")

# ════════ LEFT ════════════════════════════════════════════════════════════════
with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a cat or dog photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        if st.button("Classify →", use_container_width=True):
            if svm is None:
                st.error("❌ Model not loaded — run train_svm.py first.")
            else:
                with st.spinner("Analyzing…"):
                    pred, proba, ms = predict(image, svm, scaler, pca)

                conf    = proba[pred] * 100
                cat_pct = proba[0] * 100
                dog_pct = proba[1] * 100

                verdict_cls = ("verdict-high" if conf >= 75 else
                               "verdict-mid"  if conf >= 60 else "verdict-low")
                verdict_txt = ("High confidence" if conf >= 75 else
                               "Moderate confidence" if conf >= 60 else
                               "Low confidence — try a clearer photo")

                card_cls = "result-cat" if pred == 0 else "result-dog"
                animal   = "🐱" if pred == 0 else "🐶"
                label    = "Cat" if pred == 0 else "Dog"

                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <div class="result-animal">{animal}</div>
                    <div class="result-label">{label}</div>
                    <div class="result-meta">{conf:.1f}% confidence · {ms:.0f} ms</div>
                    <div class="conf-row">
                        <div class="conf-header"><span>🐱 Cat</span><span>{cat_pct:.1f}%</span></div>
                        <div class="conf-track"><div class="conf-fill-cat" style="width:{cat_pct}%"></div></div>
                    </div>
                    <div class="conf-row">
                        <div class="conf-header"><span>🐶 Dog</span><span>{dog_pct:.1f}%</span></div>
                        <div class="conf-track"><div class="conf-fill-dog" style="width:{dog_pct}%"></div></div>
                    </div>
                    <div class="verdict {verdict_cls}">{verdict_txt}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1.5px dashed #2a2a3e;border-radius:16px;background:#0f0f1a;
                    padding:3rem 2rem;text-align:center;margin-bottom:1.5rem">
            <div style="font-size:2.5rem;margin-bottom:0.8rem">🐾</div>
            <div style="color:#6b6890;font-size:0.86rem;line-height:1.6">
                Drag & drop a photo above<br>JPG · JPEG · PNG
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ════════ RIGHT ═══════════════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Dashboard</div>', unsafe_allow_html=True)

    if results:
        acc  = results.get("accuracy", 0) * 100
        f1   = results.get("f1_score", 0)
        prec = results.get("precision", 0) * 100
        rec  = results.get("recall", 0) * 100

        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val accent">{acc:.1f}%</div><div class="stat-lbl">Accuracy</div></div>
            <div class="stat-card"><div class="stat-val">{f1:.3f}</div><div class="stat-lbl">F1 Score</div></div>
            <div class="stat-card"><div class="stat-val">{prec:.1f}%</div><div class="stat-lbl">Precision</div></div>
            <div class="stat-card"><div class="stat-val">{rec:.1f}%</div><div class="stat-lbl">Recall</div></div>
        </div>
        """, unsafe_allow_html=True)

        total    = results.get("total_images", 0)
        feat_p   = results.get("features_after_pca", 0)
        t_time   = results.get("training_time", 0)
        hi_conf  = results.get("high_conf_pct", 0)
        cats     = results.get("cats", 0)
        dogs     = results.get("dogs", 0)
        c_val    = results.get("c_value", "N/A")
        kernel   = results.get("kernel", "rbf")
        ftype    = results.get("feature_type", "HOG + Color")

        st.markdown('<div class="section-label" style="margin-top:1.2rem">Training Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val">{total:,}</div><div class="stat-lbl">Images</div></div>
            <div class="stat-card"><div class="stat-val">{feat_p}</div><div class="stat-lbl">PCA features</div></div>
            <div class="stat-card"><div class="stat-val">{t_time:.0f}s</div><div class="stat-lbl">Train time</div></div>
            <div class="stat-card"><div class="stat-val">{hi_conf:.0f}%</div><div class="stat-lbl">High conf.</div></div>
        </div>
        <div class="stat-card" style="margin-bottom:0.8rem">
            <div style="font-size:0.77rem;color:#6b6890;line-height:2.1">
                <span style="color:#c4c0e0;font-weight:500">Features</span>&nbsp; {ftype}<br>
                <span style="color:#c4c0e0;font-weight:500">Kernel</span>&nbsp;&nbsp;&nbsp; {kernel.upper()} · C = {c_val}<br>
                <span style="color:#c4c0e0;font-weight:500">Dataset</span>&nbsp;&nbsp; {cats:,} cats · {dogs:,} dogs
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif svm is None:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-val" style="color:#f87171;font-size:1rem">No model found</div>
            <div class="stat-lbl" style="margin-top:0.4rem">Run train_svm.py first</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-val accent">Loaded</div>
            <div class="stat-lbl">No results.json found</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Tips for best results</div>', unsafe_allow_html=True)
    for tip in ["Clear, well-lit photos", "Single animal filling the frame",
                "Front-facing shots work best", "Avoid heavy filters or cropping"]:
        st.markdown(f'<div class="tip-item"><div class="tip-dot"></div><span>{tip}</span></div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)