"""
HIGH CONFIDENCE SVM CLASSIFIER - HOG + Color Features
Target: 78-83% Accuracy | 70-90% Prediction Confidence
"""

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import random
import json
import time

try:
    from skimage.feature import hog
except ImportError:
    import subprocess
    print("Installing scikit-image...")
    subprocess.run(["pip", "install", "scikit-image", "-q"], check=True)
    from skimage.feature import hog

print("="*70)
print("🐱🐶 HIGH CONFIDENCE SVM - HOG + COLOR FEATURES")
print("="*70)

# ─── Configuration ────────────────────────────────────────────────────────────
RAW_DATA_PATH    = "data/raw/train/"
IMG_SIZE         = 64
IMAGES_PER_CLASS = 4000

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  ← this exact function must be copied into app.py too
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(img_path_or_array, img_size=64):
    """
    HOG (shape/edges) + HSV color histogram.
    Output size: 1764 (HOG) + 256 (color) = 2020 features.

    IMPORTANT: app.py must use this identical function or you will get
    a feature-mismatch error from StandardScaler / PCA.
    """
    # Accept a file path OR a BGR numpy array
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            return None
    else:
        img = img_path_or_array

    try:
        img = cv2.resize(img, (img_size, img_size))

        # HOG on grayscale ─ captures ear shape, snout, fur edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feats = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )                                       # → 1764 features

        # HSV color histogram ─ captures breed colour patterns
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        c_hist = cv2.calcHist([hsv], [0, 1], None,
                              [16, 16], [0, 180, 0, 256])
        c_hist = cv2.normalize(c_hist, c_hist).flatten()  # → 256 features

        return np.concatenate([hog_feats, c_hist])        # → 2020 features

    except Exception:
        return None


# ─── Load files ───────────────────────────────────────────────────────────────
print(f"\n📁 Loading {IMAGES_PER_CLASS} cats and {IMAGES_PER_CLASS} dogs...")

cat_files = [f for f in os.listdir(RAW_DATA_PATH)
             if f.startswith('cat') and f.endswith('.jpg')]
dog_files = [f for f in os.listdir(RAW_DATA_PATH)
             if f.startswith('dog') and f.endswith('.jpg')]

print(f"   ✅ Found: {len(cat_files):,} cats, {len(dog_files):,} dogs")

random.seed(42)
selected_cats = random.sample(cat_files, min(IMAGES_PER_CLASS, len(cat_files)))
selected_dogs = random.sample(dog_files, min(IMAGES_PER_CLASS, len(dog_files)))

# ─── Extract features ─────────────────────────────────────────────────────────
X, y = [], []

print("\n🐱 Processing cats:")
for i, fn in enumerate(selected_cats):
    feats = extract_features(os.path.join(RAW_DATA_PATH, fn))
    if feats is not None:
        X.append(feats); y.append(0)
    if (i + 1) % 500 == 0:
        print(f"      {i+1}/{len(selected_cats)}")

print("\n🐶 Processing dogs:")
for i, fn in enumerate(selected_dogs):
    feats = extract_features(os.path.join(RAW_DATA_PATH, fn))
    if feats is not None:
        X.append(feats); y.append(1)
    if (i + 1) % 500 == 0:
        print(f"      {i+1}/{len(selected_dogs)}")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n✅ Dataset ready: {len(X)} images, {X.shape[1]} features each")
print(f"   Cats: {sum(y==0):,}  |  Dogs: {sum(y==1):,}")

# ─── Split ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─── Scale ────────────────────────────────────────────────────────────────────
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── PCA ──────────────────────────────────────────────────────────────────────
print("\n🔄 PCA (95% variance)...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"   {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} features")

# ─── Train ────────────────────────────────────────────────────────────────────
print("\n🤖 Training SVM (C=50, RBF)...")
start = time.time()
svm = SVC(kernel='rbf', C=50.0, gamma='scale',
          random_state=42, probability=True, cache_size=1000)
svm.fit(X_train_pca, y_train)
training_time = time.time() - start
print(f"   ✅ Done in {training_time:.1f}s")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred  = svm.predict(X_test_pca)
y_proba = svm.predict_proba(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

conf_scores = np.max(y_proba, axis=1)
avg_conf  = float(np.mean(conf_scores))
high_conf = float(np.mean(conf_scores >= 0.75) * 100)
low_conf  = float(np.mean(conf_scores <  0.60) * 100)

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

print("\n" + "="*70)
print(f"✅ ACCURACY       : {accuracy*100:.2f}%")
print(f"🔥 AVG CONFIDENCE : {avg_conf*100:.1f}%")
print(f"✅ HIGH CONF ≥75% : {high_conf:.1f}%  |  ⚠️ LOW CONF <60%: {low_conf:.1f}%")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(svm,    "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca,    "models/pca.pkl")

results = {
    'accuracy':           float(accuracy),
    'avg_confidence':     avg_conf,
    'high_conf_pct':      high_conf,
    'low_conf_pct':       low_conf,
    'precision':          float(precision),
    'recall':             float(recall),
    'f1_score':           float(f1),
    'train_size':         int(len(X_train)),
    'test_size':          int(len(X_test)),
    'total_images':       int(len(X)),
    'cats':               int(sum(y==0)),
    'dogs':               int(sum(y==1)),
    'image_size':         IMG_SIZE,
    'features_original':  int(X_train_scaled.shape[1]),
    'features_after_pca': int(X_train_pca.shape[1]),
    'feature_type':       'HOG + HSV color histogram',
    'kernel':             'rbf',
    'c_value':            50.0,
    'gamma':              'scale',
    'training_time':      training_time,
    'pca_variance':       0.95,
    'confusion_matrix': {
        'true_negatives':  int(tn), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_positives':  int(tp)
    }
}
with open("models/results.json", 'w') as f:
    json.dump(results, f, indent=4)

print("\n💾 Saved: svm_model.pkl · scaler.pkl · pca.pkl · results.json")
print("="*70)
if accuracy >= 0.78:
    print(f"🎉 {accuracy*100:.1f}% — target achieved!")
elif accuracy >= 0.75:
    print(f"✅ {accuracy*100:.1f}% — above 75% goal!")
else:
    print(f"👍 {accuracy*100:.1f}% — try IMAGES_PER_CLASS=5000 or C=100")
print("="*70)
print("\nNext: streamlit run app.py")