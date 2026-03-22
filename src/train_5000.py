"""
MAXIMUM ACCURACY SVM - Optimized for Speed & 75%+ Accuracy
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

print("="*70)
print("🐱🐶 OPTIMIZED SVM CLASSIFIER - 75%+ ACCURACY")
print("="*70)
print("Goal: 75%+ Accuracy | Optimized for speed")
print("="*70)

# Configuration
RAW_DATA_PATH = "data/raw/train/"  # Change to "../data/raw/train/" if running from src
IMG_SIZE = 64
IMAGES_PER_CLASS = 2500  # 2500 cats + 2500 dogs = 5000 images

print(f"\n📁 Loading {IMAGES_PER_CLASS} cats and {IMAGES_PER_CLASS} dogs...")
print(f"📸 Image size: {IMG_SIZE}x{IMG_SIZE} pixels")

# Get all files
cat_files = [f for f in os.listdir(RAW_DATA_PATH) if f.startswith('cat') and f.endswith('.jpg')]
dog_files = [f for f in os.listdir(RAW_DATA_PATH) if f.startswith('dog') and f.endswith('.jpg')]

print(f"\n   ✅ Found: {len(cat_files):,} cats, {len(dog_files):,} dogs")

# Select maximum images
random.seed(42)
selected_cats = random.sample(cat_files, min(IMAGES_PER_CLASS, len(cat_files)))
selected_dogs = random.sample(dog_files, min(IMAGES_PER_CLASS, len(dog_files)))

print(f"   📌 Using: {len(selected_cats)} cats + {len(selected_dogs)} dogs")

X = []
y = []

print("\n🖼️ Processing images (this will take 5-8 minutes)...")

def preprocess_image(img_path):
    """Optimized preprocessing for speed and accuracy"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    try:
        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply simple histogram equalization (faster than CLAHE)
        gray = cv2.equalizeHist(gray)
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        # Flatten
        return gray.flatten()
    except:
        return None

# Process cats
print("\n🐱 Processing cats:")
cat_success = 0
for i, filename in enumerate(selected_cats):
    img_path = os.path.join(RAW_DATA_PATH, filename)
    features = preprocess_image(img_path)
    
    if features is not None:
        X.append(features)
        y.append(0)
        cat_success += 1
    
    if (i+1) % 500 == 0:
        print(f"      Processed {i+1}/{len(selected_cats)} cats")

# Process dogs
print("\n🐶 Processing dogs:")
dog_success = 0
for i, filename in enumerate(selected_dogs):
    img_path = os.path.join(RAW_DATA_PATH, filename)
    features = preprocess_image(img_path)
    
    if features is not None:
        X.append(features)
        y.append(1)
        dog_success += 1
    
    if (i+1) % 500 == 0:
        print(f"      Processed {i+1}/{len(selected_dogs)} dogs")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n✅ Dataset ready!")
print(f"   • Total images: {len(X)}")
print(f"   • Cats: {sum(y==0)}")
print(f"   • Dogs: {sum(y==1)}")
print(f"   • Features per image: {X.shape[1]:,}")
print(f"   • Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")

# Split into train and test
print("\n🔀 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   • Training: {len(X_train)} images ({len(X_train)/len(X)*100:.0f}%)")
print(f"   • Testing: {len(X_test)} images ({len(X_test)/len(X)*100:.0f}%)")

# Scale features
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaling complete")

# Apply PCA (optional but helps with speed and accuracy)
print("\n🔄 Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"   • Features before PCA: {X_train_scaled.shape[1]:,}")
print(f"   • Features after PCA: {X_train_pca.shape[1]:,}")
print(f"   • Reduced by: {(1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100:.1f}%")

# Train SVM with pre-tuned parameters (no grid search)
print("\n🤖 Training SVM with optimized parameters...")
print("   • Kernel: RBF")
print("   • C: 10.0 (balanced)")
print("   • Gamma: scale (auto)")
print("   • Training samples: {:,}".format(len(X_train)))

start = time.time()
svm = SVC(
    kernel='rbf',
    C=10.0,  # Good balance between underfitting and overfitting
    gamma='scale',  # Auto gamma based on features
    random_state=42,
    probability=True,
    verbose=False,
    cache_size=500  # Increase cache for faster training
)
svm.fit(X_train_pca, y_train)
training_time = time.time() - start

print(f"\n   ✅ Training completed in {training_time:.2f} seconds")

# Evaluate
print("\n📈 Evaluating on test set...")
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*70)
print("🎯 FINAL RESULTS")
print("="*70)
print(f"\n✅ ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['🐱 Cat', '🐶 Dog']))

print("\n📊 CONFUSION MATRIX:")
print("              Predicted")
print("              Cat    Dog")
print(f"Actual Cat   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"       Dog   {cm[1,0]:5d}   {cm[1,1]:5d}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📈 DETAILED METRICS:")
print(f"   • Precision: {precision:.4f} - How many predicted dogs were correct")
print(f"   • Recall: {recall:.4f} - How many actual dogs were found")
print(f"   • F1-Score: {f1:.4f} - Harmonic mean")
print(f"   • True Positives: {tp:,} - Correctly identified dogs")
print(f"   • True Negatives: {tn:,} - Correctly identified cats")
print(f"   • False Positives: {fp:,} - Cats mistaken as dogs")
print(f"   • False Negatives: {fn:,} - Dogs mistaken as cats")

# Save model
print("\n💾 Saving model...")
os.makedirs("models", exist_ok=True)

joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
print("   ✅ Model saved to: models/svm_model.pkl")
print("   ✅ Scaler saved to: models/scaler.pkl")
print("   ✅ PCA saved to: models/pca.pkl")

# Save detailed results
results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_images': len(X),
    'cats': int(sum(y==0)),
    'dogs': int(sum(y==1)),
    'image_size': IMG_SIZE,
    'features_original': X_train_scaled.shape[1],
    'features_after_pca': X_train_pca.shape[1],
    'kernel': 'rbf',
    'c_value': 10.0,
    'gamma': 'scale',
    'training_time': training_time,
    'pca_used': True,
    'pca_variance_retained': 0.95,
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
}

with open("models/results.json", 'w') as f:
    json.dump(results, f, indent=4)

print("   ✅ Results saved to: models/results.json")

print("\n" + "="*70)
if accuracy >= 0.75:
    print("🎉 EXCELLENT! Accuracy is 75%+ - Goal achieved!")
    print(f"   {accuracy*100:.1f}% accuracy - Great work!")
elif accuracy >= 0.70:
    print("✅ GOOD! Accuracy is 70%+ - Close to target!")
    print(f"   {accuracy*100:.1f}% accuracy - Good job!")
else:
    print("⚠️ Accuracy is below 70%. Consider:")
    print("   • Using more images")
    print("   • Adjusting C value (try C=1 or C=100)")
print("="*70)

print("\n📌 NEXT STEPS:")
print("   1. Run: streamlit run app.py")
print("   2. Upload cat/dog images to test")
print("   3. Take screenshots for LinkedIn")
print("   4. Share your results!")
print("="*70)