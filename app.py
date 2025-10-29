import io
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from pathlib import Path

# ---------------- Model Path Resolver ----------------
APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "fer2013_emotion_model.keras",
    Path(r"C:\Users\lavan\Downloads\fer2013_emotion_model.keras"),  # fallback
]

def _resolve_model_path() -> Path:
    """Finds and validates model path."""
    for p in MODEL_CANDIDATES:
        if p.exists() and p.is_file() and p.stat().st_size > 1024:
            print(f"✅ Using model: {p}")
            return p
    raise FileNotFoundError(
        "❌ Could not find 'fer2013_emotion_model.keras'. "
        "Place it next to app.py or update the path in MODEL_CANDIDATES."
    )

MODEL_PATH = str(_resolve_model_path())

# ---------------- Configuration ----------------
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 48

st.set_page_config(page_title="FER-2013 Emotion Classifier", page_icon="🎭", layout="centered")
st.title("🎭 FER-2013 Emotion Classifier")
st.caption("Upload or capture a face image to analyze emotions (CNN trained on FER-2013).")

# ---------------- Load Model & Face Detector ----------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = get_model()
face_cascade = get_face_cascade()

# ---------------- Helper Functions ----------------
def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def detect_largest_face(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, gray
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    return (x, y, w, h), gray

def predict_emotion(pil_img: Image.Image):
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    box, gray = detect_largest_face(img_bgr)

    if box is None:
        st.warning("No face detected. Using center crop instead.")
        h, w = gray.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        roi = gray[y0:y0+side, x0:x0+side]
    else:
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]

    x_in = preprocess_face(roi)
    probs = model.predict(x_in, verbose=0)[0]
    label_idx = int(np.argmax(probs))
    label = CLASS_NAMES[label_idx]
    return label, probs

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
cam_img = st.camera_input("Or capture using webcam")

img = None
if uploaded_file:
    img = Image.open(uploaded_file)
elif cam_img:
    img = Image.open(io.BytesIO(cam_img.getvalue()))

if img:
    st.image(img, caption="Input Image", use_column_width=True)
    label, probs = predict_emotion(img)

    st.subheader(f"Predicted Emotion: **{label.upper()}**")
    st.bar_chart({name: [prob] for name, prob in zip(CLASS_NAMES, probs)})

    # 🔹 Show all scores nicely
    st.markdown("### Emotion Confidence Scores")
    for name, prob in zip(CLASS_NAMES, probs):
        st.write(f"- **{name.capitalize()}**: {prob:.3f}")

st.markdown("---")
st.caption("Model trained on FER-2013 (48×48 grayscale faces). Use clear, front-facing images for best results.")
