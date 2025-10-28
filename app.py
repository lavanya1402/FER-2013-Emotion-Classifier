# app.py — FER-2013 Emotion Classifier (Snapshot + Live)

import io
import os
import cv2
import av
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model

# ---------------- Config ----------------
# Use your absolute file path — ensure the 'r' prefix (raw string)
MODEL_PATH = r"C:\Users\lavan\Downloads\fer2013_emotion_model (1).h5"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 48

st.set_page_config(page_title="FER-2013 Emotion Classifier", page_icon="🎭", layout="centered")
st.title("🎭 FER-2013 Emotion Classifier")
st.caption("Switch between **Snapshot** and **Live** webcam. CNN trained on FER-2013 (48×48 grayscale).")

# ---------------- Cached resources ----------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}. Please check the file path.")
        st.stop()
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = get_model()
face_cascade = get_face_cascade()

# ---------------- Helpers ----------------
def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)  # (48,48,1)
    face = np.expand_dims(face, axis=0)   # (1,48,48,1)
    return face

def detect_largest_face_bgr(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, gray
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    return (x, y, w, h), gray

def predict_from_pil(pil_img: Image.Image):
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    box, gray = detect_largest_face_bgr(img_bgr)
    if box is None:
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
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs

# ---------------- UI ----------------
tab_snap, tab_live, tab_about = st.tabs(["📷 Snapshot", "🎥 Live", "ℹ️ About"])

# ---- Snapshot tab ----
with tab_snap:
    c1, c2 = st.columns(2)
    with c1:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        st.caption("Tip: close-up frontal face, good lighting.")
    with c2:
        cam_img = st.camera_input("Or capture from webcam")

    img = None
    if cam_img is not None:
        img = Image.open(io.BytesIO(cam_img.getvalue()))
    elif uploaded is not None:
        img = Image.open(uploaded)

    if img:
        st.image(img, caption="Input image", use_column_width=True)
        label, probs = predict_from_pil(img)

        st.subheader(f"Prediction: **{label.upper()}**")
        st.bar_chart({"probability": probs})
        st.caption("Order: " + ", ".join(CLASS_NAMES))

        top3 = np.argsort(probs)[::-1][:3]
        st.write("**Top-3**")
        for i in top3:
            st.write(f"- {CLASS_NAMES[i]}: {probs[i]:.3f}")

# ---- Live tab ----
with tab_live:
    st.info("Click **Start** and grant camera permission. The largest detected face is analyzed each frame.")
    smoothing = st.slider("Label smoothing (EMA)", 0.0, 0.95, 0.7, 0.05)

    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            box, gray = detect_largest_face_bgr(img)

            if box is not None:
                x, y, w, h = box
                roi = gray[y:y+h, x:x+w]
                x_in = preprocess_face(roi)
                probs = model.predict(x_in, verbose=0)[0].astype(np.float32)

                self.last_probs = smoothing * self.last_probs + (1 - smoothing) * probs
                idx = int(np.argmax(self.last_probs))
                label = CLASS_NAMES[idx]
                conf = float(self.last_probs[idx])

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 180, 255), 2)
                tag = f"{label.upper()}  {conf:.2f}"
                cv2.rectangle(img, (x, y-28), (x + max(140, w), y), (0, 180, 255), -1)
                cv2.putText(img, tag, (x+6, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="fer2013-live",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=EmotionProcessor,
        async_processing=True,
    )

# ---- About tab ----
with tab_about:
    st.markdown(
        """
**Model:** CNN on FER-2013 (48×48 grayscale).  
**Classes:** angry, disgust, fear, happy, sad, surprise, neutral.  
**Pipeline:** upload/camera → face detect (Haarcascade) → resize 48×48 → normalize → predict.  
**Live mode:** WebRTC + EMA smoothing for stable labels.  
*Use good lighting and frontal faces for best results.*
        """
    )
