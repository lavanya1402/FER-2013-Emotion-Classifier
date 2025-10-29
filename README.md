# 🎭 FER-2013 Emotion Classifier

**Made by [Lavanya Srivastava](https://www.linkedin.com/in/lavanya-srivastava/)**  
Real-time & snapshot **facial emotion recognition** using a CNN trained on **FER-2013** (48×48 grayscale).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lavanya1402/FER-2013-Emotion-Classifier/blob/main/app.py)

---

## ✨ Features
- Upload **photo** or use **webcam**.
- Detects: `angry, disgust, fear, happy, sad, surprise, neutral`.
- Shows **prediction** + **probability chart** + **full scores table**.
- Robust face detection; falls back to center-crop if no face found.

---

## 🛠️ Setup (Local)

```bash
# Python 3.10/3.11 recommended
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
