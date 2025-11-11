# app.py
import os
import cv2
import joblib
import numpy as np
import streamlit as st
from datetime import datetime

from utils import (
    extract_hand_landmarks_bgr,
    extract_from_live_frame,
    is_operational_now
)

st.set_page_config(page_title="🤟 Sign Language Detector", layout="wide")
st.title("🤟 Sign Language Detection — Prediction")
st.caption("Uses a pre-trained scikit-learn model on MediaPipe hand landmarks. Predictions allowed 6–10 PM.")

MODEL_PATH = "models/sign_model.joblib"
os.makedirs("models", exist_ok=True)

with st.sidebar:
    st.header("⚙ Settings")
    st.write("Model file:", MODEL_PATH)
    st.info("Prediction tabs work only *6:00 PM – 10:00 PM* (local time).")

def check_time_window():
    now = datetime.now()
    if not is_operational_now(now):
        st.warning("⏰ The detector is available only between *6:00 PM and 10:00 PM*. Try again later.")
        return False
    return True

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train and save it as models/sign_model.joblib.")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    return model

tabs = st.tabs(["Predict: Image Upload", "Predict: Real-Time Video"])

# ---------- TAB 1: IMAGE UPLOAD ----------
with tabs[0]:
    st.subheader("🖼 Predict from Image")
    can_run = check_time_window()
    uploaded = st.file_uploader("Upload an image (hand visible)", type=["jpg", "jpeg", "png"], key="upload_img")

    if st.button("Run Prediction (Image)", key="predict_img_btn") and can_run:
        if uploaded is None:
            st.error("Please upload an image first.")
        else:
            model = load_model()
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Input", channels="RGB")

            feats = extract_hand_landmarks_bgr(frame)
            if feats is None:
                st.warning("No hands detected.")
            else:
                pred = model.predict([feats])[0]
                # Confidence if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([feats])[0]
                    classes = list(model.classes_)
                    conf = float(probs[classes.index(pred)]) * 100.0
                    st.success(f"Prediction: *{pred}* (≈ {conf:.1f}% confidence)")
                else:
                    st.success(f"Prediction: *{pred}*")

# ---------- TAB 2: REAL-TIME VIDEO ----------
with tabs[1]:
    st.subheader("🎥 Predict from Webcam (Real-Time)")
    can_run = check_time_window()
    cam_index = st.number_input("Camera index (0 if default)", 0, 5, 0, key="predict_cam_index")
    start_cam = st.button("Start Camera", key="start_cam_btn") if can_run else False

    if start_cam:
        model = load_model()
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("Could not open camera.")
        else:
            st.info("Press *Stop* to end.")
            stop = st.button("Stop", key="stop_cam_btn")
            frame_placeholder = st.empty()

            # If classifier supports probabilities, prepare once
            proba_enabled = hasattr(model, "predict_proba")

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera frame not available.")
                    break
                frame = cv2.flip(frame, 1)

                feats = extract_from_live_frame(frame)
                overlay = frame.copy()
                text = "No hands"

                if feats is not None:
                    pred = model.predict([feats])[0]
                    text = f"Pred: {pred}"
                    if proba_enabled:
                        probs = model.predict_proba([feats])[0]
                        classes = list(model.classes_)
                        conf = float(probs[classes.index(pred)]) * 100.0
                        text += f" ({conf:.0f}%)"

                cv2.putText(overlay, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                frame_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            st.success("Camera stopped.")

