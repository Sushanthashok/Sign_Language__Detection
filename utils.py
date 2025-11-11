# utils.py
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

mp_hands = mp.solutions.hands

def _extract(frame_bgr, static_mode):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:
        res = hands.process(img_rgb)

    if not res.multi_hand_landmarks:
        return None

    feats = []
    for hand_landmarks in res.multi_hand_landmarks[:2]:
        for lm in hand_landmarks.landmark:
            feats.extend([lm.x, lm.y, lm.z])
    if len(res.multi_hand_landmarks) == 1:
        feats.extend([0.0] * (21 * 3))  # pad second hand
    return np.array(feats, dtype=np.float32)

def extract_hand_landmarks_bgr(frame_bgr):
    return _extract(frame_bgr, static_mode=True)

def extract_from_live_frame(frame_bgr):
    return _extract(frame_bgr, static_mode=False)

def is_operational_now(now_dt: datetime):
    # Allowed between 18:00 (inclusive) and 22:00 (exclusive)
    return 18 <= now_dt.hour < 22