# prepare_dataset.py
import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)

def extract_landmarks(image):
    """Return 126 features (21*3*2 hands)."""
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None
    feats = []
    for hlm in results.multi_hand_landmarks[:2]:
        for lm in hlm.landmark:
            feats.extend([lm.x, lm.y, lm.z])
    if len(results.multi_hand_landmarks) == 1:
        feats.extend([0.0] * (21 * 3))
    return feats

# --- paths ---
TRAIN_DIR = "data/train"
TRAIN_CSV = "data/train.csv"
OUT_CSV   = "data/samples.csv"

df = pd.read_csv(TRAIN_CSV)
print("Loaded:", df.shape, "rows")

rows = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    label = row['label'] if 'label' in row else row.get('gesture', None)
    if label is None:
        continue
    img_path = os.path.join(TRAIN_DIR, label, filename)
    if not os.path.exists(img_path):
        img_path = os.path.join(TRAIN_DIR, filename)
        if not os.path.exists(img_path):
            continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    feats = extract_landmarks(img)
    if feats is not None:
        rows.append(feats + [label])

if rows:
    cols = [f"f{i}" for i in range(len(rows[0]) - 1)] + ["label"]
    pd.DataFrame(rows, columns=cols).to_csv(OUT_CSV, index=False)
    print(f"✅ Saved {len(rows)} samples to {OUT_CSV}")
else:
    print("⚠️ No landmarks extracted. Check dataset folder or file names.")
