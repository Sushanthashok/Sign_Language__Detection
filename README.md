

# 🤟 Sign Language Detection

---

## 🧩 Problem Statement

Communication barriers exist between hearing-impaired individuals and those unfamiliar with sign language.  
To address this, the goal of this project is to develop a **machine learning model** capable of detecting and interpreting **sign language gestures** into corresponding letters or words.

The system must:
- Recognize **hand gestures** representing English alphabets (A–Z), including "space" and "del".
- Operate through both **image input** and **real-time webcam**.
- Include a GUI for interactive use.
- (Optional) Function only during a specific **time window (6 PM – 10 PM)**.

This solution aims to promote inclusivity and ease of communication for the hearing-impaired community using accessible computer vision techniques.

---

## 🧠 Dataset

**Dataset Used:**  
[Kaggle: Sign Language Recognition Dataset](https://www.kaggle.com/datasets/gauravduttakit/sign-language-recognition)

**Dataset Description:**
- Contains images of hands performing **American Sign Language (ASL)** gestures.
- **29 classes:** A–Z, `space`, and `del`.
- Images are RGB with variations in lighting, angle, and hand position.
- Data organized into folders by label, with metadata in `train.csv` and `test.csv`.

**Dataset Preparation Steps:**
1. Each image is passed through **MediaPipe Hands** to extract **21 landmark coordinates** (x, y, z) per hand.
2. For each image:
   - If one hand detected → pad missing hand with zeros.
   - If both hands detected → concatenate their features.
3. Final features per image:  
   `21 landmarks × 3 coordinates × 2 hands = 126 features`.
4. The extracted data is saved as `data/samples.csv` for model training.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Used **MediaPipe Hands** to detect 3D hand landmarks.
- Flattened landmark coordinates into feature vectors.
- Filtered out incomplete or invalid detections.
- Saved processed data into a CSV file for machine learning input.

### 2️⃣ Model Training
- Used **K-Nearest Neighbors (KNN)** classifier from **scikit-learn**.
- Distance-weighted voting for better accuracy on similar signs.
- Split dataset into **80% training** and **20% testing** using stratified sampling.
- Filtered out any class with fewer than 5 samples.

### 3️⃣ Model Evaluation
- Metrics used:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Evaluated using `classification_report()` from scikit-learn.

### 4️⃣ Application Development
- Built an interactive **Streamlit** GUI with:
  - **Image Upload** for static predictions.
  - **Real-Time Webcam Detection** using OpenCV.
- Time-based restriction included (6 PM – 10 PM active hours).
- Displays real-time predictions with confidence score.

---

## 📈 Results

| Metric | Value |
|--------|--------|
| **Model Type** | KNeighborsClassifier (distance weighted) |
| **Training Samples** | 25,000+ |
| **Features** | 126 (x, y, z for both hands) |
| **Accuracy** | **97.4%** |
| **Precision / Recall / F1** | ~0.97 average across all classes |


<img src="assets/sign language.png" width="500"/>


**Observation:**
- Model performs consistently across most signs (A–Z).
- Minor variation in signs with similar hand orientation (e.g., M/N/U/V).
- Performs well in real-time conditions with proper lighting.

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python |
| Framework | Streamlit |
| Computer Vision | MediaPipe Hands, OpenCV |
| Machine Learning | scikit-learn (KNN) |
| Data Handling | NumPy, pandas |
| Model Storage | joblib |

---

## 🎯 Conclusion

The **Sign Language Detection Model** successfully recognizes static sign gestures for English alphabets with high accuracy.  
The model can operate both in **image** and **real-time video** modes, offering an effective and accessible solution for basic sign language interpretation.

This project demonstrates how **AI and computer vision** can be leveraged to build inclusive tools that bridge communication gaps.

---

## 🧩 Future Enhancements
- Extend support for **dynamic gestures (words or phrases)**.
- Integrate **text-to-speech** for real-time translation.
- Train with larger, multi-user datasets for generalization.
- Deploy on **Streamlit Cloud** or **HuggingFace Spaces** for public access.


 ## Data and Visual Output

 [Drive Link](https://drive.google.com/drive/folders/1J1UxAA_md36Bp-Pra0e3L67t7lAQKCBg?usp=sharing)

 


