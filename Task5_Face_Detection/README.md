# 🧠 Face Detection & Recognition AI

A complete, production-ready **Face Detection and Recognition** web application built with **Python, OpenCV, and Streamlit**.

> 🎓 Built for **CodSoft AI Internship — Task 5**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project detects and recognizes human faces in **images** and **live webcam video** using:

- **Haar Cascade Classifier** (OpenCV) for face **detection**
- **LBPH (Local Binary Patterns Histograms)** algorithm for face **recognition**
- A modern, multi-page **Streamlit** web interface

It allows users to upload images, run live webcam detection, register new people, train a recognition model, and identify registered individuals in real time.

---

## ✨ Features

| Page | Description |
|---|---|
| 🏠 **Home** | Project overview, features, and tech stack |
| 🖼️ **Image Face Detection** | Upload an image, detect faces, view count, download result |
| 🎥 **Webcam Face Detection** | Real-time face detection from your webcam |
| 🧑‍🤝‍🧑 **Face Recognition** | Register people, train the model, recognize faces live with names |
| ℹ️ **About** | Explanation of the AI concepts used (Haar Cascade, LBPH, OpenCV) |

---

## 🗂️ Project Structure

```
Face_Detection_Recognition/
│
├── app.py                         # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── registered_faces/               # Saved face samples per person + trained model
│
├── outputs/                        # Saved processed images
│
├── src/
│   ├── __init__.py
│   ├── face_detector.py            # Haar Cascade detection logic
│   ├── face_recognition.py         # LBPH registration/training/recognition logic
│   ├── image_processor.py          # Image conversion & utility helpers
│   └── webcam_detector.py          # Webcam capture & frame processing
│
└── assets/                         # Static assets (screenshots, icons, etc.)
```

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **OpenCV** (`opencv-contrib-python`) — Haar Cascade detection + LBPH recognition
- **Streamlit** — Interactive web UI
- **NumPy** — Numerical operations on image arrays
- **Pandas** — Displaying detection coordinate tables
- **Pillow (PIL)** — Image format handling for uploads/downloads

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Face_Detection_Recognition.git
cd Face_Detection_Recognition
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> ⚠️ **Note:** The Webcam Detection and Webcam-based Recognition features require running the app **locally** on a machine with a connected webcam — they will not work on a remote/cloud-hosted server without additional configuration (e.g., WebRTC).

---

## 🚀 How to Use

### Image Face Detection
1. Go to **🖼️ Image Face Detection**.
2. Upload a `.jpg`, `.jpeg`, or `.png` image.
3. View the detected faces with bounding boxes and the total count.
4. Download the processed image.

### Webcam Face Detection
1. Go to **🎥 Webcam Face Detection**.
2. Click **Start Webcam**.
3. Faces are detected live with a running face count.
4. Click **Stop Webcam** to end the session.

### Face Recognition
1. Go to **🧑‍🤝‍🧑 Face Recognition → Register Person** tab.
2. Enter a name and upload a few clear photos (or capture via webcam).
3. Go to the **Train Model** tab and click **Train Model Now**.
4. Go to the **Recognize (Webcam)** tab, click **Start Recognition**, and see registered people identified by name (unregistered faces show as **"Unknown"**).

---

## 🧠 AI Concepts Used

- **Haar Cascade Classifier** — A fast, classical object-detection algorithm trained on positive/negative face samples using AdaBoost, used here for locating faces.
- **LBPH (Local Binary Patterns Histograms)** — A texture-based face recognition algorithm that builds histogram feature vectors from local pixel patterns, robust to lighting variations.
- **Grayscale Image Processing** — Both detection and recognition operate on grayscale images for speed and consistency.

See the **About** page inside the app for a deeper explanation of each concept.

---

## 📋 Requirements

See [`requirements.txt`](./requirements.txt):
```
streamlit>=1.32.0
opencv-contrib-python>=4.9.0.80
numpy>=1.26.0
pandas>=2.2.0
Pillow>=10.2.0
```

---

## 🔮 Future Improvements

- Replace Haar Cascade with a deep-learning detector (e.g., MTCNN, RetinaFace, or DNN-based SSD) for higher accuracy
- Replace LBPH with a deep embedding model (e.g., FaceNet/ArcFace) for better recognition accuracy
- Add `streamlit-webrtc` for browser-based webcam support on cloud deployments
- Add a database (SQLite) instead of flat files for registered people
- Add face mask / age / emotion detection as additional modules

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [OpenCV](https://opencv.org/) for the Haar Cascade and LBPH implementations
- [Streamlit](https://streamlit.io/) for the rapid web app framework
- **CodSoft** for the internship task that inspired this project
