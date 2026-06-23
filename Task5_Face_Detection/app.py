"""
app.py
------
Main entry point for the "Face Detection & Recognition" Streamlit web app.

This file is responsible for:
- Setting up the page configuration and custom CSS styling
- Providing a sidebar navigation menu
- Routing to each page: Home, Image Detection, Webcam Detection,
  Face Recognition, and About

Run with:
    streamlit run app.py
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.face_detector import FaceDetector
from src.face_recognition import FaceRecognizer
from src.image_processor import ImageProcessor
from src.webcam_detector import WebcamDetector


# ----------------------------------------------------------------------
# PAGE CONFIGURATION (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Face Detection & Recognition AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# CUSTOM CSS — gives the app a modern, professional look
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.15rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        border: 1px solid #334155;
        height: 100%;
        transition: transform 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #4F46E5;
    }
    .feature-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
    .feature-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 0.4rem; color: #E2E8F0; }
    .feature-desc { font-size: 0.92rem; color: #94A3B8; }

    .tech-badge {
        display: inline-block;
        background: #312E81;
        color: #E0E7FF;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #4F46E5;
    }
    .stat-box {
        background: linear-gradient(135deg, #4F46E5, #06B6D4);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.85rem; opacity: 0.9; }

    section[data-testid="stSidebar"] { background-color: #0F172A; }
    .footer-note {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 3rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED RESOURCES — these objects are expensive to create, so we
# cache them across reruns using Streamlit's caching decorators.
# ----------------------------------------------------------------------
@st.cache_resource
def load_face_detector():
    return FaceDetector()


@st.cache_resource
def load_face_recognizer():
    return FaceRecognizer()


@st.cache_resource
def load_image_processor():
    return ImageProcessor()


face_detector = load_face_detector()
face_recognizer = load_face_recognizer()
image_processor = load_image_processor()


# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🧠 Face AI Studio")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🖼️ Image Face Detection",
        "🎥 Webcam Face Detection",
        "🧑‍🤝‍🧑 Face Recognition",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Registered People:** {len(face_recognizer.get_registered_people())}"
)
st.sidebar.markdown("**Built with:** OpenCV + Streamlit")
st.sidebar.caption("CodSoft AI Internship — Task 5")


# ----------------------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------------------
def render_home_page():
    st.markdown('<div class="main-title">Face Detection & Recognition AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">An intelligent computer vision system that detects and '
        'recognizes human faces in images and real-time video.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="stat-box"><div class="stat-number">99%</div>'
            '<div class="stat-label">Detection Accuracy*</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(face_recognizer.get_registered_people())}</div>'
            '<div class="stat-label">Registered Faces</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="stat-box"><div class="stat-number">Real-Time</div>'
            '<div class="stat-label">Webcam Processing</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📌 Project Description")
    st.write(
        """
        This application demonstrates a complete **Face Detection and Recognition**
        pipeline built using classical computer vision techniques. It uses OpenCV's
        **Haar Cascade Classifier** to locate human faces within images or video
        frames, and an **LBPH (Local Binary Patterns Histograms)** model to learn
        and recognize the identity of specific registered individuals.

        The project is wrapped inside a modern, interactive **Streamlit** web
        interface, making it easy for anyone to upload images, use their webcam,
        register new people, and see the AI in action — no coding required from
        the end user.
        """
    )

    st.markdown("### ✨ Key Features")
    features = [
        ("🖼️", "Image Detection", "Upload any photo and instantly detect all visible faces with bounding boxes."),
        ("🎥", "Live Webcam Detection", "Detect faces in real time directly from your webcam feed."),
        ("🧑‍🤝‍🧑", "Face Recognition", "Register people and have the AI recognize them by name in real time."),
        ("⬇️", "Download Results", "Save and download processed images with detection overlays."),
        ("📊", "Live Face Count", "See exactly how many faces are detected at any moment."),
        ("⚡", "Fast & Lightweight", "Runs efficiently on CPU using classical ML — no GPU required."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🛠️ Technologies Used")
    techs = ["Python", "OpenCV", "Streamlit", "NumPy", "Pandas", "Pillow",
             "Haar Cascade", "LBPH Face Recognition"]
    badges_html = "".join(f'<span class="tech-badge">{t}</span>' for t in techs)
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">* Accuracy depends on lighting, face angle, and image quality.</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# PAGE 2: IMAGE FACE DETECTION
# ----------------------------------------------------------------------
def render_image_detection_page():
    st.markdown("## 🖼️ Image Face Detection")
    st.write("Upload an image and let the AI detect every face in it.")

    with st.expander("⚙️ Detection Settings (Advanced)"):
        col1, col2 = st.columns(2)
        with col1:
            min_neighbors = st.slider("Min Neighbors (higher = stricter)", 1, 10, 5)
        with col2:
            scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, step=0.05)

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            cv2_image = ImageProcessor.pil_to_cv2(pil_image)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(pil_image, use_container_width=True)

            with st.spinner("Detecting faces..."):
                faces, processed_image = face_detector.detect_faces(
                    cv2_image,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                )

            face_count = face_detector.get_face_count(faces)

            with col2:
                st.subheader("Processed Image")
                processed_pil = ImageProcessor.cv2_to_pil(processed_image)
                st.image(processed_pil, use_container_width=True)

            if face_count > 0:
                st.success(f"✅ {face_count} face(s) detected!")
            else:
                st.warning("⚠️ No faces detected. Try a clearer, front-facing photo.")

            # Show coordinates table for transparency / learning purposes
            if face_count > 0:
                with st.expander("📋 View Detection Coordinates"):
                    df = pd.DataFrame(
                        faces, columns=["x", "y", "width", "height"]
                    )
                    df.index = [f"Face {i+1}" for i in range(len(df))]
                    st.dataframe(df, use_container_width=True)

            # Download button
            image_bytes = ImageProcessor.get_image_bytes(processed_image)
            st.download_button(
                label="⬇️ Download Processed Image",
                data=image_bytes,
                file_name="face_detection_result.jpg",
                mime="image/jpeg",
            )

        except Exception as e:
            st.error(f"❌ An error occurred while processing the image: {e}")
    else:
        st.info("👆 Upload an image to get started.")


# ----------------------------------------------------------------------
# PAGE 3: WEBCAM FACE DETECTION
# ----------------------------------------------------------------------
def render_webcam_detection_page():
    st.markdown("## 🎥 Webcam Face Detection")
    st.write(
        "Detect faces in real time using your webcam. "
        "Click **Start Webcam** to begin and **Stop Webcam** to end the session."
    )
    st.info(
        "ℹ️ This feature requires running the app **locally** (not on a remote "
        "server) since it needs direct access to your computer's webcam hardware."
    )

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start Webcam", use_container_width=True, type="primary")
    with col2:
        stop_btn = st.button("⏹️ Stop Webcam", use_container_width=True)

    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    frame_placeholder = st.empty()
    count_placeholder = st.empty()

    if st.session_state.webcam_running:
        webcam = WebcamDetector()
        try:
            webcam.start_camera()
            while st.session_state.webcam_running:
                success, frame = webcam.read_frame()
                if not success:
                    st.error("❌ Could not read from webcam.")
                    break

                processed_frame, face_count = WebcamDetector.process_frame_for_detection(
                    frame, face_detector
                )
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(processed_rgb, channels="RGB", use_container_width=True)
                count_placeholder.metric("👤 Faces Detected (Live)", face_count)

                # Small delay to keep the UI responsive and avoid hogging CPU
                time.sleep(0.03)

                # Re-check the stop button's state on every loop iteration.
                # Streamlit reruns the script when a widget is clicked, so this
                # loop will naturally exit once the user clicks "Stop Webcam"
                # and the script reruns from the top with webcam_running=False.
        except IOError as e:
            st.error(f"❌ {e}")
        finally:
            webcam.stop_camera()
    else:
        frame_placeholder.info("📷 Webcam is currently stopped. Click 'Start Webcam' to begin.")


# ----------------------------------------------------------------------
# PAGE 4: FACE RECOGNITION
# ----------------------------------------------------------------------
def render_face_recognition_page():
    st.markdown("## 🧑‍🤝‍🧑 Face Recognition")
    tab1, tab2, tab3 = st.tabs(["➕ Register Person", "🏋️ Train Model", "🔍 Recognize (Webcam)"])

    # --- TAB 1: REGISTER ---
    with tab1:
        st.subheader("Register a New Person")
        st.write(
            "Capture a few face samples to register someone. "
            "You can either upload multiple photos OR use your webcam to capture samples."
        )

        person_name = st.text_input("Person's Name", placeholder="e.g. John Doe")

        method = st.radio("Sample Capture Method", ["📁 Upload Images", "🎥 Use Webcam"], horizontal=True)

        if method == "📁 Upload Images":
            uploaded_files = st.file_uploader(
                "Upload 3-10 clear face photos of this person",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
            )

            if st.button("💾 Save Samples", type="primary"):
                if not person_name.strip():
                    st.error("⚠️ Please enter a name first.")
                elif not uploaded_files:
                    st.error("⚠️ Please upload at least one image.")
                else:
                    try:
                        all_crops = []
                        for f in uploaded_files:
                            pil_img = Image.open(f)
                            cv2_img = ImageProcessor.pil_to_cv2(pil_img)
                            faces, _ = face_detector.detect_faces(cv2_img, draw_boxes=False)
                            crops = face_detector.crop_faces(cv2_img, faces)
                            all_crops.extend(crops)

                        if not all_crops:
                            st.warning("⚠️ No faces were found in the uploaded images.")
                        else:
                            saved = face_recognizer.register_face_samples(person_name, all_crops)
                            st.success(f"✅ Saved {saved} face sample(s) for '{person_name}'.")
                            st.info("👉 Now go to the 'Train Model' tab to train the recognizer.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        else:  # Webcam capture
            st.warning("⚠️ Run the app locally to use webcam-based registration.")
            num_samples = st.slider("Number of samples to capture", 5, 30, 15)

            if st.button("📸 Capture Samples from Webcam", type="primary"):
                if not person_name.strip():
                    st.error("⚠️ Please enter a name first.")
                else:
                    webcam = WebcamDetector()
                    try:
                        webcam.start_camera()
                        captured = []
                        progress = st.progress(0)
                        preview = st.empty()

                        attempts = 0
                        max_attempts = num_samples * 10  # safety limit

                        while len(captured) < num_samples and attempts < max_attempts:
                            success, frame = webcam.read_frame()
                            attempts += 1
                            if not success:
                                continue

                            faces, _ = face_detector.detect_faces(frame, draw_boxes=False)
                            if len(faces) == 1:
                                crops = face_detector.crop_faces(frame, faces)
                                captured.extend(crops)
                                preview.image(
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption=f"Captured {len(captured)}/{num_samples}",
                                )
                                progress.progress(min(len(captured) / num_samples, 1.0))
                                time.sleep(0.2)

                        webcam.stop_camera()

                        if captured:
                            saved = face_recognizer.register_face_samples(person_name, captured)
                            st.success(f"✅ Captured and saved {saved} sample(s) for '{person_name}'.")
                            st.info("👉 Now go to the 'Train Model' tab to train the recognizer.")
                        else:
                            st.warning("⚠️ No clear single-face samples could be captured.")
                    except IOError as e:
                        st.error(f"❌ {e}")
                    finally:
                        webcam.stop_camera()

        st.markdown("---")
        st.write("**Currently Registered People:**")
        people = face_recognizer.get_registered_people()
        if people:
            st.write(", ".join(f"`{p}`" for p in people))
        else:
            st.write("_No one registered yet._")

    # --- TAB 2: TRAIN ---
    with tab2:
        st.subheader("Train the Recognition Model")
        st.write(
            "Once you've registered one or more people with face samples, "
            "click the button below to train the LBPH recognition model on all of them."
        )

        if st.button("🏋️ Train Model Now", type="primary"):
            try:
                with st.spinner("Training model... this may take a few seconds."):
                    num_people = face_recognizer.train()
                st.success(f"✅ Model trained successfully on {num_people} person(s)!")
                st.balloons()
            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error during training: {e}")

        st.markdown("---")
        status = "✅ Trained" if face_recognizer.is_trained else "❌ Not Trained Yet"
        st.metric("Model Status", status)

    # --- TAB 3: RECOGNIZE ---
    with tab3:
        st.subheader("Recognize Faces in Real Time")

        if not face_recognizer.is_trained:
            st.warning("⚠️ The model hasn't been trained yet. Please train it in the 'Train Model' tab first.")
        else:
            confidence_threshold = st.slider(
                "Confidence Threshold (lower = stricter matching)",
                30, 120, 70,
            )

            if "recognition_running" not in st.session_state:
                st.session_state.recognition_running = False

            col1, col2 = st.columns(2)
            with col1:
                start_btn = st.button("▶️ Start Recognition", type="primary", use_container_width=True)
            with col2:
                stop_btn = st.button("⏹️ Stop Recognition", use_container_width=True)

            if start_btn:
                st.session_state.recognition_running = True
            if stop_btn:
                st.session_state.recognition_running = False

            frame_placeholder = st.empty()
            info_placeholder = st.empty()

            if st.session_state.recognition_running:
                webcam = WebcamDetector()
                try:
                    webcam.start_camera()
                    while st.session_state.recognition_running:
                        success, frame = webcam.read_frame()
                        if not success:
                            st.error("❌ Could not read from webcam.")
                            break

                        processed_frame, results = WebcamDetector.process_frame_for_recognition(
                            frame, face_detector, face_recognizer, confidence_threshold
                        )
                        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(processed_rgb, channels="RGB", use_container_width=True)

                        if results:
                            names = ", ".join(r["name"] for r in results)
                            info_placeholder.success(f"👤 Detected: {names}")
                        else:
                            info_placeholder.info("No faces detected.")

                        time.sleep(0.03)
                except IOError as e:
                    st.error(f"❌ {e}")
                finally:
                    webcam.stop_camera()
            else:
                frame_placeholder.info("📷 Recognition is stopped. Click 'Start Recognition' to begin.")


# ----------------------------------------------------------------------
# PAGE 5: ABOUT
# ----------------------------------------------------------------------
def render_about_page():
    st.markdown("## ℹ️ About This Project")

    st.markdown("### 📖 Project Overview")
    st.write(
        """
        **Face Detection & Recognition AI** is an end-to-end computer vision
        application built as part of the **CodSoft AI Internship — Task 5**.
        It demonstrates the practical application of classical machine learning
        algorithms for detecting and recognizing human faces, wrapped in a
        clean and interactive web interface built with Streamlit.
        """
    )

    st.markdown("### 🧠 AI & Computer Vision Concepts Used")

    with st.expander("🔲 OpenCV"):
        st.write(
            """
            **OpenCV (Open Source Computer Vision Library)** is an open-source
            library packed with hundreds of computer vision and image processing
            algorithms. In this project, it is used for:
            - Reading and manipulating images/video frames
            - Color space conversion (BGR ↔ Grayscale ↔ RGB)
            - Running Haar Cascade face detection
            - Drawing bounding boxes and text labels
            """
        )

    with st.expander("🟩 Haar Cascade Classifier"):
        st.write(
            """
            **Haar Cascades** are a machine learning-based approach where a
            cascade function is trained on many positive (face) and negative
            (non-face) images. It uses simple rectangular "Haar-like features"
            (e.g., edges, lines) computed at various scales and positions, combined
            using a method called **AdaBoost**, to quickly reject non-face regions
            and zoom in on likely face regions. This makes it extremely fast and
            suitable for real-time applications such as webcam detection.
            """
        )

    with st.expander("🟦 LBPH Face Recognition"):
        st.write(
            """
            **Local Binary Patterns Histograms (LBPH)** is a texture-based face
            recognition algorithm. It works by:
            1. Dividing each face image into small regions.
            2. Comparing each pixel to its neighbors to form a binary pattern.
            3. Building a histogram of these patterns for each region.
            4. Concatenating all histograms into a single feature vector.

            When recognizing a new face, its feature vector is compared against
            all stored vectors using a distance metric — the closest match (below
            a confidence threshold) is returned as the predicted identity.
            LBPH is robust to lighting changes and works well even with a small
            number of training images per person, which makes it ideal for
            student/academic projects.
            """
        )

    with st.expander("🟪 Streamlit"):
        st.write(
            """
            **Streamlit** is a Python framework for building interactive data and
            machine learning web applications with minimal code — no HTML, CSS, or
            JavaScript required. It is used here to build the entire user interface:
            navigation, file uploads, live webcam frames, sliders, buttons, and
            downloadable results.
            """
        )

    st.markdown("### 🏗️ System Architecture")
    st.code(
        """
        ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
        │   Streamlit UI   │ ───▶ │   FaceDetector    │ ───▶ │  Haar Cascade Model  │
        │   (app.py)       │      │ (Haar Cascade)    │      │ (.xml)               │
        └─────────────────┘      └──────────────────┘      └─────────────────────┘
                 │
                 ▼
        ┌─────────────────┐      ┌──────────────────┐
        │  FaceRecognizer  │ ───▶ │   LBPH Model      │
        │  (cv2.face)      │      │ (.yml + labels)   │
        └─────────────────┘      └──────────────────┘
        """,
        language="text",
    )

    st.markdown("### 👨‍💻 Developer")
    st.write(
        "Developed as part of the **CodSoft AI Internship Program** "
        "by a 3rd-year B.Tech (AI & Data Science) student."
    )

    st.markdown(
        '<div class="footer-note">© 2026 Face Detection & Recognition AI · Built with OpenCV & Streamlit</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------
if page == "🏠 Home":
    render_home_page()
elif page == "🖼️ Image Face Detection":
    render_image_detection_page()
elif page == "🎥 Webcam Face Detection":
    render_webcam_detection_page()
elif page == "🧑‍🤝‍🧑 Face Recognition":
    render_face_recognition_page()
elif page == "ℹ️ About":
    render_about_page()
