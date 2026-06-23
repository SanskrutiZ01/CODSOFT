"""
face_recognition.py
--------------------
This module implements face RECOGNITION (identifying WHO a face belongs to),
which is a step beyond face DETECTION (finding WHERE faces are).

We use OpenCV's built-in LBPH (Local Binary Patterns Histograms) Face
Recognizer, available through the `opencv-contrib-python` package
(cv2.face module). LBPH is a classic, lightweight, and beginner-friendly
algorithm that works well for small custom datasets — perfect for an
academic/internship project.

Workflow:
1. REGISTER  -> Capture multiple face images of a person and save them.
2. TRAIN     -> Train the LBPH model on all registered faces.
3. RECOGNIZE -> Predict the identity of a new/unseen face.
"""

import os
import cv2
import numpy as np
import json


class FaceRecognizer:
    """Handles registration, training, and recognition of faces."""

    def __init__(self, data_dir: str = None, model_path: str = None, labels_path: str = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.data_dir = data_dir or os.path.join(project_root, "registered_faces")
        self.model_path = model_path or os.path.join(self.data_dir, "trained_model.yml")
        self.labels_path = labels_path or os.path.join(self.data_dir, "labels.json")

        os.makedirs(self.data_dir, exist_ok=True)

        # LBPH Face Recognizer comes from the 'opencv-contrib-python' package
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Maps integer label IDs <-> person names (LBPH needs integer labels)
        self.label_to_name = {}
        self.name_to_label = {}

        self._load_labels()

        # If a trained model already exists, load it so recognition works
        # immediately without needing to retrain every time the app restarts.
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                self.is_trained = True
            except cv2.error:
                self.is_trained = False
        else:
            self.is_trained = False

    # ------------------------------------------------------------------
    # Label management helpers
    # ------------------------------------------------------------------
    def _load_labels(self):
        """Load the label<->name mapping from disk, if it exists."""
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                self.label_to_name = {int(k): v for k, v in json.load(f).items()}
            self.name_to_label = {v: k for k, v in self.label_to_name.items()}

    def _save_labels(self):
        """Persist the label<->name mapping to disk as JSON."""
        with open(self.labels_path, "w") as f:
            json.dump(self.label_to_name, f, indent=4)

    def get_registered_people(self) -> list:
        """Return a list of all registered person names."""
        return list(self.name_to_label.keys())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_face_samples(self, name: str, face_images: list) -> int:
        """
        Save grayscale face samples for a person to disk under
        registered_faces/<name>/.

        Args:
            name (str): Person's name.
            face_images (list of np.ndarray): Grayscale cropped face images.

        Returns:
            int: Number of samples saved.
        """
        if not name or not name.strip():
            raise ValueError("Name cannot be empty.")

        person_dir = os.path.join(self.data_dir, name.strip())
        os.makedirs(person_dir, exist_ok=True)

        existing_count = len(
            [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))]
        )

        saved = 0
        for i, face_img in enumerate(face_images):
            filename = f"sample_{existing_count + i + 1}.jpg"
            cv2.imwrite(os.path.join(person_dir, filename), face_img)
            saved += 1

        return saved

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> int:
        """
        Train the LBPH recognizer using every registered person's saved
        face samples found inside registered_faces/.

        Returns:
            int: Number of people the model was trained on.
        """
        faces = []
        labels = []
        self.label_to_name = {}
        self.name_to_label = {}

        people = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]

        if not people:
            raise ValueError(
                "No registered people found. Please register at least one "
                "person with face samples before training."
            )

        next_label_id = 0
        for person_name in sorted(people):
            person_dir = os.path.join(self.data_dir, person_name)
            image_files = [
                f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))
            ]

            if not image_files:
                continue  # Skip people who have no saved samples

            label_id = next_label_id
            self.label_to_name[label_id] = person_name
            self.name_to_label[person_name] = label_id
            next_label_id += 1

            for image_file in image_files:
                img_path = os.path.join(person_dir, image_file)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is None:
                    continue
                faces.append(gray_img)
                labels.append(label_id)

        if not faces:
            raise ValueError("No valid face samples found to train on.")

        # Train the LBPH model
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_path)
        self._save_labels()
        self.is_trained = True

        return len(self.label_to_name)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def recognize_face(self, gray_face_img: np.ndarray, confidence_threshold: float = 70.0):
        """
        Predict the identity of a single cropped, grayscale face image.

        LBPH returns a "confidence" score where LOWER means a BETTER match
        (it's actually a distance metric, not a probability).

        Args:
            gray_face_img (np.ndarray): Cropped grayscale face, resized to 200x200.
            confidence_threshold (float): Max distance to still consider it a match.
                                           Anything above this is labeled "Unknown".

        Returns:
            (name (str), confidence (float))
        """
        if not self.is_trained:
            return "Unknown", 0.0

        gray_face_img = cv2.resize(gray_face_img, (200, 200))
        label_id, confidence = self.recognizer.predict(gray_face_img)

        if confidence <= confidence_threshold and label_id in self.label_to_name:
            return self.label_to_name[label_id], confidence
        else:
            return "Unknown", confidence
