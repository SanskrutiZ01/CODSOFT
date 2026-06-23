"""
webcam_detector.py
-------------------
Handles real-time webcam capture and frame-by-frame face detection /
recognition. Designed to be used inside a Streamlit app's run-loop.

NOTE: Streamlit re-runs the whole script on every interaction, so true
"live video" is achieved by running a controlled while-loop inside a
single script execution (triggered by the "Start Webcam" button) that
keeps refreshing an st.empty() placeholder until the "Stop Webcam"
button's state flag is detected.
"""

import cv2
import time
import numpy as np


class WebcamDetector:
    """Manages webcam access and per-frame face detection/recognition."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None

    def start_camera(self):
        """Open the webcam device."""
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise IOError(
                "Could not access the webcam. Please make sure it is connected "
                "and not being used by another application."
            )
        return self.capture

    def read_frame(self):
        """
        Read a single frame from the webcam.

        Returns:
            (success (bool), frame (np.ndarray or None))
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None
        success, frame = self.capture.read()
        return success, frame

    def stop_camera(self):
        """Release the webcam device so other apps can use it again."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @staticmethod
    def process_frame_for_detection(frame, face_detector):
        """
        Run face detection on a single BGR frame.

        Returns:
            (processed_frame (np.ndarray), face_count (int))
        """
        faces, processed_frame = face_detector.detect_faces(frame)
        return processed_frame, len(faces)

    @staticmethod
    def process_frame_for_recognition(frame, face_detector, face_recognizer, confidence_threshold=70.0):
        """
        Run face detection AND recognition on a single BGR frame, drawing
        each person's name (or "Unknown") above their bounding box.

        Returns:
            (processed_frame (np.ndarray), results (list of dict))
            results: [{"name": str, "confidence": float, "box": (x, y, w, h)}, ...]
        """
        faces, _ = face_detector.detect_faces(frame, draw_boxes=False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = frame.copy()
        results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            try:
                face_roi_resized = cv2.resize(face_roi, (200, 200))
            except cv2.error:
                continue

            name, confidence = face_recognizer.recognize_face(
                face_roi_resized, confidence_threshold=confidence_threshold
            )

            # Choose box color: green for known person, red for unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)

            label = f"{name}"
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(
                output_frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
            )

            results.append({"name": name, "confidence": float(confidence), "box": (x, y, w, h)})

        return output_frame, results
