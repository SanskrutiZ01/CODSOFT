"""
face_detector.py
-----------------
This module contains the FaceDetector class, which is responsible for
detecting faces inside images or video frames using OpenCV's Haar Cascade
classifier.

Haar Cascade is a pre-trained machine learning model that OpenCV ships with.
It works by sliding small rectangular "feature windows" over the image and
checking for patterns that look like human facial features (eyes, nose,
mouth edges, etc.). It is fast and works well for frontal face detection.
"""

import os
import cv2
import numpy as np


class FaceDetector:
    """
    A simple, reusable wrapper around OpenCV's Haar Cascade face detector.

    Usage:
        detector = FaceDetector()
        faces, processed_image = detector.detect_faces(image)
    """

    def __init__(self, cascade_path: str = None):
        """
        Initialize the FaceDetector.

        Args:
            cascade_path (str): Path to the haarcascade XML file.
                                 If None, a default project path is used.
        """
        if cascade_path is None:
            # Default location inside our project structure
            cascade_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "haarcascade",
                "haarcascade_frontalface_default.xml",
            )

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar Cascade file not found at: {cascade_path}. "
                "Please make sure 'haarcascade_frontalface_default.xml' "
                "exists inside the 'haarcascade/' folder."
            )

        # Load the pre-trained Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade classifier. The XML file may be corrupted.")

    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple = (30, 30),
        draw_boxes: bool = True,
        box_color: tuple = (0, 255, 0),
        box_thickness: int = 2,
    ):
        """
        Detect faces in a given image (BGR format, as used by OpenCV).

        Args:
            image (np.ndarray): Input image in BGR format.
            scale_factor (float): How much the image size is reduced at each
                                   image scale (smaller = more accurate but slower).
            min_neighbors (int): How many neighbors each candidate rectangle
                                  should have to retain it (higher = fewer false positives).
            min_size (tuple): Minimum possible face size. Smaller faces are ignored.
            draw_boxes (bool): Whether to draw rectangles around detected faces.
            box_color (tuple): BGR color for the rectangle.
            box_thickness (int): Thickness of the rectangle border.

        Returns:
            faces (list): List of (x, y, w, h) tuples for each detected face.
            output_image (np.ndarray): Copy of the input image with rectangles drawn
                                        (if draw_boxes is True), otherwise the original image.
        """
        if image is None:
            raise ValueError("Input image is None. Please provide a valid image.")

        # Make a copy so we never modify the original image accidentally
        output_image = image.copy()

        # Haar Cascades work on grayscale images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # The actual face detection step
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        if draw_boxes:
            for (x, y, w, h) in faces:
                cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, box_thickness)

        return faces, output_image

    def get_face_count(self, faces) -> int:
        """Return the number of faces detected."""
        return len(faces)

    def crop_faces(self, image: np.ndarray, faces, target_size: tuple = (200, 200)):
        """
        Crop each detected face region out of the image and resize it.

        Args:
            image (np.ndarray): Original image.
            faces (list): List of (x, y, w, h) tuples.
            target_size (tuple): Size to resize each cropped face to.

        Returns:
            List of cropped (and resized) face images in grayscale.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_faces = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, target_size)
            cropped_faces.append(face_roi)
        return cropped_faces
