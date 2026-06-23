"""
image_processor.py
-------------------
Utility class for handling image-related operations such as:
- Converting between PIL <-> OpenCV formats
- Saving processed images to disk
- Preparing images for download in Streamlit

Keeping these utilities in a separate file keeps app.py clean and
follows the "separation of concerns" principle of clean code.
"""

import io
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime


class ImageProcessor:
    """Helper class for image conversions, saving, and downloads."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs",
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert a PIL Image (RGB) to an OpenCV image (BGR)."""
        rgb_array = np.array(pil_image.convert("RGB"))
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert an OpenCV image (BGR) to a PIL Image (RGB)."""
        rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array)

    def save_image(self, cv2_image: np.ndarray, prefix: str = "processed") -> str:
        """
        Save an OpenCV image to the outputs folder with a timestamped name.

        Returns:
            str: Full path to the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, cv2_image)
        return filepath

    @staticmethod
    def get_image_bytes(cv2_image: np.ndarray) -> bytes:
        """
        Convert an OpenCV image into JPEG bytes, ready for Streamlit's
        st.download_button.
        """
        pil_image = ImageProcessor.cv2_to_pil(cv2_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return buffer.getvalue()
