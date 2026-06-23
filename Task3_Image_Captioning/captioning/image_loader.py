"""
image_loader.py
----------------
Defines the ImageLoader class: responsible for ALL image I/O concerns --
locating files, validating paths/extensions, opening images safely with
PIL, and converting them to a consistent format (RGB) that the BLIP
processor expects. Keeping this separate from CaptionGenerator means the
captioning logic never has to deal with file-system or decoding errors
directly.
"""

import os
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError

from captioning.config import SUPPORTED_EXTENSIONS


class ImageLoadError(Exception):
    """Custom exception raised for any image loading failure.

    Using a custom exception type (rather than letting raw PIL/OS
    exceptions propagate) lets calling code catch ALL image-loading
    problems with a single except clause, while still printing a
    specific, user-friendly message.
    """
    pass


class ImageLoader:
    """Handles locating, validating, and opening image files."""

    def load_single_image(self, path: str) -> Tuple[Image.Image, str]:
        """
        Loads a single image from disk.

        Returns:
            (PIL.Image in RGB mode, absolute file path)

        Raises:
            ImageLoadError -- if the path doesn't exist, isn't a file,
            has an unsupported extension, or isn't a valid image.
        """
        if not path or not path.strip():
            raise ImageLoadError("No image path was provided.")

        path = path.strip().strip('"').strip("'")  # tolerate pasted quoted paths

        if not os.path.exists(path):
            raise ImageLoadError(f"File not found: '{path}'. Please check the path.")

        if not os.path.isfile(path):
            raise ImageLoadError(f"'{path}' is a directory, not an image file.")

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                f"Unsupported file type '{ext}'. Supported types: {SUPPORTED_EXTENSIONS}"
            )

        try:
            image = Image.open(path)
            image = image.convert("RGB")  # BLIP's processor expects RGB
        except UnidentifiedImageError:
            raise ImageLoadError(f"'{path}' could not be read as a valid image (corrupted file?).")
        except OSError as e:
            raise ImageLoadError(f"OS error while opening '{path}': {e}")

        return image, os.path.abspath(path)

    def load_images_from_folder(self, folder_path: str) -> List[Tuple[Image.Image, str]]:
        """
        Loads ALL supported images from a given folder (for batch
        captioning). Skips files that fail to load individually rather
        than aborting the whole batch, printing a warning for each skip.

        Returns:
            List of (PIL.Image, file_path) tuples for successfully loaded images.

        Raises:
            ImageLoadError -- if the folder itself doesn't exist or contains
            no supported image files at all.
        """
        if not os.path.isdir(folder_path):
            raise ImageLoadError(f"Folder not found: '{folder_path}'.")

        filenames = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )

        if not filenames:
            raise ImageLoadError(
                f"No supported image files found in '{folder_path}'. "
                f"Supported types: {SUPPORTED_EXTENSIONS}"
            )

        loaded_images = []
        for filename in filenames:
            full_path = os.path.join(folder_path, filename)
            try:
                image, abs_path = self.load_single_image(full_path)
                loaded_images.append((image, abs_path))
            except ImageLoadError as e:
                print(f"⚠  Skipping '{filename}': {e}")

        if not loaded_images:
            raise ImageLoadError("All images in the folder failed to load.")

        return loaded_images
