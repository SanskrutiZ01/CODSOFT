"""
utils.py
--------
Shared utility functions: saving captions to disk, displaying images
with their generated captions using matplotlib, and terminal UI helpers.
Separated from caption_generator.py and image_loader.py to keep each
module focused on a single responsibility (display/IO vs. inference vs.
file loading).
"""

import os
from datetime import datetime
from typing import List, Tuple
from PIL import Image

from captioning.config import CAPTIONS_OUTPUT_FILE


def print_banner(text: str) -> None:
    border = "=" * (len(text) + 4)
    print(border)
    print(f"| {text} |")
    print(border)


def print_divider(char: str = "-", length: int = 50) -> None:
    print(char * length)


def save_captions_to_file(results: List[Tuple[str, str]],
                           output_path: str = CAPTIONS_OUTPUT_FILE) -> str:
    """
    Appends a batch of (image_path, caption) results to a text file,
    with a timestamp header for traceability across multiple runs.

    Parameters:
        results     -- list of (image_path, caption) tuples
        output_path -- destination .txt file path

    Returns:
        The absolute path of the file written to.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n")
        for image_path, caption in results:
            f.write(f"Image  : {image_path}\n")
            f.write(f"Caption: {caption}\n")
            f.write("-" * 60 + "\n")

    return os.path.abspath(output_path)


def display_image_with_caption(image: Image.Image, caption: str, image_path: str = "") -> None:
    """
    Displays the image with its generated caption overlaid as a title,
    using matplotlib. Wrapped in a try/except because headless
    environments (e.g., remote servers/CI without a display) cannot
    open a GUI window -- in that case we degrade gracefully and just
    print a message instead of crashing the program.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        title = caption if len(caption) < 80 else caption[:77] + "..."
        plt.title(title, fontsize=11, wrap=True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"ℹ  Could not open a display window ({e}). "
              f"Caption for '{os.path.basename(image_path)}': {caption}")
