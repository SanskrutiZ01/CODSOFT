"""
config.py
---------
Centralized configuration constants for the Image Captioning project.
Keeping these in one place means model names, device selection, and
folder paths can be changed without touching logic in other modules.
"""

import torch

# Hugging Face model checkpoint. Using the base BLIP captioning model
# pretrained by Salesforce -- a genuine pre-trained vision-language
# transformer, not a placeholder or mock.
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Automatically use a GPU if available, otherwise fall back to CPU.
# This makes the project portable across laptops (CPU-only) and any
# machine with CUDA available, without code changes.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default folders
SAMPLE_IMAGES_DIR = "sample_images"
OUTPUTS_DIR = "outputs"
CAPTIONS_OUTPUT_FILE = "outputs/generated_captions.txt"

# Supported image file extensions for batch loading
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Caption generation parameters
MAX_CAPTION_LENGTH = 50
NUM_BEAMS = 4  # beam search width -- higher = more thorough but slower
