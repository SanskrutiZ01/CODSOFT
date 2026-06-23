"""
captioning package
-------------------
A modular Image Captioning system combining Computer Vision (image
preprocessing via a Vision Transformer encoder inside BLIP) and NLP
(transformer-based caption decoding).

Modules:
    config.py            -> centralized configuration constants
    image_loader.py      -> ImageLoader: file I/O & validation
    caption_generator.py -> CaptionGenerator: BLIP model wrapper
    utils.py             -> display, file-saving, terminal UI helpers
"""

__version__ = "1.0.0"
