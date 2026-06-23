"""
caption_generator.py
---------------------
Defines the CaptionGenerator class: wraps Hugging Face's pre-trained BLIP
(Bootstrapping Language-Image Pretraining) model and processor, exposing
a simple `generate_caption()` / `generate_captions_batch()` API.

This is the only module that touches the actual deep learning model --
everything else in the project (image_loader, utils, main) is decoupled
from the specific model implementation. This means BLIP could later be
swapped for another vision-language model (e.g., GIT, BLIP-2) by editing
only this file.
"""

from typing import List
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from captioning.config import BLIP_MODEL_NAME, DEVICE, MAX_CAPTION_LENGTH, NUM_BEAMS


class CaptionGenerator:
    """
    Loads the pre-trained BLIP model ONCE at initialization (model
    loading is expensive -- a few hundred MB of weights), then reuses
    it for every caption generation call. This is far more efficient
    than reloading the model per image, which a naive script would do.
    """

    def __init__(self, model_name: str = BLIP_MODEL_NAME, device: str = DEVICE):
        self.device = device
        print(f"⏳ Loading BLIP model '{model_name}' on device '{device}'... "
              f"(first run downloads weights from Hugging Face Hub)")

        # The processor handles image preprocessing (resize/normalize)
        # AND text tokenization -- mirroring how BLIP was trained.
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()  # inference mode: disables dropout etc.

        print("✅ Model loaded successfully.\n")

    def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        """
        Generates a single caption for one PIL image.

        Parameters:
            image  -- a PIL.Image in RGB mode
            prompt -- optional text prompt for "conditional" captioning
                      (e.g., "a photography of"); if None, BLIP performs
                      unconditional captioning purely from the image.

        Returns:
            The generated caption as a plain string.
        """
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        # torch.no_grad() disables gradient tracking since we're only
        # doing inference, not training -- saves memory and computation.
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=MAX_CAPTION_LENGTH,
                num_beams=NUM_BEAMS,
            )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def generate_captions_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Generates captions for a list of images one at a time.

        Note: this loops rather than feeding all images as one padded
        tensor batch. This is an intentional simplicity/robustness
        trade-off for a learning project -- see README's "Future
        Enhancements" section for how true batched tensor inference
        could be added for higher throughput on large datasets.
        """
        captions = []
        for idx, image in enumerate(images, start=1):
            caption = self.generate_caption(image)
            captions.append(caption)
            print(f"  [{idx}/{len(images)}] Caption generated.")
        return captions
