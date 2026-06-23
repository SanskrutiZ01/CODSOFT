"""
main.py
-------
Main Application Module. Provides a clean, menu-driven terminal interface
that ties together ImageLoader, CaptionGenerator, and the utility
functions into a complete, usable application.

The CaptionGenerator (and its underlying BLIP model) is loaded ONCE when
the application starts, then reused across single-image and batch
captioning requests for the rest of the session -- avoiding the major
inefficiency of reloading a deep learning model on every request.
"""

from captioning.image_loader import ImageLoader, ImageLoadError
from captioning.caption_generator import CaptionGenerator
from captioning.utils import (
    print_banner, print_divider,
    save_captions_to_file, display_image_with_caption,
)
from captioning.config import SAMPLE_IMAGES_DIR


class ImageCaptioningApp:
    """Top-level application controller -- the entry point's only job
    is to instantiate this class and call `run()`."""

    def __init__(self):
        self.loader = ImageLoader()
        self.generator = None  # lazy-loaded on first actual use (see _ensure_model_loaded)

    def _ensure_model_loaded(self) -> None:
        """
        Loads the BLIP model only when it's actually needed (i.e., the
        user chose a captioning action, not just browsing the menu).
        This makes the app start up instantly and only pays the model
        loading cost when captioning is genuinely about to happen.
        """
        if self.generator is None:
            self.generator = CaptionGenerator()

    # MENU
    
    def display_main_menu(self) -> None:
        print_divider("=")
        print("MAIN MENU")
        print("  [1] Caption a Single Image")
        print("  [2] Caption Multiple Images (Batch -- from a folder)")
        print("  [3] Exit")
        print_divider("=")

    # SINGLE IMAGE FLOW
   
    def handle_single_image(self) -> None:
        path = input("Enter the image file path: ").strip()

        try:
            image, abs_path = self.loader.load_single_image(path)
        except ImageLoadError as e:
            print(f"❌ {e}")
            return

        self._ensure_model_loaded()

        print("⏳ Generating caption...")
        try:
            caption = self.generator.generate_caption(image)
        except Exception as e:
            print(f"❌ Caption generation failed: {e}")
            return

        print(f"\n📝 Caption: {caption}\n")

        saved_path = save_captions_to_file([(abs_path, caption)])
        print(f"💾 Caption saved to: {saved_path}")

        show = input("Display the image with its caption? (y/n): ").strip().lower()
        if show in ("y", "yes"):
            display_image_with_caption(image, caption, abs_path)

    # BATCH FLOW
  
    def handle_batch_images(self) -> None:
        folder = input(
            f"Enter folder path containing images (default: '{SAMPLE_IMAGES_DIR}'): "
        ).strip() or SAMPLE_IMAGES_DIR

        try:
            loaded_images = self.loader.load_images_from_folder(folder)
        except ImageLoadError as e:
            print(f"❌ {e}")
            return

        print(f"📂 Found {len(loaded_images)} valid image(s) in '{folder}'.")
        self._ensure_model_loaded()

        print("⏳ Generating captions for all images...")
        results = []
        for image, abs_path in loaded_images:
            try:
                caption = self.generator.generate_caption(image)
            except Exception as e:
                caption = f"[ERROR generating caption: {e}]"
            results.append((abs_path, caption))
            print(f"  ✓ {abs_path} -> {caption}")

        saved_path = save_captions_to_file(results)
        print(f"\n💾 All {len(results)} captions saved to: {saved_path}")

        show = input("Display images one-by-one with captions? (y/n): ").strip().lower()
        if show in ("y", "yes"):
            images_by_path = {path: img for img, path in loaded_images}
            for abs_path, caption in results:
                display_image_with_caption(images_by_path[abs_path], caption, abs_path)

    # MAIN LOOP
  
    def run(self) -> None:
        print_banner("IMAGE CAPTIONING AI -- BLIP (Vision + Language Transformer)")

        while True:
            self.display_main_menu()
            choice = input("Enter your choice (1-3): ").strip()

            if choice == "1":
                self.handle_single_image()
            elif choice == "2":
                self.handle_batch_images()
            elif choice == "3":
                print("Goodbye! 👋")
                break
            else:
                print("⚠  Invalid choice. Please enter 1, 2, or 3.")


def main() -> None:
    try:
        app = ImageCaptioningApp()
        app.run()
    except (EOFError, KeyboardInterrupt):
        print("\nSession interrupted. Goodbye!")
    except Exception as e:
        # Top-level safety net so the user never sees a raw traceback.
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
