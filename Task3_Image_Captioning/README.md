# 🖼️📝 Image Captioning AI — BLIP (Vision + Language Transformer)

A professional, object-oriented **Image Captioning system** that combines **Computer Vision** and **Natural Language Processing** using a genuine pre-trained transformer model — **BLIP** (Bootstrapping Language-Image Pretraining) via Hugging Face Transformers. No paid APIs, no external captioning services — everything runs locally using open-source pretrained weights.

---

## 📌 Project Overview

This project implements **Task 3: Image Captioning AI** from the internship brief. Rather than a single script that loads an image and prints one caption, it is structured as a proper Python package with dedicated modules for image I/O, model inference, and utilities, wrapped in a menu-driven application supporting both single-image and batch captioning — reflecting how a real CV/NLP engineer would structure a small production-style inference service.

---

## ✨ Features

- 🤖 **Genuine Pre-trained Model** — uses `Salesforce/blip-image-captioning-base` via Hugging Face Transformers (real Vision Transformer encoder + transformer text decoder, not a mock)
- 🖼️ **Single Image Captioning** — caption any local image file by path
- 📦 **Batch Processing** — caption every supported image in a folder in one run
- 💾 **Caption Persistence** — every result is appended to a timestamped `outputs/generated_captions.txt` log
- 🎨 **Image + Caption Display** — renders the image with its generated caption as a title using matplotlib (with graceful fallback to text-only output on headless/no-display environments)
- 🛡️ **Robust Error Handling** — custom `ImageLoadError` cleanly handles missing files, wrong file types, corrupted images, and empty/invalid folders without crashing
- 🧩 **Clean OOP Architecture** — `ImageLoader`, `CaptionGenerator`, `ImageCaptioningApp` each have one clear responsibility
- ⚡ **Lazy Model Loading** — the (large) BLIP model is only loaded into memory once the user actually requests captioning, not at program startup
- 🗂️ **Sample Images Included** — ships with placeholder images so the project can be demoed immediately
- 📝 **Thoroughly Commented Code** — every module and function documents both its purpose and its design rationale

---

## 📁 Folder Structure

```
image_captioning_ai/
│
├── main.py                       # Main Application Module -- menu-driven CLI
├── requirements.txt              # torch, transformers, Pillow, matplotlib
├── README.md                     # Project documentation
│
├── sample_images/                # Sample images for demoing the project
│   ├── red_circle.jpg
│   ├── blue_square.jpg
│   └── green_triangle.jpg
│
├── outputs/                      # Auto-created -- stores generated_captions.txt
│
└── captioning/                    # Core package
    ├── __init__.py
    ├── config.py                 # Model name, device selection, constants
    ├── image_loader.py            # ImageLoader -- file I/O & validation
    ├── caption_generator.py       # CaptionGenerator -- BLIP model wrapper
    └── utils.py                   # Display, file-saving, terminal helpers
```

---

## 🧩 Module-by-Module Explanation

| Module | Responsibility |
|---|---|
| **`config.py`** | Centralizes the model checkpoint name, automatic CPU/GPU device selection, folder paths, and generation hyperparameters (max caption length, beam search width) so they're configurable in one place. |
| **`image_loader.py`** | The `ImageLoader` class handles ALL file-system concerns: validating that a path exists, is a file (not a folder), has a supported extension, and can actually be decoded as an image — converting everything to RGB for consistent model input. Batch loading skips individually broken files with a warning instead of aborting the whole run. |
| **`caption_generator.py`** | The `CaptionGenerator` class loads the BLIP processor + model once, then exposes `generate_caption()` (single image) and `generate_captions_batch()` (list of images). This is the only module that touches the actual deep learning model. |
| **`utils.py`** | `save_captions_to_file()` appends timestamped results to a log file; `display_image_with_caption()` renders the image with matplotlib and degrades gracefully if no display is available; plus shared terminal print helpers. |
| **`main.py`** | The `ImageCaptioningApp` class orchestrates everything: menu loop, single-image flow, batch flow, and lazy model loading — the only file a new contributor needs to read to understand the overall application flow. |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/image-captioning-ai.git
cd image-captioning-ai/image_captioning_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python main.py
```

> ℹ️ On first run, the BLIP model weights (~990 MB) are automatically downloaded from the Hugging Face Hub and cached locally — this requires an internet connection only the first time.

---

## 🎬 Sample Output

```
===============================================================
| IMAGE CAPTIONING AI -- BLIP (Vision + Language Transformer) |
===============================================================
==================================================
MAIN MENU
  [1] Caption a Single Image
  [2] Caption Multiple Images (Batch -- from a folder)
  [3] Exit
==================================================
Enter your choice (1-3): 1
Enter the image file path: sample_images/red_circle.jpg
⏳ Loading BLIP model 'Salesforce/blip-image-captioning-base' on device 'cpu'...
✅ Model loaded successfully.

⏳ Generating caption...

📝 Caption: a red circle on a white background

💾 Caption saved to: /path/to/outputs/generated_captions.txt
Display the image with its caption? (y/n): y
```

**`outputs/generated_captions.txt` contents:**
```
============================================================
Run timestamp: 2026-06-20 07:53:05
============================================================
Image  : /path/to/sample_images/red_circle.jpg
Caption: a red circle on a white background
------------------------------------------------------------
```

*(Note: the bundled sample images are simple synthetic shapes for out-of-the-box demoing. BLIP produces noticeably richer, more natural captions on real-world photographs — swap in a few real images for the most impressive portfolio demo.)*

---

## 🧠 BLIP Architecture Explanation

**BLIP (Bootstrapping Language-Image Pretraining)**, developed by Salesforce Research, is a vision-language model designed to unify visual understanding and text generation. Its architecture has two main components:

1. **Vision Encoder (Image Transformer / ViT)**: The input image is split into fixed-size patches, each treated like a "token." A Vision Transformer encodes these patches (plus positional information) into a sequence of image feature embeddings — capturing both local details (edges, colors, shapes) and global context (object relationships, scene layout).

2. **Text Decoder (Transformer Decoder)**: A transformer-based language decoder takes the image embeddings as cross-attention context and generates a caption **autoregressively** — predicting one word at a time, where each new word is conditioned on both the image features and all previously generated words.

BLIP was pretrained on large-scale noisy web image-text pairs using a **"bootstrapping"** technique: a captioner model generates synthetic captions for web images, and a filter model removes low-quality/noisy pairs, iteratively cleaning the training data. This is what makes BLIP especially good at producing fluent, accurate captions compared to models trained on raw, unfiltered web data.

In this project, `BlipProcessor` handles converting a raw PIL image into the normalized pixel tensor the Vision Encoder expects (and tokenizing any optional text prompt), while `BlipForConditionalGeneration` wraps both the vision encoder and text decoder together for end-to-end caption generation via `.generate()`.

---

## 🔗 Computer Vision + NLP Workflow

This project is a direct, working example of **multimodal AI** — combining two traditionally separate fields into one pipeline:

```
   [Raw Image File]
          │
          ▼
   ImageLoader (CV: I/O + preprocessing)
   - validates path/extension
   - decodes image bytes -> pixel array
   - converts to RGB
          │
          ▼
   BlipProcessor (CV: feature extraction)
   - resizes/normalizes pixels
   - splits into patches for the Vision Transformer
          │
          ▼
   Vision Transformer Encoder (CV)
   - encodes image patches into feature embeddings
          │
          ▼
   Transformer Text Decoder (NLP)
   - cross-attends to image embeddings
   - autoregressively generates caption tokens
          │
          ▼
   BlipProcessor.decode() (NLP: post-processing)
   - converts token IDs back into a human-readable sentence
          │
          ▼
   [Generated Caption String] -> saved to file + displayed with image
```

The **Computer Vision** half (image decoding, patch embedding, visual feature extraction) produces a numerical *representation* of "what is in the image." The **NLP** half (the transformer decoder) takes that representation and produces a *grammatically correct, semantically relevant English sentence* describing it. Neither half alone could do this task — CV without NLP would only classify/detect objects without forming sentences, and NLP without CV would have no visual grounding at all.

---

## 🔢 Transformer-Based Caption Generation (How `.generate()` Works)

Caption generation is **autoregressive**: the decoder generates one token at a time, and each new token is chosen based on:
1. The encoded image features (via cross-attention layers), and
2. All tokens generated so far in the caption.

This project uses **beam search** (`num_beams=4` in `config.py`) rather than always picking the single most likely next word ("greedy decoding"). Beam search keeps track of the top-*k* (here, 4) most promising partial captions at each step, expanding all of them and keeping only the best *k* overall — this avoids getting stuck on a locally-optimal-but-globally-poor word choice early in the sentence, generally producing more fluent and accurate captions than greedy decoding at a modest extra computational cost.

Generation stops when the decoder produces an end-of-sequence token, or when `MAX_CAPTION_LENGTH` (50 tokens) is reached, whichever comes first.

---

## 📊 Complexity Discussion

- **Vision Encoder (ViT) time complexity**: `O(n² · d)` per image, where `n` is the number of image patches (e.g., a 384×384 image with 16×16 patches yields n=576 patches) and `d` is the embedding dimension — this comes from the self-attention mechanism computing pairwise attention scores between all patches.
- **Text Decoder time complexity**: `O(L² · d)` for generating a caption of length `L`, since each generation step attends over all previously generated tokens (and cross-attends to all `n` image patch embeddings).
- **Overall per-image inference**: dominated by these transformer attention costs, scaled by `num_beams` for beam search (`O(num_beams)` parallel hypothesis tracking) — meaning Hard captioning quality and runtime scale together, which is why `NUM_BEAMS` is exposed as a tunable constant in `config.py`.
- **Space complexity**: model weights are fixed-size (~990 MB for the base BLIP checkpoint) regardless of input; per-image memory scales with image resolution (patch count) and beam width, but this project processes one image at a time (not a single giant padded batch tensor), keeping peak memory usage low and predictable — an explicit engineering trade-off favoring robustness/simplicity over maximum throughput (see Future Enhancements).

---

## ✅ Why This Project Satisfies the Internship Task

| Internship Requirement | How This Project Fulfills It |
|---|---|
| Combine Computer Vision and NLP | `ImageLoader` + BLIP's Vision Transformer (CV) feed into BLIP's transformer text decoder (NLP) — a genuine multimodal pipeline |
| Use pre-trained image recognition models / vision encoders | Uses BLIP's pretrained Vision Transformer encoder (`Salesforce/blip-image-captioning-base`) via Hugging Face Transformers |
| Use a transformer-based model to generate captions | `BlipForConditionalGeneration.generate()` performs autoregressive, beam-search transformer decoding |
| Professional, non-trivial implementation | OOP architecture across 4 focused modules, batch processing, error handling, captions persisted to disk, image+caption display, full README with architecture explanation |

---

## 🔮 Future Enhancements

- **True batched tensor inference**: pad and stack multiple images into a single batched tensor (rather than looping one at a time) for significantly higher throughput on large datasets/GPUs
- **Web UI**: wrap the app in a Streamlit or Flask interface with drag-and-drop image upload
- **Caption quality metrics**: integrate BLEU/CIDEr scoring against a labeled dataset (e.g., a subset of MS-COCO) to quantitatively evaluate caption quality
- **Multiple model comparison**: add support for swapping in BLIP-2, GIT, or ViT-GPT2 and comparing their captions side-by-side on the same images
- **Conditional captioning UI**: expose BLIP's prompt-conditioned captioning (e.g., "a photo of...") as a user-facing option for more controllable outputs
- **Multilingual captions**: pipe the generated English caption through a translation model for multilingual caption output
- **Unit tests**: add `pytest` coverage for `ImageLoader`'s error paths and a mocked `CaptionGenerator` for CI without requiring GPU/model downloads

---

## 🏷️ Tech Stack

`Python 3` · PyTorch · Hugging Face Transformers · BLIP (Vision Transformer + Transformer Decoder) · Pillow · Matplotlib · Object-Oriented Programming

---

## 📄 License

This project is open-source and free to use for educational purposes. The BLIP model weights are distributed by Salesforce Research under their respective license via the Hugging Face Hub.
