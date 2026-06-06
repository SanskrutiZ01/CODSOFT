from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

# Load BLIP model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Open image
image = Image.open(
    "Task3_Image_Captioning/sample.jpg"
).convert("RGB")

# Process image
inputs = processor(
    image,
    return_tensors="pt"
)

# Generate caption
output = model.generate(
    **inputs,
    max_new_tokens=25,
    num_beams=5
)

# Decode + Capitalize first letter of every word
caption = processor.decode(
    output[0],
    skip_special_tokens=True
).title()

# Print in terminal
print("Generated Caption:")
print(caption)

# Show image with caption
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title(caption, fontsize=14)
plt.axis("off")
plt.show()