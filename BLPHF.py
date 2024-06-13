from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import os

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image from a local file or URL
image_path = r"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQubckEe7hzwY1nVOFv2vawqn6pFgYS0h47KQ&s"  # Change to your image path or URL

try:
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
except (FileNotFoundError, requests.exceptions.RequestException) as e:
    print(f"Error loading image: {e}")
    exit(1)

# Prepare the image
inputs = processor(images=image, return_tensors="pt")

# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
