from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

print("=== Testing umm-maybe/AI-image-detector ===")
model_name = "umm-maybe/AI-image-detector"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
print(f"Labels: {model.config.id2label}")

test_images = [
    ("/tmp/test_images/Deepfake.png", "Deepfake.png"),
    ("/tmp/test_images/camera_image1.webp", "camera_image1.webp"),
]

for path, name in test_images:
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        print(f"{name}:")
        for idx, label in model.config.id2label.items():
            print(f"  {label}: {probs[0, idx].item():.4f}")
        print()
    except Exception as e:
        print(f"{name}: ERROR - {e}")
        print()

print("=== Testing Organika/sdxl-detector ===")
model_name2 = "Organika/sdxl-detector"
processor2 = AutoImageProcessor.from_pretrained(model_name2)
model2 = AutoModelForImageClassification.from_pretrained(model_name2)
print(f"Labels: {model2.config.id2label}")

for path, name in test_images:
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor2(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model2(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        print(f"{name}:")
        for idx, label in model2.config.id2label.items():
            print(f"  {label}: {probs[0, idx].item():.4f}")
        print()
    except Exception as e:
        print(f"{name}: ERROR - {e}")
        print()
