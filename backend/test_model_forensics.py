#!/usr/bin/env python3
"""Forensic analysis of AI/Real detection models."""

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import onnxruntime as ort

def test_pytorch_model():
    """Test the PyTorch model with transformers processor."""
    model_dir = "/models/ai_real_detector"
    print("=" * 60)
    print("TESTING PYTORCH MODEL (Organika/sdxl-detector)")
    print("=" * 60)
    
    print("\nLoading model and processor...")
    model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    
    print(f"Model id2label: {model.config.id2label}")
    print(f"Processor image_mean: {processor.image_mean}")
    print(f"Processor image_std: {processor.image_std}")
    print(f"Processor size: {processor.size}")
    
    # Create test images
    test_images = []
    
    # Test 1: Pure noise image
    np.random.seed(42)
    noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_images.append(("Noise", Image.fromarray(noise_img)))
    
    # Test 2: Gradient image
    gradient = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient[i, :] = [i, 255-i, 128]
    test_images.append(("Gradient", Image.fromarray(gradient)))
    
    # Test 3: Solid gray
    test_images.append(("Solid Gray", Image.new('RGB', (224, 224), (128, 128, 128))))
    
    # Test 4: Solid white
    test_images.append(("Solid White", Image.new('RGB', (224, 224), (255, 255, 255))))
    
    # Test 5: Solid black
    test_images.append(("Solid Black", Image.new('RGB', (224, 224), (0, 0, 0))))
    
    print("\n--- Testing with Transformers Processor ---")
    for name, img in test_images:
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        print(f"\n{name}:")
        print(f"  Input shape: {pixel_values.shape}")
        print(f"  Input range: [{pixel_values.min().item():.4f}, {pixel_values.max().item():.4f}]")
        print(f"  Input mean: {pixel_values.mean().item():.4f}")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
        
        print(f"  Logits: {logits.detach().numpy()}")
        print(f"  Probs: {probs.detach().numpy()}")
        # id2label uses integer keys
        label = model.config.id2label.get(predicted_class, f"class_{predicted_class}")
        print(f"  Predicted: {predicted_class} ({label})")


def compare_preprocessing():
    """Compare our preprocessing with transformers processor."""
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPARISON")
    print("=" * 60)
    
    model_dir = "/models/ai_real_detector"
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    
    # Create a test image
    np.random.seed(42)
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_img)
    
    # Method 1: Transformers processor
    inputs = processor(images=test_pil, return_tensors="pt")
    tf_result = inputs['pixel_values'].numpy()
    
    # Method 2: Manual preprocessing (what our code does)
    manual_float = test_img.astype(np.float32) / 255.0
    manual_norm = (manual_float - 0.5) / 0.5
    manual_nchw = np.transpose(manual_norm, (2, 0, 1))
    manual_batch = np.expand_dims(manual_nchw, 0)
    
    print(f"\nTransformers processor result shape: {tf_result.shape}")
    print(f"Manual preprocessing result shape: {manual_batch.shape}")
    print(f"\nDifference (should be near zero): {np.abs(tf_result - manual_batch).max():.6f}")
    
    if np.allclose(tf_result, manual_batch, atol=1e-5):
        print("PREPROCESSING MATCHES!")
    else:
        print("WARNING: PREPROCESSING MISMATCH!")
        print(f"Transformers range: [{tf_result.min():.4f}, {tf_result.max():.4f}]")
        print(f"Manual range: [{manual_batch.min():.4f}, {manual_batch.max():.4f}]")


if __name__ == "__main__":
    test_pytorch_model()
    compare_preprocessing()
