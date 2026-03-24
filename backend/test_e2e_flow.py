#!/usr/bin/env python3
"""End-to-end flow test for deepfake detection pipeline."""

import asyncio
import sys
sys.path.insert(0, '/app')

import numpy as np
from PIL import Image
import torch
import pytest


@pytest.mark.asyncio
async def test_ensemble():
    """Test the AI/Real detection directly."""
    from models.manager import get_model_manager
    
    # Get model manager
    manager = get_model_manager()
    
    # Load the unified AI/Real detector model
    print('Loading ai_real_detector model...')
    model_session = await manager.get_model('ai_real_detector')
    
    if model_session is None:
        print('ERROR: Model not available')
        return
    
    # model_session is a tuple (model, processor) for PyTorch models
    model, processor = model_session
    print(f'Model loaded: {type(model)}')
    print(f'Processor loaded: {type(processor)}')
    print(f'Model id2label: {model.config.id2label}')
    
    # Load AI-generated face
    img = Image.open('/tmp/ai_face2.jpg').convert('RGB')
    print(f'Image size: {img.size}')
    
    # Process with processor
    inputs = processor(images=img, return_tensors='pt')
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    print(f'\nModel Output:')
    print(f'  Logits: {logits.detach().cpu().numpy()}')
    print(f'  Probabilities: {probs.detach().cpu().numpy()}')
    print(f'  Real (class 0): {probs[0, 0].item():.4f}')
    print(f'  Fake (class 1): {probs[0, 1].item():.4f}')
    
    # This is what _run_primary_detection returns
    fake_prob = probs[0, 1].item()
    print(f'\nReturned fake probability: {fake_prob:.4f}')
    
    # Calculate trust score
    trust_score = (1 - fake_prob) * 100
    print(f'Trust score (should be ~0.4): {trust_score:.1f}')
    
    if trust_score < 20:
        verdict = "fake"
    elif trust_score < 40:
        verdict = "likely_fake"
    elif trust_score < 60:
        verdict = "uncertain"
    elif trust_score < 80:
        verdict = "likely_authentic"
    else:
        verdict = "authentic"
    
    print(f'Verdict: {verdict}')


if __name__ == "__main__":
    asyncio.run(test_ensemble())
