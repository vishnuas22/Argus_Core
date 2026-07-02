import torch
import torch.nn as nn
from transformers import VideoMAEModel
import os
from utils.logging import get_logger

logger = get_logger(__name__)

class VideoMAEDeepfakeDetector(nn.Module):
    def __init__(self, model_name="MCG-NJU/videomae-base", num_classes=2):
        super().__init__()
        # We only use the backbone (encoder) to extract features
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        
        hidden_size = self.backbone.config.hidden_size
        # Simple classification head over the pooled output
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, pixel_values):
        # pixel_values shape: (batch_size, num_frames, num_channels, height, width)
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Mean pooling over the sequence dimension (tokens)
        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        logits = self.classifier(pooled_output)
        return logits
