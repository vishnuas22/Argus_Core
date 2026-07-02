import torch
import torch.nn as nn
from transformers import AutoModel
import os
from utils.logging import get_logger

logger = get_logger(__name__)

class DinoV2DeepfakeDetector(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # We will add LoRA to the backbone if peft is available
        try:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"]
            )
            self.backbone = get_peft_model(self.backbone, config)
            logger.info("Successfully applied LoRA to DINOv2 backbone")
        except ImportError:
            logger.warning("peft not installed, using standard DINOv2 backbone without LoRA")
            
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits
