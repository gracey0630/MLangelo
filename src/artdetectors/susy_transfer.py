# src/artdetectors/susy_transfer.py

import torch
import torch.nn as nn


class SuSyTransferLearning(nn.Module):
    """
    Transfer learning model for 3-class classification.
    Maps SuSy's 6-class output to 3 classes: Authentic, Midjourney, DALL-E 3
    """
    def __init__(self, susy_model_path, freeze_backbone=True):
        super().__init__()
        
        self.susy_model = torch.jit.load(susy_model_path)
        
        if freeze_backbone:
            for param in self.susy_model.parameters():
                param.requires_grad = False
        
        self.projection = nn.Linear(6, 3)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.training):
            susy_logits = self.susy_model(x)
        output = self.projection(susy_logits)
        return output