
import torch
from torch.optim import AdamW

class AdamATan2(AdamW):
    """Fallback to AdamW optimizer when adam-atan2 is not available"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, **kwargs):
        print("Warning: Using AdamW as fallback for AdamATan2")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
