from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss, multi_VGGPerceptualLoss)
from .onerestore_loss import Total_loss

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss', 'multi_VGGPerceptualLoss', 'Total_loss'
]
