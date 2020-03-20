import torch.nn.functional as F
import torch

def cross_entropy_loss(output, target):
    target = (target * 255).to(torch.long)
    target = target.squeeze(dim=1)

    return F.cross_entropy(output, target, ignore_index=255)
