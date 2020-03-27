import torch.nn.functional as F
from model.metric import meanDice
import torch

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target, ignore_index=255)
