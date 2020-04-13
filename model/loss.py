import torch.nn.functional as F
import torch
import torch.nn as nn
import model.loss_implementations as i

focal_loss_instance = i.FocalLoss()
dice_loss_instance = i.DiceLoss()

def losvasz_softmax(output, target):
    return i.lovasz_softmax(output, target, ignore=255)


def cross_entropy_loss(output, target, ignore_background=True):
    if ignore_background:
        target[target == 255] = 0
        index = 0
    else:
        index = 255
    return F.cross_entropy(output, target, ignore_index=index)


def dice_loss(output, target):
    return dice_loss_instance(output, target)


def focal_loss(output, target):
    return focal_loss_instance(output, target)
    

def dice_weighted_ce_loss(output, target):
    return dice_loss(output, target) * cross_entropy_loss(output, target)


def binary_cross_entropy_loss(output, target):
    output = torch.sigmoid(output.squeeze(dim=1))
    return F.binary_cross_entropy(output, target.to(torch.float32))
