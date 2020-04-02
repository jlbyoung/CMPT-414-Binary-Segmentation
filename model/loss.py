import torch.nn.functional as F
import torch
import torch.nn as nn
import model.loss_implementations as i

focal_loss_instance = i.FocalLoss()
dice_loss_instance = i.DiceLoss()

def losvasz_softmax(output, target):
    return i.lovasz_softmax(output, target, ignore=255)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target, ignore_index=255)

def dice_loss(output, target):
    return dice_loss_instance(output, target)

<<<<<<< HEAD
def focal_loss(output, target):
    return focal_loss_instance(output, target)
    

def dice_weighted_ce_loss(output, target):
    return dice_loss(output, target) * cross_entropy_loss(output, target)
