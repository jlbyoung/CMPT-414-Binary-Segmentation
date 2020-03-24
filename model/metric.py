import torch
import numpy as np

def _preprocess(output, target):
    output = output.detach().cpu()
    output = output.argmax(dim=1)
    output = output.to(torch.uint8)

    target = (target * 255).squeeze(dim=1)
    target = target.to(torch.uint8)
    target = target.cpu()

    return output, target


def _thresholded(val):
    # This is equal to comparing with thresolds
    # The minimum value of a correct metric is 0.5, the following will measure how correct a corect
    # segmentation is (segmentation with metric > 0.5).
    return torch.clamp(20 * (val - 0.5), 0, 10).ceil() / 10

def classwise_iou(output, target, n_class=21):
    SMOOTH = 1e-10

    output, target = _preprocess(output, target)
    class_wise = torch.zeros(n_class)
    for cl in range(n_class):
        intersection = ((target == cl) & (output == cl)).sum().float()
        union = ((target == cl) | (output == cl)).sum().float()

        if union.item() == 0: # A union of zero should be ignored
            class_wise[cl] = float('NaN')
        else:
            iou = intersection / (union + 1e-10)
            class_wise[cl] = iou
    return class_wise

def classwise_dice(output, target, n_class=21):
    SMOOTH = 1e-10

    output, target = _preprocess(output, target)
    class_wise = torch.zeros(n_class)
    for cl in range(n_class):
        intersection = ((target == cl) & (output == cl)).sum().float()
        total = ((target == cl).sum() + (output == cl).sum()).float()

        if total.item() == 0: # A union of zero should be ignored
            class_wise[cl] = float('NaN')
        else:
            dice = (intersection * 2) / (total + 1e-10)
            class_wise[cl] = dice
    return class_wise


def meanIoU(output, target, n_class=21):
    classwise = classwise_iou(output, target, n_class)

    # thresholded = _thresholded(classwise)
    return np.nanmean(classwise)

def meanDice(output, target, n_class=21):
    classwise = classwise_dice(output, target, n_class)

    # thresholded = _thresholded(classwise)
    return np.nanmean(classwise)
