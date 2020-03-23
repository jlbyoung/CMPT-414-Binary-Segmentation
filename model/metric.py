import torch
import numpy as np

def _preprocess(output, target):
    output = output.detach().cpu()
    output = output.argmax(dim=1)
    output = output.to(torch.uint8)

    target = (target * 255).squeeze(dim=1)
    target = target.to(torch.uint8)
    return output, target

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


def meanIoU(output, target, n_class=21):
    SMOOTH = 1e-6
    
    classwise = classwise_iou(output, target, n_class)

    # This is equal to comparing with thresolds
    # The minimum value of a correct IoU is 0.5, the following will measure how correct a corect
    # segmentation is (segmentation with IoU > 0.5).
    # thresholded = torch.clamp(20 * (classwise - 0.5), 0, 10).ceil() / 10
    return np.nanmean(classwise)
