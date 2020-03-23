import torch


def _preprocess(output, target):
    output = output.argmax(dim=1)
    output = output.to(torch.uint8)
    # output = output.squeeze(dim=1)  # BATCH x 1 x H x W => BATCH x H x W

    target = (target * 255).squeeze(dim=1)
    target = target.to(torch.uint8)
    target[target == 255] = 0
    return output, target


def meanIoU(output, target):
    SMOOTH = 1e-6
    
    output, target = _preprocess(output, target)
    intersection = (output & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (output | target).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # This is equal to comparing with thresolds
    # The minimum value of a correct IoU is 0.5, the following will measure how correct a corect
    # segmentation is (segmentation with IoU > 0.5).
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    return iou.mean()
