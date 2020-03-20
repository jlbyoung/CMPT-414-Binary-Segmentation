import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def meanIoU(output, target):
    SMOOTH = 1e-6
    output = output.argmax(dim=1)
    output = output.to(torch.uint8)

    target = (target * 255).squeeze(dim=1)
    target = target.to(torch.uint8)


    output = output.squeeze(dim=1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (output & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (output | target).float().sum((1, 2))         # Will be zero if both are 0
    

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou.mean().item()
