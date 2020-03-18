import torch
import numpy as np
import random
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

"""PyTorch"""

def get_best_device():
    # return torch.device('cpu')

    # Uncomment when option to run on better GPU is available
    if torch.cuda.is_available(): 
        print(f"Running on the GPU: {torch.cuda.get_device_name()}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        print(f"Running on the CPU, certain tasks might run very slow")
        return torch.device('cpu')


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou


""" Semantic Segementation """
def decode_segmap(image, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb


""" Instance Segementation """
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_masks(pred, threshold=0.5):
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    
    masks = masks[:pred_t+1]
    return masks


def instance_segmentation_plot(img, masks):
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        rgb_mask = Image.fromarray(rgb_mask, 'RGB')
        img = ImageChops.add(img, rgb_mask)

    display_image(img)


def display_image(img, size=(8, 8)):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
