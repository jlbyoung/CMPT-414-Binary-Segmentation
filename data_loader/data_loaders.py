from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from base import BaseDataLoader
import numpy as np
import random
import torch

def transform_pipeline(image, mask):
    # Note resize and crop first if scaling down to make the
    # other transformations more efficient

    # Resize 
    resize = transforms.Resize(size=(256, 256))
    image = resize(image)
    mask = resize(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Apply one by using elif for subsequent if statements

    # # Randomly change brightness
    # if random.random() > 0.5:
    #     # Brightness factor with a range of (0.5, 1.5)
    #     image = TF.adjust_brightness(image, random.random() + 0.5)

    # # Randomly change contrast
    # if random.random() > 0.5:
    #     # Contrast factor with a range of (0.5, 2.0)
    #     image = TF.adjust_contrast(image, random.randint(5, 20) / 10)

    # # Randomly change gamma
    # if random.random() > 0.5:
    #     # Contrast factor with a range of (0.5, 1.5)
    #     image = TF.adjust_gamma(image, random.random() + 0.5)

    # # Randomly change hue
    # if random.random() > 0.5:
    #     # Contrast factor with a range of (-0.5, 0.5)
    #     image = TF.adjust_hue(image, random.random() - 0.5)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Random rotation between -10 and 10 degrees
    if random.random() > 0.5:
        angle = random.randint(-10, 10)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = np.array(mask)
    mask = torch.from_numpy(mask).to(torch.long)

    # Normalize image using values from ImageNet
    image = TF.normalize(image, 
                         mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225]) 
    return image, mask
    

class VOCDataLoader(BaseDataLoader):
    """
    Pascal VOC data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        image_set = 'train' if training else 'val'
        self.data_dir = data_dir
        self.dataset = datasets.VOCSegmentation(self.data_dir, image_set=image_set, download=True,
                                                transforms=transform_pipeline)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
