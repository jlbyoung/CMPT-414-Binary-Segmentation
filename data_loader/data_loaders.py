from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from base import BaseDataLoader
import numpy as np
import random
import torch

from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms
from albumentations.pytorch import (
    ToTensor,
    ToTensorV2
)

import albumentations as AB

albumentations_transform = AB.Compose([
        
    # Note resize and crop first if scaling down to make the
    # other transformations more efficient
    AB.Resize(256, 256),

    # Random crop
    AB.RandomSizedCrop(
        p=1,
        min_max_height=(224, 224),
        height=256,
        width=256
    ),

    #TODO: Implement ShiftScaleRotate

    # Image Flips
    AB.OneOf([
        AB.VerticalFlip(p=0.3),
        AB.HorizontalFlip(p=0.3),
        AB.RandomRotate90(p=0.5),
        AB.Transpose(p=0.4),
        ], p=0.5
    ),

    # Blurring, causing isues with tensor conversion
    AB.OneOf([
       AB.MotionBlur(p=0.2),
       AB.MedianBlur(blur_limit=3, p=0.4),
       AB.Blur(blur_limit=3, p=0.2),
       ], p=0.2
    ),

    # Image Warping/ Distortion
    AB.OneOf([
        AB.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        AB.OpticalDistortion(p=0.3),
        AB.GridDistortion(p=0.1),
        #IAAPiecewiseAffine(p=0.3),
        ], p=0.2
    )

    # Convert to Tensor
    #ToTensor()
])

image_color_transformations = AB.Compose([

    #Add varying levels and types of noise
    AB.OneOf([
        AB.IAAAdditiveGaussianNoise(p=0.4),
        AB.GaussNoise(p=0.4),
        AB.MultiplicativeNoise(multiplier=0.5, p=0.2),
        AB.MultiplicativeNoise(multiplier=1.5, p=0.1),
        AB.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),
        AB.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.2),
        AB.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
        ],p=0.3
    ),


    # Randomly Change Brightness/Sharpness/Embossment
    AB.OneOf([
        AB.CLAHE(clip_limit=2),
        AB.IAASharpen(),
        AB.IAAEmboss(),
        AB.RandomBrightnessContrast(),
        ], p=0.4
    ),

    # Change Gamma and Saturation
    AB.HueSaturationValue(p=0.3),
    AB.RandomGamma(p=0.4),

    # Random Color Channel manipulation
    AB.OneOf([
        AB.ChannelShuffle(p=0.4),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.2),
        AB.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.2),
        ], p=0.2
    ),

    # Normalize image using values from ImageNet
    AB.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


def transform_pipeline(image, mask):

    # Convert image and mask to numpy arrays
    image = np.array(image)
    mask = np.array(mask)

    # Get the composed albumentation transforms
    # augmentation = albumentations_transform()
    # data = {"image": image, "mask": mask}
    # augmented = augmentation(**data)

    augmented = albumentations_transform(image=image,mask=mask)

    # Extract transformed image and mask
    image = augmented['image']
    mask = augmented['mask']

    img_col = image_color_transformations(image=image)
    image = img_col['image']

    image = TF.to_tensor(image)
    mask = torch.from_numpy(mask).to(torch.long)
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
