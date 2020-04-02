from torchvision import datasets, transforms
#import torchvision.transforms.functional as TF
from base import BaseDataLoader
import numpy as np
import random
import torch

from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms
from albumentations.pytorch import ToTensor

from albumentations import (
    PadIfNeeded,
    IAAAdditiveGaussianNoise,
    IAASharpen,
    IAAEmboss,
    IAAPiecewiseAffine,
    GaussNoise,
    MedianBlur,
    MotionBlur,
    Blur,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    Normalize,
    Resize,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    HueSaturationValue,  
    RandomGamma,
    MultiplicativeNoise,
    ChannelDropout,
    ChannelShuffle    
)

def albumentations_transform(p=0.5):
    return Compose([
        # Note resize and crop first if scaling down to make the
        # other transformations more efficient
        Resize(256, 256),

        # Random crop
        RandomSizedCrop(
            p=1,
            min_max_height=(224, 224),
            height=256,
            width=256
        ),

        #TODO: Implement ShiftScaleRotate

        # Image Flips
        OneOf([
            VerticalFlip(p=0.3),
            HorizontalFlip(p=0.3),
            RandomRotate90(p=0.5),
            Transpose(p=0.4),
            ], p=0.5
        ),

        # Blurring, causing isues with tensor conversion
        #OneOf([
        #    MotionBlur(p=0.2),
        #    MedianBlur(blur_limit=3, p=0.4),
        #    Blur(blur_limit=3, p=0.2),
        #    ], p=0.2
        #),

        # Image Warping/ Distortion
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
            ], p=0.2
        ),

        #Add varying levels and types of noise
        OneOf([
            IAAAdditiveGaussianNoise(p=0.4),
            GaussNoise(p=0.4),
            MultiplicativeNoise(multiplier=0.5, p=0.2),
            MultiplicativeNoise(multiplier=1.5, p=0.1),
            MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),
            MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.2),
            MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
            ],p=0.3
        ),


        # Randomly Change Brightness/Sharpness/Embossment
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
            ], p=0.4
        ),

        # Change Gamma and Saturation
        HueSaturationValue(p=0.3),
        RandomGamma(p=0.4),

        # Random Color Channel manipulation
        OneOf([
            ChannelShuffle(p=0.4),
            ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.2),
            ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.2),
            ], p=0.2
        ),

        # Normalize image using values from ImageNet
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        # Convert to Tensor
        ToTensor()
    ], p=p)


def transform_pipeline(image, mask):

    # Convert image and mask to numpy arrays
    image = np.array(image)
    mask = np.array(mask)

    # Get the composed albumentation transforms
    augmentation = albumentations_transform(p=0.9)
    data = {"image": image, "mask": mask}
    augmented = augmentation(**data)

    # Extract transformed image and mask
    image = augmented['image']
    mask = augmented['mask']

    
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
