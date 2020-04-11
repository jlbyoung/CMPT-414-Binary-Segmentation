from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from base import BaseDataLoader, BaseDataSet
import numpy as np
import random
import torch
import os

from torch.utils.data import Dataset
from PIL import Image
import cv2
from albumentations.pytorch import (
    ToTensor,
    ToTensorV2
)

import albumentations as AB

albumentations_transform = AB.Compose([
        
    # Note resize and crop first if scaling down to make the
    # other transformations more efficient
    AB.SmallestMaxSize(256),

    # Random crop
    AB.RandomSizedCrop(
        p=1,
        min_max_height=(200, 224),
        height=256,
        width=256
    ),

    # ShiftScaleRotate
    AB.ShiftScaleRotate(p=0.5),

    # Image Flips
    AB.OneOf([
        AB.VerticalFlip(p=0.5),
        AB.HorizontalFlip(p=0.5),
        AB.RandomRotate90(p=0.5),
        AB.Transpose(p=0.5),
        ], p=0.5
    ),

    # Blurring, causing isues with tensor conversion
    AB.OneOf([
       AB.MotionBlur(p=0.5),
       AB.MedianBlur(blur_limit=3, p=0.5),
       AB.Blur(blur_limit=3, p=0.5),
       ], p=0.5
    ),

    # Image Warping/ Distortion
    AB.OneOf([
        AB.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        AB.OpticalDistortion(p=0.5),
        AB.GridDistortion(p=0.5),
        #IAAPiecewiseAffine(p=0.3),
        ], p=0.5
    )

    # Convert to Tensor
    #ToTensor()
])

image_color_transformations = AB.Compose([

    #Add varying levels and types of noise
    AB.OneOf([
        AB.IAAAdditiveGaussianNoise(p=0.5),
        AB.GaussNoise(p=0.5),
        AB.MultiplicativeNoise(multiplier=0.5, p=0.5),
        AB.MultiplicativeNoise(multiplier=1.5, p=0.5),
        AB.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.5),
        AB.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.5),
        AB.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.5),
        ],p=0.2
    ),


    # Randomly Change Brightness/Sharpness/Embossment
    AB.OneOf([
        AB.CLAHE(clip_limit=2),
        AB.IAASharpen(),
        AB.IAAEmboss(),
        AB.RandomBrightnessContrast(),
        ], p=0.2
    ),

    # Change Gamma and Saturation
    AB.HueSaturationValue(p=0.2),
    AB.RandomGamma(p=0.2),

    # Random Color Channel manipulation
    AB.OneOf([
        AB.ChannelShuffle(p=0.5),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.5),
        AB.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=0.5),
        ], p=0.2
    ),

    # Normalize image using values from ImageNet
    AB.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

valid_transforms = AB.Compose([
    # Note resize and crop first if scaling down to make the
    # other transformations more efficient
    AB.SmallestMaxSize(512),
    # Center crop
    AB.RandomCrop(
        height=480,
        width=480
    )
])

valid_image_transforms = AB.Compose([
    # Normalize image using values from ImageNet
    AB.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


def transform_pipeline(image, mask, training=True):

    # Convert image and mask to numpy arrays
    image = np.array(image)
    mask = np.array(mask)

    if training:
        augmented = albumentations_transform(image=image,mask=mask)

        # Extract transformed image and mask
        image = augmented['image']
        mask = augmented['mask']

        img_col = image_color_transformations(image=image)
        image = img_col['image']
    else:
        augmented = valid_transforms(image=image,mask=mask)

        # Extract transformed image and mask
        image = augmented['image']
        mask = augmented['mask']

        img_col = valid_image_transforms(image=image)
        image = img_col['image']
        
    image = TF.to_tensor(image)
    mask = torch.from_numpy(mask).to(torch.long)
    
    mask[mask != 15] = 0
    mask[mask == 15] = 1
    return image, mask
    

class VOCDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.base = os.path.join(self.root, 'VOCdevkit/VOC2012')
        self.image_dir = os.path.join(self.base, 'JPEGImages')
        self.label_dir = os.path.join(self.base, 'SegmentationClass')

        file_list = os.path.join(self.base, "ImageSets/Segmentation", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = Image.open(image_path)
        label = Image.open(label_path)
        # image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label
      
class VOCDataLoader(BaseDataLoader):
    """
    Pascal VOC data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        train_pipeline = lambda im, target: transform_pipeline(im, target)
        valid_pipeline = lambda im, target: transform_pipeline(im, target, training=False)

        image_set = 'person_train' if training else 'person_val'
        self.data_dir = data_dir
        self.dataset = VOCDataset(root=self.data_dir, split=image_set, 
                                  transforms=train_pipeline if training else valid_pipeline,
                                  val_transforms=valid_pipeline)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
