from torchvision import datasets, transforms
from base import BaseDataLoader


class VOCDataLoader(BaseDataLoader):
    """
    Pascal VOC data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # NOTE: we are scaling down to much we will need to increase this significantly for good model results
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # Values from ImageNet
        ])

        target_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor()
        ])

        image_set = 'train' if training else 'val'
        self.data_dir = data_dir
        self.dataset = datasets.VOCSegmentation(self.data_dir, image_set=image_set, download=True,
                                                transform=transform, target_transform=target_transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
