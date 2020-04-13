from torch.utils.data import Dataset

class BaseDataSet(Dataset):
    def __init__(self, root, split, transforms, val_transforms=None):
        self.root = root
        self.split = split
        self.files = []
        self.transforms = transforms
        self.val_transforms = val_transforms
        self._set_files()

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label = self._load_data(index)
        return self.transforms(image, label)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
    
    def val(self):
        return self.__class__(root=self.root, split=self.split, 
                              transforms=self.val_transforms, 
<<<<<<< HEAD
                              val_transforms=None)
=======
                              val_transforms=None)
>>>>>>> 68bd97f826d5862bc77be3dbf8f370c1f1f35b87
