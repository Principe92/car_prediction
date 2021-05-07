import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision import models
from src.utils import get_vector


class ImageFolderWithPaths(datasets.ImageFolder):
    """
        Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder

        Args:
            root (string): Root directory path
            imgPaths (string []): List of image paths 
    """

    def __init__(self, root, imgPaths, transform=None):
        super(ImageFolderWithPaths, self).__init__(root, transform=transform)
        self.imgPaths = imgPaths

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgPaths[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


class TestSet(data.Dataset):

    def __init__(self, root: str, paths: np.ndarray, device, transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.paths = paths
        self.root = root
        self.device = device

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet.to(device)

    def __getitem__(self, index):

        image_1_path, image_2_path, label = self.paths[index]

        image_1 = Image.open(self.root + '/' + image_1_path, mode='r')
        image_2 = Image.open(self.root + '/' + image_2_path, mode='r')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        v1 = get_vector(self.resnet, image_1, self.device).reshape((1, 512))
        v2 = get_vector(self.resnet, image_2, self.device).reshape((1, 512))
        

        return v1, v2, image_1, image_2, label

    def __len__(self):
        return self.paths.shape[0]


class TrainSet(data.Dataset):

    def __init__(self, root: str, paths: np.ndarray, device, transform: transforms.Compose = None) -> None:
        super().__init__()

        self.transform = transform
        self.paths = paths
        self.root = root
        self.device = device
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet.to(device)

    def __getitem__(self, index):

        path = self.paths[index][0]

        model = int(path.split('/')[1])

        path = self.root + '/' + path

        image = Image.open(path, mode='r')

        if self.transform is not None:
            image = self.transform(image)

        
        v = get_vector(self.resnet, image, self.device).reshape((1, 512))

        return v, image, model

    def __len__(self):
        return self.paths.shape[0]
