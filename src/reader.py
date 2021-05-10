import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision import models
from src.utils import get_vector, load_test_images, load_train_images


class TestSet(data.Dataset):
    """
        Pytorch dataset that loads testing images,
        Generates features vectors extracted using resnet
    """

    def __init__(self, root: str, device: str, transform: transforms.Compose = None) -> None:
        super().__init__()

        self.transform = transform
        self.root = root
        self.device = device
        self.paths = load_test_images(self.root + '/train_test_split/verification/verification_pairs_easy.txt')

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet.to(self.device)
        self.layer = self.resnet._modules.get('avgpool')

    def __getitem__(self, index):

        image_1_path, image_2_path, label = self.paths[index]

        image_1_path = self.root + '/image/' + image_1_path
        image_2_path = self.root + '/image/' + image_2_path
        path = image_1_path + '||' + image_2_path

        image_1 = Image.open(image_1_path, mode='r')
        image_2 = Image.open(image_2_path, mode='r')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # using resnet, extract the feature vectors required for PCA for each image
        v1 = get_vector(self.resnet, image_1, self.device, self.layer).reshape((1, 512))
        v2 = get_vector(self.resnet, image_2, self.device, self.layer).reshape((1, 512))
        

        return v1, v2, image_1, image_2, int(label), path

    def __len__(self):
        return self.paths.shape[0]


class TrainSet(data.Dataset):
    """
        Pytorch dataset that loads training images,
        Generates feature vectors extracted using resnet
    """

    def __init__(self, root: str, device: str, transform: transforms.Compose = None) -> None:
        super().__init__()

        self.transform = transform
        self.root = root
        self.device = device
        self.paths = load_train_images(self.root + '/train_test_split/verification/verification_train.txt')
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet.to(self.device)
        self.layer = self.resnet._modules.get('avgpool')

    def __getitem__(self, index):

        path = self.paths[index][0]

        label = int(path.split('/')[1])

        path = self.root + '/image/' + path

        image = Image.open(path, mode='r')

        if self.transform is not None:
            image = self.transform(image)

        # using resnet, extract the feature vector required for PCA
        v = get_vector(self.resnet, image, self.device, self.layer).reshape((1, 512))

        return v, image, int(label), path

    def __len__(self):
        return self.paths.shape[0]
