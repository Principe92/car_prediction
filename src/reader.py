import torch
from torchvision import datasets

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

    def load_images(filename):

        file = open(filename, 'r')

        while True:
            line = file.readline()

            if not line:
                break

            image_1, image_2, label = line.split(' ')

        
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgPaths[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path