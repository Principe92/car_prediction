import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from torch.autograd import Variable


def show_images(images: np.ndarray, color=False) -> None:
    """
        Function originally created by Dr. Stylianou
        Displays images from a numpy array in a grid
    """
    if color:
        sqrtimg = int(np.ceil(np.sqrt(images.shape[2]*images.shape[3])))
    else:
        # images reshape to (batch_size, D)
        images = np.reshape(images, [images.shape[0], -1])
        sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if color:
            plt.imshow(np.swapaxes(np.swapaxes(img, 0, 1), 1, 2))
        else:
            plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


def show_prediction_images(
        expected: np.ndarray,
        actual: np.ndarray,
        images: np.ndarray,
        name: str,
        wrong: bool = True) -> None:
    """
        Displays pairs of images predicted as equal or not
    """

    itr = 0

    size = 20

    img_vectors = []

    for i in range(expected.shape[0]):
        same = expected[i] == actual[i]

        if wrong and not same and itr < size:
            paths = images[i].split('||')
            img_vectors.append(mpimg.imread(paths[0]))
            img_vectors.append(mpimg.imread(paths[1]))
            itr += 1

        if not wrong and same and itr < size:
            paths = images[i].split('||')
            img_vectors.append(mpimg.imread(paths[0]))
            img_vectors.append(mpimg.imread(paths[1]))
            itr += 1

        if itr == size:
            break

    img_vectors = np.array(img_vectors)
    sqrtn = int(np.ceil(np.sqrt(img_vectors.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(size, 2)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(len(img_vectors)):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        print(img_vectors[i].shape)
        plt.imshow(img_vectors[i])
    
    figure = "{0}.png".format(name)
    plt.savefig(figure, bbox_inches='tight', dpi=600)


def load_test_images(filename: str) -> np.ndarray:
    """
        Opens the test set file and returns a numpy array of images paths
    """

    file = open(filename, 'r')
    paths = []
    itr = 0

    while True:
        line = file.readline().replace('\n', '')
        item = []

        if not line:
            break

        image_1, image_2, label = line.split(' ')

        item.append(image_1)
        item.append(image_2)
        item.append(label)
        paths.append(item)

        itr += 1

    return np.array(paths)


def load_train_images(filename: str) -> np.ndarray:
    """
        Opens the training set file and returns a numpy array of image paths
    """

    file = open(filename, 'r')
    paths = []

    while True:
        line = file.readline().replace('\n', '')

        if not line:
            break

        paths.append([line])

    return np.array(paths)


def get_vector(model, img: torch.Tensor, device: str, layer) -> torch.Tensor:
    """
        Extracts feature vectors of an image at the avgpool layer of a resnet model
    """

    t_img = Variable(img.unsqueeze(0)).to(device)

    embedding = torch.zeros(512)

    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_data)

    model(t_img)

    h.remove()

    return embedding


def get_acc(pred: np.ndarray, actual: np.ndarray) -> float:
    """
        Calculates the accuracy of prediction
    """
    return np.sum(actual == pred)/len(actual)*100
