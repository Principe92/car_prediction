import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable


def show_images(images, color=False):
    if color:
        sqrtimg = int(np.ceil(np.sqrt(images.shape[2]*images.shape[3])))
    else:
        images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
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
            plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def loss_fn(x, y):

    return None

def load_test_images(filename) -> np.ndarray:

    file = open(filename, 'r')
    paths = []
    itr = 0

    while True:
        line = file.readline().replace('\n', '')
        item = []

        if not line or itr == 5000:
            break

        image_1, image_2, label = line.split(' ')

        item.append(image_1)
        item.append(image_2)
        item.append(label)
        paths.append(item)

        itr += 1

    return np.array(paths)
        
def load_train_images(filename) -> np.ndarray:

    file = open(filename, 'r')
    paths = []

    while True:
        line = file.readline().replace('\n', '')

        if not line:
            break

        paths.append([line])
    
    return np.array(paths)


def get_vector(model, img, device, layer) -> torch.Tensor:

    t_img = Variable(img.unsqueeze(0)).to(device)

    embedding = torch.zeros(512)

    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_data)

    model(t_img)

    h.remove()

    return embedding

def get_acc(pred, y_test) -> float:
    return np.sum(y_test==pred)/len(y_test)*100