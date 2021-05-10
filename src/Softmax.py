"""Softmax model."""

import numpy as np
from numpy.random import beta
import torch


class Softmax(torch.nn.Module):
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        super(Softmax, self).__init__()

        self.w = []  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        # Initializing weight array
        gradients = np.random.rand(self.n_class, X_train.shape[1])
        loss = 0

        # Calculating gradients (credit to https://tomaxent.com/2017/03/05/cs231n-Assignment-1-softmax/ for some help)

        for i in range(0, len(X_train)):
            # Make initial prediction
            # Calculating initial prediction values
            wyx = np.matmul(self.w, np.transpose(X_train[i]))

            # making everything non=positive (max of zero) (because e^(non-positive) is between 0 and 1)
            wyx -= np.max(wyx)

            # getting the prediction value for the correct class
            correct_wyx = wyx[y_train[i]]

            sigma = np.sum(np.exp(wyx))  # the summed denominator

            # The model's calcualted probability for the correct class is correct for loss function
            prob = np.exp(correct_wyx)/sigma
            loss += -np.log(prob)  # Adjusting sum of the loss

            for c in range(0, self.n_class):
                if c == y_train[i]:
                    # Adjusting the gradients of the actual class
                    gradients[y_train[i], :] -= ((np.exp(wyx[y_train[i]])/sigma) - 1)*X_train[i]

                else:
                    gradients[c, :] -= (np.exp(wyx[c])/sigma)*X_train[i]
             
        # Averaging loss
        gradients /= len(X_train)
        loss /= len(X_train)

        # Regularization
        gradients += self.reg_const*self.w
        loss += 0.5*self.reg_const*np.sum(self.w**2)

        return gradients, loss

    def forward(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.random.rand(self.n_class, X_train.shape[1])/1000

        # Thinking something like: https://cs231n.github.io/optimization-1/
        num_batches = 10  # where does this fit in
        batch_size = len(X_train)/num_batches
        loss_list = []

        for e in range(0, self.epochs):
            loss_epoch = 0

            # if e % 10 == 0:
            #     self.lr = self.lr/5 


            for i in range(0, num_batches):


                batch_x = X_train[int(i*(batch_size)) :int(i*batch_size+batch_size-1)]
                batch_y = y_train[int(i*(batch_size)) :int(i*batch_size+batch_size-1)]
                gradients, loss = Softmax.calc_gradient(self, batch_x, batch_y)


                self.w += self.lr*gradients
                loss_epoch = loss

            loss_list.append(loss_epoch)

        # pass
        return loss_list

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me

        y_test = []
        for i in range(0, len(X_test)):
            fx = np.matmul(self.w, np.transpose(X_test[i]))
            index = np.argmax(fx)
            y_test.append(index)

        return y_test
