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

        # num_batches=10
        # data_shuffle=np.random.shuffle(data)
        #batch = data.shape[0] // batch_size

        # Calculating gradients (credit to https://tomaxent.com/2017/03/05/cs231n-Assignment-1-softmax/ for some help)

        for i in range(0, len(X_train)):
            # print(i)
            # Make initial prediction
            # Calculating initial prediction values
            wyx = np.matmul(self.w, np.transpose(X_train[i]))
            # print('wyx')
            # print(wyx)
            # print(X_train[i].dot(np.transpose(self.w)))
            # making everything non=positive (max of zero) (because e^(non-positive) is between 0 and 1)
            wyx -= np.max(wyx)
            # getting the prediction value for the correct class
            correct_wyx = wyx[y_train[i]]
            # print('Correct_wyx')
            # print(correct_wyx)
            sigma = np.sum(np.exp(wyx))  # the summed denominator
            # print('Sigma')
            # print(sigma)
            # print('e^wyx')
            # print(np.exp(wyx))
            # print('y_train')
            # print(y_train[i])

            # The model's calcualted probability for the correct class is correct for loss function
            prob = np.exp(correct_wyx)/sigma
            loss += -np.log(prob)  # Adjusting sum of the loss
            # print('loss')

            for c in range(0, self.n_class):
                if c == y_train[i]:
                    # Adjusting the gradients of the actual class
                    gradients[y_train[i],
                              :] -= ((np.exp(wyx[y_train[i]])/sigma) - 1)*X_train[i]
                    # print(gradients[y_train[i],:])
                    #print('wow incredible')
                    # print(gradients[:,y_train[i]])

                else:
                    #print('Gradients before')
                    # print(gradients[c,:])
                    # print('wyx[c]')
                    # print(wyx[c])
                    #print('the gradient eq')
                    # print((np.exp(wyx[c])/sigma)*X_train[i])
                    # Adjusting gradients of the incorrect classes
                    gradients[c, :] -= (np.exp(wyx[c])/sigma)*X_train[i]
                    #print('Gradients after')
                    # print(gradients[c,:])
                # Calculate gradients
                # gwy=-X_train[i]+((np.exp(wyx)*X_train[i])/np.sum(sigma))

                # gwc=(np.exp(wyx)*X_train[i])/(np.sum(sigma)) #do i have to change for the j?

        # Averaging loss
        gradients /= len(X_train)
        loss /= len(X_train)

        # Regularization
        gradients += self.reg_const*self.w
        loss += 0.5*self.reg_const*np.sum(self.w**2)
        # print('loss')
        # print(loss)

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
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 0.0001
        m = v = 0

        for e in range(0, self.epochs):
            loss_epoch = 0

            # if e % 10 == 0:
            #     self.lr = self.lr/5 


            for i in range(0, num_batches):


                batch_x = X_train[int(i*(batch_size)) :int(i*batch_size+batch_size-1)]
                batch_y = y_train[int(i*(batch_size)) :int(i*batch_size+batch_size-1)]
                # print(batch_x)
                # print(batch_y)
                # print(type(batch_y))
                gradients, loss = Softmax.calc_gradient(self, batch_x, batch_y)


                self.w += self.lr*gradients
                loss_epoch = loss
                # print(self.w)

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
            # print(index)
            y_test.append(index)
            # print(y_test)

        return y_test
