"""Support Vector Machine (SVM) model."""

import numpy as np
import torch


class SVM(torch.nn.Module):
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        super(SVM, self).__init__()

        self.w = []  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples. (I think my w array is (C,D))

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # Initializing weight array
        gradients = np.random.rand(self.n_class, X_train.shape[1])
        loss = 0

        # Calculating gradients (credit to https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html for some help)

        for i in range(0, len(X_train)):
            # Make initial prediction

            scores = np.matmul(self.w, np.transpose(
                X_train[i]))  # 2 * 200 X 200 * 216

            for c in range(0, self.n_class):
                if c == y_train[i]:
                    continue
                # want min distance 1
                margin = scores[c]-scores[int(y_train[i])]+1

                # Update function
                if margin > 0:
                    loss += margin
                    gradients[y_train[i]] += X_train[i]
                    gradients[c] -= X_train[i]

        # Averaging loss
        loss /= len(X_train)
        gradients /= len(X_train)

        # Regularization
        loss += 0.5*self.reg_const * np.sum(self.w**2)
        gradients += self.reg_const*self.w

        # TODO: implement me
        # do I even need to calculate losses? (only for sanity check I think)
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
        self.w = np.random.rand(self.n_class, X_train.shape[1])
        
        # Thinking something like: https://cs231n.github.io/optimization-1/
        num_batches = 5  # where does this fit in
        batch_size = len(X_train)/num_batches
        loss_list = []

        for e in range(0, self.epochs):
            loss_epoch = 0

            # if e % 10 == 0:
            #     self.alpha = self.alpha/2

            for i in range(0, num_batches):

                batch_x = X_train[int(i*(batch_size))
                                      :int(i*batch_size+batch_size-1)]
                batch_y = y_train[int(i*(batch_size))
                                      :int(i*batch_size+batch_size-1)]

                gradients, loss = SVM.calc_gradient(self, batch_x, batch_y)
                self.w += self.alpha*gradients
                loss_epoch = loss

            loss_list.append(loss_epoch)

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
