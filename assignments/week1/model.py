import numpy as np
from numpy.linalg import inv
import matplotlib.pylab as plt


class LinearRegression:
    """
    A linear regression model that closed form to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.ndarray
        self.b = float

    def fit(self, X: np.ndarray, y: np.ndarray) -> (None):
        """
        fit the function by closed form

         Arguments:
             X (np.ndarray): The input data.
             y (np.ndarray): The input data




         Returns:
             None

        """

        n = y.shape[0]
        p = X.shape[1]
        self.index = np.ones((n, 1), dtype=int)
        X = np.column_stack((X, self.index))
        self.beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = self.beta_hat[p]
        self.w = self.beta_hat[0:p]

    def predict(self, X: np.ndarray) -> (np.ndarray):
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.




        Returns:
            np.ndarray: The predicted output.

        """

        y = np.matmul(X, self.w) + self.b
        return y


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> (np.ndarray):
        """
        fit the function by gradient descent

         Arguments:
             X (np.ndarray): The input data.
             y (np.ndarray): The input data
             lr (float): learning rate
             epochs (int): number of epochs



         Returns:
             np.ndarray: The fitted value output.

        """

        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = np.zeros((1))
        y = y.reshape(m, 1)
        losses = []
        for i in range(epochs):

            y_hat = np.matmul(X, self.weights) + self.bias

            # Calculting loss
            loss = np.mean((y_hat - y) ** 2)

            # Appending loss in list: losses
            losses.append(loss)

            # Calculating derivatives of parameters(weights, and
            # bias)
            dw = (2 / m) * np.matmul(X.T, (y_hat - y))
            db = (2 / m) * np.sum((y_hat - y))
            # Updating the parameters: parameter := parameter - lr*derivative
            # of loss/cost w.r.t parameter)
            self.weights -= lr * dw
            self.bias -= lr * db

            # y = np.matmul(X, self.weights) + self.bias
        y_fitted = np.matmul(X, self.weights) + self.bias
        self.out = loss

        # print(loss)
        return y_fitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.





        Returns:
            np.ndarray: The predicted output.

        """

        product = np.matmul(X, self.weights) + self.bias
        return product
