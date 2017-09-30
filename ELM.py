import numpy as np
from math import sqrt


class ELM:
    def __init__(self, alpha_L2=1e-6, tanh_neurons=20, linear_neurons=True):
        self.alpha_L2 = alpha_L2
        self.tanh_neurons = tanh_neurons
        # self.linear_neurons = linear_neurons
        self._neuron_width = sqrt(3)
        self._beta = None
        self._W = None
        self._W0 = None
        # self._rand_linear = None

    def _project(self, x: np.array) -> np.array:
        return np.tanh(np.dot(x, self._W) + self._W0)

    def train(self, x: np.array, y: np.array) -> None:
        self._W = np.random.randn(x.shape[1], self.tanh_neurons) * self._neuron_width
        self._W0 = np.random.randn(1, self.tanh_neurons)
        self._beta = np.dot(np.linalg.pinv(self._project(x)), y)

    def predict(self, x: np.array) -> np.array:
        return np.dot(self._project(x), self._beta)

    def score(self, x: np.array, y: np.array) -> float:
        pass
