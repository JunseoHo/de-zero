import numpy as np

from Function import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
