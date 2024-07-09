import unittest

import numpy as np

from Function import square
from Variable import Variable


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (eps * 2)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        self.assertEqual(y.data, np.array(4.0))

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        self.assertEqual(x.grad, np.array(6.0))

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
