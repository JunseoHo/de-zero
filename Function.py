import numpy as np

from Variable import Variable


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(
            np.array(y) if np.isscalar(y) else y
        )  # numpy는 0차원 배열에 대한 연산 결과를 스칼라로 반환하므로 이를 다시 numpy 배열로 변환
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)
