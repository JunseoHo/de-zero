import numpy as np

from Variable import Variable


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(
            np.array(y) if np.isscalar(y) else y
        ) for y in ys]  # numpy는 0차원 배열에 대한 연산 결과를 스칼라로 반환하므로 이를 다시 numpy 배열로 변환

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]  # 원소가 하나라면 구태여 리스트로 반환할 필요가 없다.

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def add(x0, x1):
    f = Add()
    return f(x0, x1)
