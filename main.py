import numpy as np

from Exp import Exp
from Square import Square
from Variable import Variable

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
print(f"{x.data} -> A -> {a.data}")
b = B(a)
print(f"{a.data} -> B -> {b.data}")
y = C(b)
print(f"{b.data} -> C -> {y.data}")

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
