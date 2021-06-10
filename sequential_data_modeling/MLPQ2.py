import math
from mxnet import np, npx, gluon, autograd
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()

#とりあえずNo1だけについて
W1 = np.array([[0.9, 0.3, 0.9], [-0.7, 0.3, -0.7]])
W2 = np.array([-0.3, -0.9, -0.7])
b1 = np.array([1])
b2 = np.array([1])
params = [W1, b1, W2, b2]
for param in params:
    param.attach_grad()

X = np.array([1, 1, 1])
X.attach_grad()
y_true = np.array([1])

with autograd.record():
    H = np.tanh(np.dot(W1, X))
    O = np.tanh(np.dot(W2, np.append(H, np.array([1]))))
    L = (y_true - O) ** 2
    L = 1/2 * L
L.backward()
print('predicted value:', O)
print('updated W1', W1 - W1.grad)
print('updated W2', W2 - W2.grad)
