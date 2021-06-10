import math
from mxnet import np, npx, gluon, autograd
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()

#とりあえずNo1だけについて
W1 = np.array([[ 0.8794594 ,  0.2794594  , 0.8794594 ],
 [-1.0801702,  -0.08017018, -1.0801702 ]]) #updated based on Q2
W2 = np.array([ 0.8412257 , -1.8413655  , 0.47597343])
b1 = np.array([1])
b2 = np.array([1])
params = [W1, b1, W2, b2]
for param in params:
    param.attach_grad()

X = np.array([0, 1, 1]) #No2
X.attach_grad()
y_true = np.array([-1])#True Label of No2

with autograd.record():
    H = np.tanh(np.dot(W1, X))
    O = np.tanh(np.dot(W2, np.append(H, np.array([1]))))
    L = (y_true - O) ** 2
    L = 1/2 * L
L.backward()
print('predicted value:', O)
print('updated W1', W1 - W1.grad)
print('updated W2', W2 - W2.grad)
