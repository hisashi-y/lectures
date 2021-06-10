import math
from mxnet import np, npx, gluon, autograd
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()

initial_w1 = np.array([[0.9, 0.3, 0.9], [-0.7, 0.3, -0.7]])
initial_w2 = np.array([-0.3, -0.9, -0.7])

true_labels = np.array([1, -1, 1, -1])
inputs = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]])

#No1についての予測, 以下No. n のm層目に関してh_n_m などと記述
h_1_1 = np.dot(initial_w1, inputs[0])
z_1_1 = np.append(np.tanh(h1), np.array([1]))
z_1_3 = np.dot(initial_w2, z_1_1)
y_hat_1 = np.tanh(z_1_3)

y_hat_1

#No1についてのロス
