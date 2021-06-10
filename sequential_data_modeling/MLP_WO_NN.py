import numpy as np
input_size = 2
hidden_size = 2
out_size = 1
# network definition w/ initial value
w_1 = np.array([[0.9, 0.3, 0.9], [-0.7, 0.3, -0.7]])
b_1 = np.array([1])
w_2 = np.array([-0.3, -0.9, -0.7])
b_2 = np.array([1])
network = w_1, b_1, w_2, b_2
rlate = 1

def forward(network, x):
    phi1 = np.tanh(np.dot(w_1, x) + b_1)
    phi2 = np.tanh(np.dot(w_2, x) + b_2)
    return phi1, phi2

def backward(network, x, phi1, phi2, y):
    theta = np.array([0, 0, y-phi2])
    theta_dash = np.zeros(3)
    # backward for w_02
    loss = 1 - phi2 ** 2 #MSError
    theta_dash[2] = theta[2] * loss
    theta[1] = np.dot(theta_dash[2], w_2) # backward for w_01
    loss = 1 - phi1 ** 2
    theta_dash[1] = theta[1] * loss
    theta[0] = np.dot(theta_dash[1], w_1)
    return theta_dash

def update_weights(network, x, phi1, phi2, theta_dash, l_rate):
    w_1 += rlate * np.outer(theta_dash[1], x)
    b_1 += rlate * theta_dash[1]
    w_2 += rlate * np.outer(theta_dash[2], phi1)
    b_2 += rlate * theta_dash[2]

X = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]])
Y = np.array([1, -1, 1, -1])

for i in range(len(Y)):
    phi1, phi2 = forward(network, X[i])
    theta_dash = backward(network, X[i], phi1, phi2, Y[i])
    update_weights(network, X[i], phi1, phi2, theta_dash,l_rate)
