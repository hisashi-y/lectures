from mxnet import autograd, np, npx
npx.set_np()

W1 = np.array([[0.9, 0.3], [-0.7, 0.3]])
W2 = np.array([-0.3, -0.9])
b1 = np.array([0.9,-0.7])
b2 = np.array([-0.7])
params = [W1, b1, W2, b2]
for param in params:
    param.attach_grad()

X = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
X.attach_grad()
y_true = np.array([1, -1, 1, -1])

for i in range(len(y_true)):
    with autograd.record():
        H = np.tanh(np.dot(W1, X[i]) + b1)
        O = np.tanh(np.dot(W2, H) + b2)
        L = (y_true[i] - O) ** 2
        L = 1/2 * L
    L.backward()
    W1 -= W1.grad
    W2 -= W2.grad
    b1 -= b1.grad
    b2 -= b2.grad
    print('iteration:', i+1)
    print('true label', y_true[i])
    print('input', X[i])
    print('predicted label:', O)
    print('updated W1', W1)
    print('updated b1', b1)
    print('updated W2', W2)
    print('updated b1', b2)
