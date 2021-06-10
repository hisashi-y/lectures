from mxnet import autograd, np, npx
npx.set_np()

W1 = np.array([[0.9, 0.3, -0.8], [-0.3, -0.5, 0.7]])
Wh = np.array([[0.7, -0.7], [-0.9, 0.4]])
W2 = np.array([[-0.3, -0.8], [-0.6, 0.2], [-0.9, 0.2]])
params = [W1, Wh, W2]
#for param in params:
#    param.attach_grad()

X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#X.attach_grad()

fh = np.array([0, 0])

def ReLu(X):
    return np.maximum(0, X)
def SoftMax(X):
    x1 = np.exp(X[0])
    x2 = np.exp(X[1])
    x3 = np.exp(X[2])
    denominator = x1 + x2 + x3
    y1 = x1 / denominator
    y2 = x2 / denominator
    y3 = x3 / denominator
    return np.array([y1, y2, y3])

for i in range(len(X)):
    print('time', i+1)
    x = X[i]
    hi = np.dot(W1, x) + np.dot(Wh, fh)
    print('result of 1st layer', hi)
    fh = ReLu(hi)
    print('apply ReLu', fh)
    Y = np.dot(W2, fh)
    print('result of 2nd layer', Y)
    Y = SoftMax(Y)
    print('apply SoftMax', Y)
