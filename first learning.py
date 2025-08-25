import numpy as np


def sigmoid(array):
    return 1/(1+np.exp(array))


def loss(predictions, actual):
    return -(actual*np.log(predictions) + (1-actual)*np.log(1-predictions))


m = 20  # number of training set
y = np.random.rand(1, m)
rate_of_learning = 0.1
x = np.random.rand(2, m)  # 1 input img
w = np.random.rand(m, 1)
b = np.random.rand(1, m)
J = 0
'''dw1 = 0
dw2 = 0'''
db = 0
dw = 0
for i in range(m):
    z_i = w[i]*x[:, i] + b[1, i]
    a_i = sigmoid(z_i)
    J += loss(a_i, y[1, i])
    dz_i = a_i - y[1, i]
    '''dw1 += x[i, 0] * dz_i[0]  # non vectorised
    dw2 += x[i, 1] * dz_i[1]'''
    dw += x[i]*dz_i
    db += dz_i
J = J/m
'''dw1 = dw1/m
dw2 = dw2/m'''
dw = dw/m
db = db/m

# complete vectorisation
Z = np.dot(w.transpose(), x) + b
A = sigmoid(Z)
dZ = A - y
dw = np.dot(x, dZ.tranpose())/m
db = np.sum(dZ, axis=0)/m
new_w = w - rate_of_learning * dw
new_b = b - rate_of_learning * db
