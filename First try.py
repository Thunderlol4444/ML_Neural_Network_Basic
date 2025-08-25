import numpy as np
import matplotlib.pyplot as plt

a = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]
b = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0]
c = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]

y = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

plt.imshow(np.array(a).reshape(5, 6))
plt.show()
x = [a, b, c]
y = np.array(y)
x = np.array(x).T


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(X, parameters):
    # retrieve the parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


def binary_cross_entropy_loss(A2, y):
    m = y.shape[0]
    loss = -(1/m) * np.sum(y*np.log(A2) + (1-y)*np.log(1-np.array(A2)))
    return loss


def backward_propagation(parameters, cache, X, y):
    m = y.shape[1]
    gradients = []
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    # compute the derivative of the activation function of the output layer
    dZ2 = A2 - y
    # compute the derivative of the weights and biases of the output layer
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(parameters["W2"].T, dZ2) * relu_derivative(Z1)

    # compute the derivative of the weights and biases of the hidden layer
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return gradients


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def update_parameters(parameters, gradients, learning_rate):
    # retrieve the gradients
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    # retrieve the weights and biases
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # update the weights and biases
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


# train the neural network
def train(X, y, hidden_layer_size, num_iterations, learning_rate):
    # initialize the weights and biases
    parameters = initialize_parameters(30, hidden_layer_size, 3)

    for i in range(num_iterations):
        # forward propagation
        A2, cache = forward_propagation(X, parameters)

        # compute the loss
        loss = binary_cross_entropy_loss(A2, y)

        # backward propagation
        gradients = backward_propagation(parameters, cache, X, y)

        # update the parameters
        parameters = update_parameters(parameters, gradients, learning_rate)

        if i % 1000 == 0 and i != 0:
            print(f"iteration {i}: loss = {loss}")

    return parameters


parameters = train(x, y, hidden_layer_size=10, num_iterations=10000, learning_rate=0.01)


def predict(x, parameters):
    x = np.array(x).reshape(30, 1)
    out, _ = forward_propagation(x, parameters)
    print(out)
    maxm = 0
    k = 0
    for i in range(len(out)):
        if out[i] > maxm:
            maxm = out[i]
            k = i
    if k == 0:
        print("Image is of letter A.")
    elif k == 1:
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")
    plt.imshow(x.reshape(5, 6))
    plt.imshow(parameters["W1"][1].reshape(5, 6))
    plt.show()
    return _


cache = predict(x[:, 1], parameters)
