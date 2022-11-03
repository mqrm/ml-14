import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame({"x0": [1, 1, 1, 1],
                   "x1": [2, 3, -4, -2],
                   "x2": [4, 3, -2, -6],
                    "y": [1, 1, 0, 0],
                  })

X = df.loc[:,["x0", "x1", "x2"]].to_numpy()
y = df.loc[:,["y"]].to_numpy()
x0 = df.loc[:, ["x0"]].to_numpy()
x1 = df.loc[:, ["x1"]].to_numpy()
x2 = df.loc[:, ["x2"]].to_numpy()
w = np.asmatrix([0, 0, 0]) # initial weights
a = 0.5 # the learning rate
num_of_iterations = 1

# define the sigmoid function g(z)
def g(z):
    return 1. / (1 + np.exp(-z))

# define the hypothesis function h(x, w)
def H(X, w):
    return g(X.dot(w.T))

# define the prediction function returning 0 or 1 depending on the predicted class
def predict(X, w):
    return np.around(H(X, w))

# define the loss function
def J(h, y):
    #return - np.mean(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
    return - np.sum(np.multiply(y, np.log(h)) + np.multiply((1-y), np.log(1-h)))

# define the gradient function
def grad(w, X, y):
    return (H(X, w) - y).T * X

# define the update function of w
def delta_w(w, X, y, a):
    return a * grad(w, X, y)

h = H(X, w)
j = J(h, y)
j_iter = [j] # List to store the loss values over the iterations
w_iter = [w] # List to store the weight values over the iterations

for i in range(num_of_iterations):
    dw = delta_w(w, X, y, a)
    w = w - dw
    h = H(X, w)
    j = J(h, y)
    w_iter.append(w)
    j_iter.append(j)

print(j_iter, w_iter)