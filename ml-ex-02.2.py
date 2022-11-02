from sklearnex import patch_sklearn
patch_sklearn()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps
"""
# Define and generate the samples
nb_of_samples_per_class = 20  # The number of sample in each class
red_mean = (-1., 0.)  # The mean of the red class
blue_mean = (1., 0.)  # The mean of the blue class
# Generate samples from both classes
x_red = np.random.randn(nb_of_samples_per_class, 2) + red_mean
x_blue = np.random.randn(nb_of_samples_per_class, 2)  + blue_mean

# Merge samples in set of input variables x, and corresponding 
# set of output variables t
x = np.vstack((x_red, x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class,1)), 
               np.ones((nb_of_samples_per_class,1))))
#print(X, type(X))
#
"""

df = pd.DataFrame({"x1": [2, 3, -4, -2],
                   "x2": [4, 3, -2, -6],
                    "y": [1, 1, 0, 0],
                })

x = df.loc[:,["x1", "x2"]].to_numpy()
t = df.loc[:,["y"]].to_numpy()
x1 = df.loc[:, ["x1"]].to_numpy()
x2 = df.loc[:, ["x2"]].to_numpy()
#w = np.asmatrix([0, 0])
w = np.asmatrix([-4, -2])
a = 0.05
num_of_iterations = 10

# Define the sigmoid function g(z)
def g(z):
    return 1. / (1 + np.exp(-z))

# Define the probability function p(x, w)
def p(x, w):
    return g(x.dot(w.T))

y = p(x, w)

# Define the prediction function returning 0 or 1 depending on the predicted class
def predict(x, w):
    return np.around(p(x, w))

# Define the loss function
def J(y, t):
    #return - np.mean(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
b = J(y, t)
print(b)
print(type(b))

# Define the gradient function
def grad(w,x,t):
    return (p(x, w) -t).T * x

# Define the update function of w
def delta_w(w_k, x, t, a):
    return a * grad(w_k, x, t)

w_iter = [w] # List to store the weight values over the iterations

for i in range(num_of_iterations):
    dw = delta_w(w, x, t, a)
    w = w - dw
    w_iter.append(w)

# Plot the loss in function of the weights
# Define a vector of weights for which we want to plot the loss
nb_of_ws = 25 # compute the loss nb_of_ws times in each dimension
wsa = np.linspace(-5, 2, num=nb_of_ws) # weight a
wsb = np.linspace(-5, 2, num=nb_of_ws) # weight b
ws_x, ws_y = np.meshgrid(wsa, wsb) # generate grid
loss_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize loss matrix
# Fill the loss matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        loss_ws[i,j] = J(
            p(x, np.asmatrix([ws_x[i,j], ws_y[i,j]])) , t)
# Plot the loss function surface
plt.figure(figsize=(6, 4))
plt.contourf(ws_x, ws_y, loss_ws, 20, cmap=cm.viridis)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$', fontsize=12)
plt.xlabel('$w_1$', fontsize=12)
plt.ylabel('$w_2$', fontsize=12)
plt.title('Loss function surface')
plt.grid()
#plt.show()
#
# Plot the first weight updates on the error surface
# Plot the error surface
plt.figure(figsize=(6, 4))
plt.contourf(ws_x, ws_y, loss_ws, 20, alpha=0.75, cmap=cm.viridis)
cbar = plt.colorbar()
cbar.ax.set_ylabel('loss')

# Plot the updates
for i in range(1, 4): 
    w1 = w_iter[i-1]
    w2 = w_iter[i]
    # Plot the weight-loss values that represents the update
    plt.plot(w1[0,0], w1[0,1], marker='o', color='#3f0000')  # Plot the weight-loss value
    plt.plot([w1[0,0], w2[0,0]], [w1[0,1], w2[0,1]], linestyle='-', color='#3f0000')
    plt.text(w1[0,0]-0.2, w1[0,1]+0.4, f'$w({i-1})$', color='#3f0000')
# Plot the last weight
w1 = w_iter[3]  
plt.plot(w1[0,0], w1[0,1], marker='o', color='#3f0000')
plt.text(w1[0,0]-0.2, w1[0,1]+0.4, f'$w({i})$', color='#3f0000') 
# Show figure
plt.xlabel('$w_1$', fontsize=12)
plt.ylabel('$w_2$', fontsize=12)
plt.title('Gradient descent updates on loss surface')
plt.show()
#
""""
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.grid
ax.scatter(x1, x2, y, c="red", marker="o")
x1_surf, x2_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 100), np.linspace(x1.min(), x1.max(), 100))
y_surf = model.predict(pd.DataFrame({"x1": x1_surf.ravel(), "x2": x2_surf.ravel()}).to_numpy())
ax.plot_surface(x1_surf, x2_surf, y_surf.reshape(x1_surf.shape), color="b")
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
plt.show()
"""
