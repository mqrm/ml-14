"""
Michael Roth <rothm@informatik.uni-freiburg.de>
ml-ex-02.1
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.DataFrame({"Age": [18, 58, 23, 45, 63, 36],
                    "BMI": [53.13, 49.06, 17.38, 21, 21.66, 28.59],
                    "Charges": [1163.43, 11381.33, 2775, 7222, 14349, 6548],
                  })

X = df.loc[:, ["Age", "BMI"]].to_numpy()
y = df.loc[:, ["Charges"]].to_numpy()
x1 = df.loc[:, ["Age"]].to_numpy()
x2 = df.loc[:, ["BMI"]].to_numpy()

def theta(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

def predict(x):
    return np.dot(x, theta(X, y))

# task of the exercise
x = np.array([40, 32.5], ndmin=2)
p = predict(x)

# checking the results
model = LinearRegression(fit_intercept=False, n_jobs=-1).fit(X, y)
pm = model.predict(x)
np.testing.assert_allclose(p, pm, rtol=1e-5)

weights = model.coef_
bias = model.intercept_

# printing
print("Prediction for ", x, "is ", p, ".")
print("Weightss:  ", weights)
print("Bias:          ", bias)
print("Equation: y = {:.2f} + {:.2f} * x1 + {:.2f} * x2".format(bias, weights[0,0], weights[0,1]))

# plotting
fig = plt.figure()
# plot axes
ax = plt.axes(projection="3d")
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')  # type: ignore
# plot training data (red points)
ax.scatter(x1, x2, y, c='red')
# create data for plane
xs, ys = np.meshgrid(np.linspace(x1.min(), x1.max(), 100), np.linspace(x1.min(), x1.max(), 100))
zs = model.predict(pd.DataFrame({'x1': xs.ravel(), 'x2': ys.ravel()}).to_numpy())
zs = zs.reshape(xs.shape)
# plot plane
ax.plot_surface(xs, ys, zs, color="b")  # type: ignore

plt.show()