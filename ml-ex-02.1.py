from sklearnex import patch_sklearn
patch_sklearn()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

df = pd.DataFrame({"Age": [18, 58, 23, 45, 63, 36],
                    "BMI": [53.13, 49.06, 17.38, 21, 21.66, 28.59],
                    "Charges": [1163.43, 11381.33, 2775, 7222, 14349, 6548],
                })

X = df.loc[:,["Age", "BMI"]].to_numpy()
y = df.loc[:,["Charges"]].to_numpy()
x1 = df.loc[:, ["Age"]].to_numpy()
x2 = df.loc[:, ["BMI"]].to_numpy()

def find_theta(X, y):
    m = X.shape[0]
    y = y.reshape(m,1)
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta

theta =find_theta(X,y)

def predict(X):
    preds = np.dot(X, theta)
    return preds

preds_m = predict(X)
model = LinearRegression(fit_intercept=False, n_jobs=-1).fit(X, y)
preds_a = model.predict(X)
x_task = [40, 32.5]
y_task = predict(x_task)
print("Prediction for {} is {}.".format(str(x_task), str(y_task)))

coefs = model.coef_
intercept = model.intercept_
print("Coefficients:  {}".format(coefs))
print("Bias:          {}".format(intercept))
print("Equation: y = {:.2f} + {:.2f} * x1 + {:.2f} * x2".format(intercept[0], coefs[0,0], coefs[0,1]))
#print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(0, coefs[0], coefs[1]))
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

