# %%
import numpy as np 


# %%
X = 2*np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# %%
import matplotlib.pyplot as plt
plt.plot(X, y, 'b.')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
plt.axis([0, 2, 0, 15])
plt.show()

# %%
X_b = np.c_[np.ones((100, 1)), X]

# %%
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# %%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pre = X_new_b.dot(theta)
y_pre

# %%
plt.plot(X_new, y_pre, 'r--')
plt.plot(X, y, 'b.')
plt.show()

# %%
from sklearn.linear_model import LinearRegression

# %%
linear_reg = LinearRegression()
linear_reg.fit(X, y)
print(linear_reg.coef_)
print(linear_reg.intercept_)

# %%
theta = np.random.randn(2, 1)
m = 100
iterations = 100
rate = 0.01
for ite in range(iterations):
    gradit = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= gradit * rate

# %%
X_new_b.dot(theta)

# %%
theta_path_bgd = []
def plot_gradient_descent(theta, eta, theta_path = None):
    m = len(X_b)
    plt.plot(X, y, 'b.')
    n_itera = 1000
    for i in range(n_itera):
        y_pre = X_new_b.dot(theta)
        plt.plot(X_new, y_pre, 'b-')
        gradit = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= gradit * rate
        if theta_path:
            theta_path.append(theta)
    plt.xlabel("X_1")
    plt.axis([0,2, 0,15])
    plt.title('eta = {}'.format(eta))


# %%
theta = np.random.randn(2,1)
plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta = 0.02)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1)
plt.subplot(133)
plot_gradient_descent(theta, eta = 0.4)
plt.show()

# %%



