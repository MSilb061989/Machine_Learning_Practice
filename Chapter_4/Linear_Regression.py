import numpy as np
import matplotlib.pyplot as plt

#Linear Regression Example
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#Compute parameter vector that minimizes the cost function using the normal equation. Use the inv() function from
#NumPy's Linear Algebra module (np.linalg) to compute the inverse matrix and dot() method for matrix multiplication
X_b = np.c_[np.ones((100,1)), X] #add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #Parameter vector that minimizes cost function
#Actual function is y = 4 + 3x1 + Gaussian Noise
print(theta_best)
#Noise made it impossible to recover the exact parameters of the original function

#Now we can make predictions using theta_best
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] #add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print(y_predict)

#Plot this model's predictions
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.show()
#NOTE: np.c_ function concatenates along the column vector. So if we have a column vector x and a column vector y, using
#np.c(x, y) will return a matrix with x as the first column and y as the second column