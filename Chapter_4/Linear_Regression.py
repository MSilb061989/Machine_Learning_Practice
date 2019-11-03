import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

#Now we can perform Linear Regression using Scikit-Learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))
#LinearRegression class is based on the scipy.linalg.lstsq() function ("Least Squares"). We can call this function
#directly
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best_svd) #This function computes theta_hat = (X^+)y --. X^+ is the pseudoinverse of X (Moore-Penrose
#inverse). We can use np.linalg.pinv() to compute the pseudoinverse directly:
print(np.linalg.pinv(X_b).dot(y))

#The pseudoinverse is computed using Singular Value Decomposition (SVD). It can decompose the training set matrix X
#into the matrix multiplication of three matrices (U Sigma and V^T). The pseudoinverse X^+ is computed as
#   X^+ = Vsigma^+U^T, where for sigma^+ the algorithm takes sigma and sets to zero all values smaller than a tiny
#threshold value and replaces all the non-zero values with their inverse and finally transposes the resulting matrix.
#This approach is more efficient than computing the Normal Equation and handles edge cases nicely
#NOTE: the Normal Equation many not work if X^TX is not invertible (singular), such as if some features are redundant
#HOWEVER: the pseudoinverse is ALWAYS defined!
