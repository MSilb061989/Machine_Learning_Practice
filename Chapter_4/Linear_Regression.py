import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

###################################################Pseudoinverse########################################################

#The pseudoinverse is computed using Singular Value Decomposition (SVD). It can decompose the training set matrix X
#into the matrix multiplication of three matrices (U Sigma and V^T). The pseudoinverse X^+ is computed as
#   X^+ = Vsigma^+U^T, where for sigma^+ the algorithm takes sigma and sets to zero all values smaller than a tiny
#threshold value and replaces all the non-zero values with their inverse and finally transposes the resulting matrix.
#This approach is more efficient than computing the Normal Equation and handles edge cases nicely
#NOTE: the Normal Equation many not work if X^TX is not invertible (singular), such as if some features are redundant
#HOWEVER: the pseudoinverse is ALWAYS defined!

########################################################################################################################

#In essence, instead of computing X^Ty we compute X^+y, where X^+ is the pseudoinverse matrix computed from matrix
#decomposition of X into three separate matrices, U, sigma and V^T --> have to compute the pseudoinverse of sigma
#as described above and then X^+ is computed

#Implementation of Gradient Descent algorithm, which is supposidly faster than the Normal Equation or SVD for very
#large data sets
eta = 0.1 #Learning Rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) #Random Initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta) #Same result as when we used the Normal Equation
#In the book, it is shown that increasing or decreasing eta can reduce the accuracy of the learning algorithm

#The code below implements Stochastic Gradient Descent, which uses randomness in selecting the training set when
#computing the gradient. Although it is faster, it can get caught in local minima, depending on the training rate and
#may never reach the optimal solution. This code below uses a learning function to determine the learning rate
n_epochs = 50
t0, t1 = 5, 50 #Learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.rand(2, 1) #Random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

#Iterate by rounds of m iterations with each round called an epoch --> this code only iterated 50 times compared with
#Batch Gradient Descent that iterated for 1000 iterations and reached a fairly good solution

print(theta)

#To perform Linear Regression with Stochastic Gradient Descent using Scikit-Learn, we can use the SGDRegressor class,
#which defaults to optimizing the squared error cost function. The following code will run 50 epochs starting with a
#learning rate of 0.1 and using the default learning schedule. This DOES NOT use regularization with a penalty set to
#"None"
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.01)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)

#We can utilize linear models for nonlinear data by adding powers to each feature --> this is called Polynomial
#Regression and will be explored below

#First, we generate some nonlinear data
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
plt.show(plt.plot(X, y, "b.")) #Nonlinear dataset

#Use PolynomialFeatures from Scikit-Learn to transform the training data by adding the square (2nd degree polynomial)
#of each feature in the training set as new features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0]) #X_poly now contains the original feature plus the square of the feature. Now we can fit a
#LinearRegression model to the extended training data
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

#We can use learning curves to determine if a model is underfitting or overfitting the data by training the model
#several times on different sized subsets of the training set. The following code defines a function that plots the
#learning curves of a model given some training data
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

#What is going here in te results? When there just one or two instances in the training set the model can fit them
#perfectly, but as new instances are added to the training set  it becomes impossible for the model to fit the training
#data because of the noise in the data and because it is not linear --> therefore the error goes up until it reaches
#a plateau where adding new instances to the training set doesn't make the average error much better or worse

#Now considering the validation data, when the model is trained on just a few instances it is incapable of
#generalizing properly leading to a large validation error but learns as it is shown more training examples, reducing
#the error

#Now we look at the learning curves for a 10th degree polynomial
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])

plot_learning_curves(polynomial_regression, X, y)