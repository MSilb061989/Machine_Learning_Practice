import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn import datasets

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

#What is going here in the results? When there just one or two instances in the training set the model can fit them
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

#The learning curves for the 10th order polynomial are similar to the learning curves for a Linear Regression fit, with
#two important differences
#   1.) The error on the training data is much lower than that for the Linear Regression model
#   2.) There is a gap between the curves -> the model performs much better on the training data than on the
#       validation data which is a symptom of overfitting the data (a model that is overfitting). However, a much
#       larger training set would bring the curves closer together

#Note: we can cure an overfitting model so that the validation curve error approach the training curve error by feeding
#      the model more data

#Here we will perform Ridge Regression, a regularized version of Linear Regression, using a variant of the closed-form
#equation on page 132 using a matrix factorization technique by Andre-Louis Cholesky
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))

#Now applying Ridge Regression for Stochastic Gradient Descent
sgd_reg = SGDRegressor(penalty="l2") #L2 norm, NOT 12!
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

#The penalty hyperparameter sets the type of regularization to use -> in this case L2, the L2 norm, indicates that
#you want SGD to add a regularization term to the cost function which is half the square of the L2 norm of the weight
#vector --> this is simply Ridge Regression

#Note: Rule of thumb -> if you have to specify a model parameter manually then it is probably a model hyperparameter
#Some examples of model hyperparameters:
#   1.) Learning rate for training a neural network
#   2.) C and sigma hyperparameters for Support Vector Machines
#   3.) The "K" in K-Nearest Neighbors
#Note: A model parameter is a configuration variable that is internal to the model and whose value can be estimated
#from the data
#   1.) They are required by the model when making predictions
#   2.) The values define the skill of the model on your problem
#   3.) They are estimated or learned from the data
#   4.) They are often NOT manually set by the practitioner
#   5.) They are often saved as part of the learned model
#Model parameters are the part of the model that are learned from historical training data (in classical machine
#learning we think of the model as the hypothesis and the parameters as the tailoring of the hypothesis to a particular
#set of data)
#Model parameters are often estimated using an optimization algorithm
#Examples of model parameters are:
#   1.) The weights in an artifical neural network
#   2.) The support vectors in a support vector machine
#   3.) Coefficients in a linear or logistic regression

#NOTE: In Ridge Regression we're essentially adding a penalty so that the fit to the training data has a little bit of
#bias, and in return the variance is reduced when fit to the test data <- Eureka!
#To determine hyperparameter in Ridge Regression cost function, just try a bunch of values and use Cross_Validation
#(typically ten-fold Cross Validation) to determine which one minimizes variance

#Think of Regularization term as a penalty...

#Here is a small example using the Lasso class (we could also use an SGDRegressor(penalty="l1"))
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[1.5]]))

#Example of Elastic Net (l1_ratio is the mix ratio)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print(elastic_net.predict([[1.5]]))

#Recall: An epoch is an instance of training on Gradient Descent "m" number of times
#Below is an implementation of Early Stopping

#Prepare the data

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10) #Got this
#from Geron's Github

poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) #Continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

print(best_epoch, best_model)
#numpy.ravel() flattens a multi-dimensional array into a 1-D array

#Now, using Logistic Regression, we will build a classifier to detect the Iris-Virginica type flower based on the
#pedal width feature. This is from the "Iris Dataset"
iris = datasets.load_iris()
print(list(iris.keys()))
X = iris["data"][:, 3:] #petal width
print(iris["data"])
y = (iris["target"] == 2).astype(np.int) #1 if Iris-Virginica, else0

#Now, train a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X, y) #Trained a Logistic Regression model based on the pedal sizes (0.2 - 1.5?) from the dataset above

#Now, we look at the model's esimated probabilities for flowers with petal widths varying from 0 to 3 cm
X_new = np.linspace(0, 3, 1000).reshape(-1, 1) #Create an evenly spaced vector with values from 0 to 3
y_proba = log_reg.predict_proba(X_new) #Do the values in X_new pertain to a width that means an Iris-Virginica or not?
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
# save_fig("logistic_regression_plot") Need function from his Github for this to work...
plt.show()

#Decision Boundary at 1.6 cm where the probability that it is an Iris-Virginica and not an Iris-Viriginica are equally
#likely (equal to 50%)
print(log_reg.predict([[1.7], [1.5]])) #1.7 should result in "1" since the classifier is sure above 1.6 cm the flower
#is an Iris-Viriginica and should be a "0" for 1.5 since below 1.6 cm the classifier is confident the flower is not
#an Iris-Virginica
#Output is, indeed, [1 0]

#The code below, taken straight from the Github, uses Logistic Regression to find the probability of whether a flower
#is an Iris-Virginica based on two features -> the dashed line in the figure represents the 50% probability point is
#called the Decision Boundary. It is a parallel line because it is a linear boundary and each parallel line represents
#the points where the model outputs a specific probability, from 15% at the bottom left to 90% at top right. All flowers
#beyond the top-right line have an over 90% chance of being Iris-Virginica according to the model
X = iris["data"][:, (2,3)]

lop_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2,7, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10,4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmamp=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
# save_fig("logistic_regression_contour_plot") #Gotta download that function off of Github... :/
plt.show()

#We're going to use Softmax Regression to classify the iris flowers into all three classes. Scikit-Learn's
#LogisticRegression uses one-versus-all by default when you train it on more than two classes, but you can set the
#multi_class hyperparameter to "multinomial" to swtich it to Softmax Regression instead. The solver also has to be
#specified and it must be one that supports Softmax Regression ("lbfgs"). It also applies L2 regularization by default
#which can be controlled with hyperparameter C.

#Also, the cost function is a Cross-Entropy cost function, utilizing the minimization of transmitted information over
#bits (see the weather example on Youtube by the author)
X = iris["data"][:, (2, 3)] #petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
#The results show, for example, that an iris with 5 cm long and 2 cm wide petals is an Iris-Virginica, according to the
#model (class 2)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))

#The figure on page 145 shows a detailed view of the output with Decision Boundaries (any two Decision Boundaries are
#linear)

#Some notes in review:
#1.) Use the LinearRegression implementation in Scikit-Learn for training instead of the Normal Equation because it
#implements the Moore-Penrose inverse, which always works, as opposed to the Normal Equation that may not if X^TX is
#singular (not invertible)

#2.) Also, the Normal Equation is computationally complex (n + 1) x (n + 1) matrix --> O(n^3) but SVD is only O(n^2)

#3.) Other ways to train a Linear Regresion model when there are too many training instances to fit in memory
#    - Gradient Descent - use learning rate hyperparameter and compute derivative with respect to parameter vector
#    - Stochastic Gradient Descent
#    - Batch and Mini-Batch Gradient Descent

#We're in luck when minimizing MSE cost function for Linear Regression because it's a convex function --> implies no
#local minima (just one global minima) and is a continuous function
#Using Gradient Descent, we should scale the features. Otherwise we will run into long convergence times

#Batch Gradient Descent uses the whole training set on every step to compute the gradients --> it is slow
#Solution? Use Stochastic Gradient Descent --> picks random instance in the trainin set at every step and computes the
#gradient based only on that instance
#   - Only one instance needs to be in memory at every iteration
#   - A strategy with SGD is to gradually reduce the learning rate to allow the algorithm to settle at a global minimum
#   - This is pretty much what simulated annealing is...
#   - The learning rate at each iteration is determined from the learning schedule function

#Polynomial Regression can also be achieved using Linear Regression with Scikit-Learn by using PolynomialFeatures to
#to add a power to each feature in the training set (for the example on page 124 there's only one feature)
#   - PolynomialFeatures transforms an array containing "n" features into an array containing (n+d)!/(d!n!) features

#How can we tell if a model if overfitting or underfittig data?
#   - Used cross-validation previously to get an estimate of a model's generalization performance
#   - If a model performs well on training data but generalizes poorly - overfitting according to cross-validation
#   - If it performs poorly on both - underfitting according to cross-validation
#Another way? Look at learning curves, which are plots of the model's performance on the training set and the validation
#set as a function of the training set size
#   - To generate, train the model several times on different sized subsets of the training set

#A model's generalization error can be expressed as the sum of three different errors:
#   - Bias: generalization error due to wrong assumptions (assume data is linear when it isn't) -> likely to underfit
#           data
#   - Variance: due to model's excessive sensitivity to small variations in the training data
#   - Irreducible Error: due to noisiness of the data itself -> only way to reduce is to clean up the data and outliers

#A good way to reduce overfitting is to regularize a model (constrain it) -> remove degrees of freedom such as in a
#polynomial model
#For linear model, regularization can be achieved by constraining the weights of the model
#   - Ridge Regression
#   - Lasso Regression
#   - Elastic Net --> All three of these constrain the weights!

#Ridge Regression is a regularized version of Linear Regression with a regularization term added that is the sum of the
#parameter weights squared multiplied by a hyperparameter -> forces learning algorithm to fit data and keep weights
#small
#   - The regularization term is only added during training
#   - Hyperparameter controls how much regularization is applied
#   - The vector of feature weights is represented as the L2 norm of the weight vector
#   - Don't forget to scale the data!

#Note: Ridge Regression can be performed from either the Normal Equation (page 132) or from Stochastic Gradient Descent
#by adding a penalty equal to the L2 norm

#Lasso Regression is another regularized version of Linear Regression that adds a regularization term to the cost
#function, but instead of the L2 norm adds the L1 norm of the weight vector
#   - Can eliminate weights that are not important (remove degrees in polynomial) -> feature selection
#   - Lasso Regression can be implemented with Lasso class or by implementing SGDRegressor with an L1 penalty

#The plot on page 134 highlights how weights are eliminated are evaluated by adding a penalty (L1 or L2)

#Elastic Net is a middle ground between Ridge and Lasso Regression -> the regularization term is a mix of both Ridge
#and Lasso regularization terms, which are controlled with the mix ratio "r"
#   - When it is suspected not all of the features are useful -> use Lasso or Elastic Net
#   - Otherwise -> use Ridge
#   - In general -> Elastic Ridge is preferred over Lasso Regression because of its erratic behavior when the number of
#                   features exceeds the number of training instances or when several features are strongly correlated