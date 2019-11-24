import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

#This code loads the Iris dataset, scales the features and then trains a linear SVM (Using the LinearSVM class with
#C = 1 and the hinge loss function) to detect the Iris-Virginica flowers

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] #petal length, petal width
y = (iris["target"] == 2).astype(np.float64) #Iris-Virginica

#Support Vector Machine Classifier Pipeline
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svm", LinearSVC(C=1, loss="hinge")),
])

#Train the SVM model
svm_clf.fit(X, y)

#Use the model to make predictions
print(svm_clf.predict([[5.5, 1.7]]))

#Unlike Logistic Regression, SVM classifiers do not output probabilities for each class

#As an alternative, the SVC class could be used. We can use SVC(kernel="linear", C=1), but is much slower (especially
#large training sets) and is not recommended. Another option is to use SGDClassifier class with the
#SGDClassifier(loss="hinge", alpha=1/(m*C)) --> This applies regular Stochastic Gradient Descent to train a linear
#SVM classifier. This is useful for huge datasets that don't fit in memory (out-of-core training) or to handle online
#classification tasks

#To implement an approach for handling nonlinear datasets using SVM, we can use Scikit-Learn by creating a Pipeline
#containing a PolynomialFeatures transformer (see "Polynomial Regression" on page 124), followed by a StandardScaler
#and a LinearSVC

#We will test this on the moons dataset: this is a toy dataset for binary classification in which the data points are
#shaped as two interleaving half circles

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

#Train the model
polynomial_svm_clf.fit(X, y)

#Here we're implementing the kernel trick, which makes it possible to achieve the same result as f we added many
#polynomial features without actually having to add them --> no combinatorial explosion of features since we're not
#adding any. This trick is supported by the SVM class

poly_kernal_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

#Train the model
poly_kernal_clf.fit(X, y)

#This code section above train an SVM classifier using a 3rd degree polynomial kernel (there is also an SVM classifier
#using a 10th degree polynomial kernel on page 153).
#   If the model is overfitting: reduce the polynomial degree
#   If the model is underfitting: increase the polynomial degree
#   Hyperparameter coef0 controls how much the model is influenced by high-degree polynomials versus low-degree
#   polynomials

#A common approach to find the right hyperparameter values is to use grid search --> faster to do a coarse grid search
#first and then a finer grid search aroudn the best values found

#The code below implements the kernel trick again for adding polynomial features using the similarity features method
#without actually having to add them. To recall, the kernel trick makes it so that we can obtain the same result as
#if we had added many features without actually having to add them. Below, the Gaussian RBF kernel achieves this as if
#we had used the similarity features function with a Gaussian Radial Basis Function
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)) #gamma and C are hyperparameters
])

#Train the model
rbf_kernel_svm_clf.fit(X, y)

#Some notes from page 155:
#   Increasing gamma makes the bell-shape barrower and as a result each instance's range of influences is smaller -->
#   the decision boundary ends up being more irregular, wiggling around individual instances
#   Small gamma values makes the bell-shaped curve wider, so instances have a larger range of influence and the
#   decision boundary ends up smoother
#   Gamma winds up acting like a regularization hyperparameter --> if the model is overfitting then reduce it and
#   increase if it is underfitting (similar to the C hyperparameter)