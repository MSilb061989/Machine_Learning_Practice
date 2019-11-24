import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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