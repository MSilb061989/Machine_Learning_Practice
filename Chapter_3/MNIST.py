from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone, BaseEstimator

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

#We're exploring classification in this chapter

#Datasets loaded by Scikit-Learn generally have a similar dictionary structure including:
# 1.) A DESCR key describing the dataset
# 2.) A data key containing an array with one row per instance and one column per feature
# 3.) A target key containing an array with the labels

#Recall: fetch is used to pull down a program or dataset
X, y  = mnist["data"], mnist["target"]
print(X.shape)
print("\n")
print(y.shape)

#There are 70,000 images and each image has 784 features -> this is because each image is 28x28 pixels and each feature
#represents one pixel's intensity, from 0 (white) to 255 (black)

#Look at one digit from dataset by grabbing an instance's feature vector, reshape it to 28x28 array and display is using
#Matplotlib's imshow() function

#One image is an instance (70,000 instances) with 784 features across for 28x28 pixels -> need this as a vector reshaped
#with a 28x28 array
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

print(y[0])
#Cast labels as integers from strings
y = y.astype(np.uint8)

#REMEMBER! You should ALWAYS create a test set and set it aside before inspecting the data closely
#Luckily, the MNIST dataset is already split into a training set (the first 60,000 images) and a test set
#(the last 10,000 images)

#Do note that this is a complicated classification task

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Training set is already shuffled, which guarantees cross-validation folds will be similar (don't want one fold to be
#missing digits)
#Note: don't want to shuffle some types of data, such as time series data (stock market prices or weather conditions)

#Simplify the problem and only identify the digit "5" using a binary classifier, capable of distinguishing between
#just two classes -> 5 and not 5

#Creating target vectors for classification task:
y_train_5 = (y_train == 5) #True for all 5's. False for all other digits
y_test_5 = (y_test == 5) #Set test set equal to all instances in test set that are the image of 5

#Time to pick a classifier and train it. A good place to start is with a Stochastic Gradient Descent (SGD) classifier
#using Scikit-Learn's SGDCliassifier class -> has the advantage of being capable of handling very large datasets well
#It also handles instances one at a time and is great for online learning
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5) #SGDClassifier relies on randomness during training ("stochastic") -> for reproducable
#results set random_state parameter_
#Train data set against all instances of 5

#Now we can use it to detect images of the number 5:
print(sgd_clf.predict([some_digit])) #some_digit was the first instance and is indeed the number 5 -> now we want to see
#if the classifier detects that this is true or not (is it true that this a 5 or not?)

#Evaluating a classifier can be trickier than evaluating a regressor -> there are many performance measures available

#Good way to evaluate accuracy is to use cross-validation

#Somestimes you need to implement your own cross-validation process because the one offered by Scikit-Learn's
#cross_val_score() doesn't provide as much control as required -> we can make a cross_validation function ourselves
#Creating home-grown cross validation function
skfolds = StratifiedKFold(n_splits=3, random_state=42) #How many different folds (3)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf) #Clone creates a deep copy of the model in the estimator without actually copying
    #attached data -> yields new estimator with the same parameters that has not been fit to any data
    #A deep copy copies all fields and makes copies of dynamically allocated memory pointed to by the fields. A deep
    #copy occurs when an object is copied along with the objects to which it refers
    #Dyanmic memory allocation refers to manually memory allocation by programmer -> managing system memory at runtime

    #See this link on dynamic memory allocation: https://www.cs.fsu.edu/~myers/c++/notes/dma.html
    #Basically, memory is allocated "on the fly" during runtime where the exact amount of space or memory or number
    #of items does not need to be known by the compiler in advance -> pointers are crucial

    ###########################Description##############################################################################
    #The StratifiedKFold class performs stratififed sampling (Recall: stratified sampling) to produce folds that contain
    #a representative ratio of each class. At each iteration the code creates a clone of the classifier, trains that
    #clone on the training folds and makes predictions on the test fold. Then it counts the number of correct
    #predictions and outputs the ratio of correct predictions
    X_train_folds = X_train[train_index] #Create training set for cross-validation
    y_train_folds = y_train_5[train_index] #Create training set for cross-validation
    X_test_fold = X_train[test_index] #Create test set for cross-validation
    y_test_fold = y_train_5[test_index] #Create test set for cross-validation

    clone_clf.fit(X_train_folds, y_train_folds) #Build model based on training set (do this three times since there
    #are three folds)
    y_pred = clone_clf.predict(X_test_fold) #Attempt to see how well the model performs on the test set (again, this is
    #done three times for three folds)
    n_correct = sum(y_pred == y_test_fold) #How many are correct by seeing if True = True, False = False etc.)
    print((n_correct / len(y_pred))) #prints 0.9502, 0.96565 and 0.96495
    ####################################################################################################################

print("Here!")

#Now we will use the cross_val_score() function to evaluate the SGDClassifier model using k-folds cross-validation
#RECALL: k-folds cross validation means splitting the training set into k-folds (in this case, three), then making
#predictions and evaluating them on each fold using a model trained on the remaining folds
scoreCV = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(scoreCV)

#This function gives 95% accuracy (ratio of correct predictions) on all cross-validation folds! However, this is not
#entirely correct -> need to ensure that we are, in fact, predicting at 95% accurate. We can do this by constructing
#a dumb classifier that classifies every single image in the "not a five" class
class Never5Classifier(BaseEstimator): #BaseEstimator is the base class for all estimators in Scikit-Learn
#This base class enables to set and get parameters of the estimators. More specifically, BaseEstimator provides an
#implementation of the get_params and set_params methods -> Why is this needed? It can make a model applicable to
#GridSearchCV, which ensures it behaves well when placed in a pipeline
#GridSearchCV is one of the generic approaches to sampling search parameter candidates, such as hyper-parameters which
#are not directly learnt within estimators
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

#"Now we try and guess the "Never5Classifier" model accuracy
never_5_clf = Never5Classifier()
scoreCV = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(scoreCV) #The results, according to book, is 0.909, 0.90715 and 0.9128 -> Over 90% accurate
#What this tells us is that it has over 90% accuracy -> this is simply because only about 10% of the images are 5's.
#Therefore, if you always guess that an image is NOT a 5 you would be right 90% of the time

#IMPORTANT: This tells us that accuracy is not the preferred performance metric for classifiers, especially when
#dealing with skewed datasets (skewed datasets are those where some classes appear much more frequency than others)