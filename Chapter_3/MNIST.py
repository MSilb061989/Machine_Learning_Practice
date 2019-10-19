from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier

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
y_test_5 = (y_test == 5)

#Time to pick a classifier and train it. A good place to start is with a Stochastic Gradient Descent (SGD) classifier
#using Scikit-Learn's SGDCliassifier class -> has the advantage of being capable of handling very large datasets well
#It also handles instances one at a time and is great for online learning
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5) #SGDClassifier relies on randomness during training ("stochastic") -> for reproducable
#results set random_state parameter