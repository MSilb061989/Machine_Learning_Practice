from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

#Datasets loaded by Scikit-Learn generally have a similar dictionary structure including:
# 1.) A DESCR key describing the dataset
# 2.) A data key containing an array with one row per instance and one column per feature
# 3.) A target key containing an array with the labels

#Recall: fetch is used to pull down a program or dataset
