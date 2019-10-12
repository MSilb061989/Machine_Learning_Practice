import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32 #For compressing data...
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# This is not the best method to generate test data...
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) #Randomly shuffles data around
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

if __name__ == "__main__":

    fetch_housing_data()

    #"housing" is a Pandas data frame
    housing = load_housing_data()
    print(housing.head())
    print(housing.info())
    # print(housing["longitude"].value_counts())
    print(housing.describe())

    housing.hist(bins=50, figsize=(20,15))
    #plt.show()

    #Split dataframe into random training and test sets
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    print(train_set)
    print(test_set)

    #Bin data into discrete intervals
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    plt.show(housing["income_cat"].hist()) #Now I can do Stratified Sampling (See Book)

    #Prepare data for stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]): #This Function Performs Stratified Sampling Based on the Income Category (Recall Male and Female Example...)
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index] #.loc can access a group of rows or columns by label(s)

    print(strat_test_set["income_cat"].value_counts()/len(strat_test_set)) #Compare to Histogram to See if Ratio of Test Set Data Matches the Height of the Bars

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame({
        "Overall": housing["income_cat"].value_counts()/len(housing),
        "Stratified": strat_test_set["income_cat"].value_counts()/len(strat_test_set),
        "Random": test_set["income_cat"].value_counts()/len(test_set)
    }).sort_index()
    compare_props["Rand. %Error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %Error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

    print(compare_props)

    for set_ in(strat_train_set, strat_test_set): #Removing the "Income Category (income_cat) Attribute...
        set_.drop("income_cat", axis=1, inplace=True)

    #Create a Copy of the Training Set so the Original is not Harmed
    housing = strat_train_set.copy()

    #Visualize the Data in a Scatterplot
    plt.show(housing.plot(kind='scatter', x="longitude", y="latitude", alpha=0.1)) #alpha Helps Highlight High Density Areas

    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                 s=housing["population"]/100, label="population", figsize=(10,7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    #plt.show()

    #Look at how each Attribute Correlates with Median House Value (Median House Value is the "Target" Attribute)
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))
    #plt.show()

    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
    #plt.show()

    #Try Different Combinations of Attributes Before Feeding Data to Machine Learning Algorithm

    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    #See How Many Attributes There Are
    # print(housing.info())
    # print(housing.describe())

    #Look at Correlation Matrix Again with Median House Value as the Target Value
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    #The Result: "bedrooms_per_room" is more correlated than "total_room" or "total_bedrooms" with Median Housing Value

    #Next, the data will be prepared for machine learning algorithms
    #First, we will revert to a clean training set. The predictors and labels will be separated since we
    #don't want to apply the same transformation to the predictors and the target values

    #Creates a Copy of the Data "strat_train_set"
    #The predictors and labels are separated since we don't want to necessarily apply the same transformations to the
    #predictors and target values
    housing = strat_train_set.drop("median_house_value", axis=1)
    print(housing)
    housing_labels = strat_train_set["median_house_value"].copy()
    print(housing.info())