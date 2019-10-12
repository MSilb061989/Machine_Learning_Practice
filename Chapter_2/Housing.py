import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32 #For compressing data...
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

####################################################################################################
#This block of code is because Scikit-Learn 0.20 replaced sklearn.preprocessing.Imputer class with
#sklearn.impute.SimpleImputer class

# try:
#     from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
# except ImportError:
#     from sklearn.preprocessing import Imputer as SimpleImputer
####################################################################################################

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#Custom transformer to add attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): #No *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self #Nothing else to do
    def transform(self, X, y=None):
        room_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, room_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, room_per_household, population_per_household]

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
    housing = strat_train_set.drop("median_house_value", axis=1) #Drop "median_house_value" from training set and creates a copy of the training set
    ###NOTE: I believe "median_house_value" was dropped because we are separating the predictors and labels...###
    print(housing)
    housing_labels = strat_train_set["median_house_value"].copy()
    print(housing.info())

    #Sample incomplete rows
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    print(sample_incomplete_rows)

    print(housing_labels) #Print Training Set (This is only the "median_house_value" attribute)

    #Recall: At this point, the "total-bedrooms" attribute is missing some values
    #There are three options to take care of the attribute's missing values:
    #1.) Get rid of the corresponding districts (rows)
    #2.) Get rid of the whole attribute
    #3.) Set the values to some value (zero, mean, median etc.)
    housing.dropna(subset=["total_bedrooms"])   #Option #1.)
    housing.drop("total_bedrooms", axis=1)      #Option #2.)
    median = housing["total_bedrooms"].median() #Option #3.)
    housing["total_bedrooms"].fillna(median, inplace=True) #Whatever this median value is, save it -> We will need it
    #later to replace missing values in the test set


    #Use Scikit-Learn modules to take care of missing values: SimpleInputer

    #First, create instance of SimpleInputer and specify that you want to replace each attribute's missing values with
    #the median of the attribute
    imputer = SimpleImputer(strategy="median")


    #Because the median can only be computed on numerical attributes, we need to copy the data without text attribute
    #"ocean_proximity"
    housing_num = housing.drop("ocean_proximity", axis=1)

    #Now fit the Imputer instance to the training date using the fit() method
    imputer.fit(housing_num) #<-- Computed the median of each attribute and stored the results in its statistics_
    #instance variable

    #Since only "total_bedrooms" attribute was missing date, it only computed median values for that attribute, but
    #once the system goes live there can be more missing attributes, so it's better to apply the Imputer to all of the
    #numerical attributes
    print(imputer.statistics_)
    print(housing_num.median().values) #<-- This is just checking to ensure manually computing the median of the
    #attribute is the same as using the imputter.fit

    #Replace missing values in training set by learned medians (Transform the training set)
    #Note: 433 is the median of the "total_bedrooms" attribute
    X = imputer.transform(housing_num)
    print(imputer.strategy)

    #The result is a plain Numpy array containing the transformed features. Now we can put it back into a Pandas
    #DataFrame using the following:
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index) #housing_num does not include
    #"ocean_proximity" attribute

    print("\nThis is the housing.index values:")
    print(housing.index)

    #Since we already stored the incomplete rows in
    #"sample_incomplete_rows", we're just checking to ensure those values were replaced with the median

    #Recall: the ".loc" locates values in a Pandas DataFrame  <-- see documentation
    print(housing_tr.loc[sample_incomplete_rows.index.values])

    #NOTE: For pushing "bare" repo to Github: $ git remote add origin https://github.com/MSilberberg0619/Machine_Learning_Practice.git

    #"ocean_proximity" was left out because it's a text attribute and so the median can't be computed
    #To fix, convert these categories from text to numbers using Scikit-Learn's OrdinalEncoder class
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(10))

    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded)

    #Can use one-hot encoding to map attributes to categories so the values of the attributes that are more similar
    #will have similar encoded values
    #We don't want the model to assume some natural ordering to the data --> could result in poor performance or
    #unexpected results
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)
    housing_cat_1hot.toarray()
    print(housing_cat_1hot)

    #List of categories using the encoder's categories instance variable
    print(cat_encoder.categories_)

    #May need to write custom transformations for tasks such as custom cleanup operations
    #This transformer class adds the combined attributes discussed earlier
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6 #Line 1.1

    # get the right column indices: safer than hard-coding indices 3, 4, 5, 6
    rooms_ix, bedrooms_ix, population_ix, household_ix = [           #Line 1.2
        list(housing.columns).index(col)
        for col in ("total_rooms", "total_bedrooms", "population", "households")]

    #NOTE: Line 1.1 and Line 1.2 provide the same result, but Line 1.2 is safer, as noted


