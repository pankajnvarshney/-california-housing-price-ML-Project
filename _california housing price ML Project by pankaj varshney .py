#!/usr/bin/env python
# coding: utf-8

# # Objective:- In this project we chose the California Housing Prices dataset from the StatLib repository2 
#  The dataset was based on data from the 1990 California census.

# # Import Libraries 
# by importing libraries it will help to run codes without any error 

# <a id='import'></a>

# In[1]:


# Common imports
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# random seed to make output stable across runs
np.random.seed(42)


# <a id='data'></a>

# # **California housing Dataset**

# ### Get the data: • In this step we work on the following:
# • Download the Data
# • Load the data
# • Take a Quick Look at the Data Structure
# • Using DataFrame info() method
# • Using DataFrame value_counts() method
# • Using DataFrame describe() method
# • Plot a Histogram
# • Create a Test Set

# In[2]:


import os
import tarfile
from six.moves import urllib

root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
path = os.path.join("datasets", "housing")
source = root + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=source, path=path):
    if not os.path.isdir(path):
        os.makedirs(path)
    tgz_path = os.path.join(path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=path)
    housing_tgz.close()


# In[3]:


fetch_housing_data()


# In[4]:


import pandas as pd

def load_data(housing_path=path):
    csv = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv)


# <a id='bigpop'></a>

# # Load the data and Take a  big picture of the data

# In[5]:


housing = load_data()
housing.head()


# Each row  represents one district. The dataset has 10 attributes: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value and ocean_proximity.

# In[6]:


housing.info()


# Now we can see a brief description of the data using the "info()" method above.

# In[7]:


housing["ocean_proximity"].value_counts()


# Now I will use the "describe()" method to show a summary of the numerical attributes:

# In[10]:


housing.describe() #summary of the numerical attributes


# The 25%, 50%, 75% are the percentiles. For an example 75% of the districts have housing_median_age of lower than 37 while 50% are lower than 29 and 25% are lower than 18. These are often called the 25th percentile (or 1st quartile), the median and the 75th percentile (or 3rd quartile).

# In[11]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()


# 1. First, the median income attribute does not look like it is expressed in US dollars (USD). 
# After checking with the team that collected the data, you are told that the data has been 
# scaled and capped at 15 (actually 15.0001) for higher median incomes, and at 0.5 (actually 
# 0.4999) for lower median incomes. 
# • Working with preprocessed attributes is common in Machine Learning, and it is not 
# necessarily a problem, but you should try to understand how the data was computed.
# 2. The housing median age and the median house value were also capped. The latter may be a 
# serious problem since it is your target attribute (your labels). Your Machine Learning 
# algorithms may learn that prices never go beyond that limit. You need to check with your 
# client team (the team that will use your system’s output) to see if this is a problem or not. If 
# they tell you that they need precise predictions even beyond $500,000.
# 
# 

# <a id='split'></a>

# # Train/Test Split
# 
# • You have only taken a quick glance at the data, and surely you should learn a whole lot 
# more about it before you decide what algorithms to use, right? 
# 
# • This is true, but your brain is an amazing pattern detection system, which means that it is 
# highly prone to overfitting: if you look at the test set, you may stumble upon some 
# seemingly interesting pattern in the test data that leads you to select a particular kind of 
# Machine Learning model. 
# 
# • When you estimate the generalization error using the test set, your estimate will be too 
# optimistic and you will launch a system that will not perform as well as expected. This is 
# called data snooping bias.
# 
# • Creating a test set is theoretically quite simple: just pick some instances randomly, typically 
# 20% of the dataset, and set them aside:

# In[46]:


import numpy as np
def splitTrainTest(data, testRatio):
 shuffledIndices = np.random.permutation(len(data))
 testSetSize = int(len(data) * testRatio)
 testIndices = shuffledIndices[:testSetSize]
 trainIndices = shuffledIndices[testSetSize:]
 return data.iloc[trainIndices], data.iloc[testIndices]


# In[47]:


#We will use the straight forward "train_test_split()" method from sklearn to split our data into train and test subsets.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "Train Instances +", len(test_set), "Test Instances")


# The following code creates an income category attribute by dividing the median income by 1.5 (to limit the number of income categories) and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5.

# In[12]:


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[13]:


plt.hist(housing["income_cat"])
fig = plt.gcf()


# Most median income values are clustered around 2–5 
# (tens of thousands of dollars), but some median 
# incomes go far beyond 6. 
# 
# It is important to have a sufficient number of instances 
# in your dataset for each stratum, or else the estimate of 
# the stratum’s importance may be biased. 
# 
# This means that you should not have too many strata, 
# and each stratum should be large enough.
# 
# ### Now you are ready to do stratified sampling based on the income category. For this you can use 
# Scikit-Learn’s StratifiedShuffleSplit class:
# 

# In[50]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for trainIndex, testIndex in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[trainIndex]
    strat_test_set = housing.loc[testIndex]


# Let’s see if this worked as expected. You can start by looking at the income category proportions 
# in the full housing dataset:

# In[51]:


housing["income_cat"].value_counts() / len(housing)


# In[52]:


strat_train_set.info()


# In[53]:


strat_test_set.info()


# In[54]:


# Now we want to remove the income_categoreis attribute because we don't need it anymore.
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# <a id='explo'></a>

# # DISCOVER AND VISUALIZE THE DATA TO GAIN INSIGHTS
# • So far you have only taken a quick glance at the data to get a general understanding of 
# the kind of data you are manipulating.
# 
# • Now the goal is to go a little bit more in depth.
# 
# • First, make sure you have put the test set aside and you are only exploring the training 
# set. Also, if the training set is very large, you may want to sample an exploration set, to 
# make manipulations easy and fast. 
# 
# • In our case, the set is quite small so you can just work directly on the full set. Let’s create 
# a copy so you can play with it without harming the training set:
# 

# In[55]:


housing = strat_train_set.copy()


# Let us work on the following points:
# 1. Visualizing Geographical Data ,2. Looking for Correlations ,3. Experimenting with Attribute Commbinations

# In[56]:


housing.plot(kind="scatter", x="longitude", y="latitude") # since there is geographical information we make scatter plot to visualize the data 


# In[57]:


# Setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points.
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# Here we can clearly see the high density areas, namely the Bay Area and around Los Angeles and San Diego, plus a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno.

# # Visualisation

# In[58]:


import seaborn as sns


# In[59]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,\
             s=housing["population"]/100, label="population",c="median_house_value", \
             cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# On the scatterplot we can see, that the housing prices are related to the location (close to the ocean) and to the population density, but we know that the housing prices of coastal districts are not that high in Northern California, so we can't make that rule as simple as that.
# 
# It will probably be useful to use a clustering algorithm to detect the main clusters and add new features that measure the proximity to the cluster centers.
# 

# In[ ]:





# <a id='corre'></a>

# #  Correlation
# 
# Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called 
# Pearson’s r) between every pair of attributes using the corr() method:

# In[14]:


corr_matrix = housing.loc[ : , housing.columns!='ocean_proximity'].corr()
#the full correlation matrix
corr_matrix


# In[15]:


#Correlation for a single variable, let us use median_income 
corr_matrix['median_income'].sort_values()


# The coefficient of the correlation ranges from 1 to -1. The closer it is to 1 the more correlated it is and vice versa.  Correlations that are close to 0, means that there is no correlation, neither negative or positive. You can see that the median_income is correlated the most with the median house value. Because of that, we will generate a more detailed scatterplot below:

# # Scatter Matrix

# In[16]:


from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# These graphs show the distribution of the feature variables along the main diagonal and the scatter plots show the relationship between pairs of vairables.
# We can even make these scatter plots separately as follows:

# In[17]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# The scatterplot reveals, that the correlation is indeed very strong because we can clearly see an upward trend and the points are not to dispersed. We can also clearly see the price-cap, we talked about earlier, at 500 000 as a horizontal line.

# # Attribute Combination

# Before we now actually prepare the data to fed it into the model, we should think about combinating a few attributes.

# In[18]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[65]:


corr_matrix = housing.loc[ : , housing.columns!='ocean_proximity'].corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# Now we see, that the bedrooms_per_room attribute is more correlated with the median house value than the total number of rooms or bedrooms. Houses with a lower bedroom/room ratio tend to be more expensive. The rooms_per_household attribute is also better than the total number of rooms in a district. Obviously the larger the house, the higher the price.
# 

# # Data Preparation
# 
# Now it is time to prepare the data so that our model can process it. We will write functions that do this instead of doing it manually. 

# In[66]:


housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# We noticed earlier that the total_bedrooms attribute has some missing values. Most Machine Learning algorithms can't work with datasets that have missing values. 
# 
# There are 3 ways to solve this problem:
# 
# 1. You could remove the whole attribute ,2. You could get rid of the districts that contain missing values.
# 3. You could replace them with zeros, the median or the mean
# 
# We chose option 3 and will compute the median on the training set. Sklearn provides you with "Imputer" to do this. You first need to specify an Imputer instance, that specifies that you want to replace each attributes missing values with the median of that attribute. Because the median can only be computed on numerical attributes, we need to make a copy of the data without the ocean_proximity attribute that contains text and no numbers.
# 
# While Preparing the Data for ML we will work on the following pints:
# 1. Data Cleaning
# 2. Handling Text and Categorical Data
# 3. Custom Transformers
# 4. Feature Scaling
# 5. Transformation Pipelines

# # DATA CLEANING
# • Most Machine Learning algorithms cannot work with missing features, so let’s create a few functions 
# to take care of them. 
# • You noticed earlier that the total_bedrooms attribute has some missing values, so let’s fix this. You 
# have three options:
# • Get rid of the corresponding districts.
# • Get rid of the whole attribute.
# • Set the values to some value (zero, the mean, the median, etc.).
# • You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna() methods:
# 

# In[19]:


housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) #option 3 #set the values to some value (like zero,mean or median)


# In[21]:


median = housing["total_bedrooms"].median()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

#Since the median can only be computed on numerical attributes, we need to create a copy of the data without the text attribute ocean_proximity:

housing_num = housing.drop('ocean_proximity', axis=1)


# Now we can fit the Imputer instance to the training data using the "fit()" method.

# In[22]:


imputer.fit(housing_num)


# In[23]:


imputer.statistics_


# In[25]:


housing_num.median().values


# Now you can use this “trained” imputer to transform the training set by replacing missing values by 
# the learned medians:
# 
# Now that we have trained the Imputer we can use it to impute values.
# After imputation we get the results in the form of numpy array so we have to convert it back to Pandas dataframe.

# In[26]:


X=imputer.transform(housing_num)
housing_tr=pd.DataFrame(X, columns=housing_num.columns, index = list(housing.index.values))


# In[27]:


housing_tr.head()


# <a id='categ'></a>

# # Handling Categorical Attributes
# 
# As already mentioned most of the machine learning algorithms can just work with numerical data. The ocean_proximity attribute is still a categorical feature and we will convert it now. We can use Pandas' factorize() method to convert this string categorical feature to an integer categorical feature.
# 

# In[28]:


from sklearn.preprocessing import LabelBinarizer 
# label binarization, which is the process of converting categorical labels into binary (0/1) representation. This is particularly useful when dealing with categorical target variables in classification tasks.
encoder = LabelBinarizer() # to transform categorical labels into binary representation
housingCat = housing['ocean_proximity']
housingCat1hot = encoder.fit_transform(housingCat)


# In[29]:


housingCat1hot
#this is a numpy array


# # Transformation Pipelines

# In[30]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),# to handle the missing values with median of that column attribute
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[32]:


housing_num_tr


# In[79]:


from sklearn.preprocessing import OneHotEncoder
# used to convert categorical variables into a "one-hot" or "dummy" encoded representation.
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[80]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot_encoder', OneHotEncoder(sparse=False)),
    ])


# In[81]:


# Now we combine the two pipelines
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[82]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# # Train Models
# 
# First, let's test whether a Linear Regression model gives us a satisfying result:

# In[83]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[84]:


ex = housing.iloc[:5]
ex_labels = housing_labels.iloc[:5]
ex_data_prepared = full_pipeline.transform(ex)

print("Predictions:", lin_reg.predict(ex_data_prepared))
print("Labels:",list(ex_labels))


# • It works, although the predictions are not exactly accurate (e.g., the second prediction is off by more 
# than 50%!). 
# 
# • Let’s measure this regression model’s RMSE on the whole training set using Scikit-Learn’s
# mean_squared_error function:
# 
# 
# Let's use [RMSE (Root Mean Squared Error)](http://www.statisticshowto.com/rmse/) to judge the quality of our predictions:

# In[85]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# okay ,This is better than nothing but clearly not a great score. Since most districts median_housing_values range between 120,000 and 265,000 dollar, a prediction error of $68,627 is not very satisfying and also an example of a model underfitting the data. This either means that the features do not provide  enough information to make proper predictions, or that the model is just not powerful enough.
# 
# 
# The main ways to fix underfitting are:
# 
# 1.) feed the model with better features
# 2.) select a more powerful model
# 3.) reduce the constraints on the model
# 
# 
# First let's try out a more powerful model since we just only tested one.

# Let's use a DecisionTreeRegressor, which can find complex nonlinear relationships in the data:

# In[86]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[87]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# This gives you no error at all, which means that we strongly overfitted our data. How can we be sure ? As we allready discussed earlier, you don't want to use the test set until you are confident about your model. But how can we test how our model performs if we can't use the test data ? One way to do this is using
# 
# **K-Fold Cross-Validation**, which uses part of the training set for training and a part for validation. The following code randomly splits the training set into 10 subset called **folds**. Then it trains and evaluates the Decision tree model 10 times, 
# picking a different fold for evaluation every time and training on the other 9 folds
# 
# The result is an array containing the 10 evaluation scores:

# In[88]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# Let's look at the result:

# In[89]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# • Now the Decision Tree doesn’t look as good as it did earlier. In fact, it seems to perform 
# worse than the Linear Regression model! 
# 
# • Notice that cross-validation allows you to get not only an estimate of the performance of 
# your model, but also a measure of how precise this estimate is (i.e., its standard deviation)
# 
# These values indicate that Regression tree is slightly overfitting the data. Let us try RandomForestRegressor.
# 
# Random Forests work by training many Decision Trees on random subsets of the features, then averaging out their predictions.

# In[90]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[91]:


housing_pred=forest_reg.predict(housing_prepared)
forest_mse=mean_squared_error(housing_labels, housing_pred)
forest_rmse=np.sqrt(forest_mse)
forest_rmse


# In[ ]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,\
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse= np.sqrt(-forest_scores)
display_scores(forest_rmse)


# This is a lot better. Note that the score on the training set is still much lower, than on the validation set, which indicates that the model is overfitting the training set and that we should optimize the model to solve this problem. 

# # Fine Tuning Of Parameters
#  You now need to fine-tune them. Let’s look at a few ways you can do that:
#  
# • Grid Search
# 
# • Randomized Search
# 
# • Ensemble Methods
# 
# • One way to do that would be to fiddle with the hyperparameters manually, until you find a great combination of hyperparameter values. 
# 
# • This would be very tedious work, and you may not have time to explore many combinations.
# 
# • Instead you should get Scikit-Learn’s GridSearchCV to search for you.
# 
# For example, the following code searches for the best combination of hyperparameter values for the RandomForestRegressor:
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[ ]:


grid_search.best_params_


# This tells us that the best solution would be by setting the max_features to 8 and the n_estimators to 30.

# # Evaluation
# 
# Now that we have our best parameter values, we can fit a final model using these.

# In[ ]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[ ]:


final_rmse


# We now have a final prediciton error of $47,873.

# <a id='summary'></a>

# # EVALUATE YOUR SYSTEM ON THE TEST SET
# • The performance will usually be slightly worse than what you measured using 
# crossvalidation if you did a lot of hyperparameter tuning (because your system ends up 
# fine-tuned to perform well on the validation data, and will likely not perform as well on 
# unknown datasets). 
# 
# • It is not the case in this example, but when this happens you must resist the temptation to 
# tweak the hyperparameters to make the numbers look good on the test set; the 
# improvements would be unlikely to generalize to new data.
# 
# • Now comes the project prelaunch phase: you need to present your solution (highlighting 
# what you have learned, what worked and what did not, what assumptions were made, and 
# what your system’s limitations are), document everything, and create nice presentations 
# with clear visualizations and easy-to-remember statements (e.g., “the median income is the 
# number one predictor of housing prices”).
# 
# 
# 

# # LAUNCH, MONITOR, AND MAINTAIN YOUR SYSTEM
# 
# • Perfect, you got approval to launch! You need to get your solution ready for production, in 
# particular by plugging the production input data sources into your system and writing tests.
# 
# • You also need to write monitoring code to check your system’s live performance at regular 
# intervals and trigger alerts when it drops. 
# 
# • This is important to catch not only sudden breakage, but also performance degradation.
# 
# • This is quite common because models tend to “rot” as data evolves over time, unless the 
# models are regularly trained on fresh data.
# 
# • Evaluating your system’s performance will require sampling the system’s predictions and 
# evaluating them. 
# 
# • This will generally require a human analysis. These analysts may be field experts, or 
# workers on a crowdsourcing platform (such as Amazon Mechanical Turk or CrowdFlower). 
# 
# • Either way, you need to plug the human evaluation pipeline into your system.

# In[ ]:




