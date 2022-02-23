import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

raw_data = pd.read_csv('1.04. Real-life example.csv')

# ## Preprocessing
raw_data.describe(include='all')
data = raw_data.drop(['Model'],axis=1)
data.describe(include='all')
# Dropped the Model category because it is insignificant to my model.

data.isnull().sum()
# This shows the missing values in my dataset
data_no_mv = data.dropna(axis=0)
# I drop the missing values here, which is acceptable because it is less than 5% of the observations.
data_no_mv.describe(include='all')

# ### PDFs
# #### Here I check the Probability Distribution Functions (PDF) of the Independent Variables Price, Year, Mileage, and Engine Volume to identify and weed out the Outliers. They can adversely affect the accuracy of my Regression model because a Regression attempts to draw a line closest to all the data; including the Outliers that might inflate/deflate my model.

sns.distplot(data_no_mv['Price'])
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')
# I decided to exclude the observations in the 99th percentile and above to get rid of the Outliers.
sns.distplot(data_1['Price'])
# Now the Price variable only includes observations up to the 98th percentile and has much fewer Outliers.

sns.distplot(data_no_mv['Mileage'])
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
# Similar to the Price variable, I decided to exclude the observations in the 99th percentile and beyond to remove the Outliers.
sns.distplot(data_2['Mileage'])

sns.distplot(data_no_mv['EngineV'])
# The PDF looks unusual compared to the previous two.
data_3 = data_2[data_2['EngineV']<6.6]
# After research, I found out that the normal interval of the Engine Volume falls between 06. to 6.5.
# The observations beyond 6.5 are mostly 99.99 - a variable that was used in the past to label missing values. It is a bad idea to label missing values in this manner in practice.
# I decided to remove such observations as they are Outliers.
sns.distplot(data_3['EngineV'])

sns.distplot(data_no_mv['Year'])
# Most cars are newer but there are a few vintage cars in the variable.
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
# I decided to remove the 1st percentile and keep the rest to remove the Outliers.
sns.distplot(data_4['Year'])

data_cleaned = data_4.reset_index(drop=True)
#I reset the index of the preprocessed data to completely forget the old index.
data_cleaned.describe(include='all')
# This excludes ~250 problematic observations that could've hindered the accuracy of my model, if left uncleaned.


# ## Checking the OLS assumptions

# ### Distribution
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()
# These are not linear regressions and show that I should first transform one or more variables before running the Regression.

sns.distplot(data_cleaned['Price'])
#Here I checked the distribution of the dependent variable Price.
log_price = np.log(data_cleaned['Price'])
# Here I used the log transformation to fix heteroscedasticity and remove Outliers from the dependent variable.
data_cleaned['log_price'] = log_price

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')
plt.show()
# After using the log transformation on Price, the PDFs now have a linear regression.

data_cleaned = data_cleaned.drop(['Price'],axis=1)
# Here I dropped the variable Price and replaced it with log_Price, after discovering that Price had no significance to my model anymore.

# ### Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
# With Statsmodels, I used the Variance Inflation Factor here to check for multicollinearity in my variables.
# While I expect multicollinearity in my data, I wanted to check the variables that introduce unacceptable correlation to my model; these variables have high VIFs.
vif

data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
# Dropped 'Year' because it has an unacceptably high VIF and is therefore a feature that introduces correlation in my data
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
# This identifies categorical variables and creates dummies automatically to avoid multicollinearity in my Model
data_with_dummies.head()

data_with_dummies.columns.values
cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()

# ## Training my Model
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)
# I removed log_price from the inputs to exclude the transformed dependent variable.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
# This standardizes my inputs; in other words, it subtracts the mean and divide by the standard deviation from each observation.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
# I did this to avoid overfitting my model to my data.
# The default setting of the train-test split is 75-25, but here I chose 80-20.
# I used 'random_state' to ensure that I get the same random shuffle every time I split my data.

reg = LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)
plt.xlabel('Targets(y_train)', size=20)
plt.ylabel('Predictions(y_hat)', size=20)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

# ### Using Residuals to check the Model
sns.distplot(y_train-y_hat)
plt.title("Residuals PDF", size=20)
#to check whether the Residuals are normally distributed and the variability of the outcome

reg.score(x_train, y_train)
reg.intercept_
# The intercept or bias calibrates the model: without it, each feature will be off the mark.
reg.coef_
# The coefficient or weight of each feature determines the significance of a feature to the model.
# A feature with a coefficient of 0 means that it has no significance to the model.
reg_summary=pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary

# To know the categorical variables of my features
data_cleaned['Engine Type'].unique()
data_cleaned['Brand'].unique()
data_cleaned['Body'].unique()
data_cleaned['Registration'].unique()

# ## Testing my Model
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets(y_test)', size=20)
plt.ylabel('Predictions(y_hat_test)', size=20)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
# This returns the exponential of Y Hat Test and removes the log. This is done to get the actual price.
y_test = y_test.reset_index(drop=True)
y_test.head()
df_pf['Target'] = np.exp(y_test)
df_pf

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
# This shows the residuals and the percentage difference between the actual price and the predicted price.

pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%2f' % x)
df_pf.sort_values(by=['Difference%'])
# This shows the difference in percentage of the prediction and the target using the test data.
# I included the Residuals because examining them is the same as examining the heart of the algorithm.
