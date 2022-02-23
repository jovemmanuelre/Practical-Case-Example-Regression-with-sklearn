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

# In[211]:


sns.distplot(data_no_mv['Price'])


# In[212]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')
# I decided to exclude the observations in the 99th percentile and above to get rid of the Outliers.


# In[213]:


sns.distplot(data_1['Price'])
# Now the Price variable only includes observations up to the 98th percentile and has much fewer Outliers.


# In[214]:


sns.distplot(data_no_mv['Mileage'])


# In[215]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
# Similar to the Price variable, I decided to exclude the observations in the 99th percentile and beyond to remove the Outliers.


# In[216]:


sns.distplot(data_2['Mileage'])


# In[217]:


sns.distplot(data_no_mv['EngineV'])
# The PDF looks unusual compared to the previous two.


# In[218]:


data_3 = data_2[data_2['EngineV']<6.6]
# After research, I found out that the normal interval of the Engine Volume falls between 06. to 6.5.
# The observations beyond 6.5 are mostly 99.99 - a variable that was used in the past to label missing values. It is a bad idea to label missing values in this manner now.
# I decided to remove such observations as they are Outliers.


# In[219]:


sns.distplot(data_3['EngineV'])


# In[220]:


sns.distplot(data_no_mv['Year'])
# Most cars are newer but there are a few vintage cars in the variable.


# In[221]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
# I decided to remove the 1st percentile and keep the rest


# In[222]:


sns.distplot(data_4['Year'])


# In[223]:


data_cleaned = data_4.reset_index(drop=True)
#I reset the index to completely forget the old index.


# In[224]:


data_cleaned.describe(include='all')
# This excludes ~250 problematic observations that could've hindered the accuracy of my model if left unchecked.


# ## Checking the OLS assumptions

# ### Distribution

# In[225]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()
# These are not linear regressions and shows that I should first transform one or more variables to run the Regression.


# In[226]:


sns.distplot(data_cleaned['Price'])
#Here I check the distribution of the dependent variable Price.


# In[227]:


log_price = np.log(data_cleaned['Price'])
# Here I used the log transformation to fix heteroscedasticity and remove outliers from the variable Price.
data_cleaned['log_price'] = log_price
data_cleaned


# In[228]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')
plt.show()
# After performing the log transformation on Price, the PDFs now show a linear regression line.


# In[229]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)
# Here I dropped the variable Price and replaced it with log_Price because the former has no statistical significance to my model.


# ### Multicollinearity

# In[230]:


data_cleaned.columns.values


# In[231]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
# Through Statsmodels, I used the Variance Inflation Factor here to check for multicollinearity in my variables.
# While I expect multicollinearity in my data, I wanted to check the variables the introduce unnacceptable correlation to my model; they have high VIFs.


# In[232]:


vif


# In[233]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
# Dropped 'Year' because it has an unacceptably high VIF and is therefore a feature that introduces correlation in my data


# In[234]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
# This identifies categorical variables and creates dummies automatically to avoid multicollinearity in my Model


# In[235]:


data_with_dummies.head()


# In[236]:


data_with_dummies.columns.values


# In[237]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[238]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()
# Here I arranged the data into a table.


# ## Training my Regression Model
# 

# In[239]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)
# I removed log_price in the inputs to exclude the transformed dependent variable from my inputs.


# In[240]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
# This standardizes my inputs; in other words, it subtractrs the mean and divide by the standard deviation from each observation.


# In[241]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
# I did this to avoid overfitting my model to my data.
# The default setting of the train-test split is 75-25, but here I chose 80-20.
# I used 'random_state' to ensure that I get the same random shuffle every time I split my data.


# In[242]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[243]:


y_hat = reg.predict(x_train)


# In[244]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets(y_train)', size=20)
plt.ylabel('Predictions(y_hat)', size=20)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# ### Using Residuals to check the Model

# In[245]:


sns.distplot(y_train-y_hat)
plt.title("Residuals PDF", size=20)
#to check whether the Residuals is normally distributed and the variability of the outcome


# In[246]:


reg.score(x_train, y_train)


# In[247]:


reg.intercept_
# The intercept or bias calibrates the model: without it, each feature will be off the mark.


# In[248]:


reg.coef_


# In[249]:


reg_summary=pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary

# A feature with a coefficient of 0 means that it has no significance to the model.


# In[250]:


#to know the categorical variables of my features
data_cleaned['Brand'].unique()


# In[251]:


data_cleaned['Body'].unique()


# In[252]:


data_cleaned['Engine Type'].unique()


# In[253]:


data_cleaned['Registration'].unique()


# ## Testing my Model

# In[254]:


y_hat_test = reg.predict(x_test)


# In[255]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets(y_test)', size=20)
plt.ylabel('Predictions(y_hat_test)', size=20)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()


# In[256]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
#this returns the exponential of Y Hat Test and removes the log. 
df_pf.head()


# In[257]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[258]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[259]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[260]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%2f' % x)
df_pf.sort_values(by=['Difference%'])
# This table shows the difference in percetage of the prediction and the target using the test data.
# I included the Residuals because examining them as the same as examining the heart of the alogirthm.


# In[ ]:




